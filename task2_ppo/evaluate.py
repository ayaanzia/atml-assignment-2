"""Fixed held-out generation and scoring for every Task 2 condition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, print_gpu_info, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.runtime import (
    generate_one, parameter_audit, response_logps, reward_input_length, stable_seed,
)


def _adapter_digest(adapter: str) -> str:
    directory = repo_path(adapter)
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / filename
        if candidate.exists():
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
    raise FileNotFoundError(f"Missing saved PPO policy adapter in {directory}")


def _existing_records(path: Path, rows: list[dict], seed: int) -> dict[int, dict]:
    if not path.exists():
        return {}
    records = read_jsonl(path)
    indexed = {int(record["file_position"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate held-out PPO evaluation file positions")
    for pos, record in indexed.items():
        if pos < 0 or pos >= len(rows) or record["prompt_id"] != rows[pos]["prompt_id"] or \
                record["seed"] != stable_seed(seed, f"ppo-eval-{rows[pos]['prompt_id']}"):
            raise ValueError("Saved held-out record does not match fixed prompt or seed")
    return indexed


def _summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No held-out PPO records")
    usable = sum(row["usable_response_tokens"] for row in records)
    if usable == 0:
        raise ValueError("Held-out responses have no usable tokens")
    lengths = [row["response_tokens"] for row in records]
    return {
        "prompts": len(records),
        "mean_raw_reward": statistics.fmean(row["raw_reward"] for row in records),
        "std_raw_reward": statistics.pstdev(row["raw_reward"] for row in records),
        "sampled_kl_token_mean": sum(row["sampled_kl_sum"] for row in records) / usable,
        "token_entropy_mean": sum(row["entropy_sum"] for row in records) / usable,
        "total_usable_response_tokens": usable,
        "mean_response_tokens": statistics.fmean(lengths),
        "std_response_tokens": statistics.pstdev(lengths),
        "generated_response_tokens": sum(lengths),
        "eos_rate": statistics.fmean(float(row["terminated_with_eos"]) for row in records),
        "generation_cap_rate": statistics.fmean(float(row["truncated_at_generation_cap"]) for row in records),
        "prompt_truncated_count": sum(bool(row["prompt_truncated"]) for row in records),
        "reward_input_truncated_count": sum(bool(row["reward_input_truncated"]) for row in records),
    }


def evaluate_run(config_path: str, adapter: str, name: str, resume: bool = False) -> dict:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a simple directory name")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 held-out evaluation requires a CUDA GPU")
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    adapter_hash = _adapter_digest(adapter)
    payload = {"cfg": cfg, "adapter_digest": adapter_hash,
               "prompt_ids": [row["prompt_id"] for row in rows]}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    result_dir = repo_path(cfg["results_dir"]) / name / "heldout_eval"
    manifest_path = result_dir / "evaluation_manifest.json"
    record_path = result_dir / "generations.jsonl"
    summary_path = result_dir / "summary.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"No evaluation manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Saved held-out evaluation differs in config, adapter, or prompt set")
    elif any(path.exists() for path in (manifest_path, record_path, summary_path)):
        raise FileExistsError(f"Evaluation output exists in {result_dir}; use --resume")
    else:
        result_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest_path, {
            "signature": signature, "adapter": str(repo_path(adapter)),
            "adapter_sha256": adapter_hash, "prompt_ids": [row["prompt_id"] for row in rows],
            "seed": cfg["seed"], "generation": cfg["generation"],
            "prompt_cap": cfg["max_prompt_length"],
            "response_cap": cfg["eval_max_response_length"],
            "reward_input_cap": cfg["reward_max_length"],
            "reward_input_truncation_side": "left",
            "prompt_truncation": "trim older user content; preserve chat wrapper and assistant suffix",
        })
    existing = _existing_records(record_path, rows, int(cfg["seed"]))
    previously_scored = len(existing)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward, reward_tokenizer = load_reward_model(cfg)
    reward_tokenizer.truncation_side = "left"
    model_audit = [parameter_audit(policy, "policy"), parameter_audit(reward, "reward_model")]
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    generation_seconds, scoring_seconds = 0.0, 0.0
    try:
        policy.eval()
        for pos, row in enumerate(tqdm(rows, desc=f"PPO {name} held-out evaluation", unit="prompt")):
            if pos in existing:
                continue
            seed = stable_seed(int(cfg["seed"]), f"ppo-eval-{row['prompt_id']}")
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = generate_one(policy, tokenizer, row, cfg, seed,
                                     int(cfg["eval_max_response_length"]))
            torch.cuda.synchronize()
            generation_seconds += time.perf_counter() - start
            mask = generated["response_mask"].float()
            usable = int(mask.sum().item())
            if usable == 0:
                raise ValueError(f"No usable generated tokens for {row['prompt_id']}")
            start = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                policy_logp, entropy_tokens = response_logps(policy, generated, entropy=True)
                with reference_mode(policy):
                    reference_logp = response_logps(policy, generated)[0]
                reward_score = score_reward_pairs(
                    reward, reward_tokenizer, [generated["messages"]], generated["responses"],
                    max_length=int(cfg["reward_max_length"]),
                ).float()
            kl_sum = float(((policy_logp - reference_logp) * mask).sum().item())
            entropy_sum = float((entropy_tokens * mask).sum().item())
            reward_tokens = reward_input_length(
                reward_tokenizer, generated["messages"], generated["responses"][0]
            )
            record = {
                "file_position": pos, "prompt_id": row["prompt_id"],
                "source_index": row["source_index"], "seed": seed,
                "prompt_tokens_full": generated["prompt_tokens_full"],
                "prompt_tokens_used": generated["prompt_tokens_used"],
                "prompt_truncated": generated["prompt_truncated"],
                "response": generated["responses"][0],
                "response_tokens": int(generated["response_lengths"][0]),
                "usable_response_tokens": usable,
                "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
                "truncated_at_generation_cap": bool(generated["truncated"][0]),
                "raw_reward": float(reward_score[0].item()),
                "reward_input_tokens_full": reward_tokens,
                "reward_input_truncated": reward_tokens > int(cfg["reward_max_length"]),
                "reward_input_truncation_side": "left",
                "sampled_kl_sum": kl_sum,
                "sampled_kl_token_mean": kl_sum / usable,
                "entropy_sum": entropy_sum,
                "token_entropy_mean": entropy_sum / usable,
            }
            if not all(math.isfinite(record[key]) for key in
                       ("raw_reward", "sampled_kl_sum", "entropy_sum")):
                raise FloatingPointError(f"Nonfinite held-out score for {row['prompt_id']}")
            append_jsonl(record_path, record)
            existing[pos] = record
            torch.cuda.synchronize()
            scoring_seconds += time.perf_counter() - start
            del generated, mask, policy_logp, entropy_tokens, reference_logp, reward_score
        if len(existing) != len(rows):
            raise RuntimeError("Held-out PPO evaluation did not cover every fixed prompt")
        summary = {
            "name": name, "adapter": str(repo_path(adapter)), "adapter_sha256": adapter_hash,
            "metrics": _summarize(list(existing.values())), "models": model_audit,
            "autocast_dtype": "torch.float16",
            "prompt_truncation": "trim older user content; preserve chat wrapper and assistant suffix",
            "model_loading_seconds": round(model_load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "scoring_seconds": round(scoring_seconds, 3),
            "new_records_this_invocation": len(existing) - previously_scored,
            "timing_note": "stage seconds cover newly processed records in this invocation; cached records are excluded",
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        save_json(summary_path, summary)
        print(f"Completed PPO {name} held-out evaluation: {len(rows)} prompts; "
              f"summary={summary_path}", flush=True)
        return summary
    finally:
        del policy, reward
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", default="standard")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.config, args.adapter, args.name, args.resume)


if __name__ == "__main__":
    main()
