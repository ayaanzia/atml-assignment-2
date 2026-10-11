"""Fixed 200-prompt GRPO held-out generation and scoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, print_gpu_info, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.runtime import (
    evaluation_seed, generate_one, parameter_audit, response_logps,
    reward_input_length, sync_time,
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
    raise FileNotFoundError(f"No saved GRPO adapter in {directory}")


def load_evaluation_bundle(config_path: str, adapter: str, cfg: dict | None = None) -> dict:
    cfg = load_yaml(config_path) if cfg is None else cfg
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward, reward_tokenizer = load_reward_model(cfg)
    reward_tokenizer.truncation_side = "left"
    policy.eval()
    return {
        "cfg": cfg, "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": tokenizer, "policy": policy,
        "reward_model": reward, "reward_tokenizer": reward_tokenizer,
        "model_audit": [parameter_audit(policy, "GRPO policy"),
                        parameter_audit(reward, "frozen reward model")],
    }


def _existing_records(path: Path, rows: list[dict], seed: int) -> dict[int, dict]:
    if not path.exists():
        return {}
    records = read_jsonl(path)
    indexed = {int(record["file_position"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate GRPO held-out record positions")
    for position, record in indexed.items():
        if position < 0 or position >= len(rows) or \
                record["prompt_id"] != rows[position]["prompt_id"] or \
                record["seed"] != evaluation_seed(seed, rows[position]["prompt_id"]):
            raise ValueError("Saved GRPO held-out record has a different prompt or seed")
    return indexed


def _summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No GRPO held-out records")
    usable = sum(row["usable_response_tokens"] for row in records)
    if usable < 1:
        raise ValueError("Held-out GRPO responses have no valid tokens")
    lengths = [row["response_tokens"] for row in records]
    return {
        "prompts": len(records),
        "mean_raw_reward": statistics.fmean(row["raw_reward"] for row in records),
        "std_raw_reward": statistics.pstdev(row["raw_reward"] for row in records),
        "sampled_kl_token_mean": sum(row["sampled_kl_sum"] for row in records) / usable,
        "token_entropy_mean": sum(row["entropy_sum"] for row in records) / usable,
        "total_usable_response_tokens": usable,
        "generated_response_tokens": sum(lengths),
        "mean_response_tokens": statistics.fmean(lengths),
        "std_response_tokens": statistics.pstdev(lengths),
        "eos_rate": statistics.fmean(float(row["terminated_with_eos"]) for row in records),
        "generation_cap_rate": statistics.fmean(float(row["truncated_at_generation_cap"]) for row in records),
        "prompt_truncated_count": sum(bool(row["prompt_truncated"]) for row in records),
        "reward_input_truncated_count": sum(bool(row["reward_input_truncated"]) for row in records),
    }


def evaluate_run(config_path: str, adapter: str, name: str = "standard",
                 resume: bool = False) -> dict:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a simple directory name")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 3 held-out evaluation requires CUDA")
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if len(rows) != 200 or len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Task 3 expects the fixed 200-row held-out prompt pool with unique IDs")
    adapter_hash = _adapter_digest(adapter)
    payload = {"config": cfg, "adapter_digest": adapter_hash,
               "prompt_ids": [row["prompt_id"] for row in rows]}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    result_dir = repo_path(cfg["results_dir"]) / name / "heldout_eval"
    manifest_path = result_dir / "evaluation_manifest.json"
    record_path = result_dir / "generations.jsonl"
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        if resume:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("signature") != signature:
                raise ValueError("Completed GRPO evaluation used different settings or adapter")
            return summary
        raise FileExistsError(f"GRPO evaluation output exists in {result_dir}; use --resume")
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"No GRPO evaluation manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise ValueError("GRPO evaluation resume settings or adapter differ")
    else:
        if manifest_path.exists() or record_path.exists():
            raise FileExistsError(f"GRPO evaluation output exists in {result_dir}; use --resume")
        result_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest_path, {
            "signature": signature, "adapter": str(repo_path(adapter)),
            "adapter_sha256": adapter_hash,
            "prompt_ids": [row["prompt_id"] for row in rows],
            "seed": cfg["seed"], "generation": cfg["generation"],
            "prompt_cap": cfg["max_prompt_length"],
            "response_cap": cfg["eval_max_completion_length"],
            "reward_input_cap": cfg["reward_max_length"],
            "prompt_truncation": "trim older user content; preserve chat wrapper and assistant suffix",
            "reward_input_truncation_side": "left",
        })
    existing = _existing_records(record_path, rows, int(cfg["seed"]))
    previous = len(existing)
    torch.cuda.reset_peak_memory_stats()
    load_start = sync_time()
    bundle = load_evaluation_bundle(config_path, adapter, cfg)
    load_seconds = sync_time() - load_start
    generation_seconds = scoring_seconds = 0.0
    try:
        policy = bundle["policy"]
        for position, row in enumerate(tqdm(rows, desc=f"GRPO {name} held-out evaluation",
                                            unit="prompt")):
            if position in existing:
                continue
            seed = evaluation_seed(int(cfg["seed"]), row["prompt_id"])
            start = sync_time()
            generated = generate_one(policy, bundle["tokenizer"], row, cfg, seed,
                                     int(cfg["eval_max_completion_length"]))
            generation_seconds += sync_time() - start
            mask = generated["response_mask"].float()
            usable = int(mask.sum().item())
            if usable == 0:
                raise ValueError(f"No GRPO held-out response tokens for {row['prompt_id']}")
            start = sync_time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                policy_logp, entropy_tokens = response_logps(policy, generated, entropy=True)
                with reference_mode(policy):
                    reference_logp = response_logps(policy, generated)[0]
                reward_score = score_reward_pairs(
                    bundle["reward_model"], bundle["reward_tokenizer"],
                    [generated["messages"]], generated["responses"],
                    max_length=int(cfg["reward_max_length"]),
                ).float()
            reward_tokens = reward_input_length(
                bundle["reward_tokenizer"], generated["messages"], generated["responses"][0]
            )
            kl_sum = float(((policy_logp - reference_logp) * mask).sum().item())
            entropy_sum = float((entropy_tokens * mask).sum().item())
            record = {
                "file_position": position, "prompt_id": row["prompt_id"],
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
                "sampled_kl_sum": kl_sum,
                "sampled_kl_token_mean": kl_sum / usable,
                "entropy_sum": entropy_sum,
                "token_entropy_mean": entropy_sum / usable,
            }
            if not all(math.isfinite(record[key]) for key in
                       ("raw_reward", "sampled_kl_sum", "entropy_sum")):
                raise FloatingPointError(f"Nonfinite GRPO held-out score for {row['prompt_id']}")
            append_jsonl(record_path, record)
            existing[position] = record
            scoring_seconds += sync_time() - start
            del generated, mask, policy_logp, entropy_tokens, reference_logp, reward_score
        if len(existing) != len(rows):
            raise RuntimeError("GRPO held-out evaluation did not cover the fixed pool")
        summary = {
            "name": name, "signature": signature,
            "adapter": str(repo_path(adapter)), "adapter_sha256": adapter_hash,
            "metrics": _summarize(list(existing.values())),
            "models": bundle["model_audit"], "autocast_dtype": "torch.float16",
            "prompt_truncation": "trim older user content; preserve chat wrapper and assistant suffix",
            "model_loading_seconds": round(load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "scoring_seconds": round(scoring_seconds, 3),
            "new_records_this_invocation": len(existing) - previous,
            "timing_note": "phase seconds cover newly processed records; resumed records are excluded",
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        save_json(summary_path, summary)
        print(f"Completed GRPO {name} held-out evaluation: {len(rows)} prompts", flush=True)
        return summary
    finally:
        del bundle
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", default="standard")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.config, args.adapter, args.name, args.resume)


if __name__ == "__main__":
    main()
