"""Held-out preference, generation, reward, and length evaluation for Task 1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import (
    chat_template_token_ids, load_yaml, pad_batch, prompt_messages,
    prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl,
)
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, print_gpu_info, save_json, set_seed
from common.metrics import sampled_kl, word_count, word_limit_compliance, parse_word_limit
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.data import prepare_preference_examples, seed_for_prompt, selection_summary
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import sequence_logps


def _adapter_digest(adapter: str) -> str:
    path = repo_path(adapter)
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / filename
        if candidate.exists():
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
    raise FileNotFoundError(f"No saved PEFT adapter weights in {path}")


def _existing_records(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    indexed = {int(row["file_position"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate file positions in {path}")
    return indexed


def _pair_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No held-out preference pairs were scored")
    output = {
        "pairs": len(rows),
        "mean_dpo_loss": statistics.fmean(row["dpo_loss"] for row in rows),
        "preference_accuracy": statistics.fmean(float(row["preference_correct"]) for row in rows),
    }
    strata = defaultdict(list)
    for row in rows:
        if row["length_stratum"] is not None:
            strata[row["length_stratum"]].append(row)
    if strata:
        output["strata"] = {
            key: {
                "pairs": len(items),
                "mean_dpo_loss": statistics.fmean(item["dpo_loss"] for item in items),
                "preference_accuracy": statistics.fmean(float(item["preference_correct"]) for item in items),
            }
            for key, items in sorted(strata.items())
        }
    return output


def _generation_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No generated responses were scored")
    lengths = [row["response_tokens"] for row in rows]
    total_tokens = sum(row["usable_response_tokens"] for row in rows)
    if total_tokens == 0:
        raise ValueError("Generated responses have no usable tokens for sampled KL")
    return {
        "prompts": len(rows),
        "mean_reward_model_score": statistics.fmean(row["reward_model_score"] for row in rows),
        "sampled_kl_token_mean": sum(row["sampled_kl_sum"] for row in rows) / total_tokens,
        "sampled_kl_averaging": "sum(policy minus reference logp over response tokens) / total response tokens",
        "mean_response_tokens": statistics.fmean(lengths),
        "std_response_tokens": statistics.pstdev(lengths),
        "total_response_tokens": total_tokens,
        "max_response_tokens": max(lengths),
        "reward_input_truncated_count": sum(row["reward_input_truncated"] for row in rows),
    }


def _word_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Word-limit prompt results are empty")
    return {
        "prompts": len(rows),
        "word_limit_compliance": statistics.fmean(row["compliant"] for row in rows),
        "mean_response_tokens": statistics.fmean(row["response_tokens"] for row in rows),
        "std_response_tokens": statistics.pstdev(row["response_tokens"] for row in rows),
        "mean_response_words": statistics.fmean(row["response_words"] for row in rows),
    }


def _generated(model, tokenizer, messages: list[dict], cfg: dict, prompt_id: str,
               max_prompt_tokens: int) -> dict:
    seed = seed_for_prompt(int(cfg["seed"]), prompt_id)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        generated = batch_generate(
            model, tokenizer, [messages], max_prompt_tokens, int(cfg["max_generation_tokens"]),
            **cfg["generation"],
        )
    prompt_tokens = len(chat_template_token_ids(tokenizer, messages))
    if int(generated["prompt_width"]) < prompt_tokens:
        raise ValueError(f"Generation truncated fixed prompt {prompt_id}")
    generated["seed"] = seed
    generated["prompt_tokens"] = prompt_tokens
    return generated


def _generated_logps(model, generated: dict) -> torch.Tensor:
    result = response_token_logprobs(
        model,
        generated["sequences"],
        generated["attention_mask"],
        generated["prompt_width"],
        generated["response_ids"],
    )
    selected = result[0].float()
    del result
    return selected


def evaluate_run(config_path: str, adapter: str, name: str, dataset: str = "standard",
                 word_limit: bool = False, resume: bool = False) -> dict:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a simple directory name")
    if dataset not in {"standard", "length"}:
        raise ValueError("dataset must be standard or length")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 1 evaluation requires a CUDA GPU")
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    data_key = "dpo_standard_eval" if dataset == "standard" else "dpo_length_eval"
    data_path = cfg["paths"][data_key]
    rows = read_jsonl(data_path)
    prompt_ids = [{"file_position": i, "prompt_id": row["prompt_id"], "source_index": row["source_index"]}
                  for i, row in enumerate(rows)]
    word_rows = read_jsonl(cfg["paths"]["word_limit_prompts"]) if word_limit else []
    train_manifest_path = repo_path(cfg["results_dir"]) / name / "run_config.json"
    trained_beta = float(cfg["beta"])
    if train_manifest_path.exists():
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        trained_beta = float(train_manifest["beta"])
    adapter_hash = _adapter_digest(adapter)
    signature_payload = {
        "config": cfg, "adapter_digest": adapter_hash, "dataset": str(repo_path(data_path)),
        "prompt_ids": prompt_ids, "word_limit_ids": [row["prompt_id"] for row in word_rows],
        "beta": trained_beta,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
    result_dir = repo_path(cfg["results_dir"]) / name / f"{dataset}_eval"
    manifest_path = result_dir / "evaluation_manifest.json"
    pair_path = result_dir / "pair_scores.jsonl"
    generation_path = result_dir / "generations.jsonl"
    word_path = result_dir / "word_limit.jsonl"
    summary_path = result_dir / "summary.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"No evaluation manifest at {manifest_path}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Saved evaluation does not match adapter, configuration, or prompt IDs")
    elif any(path.exists() for path in (manifest_path, pair_path, generation_path, word_path, summary_path)):
        raise FileExistsError(f"Evaluation output exists in {result_dir}; use --resume")
    else:
        result_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest_path, {
            "signature": signature, "name": name, "dataset": dataset,
            "adapter": str(repo_path(adapter)), "adapter_sha256": adapter_hash,
            "pair_file": str(repo_path(data_path)), "prompt_ids": prompt_ids,
            "word_limit_ids": [row["prompt_id"] for row in word_rows],
            "beta": trained_beta, "seed": cfg["seed"], "generation": cfg["generation"],
            "max_generation_tokens": cfg["max_generation_tokens"],
            "sampled_kl_averaging": "response-token mean",
        })

    tokenizer = load_tokenizer(cfg["base_model"])
    prepared, excluded = prepare_preference_examples(rows, tokenizer, int(cfg["max_sequence_length"]))
    selection = selection_summary(prepared, excluded)
    write_jsonl(result_dir / "excluded_pairs.jsonl", excluded)
    pair_records = _existing_records(pair_path)
    generation_records = _existing_records(generation_path)
    word_records = _existing_records(word_path) if word_limit else {}
    if any(pos >= len(rows) or pos < 0 for pos in pair_records) or any(
        pos >= len(rows) or pos < 0 for pos in generation_records
    ):
        raise ValueError("Saved pair or generation file position is outside this dataset")
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.eval()
    policy_dtypes = dict(Counter(str(param.dtype) for param in policy.parameters()))
    policy_devices = dict(Counter(str(param.device) for param in policy.parameters()))
    model_load_seconds = time.perf_counter() - load_start
    pair_start = time.perf_counter()
    try:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            for item in tqdm(prepared, desc=f"DPO {name} {dataset} pairs", unit="pair"):
                pos = item["file_position"]
                if pos in pair_records:
                    continue
                chosen = {key: val.cuda() for key, val in pad_batch(tokenizer, [item["chosen"]]).items()}
                rejected = {key: val.cuda() for key, val in pad_batch(tokenizer, [item["rejected"]]).items()}
                with reference_mode(policy):
                    rc = sequence_logps(policy, chosen)
                    rr = sequence_logps(policy, rejected)
                pc = sequence_logps(policy, chosen)
                pr = sequence_logps(policy, rejected)
                loss, diag = dpo_loss(pc, pr, rc, rr, trained_beta)
                margin = (pc - pr) - (rc - rr)
                record = {
                    "file_position": pos, "source_index": item["source_index"],
                    "prompt_id": item["prompt_id"], "length_stratum": item["length_stratum"],
                    "prompt_tokens": item["prompt_tokens"],
                    "chosen_tokens_used": item["chosen_tokens_used"],
                    "rejected_tokens_used": item["rejected_tokens_used"],
                    "chosen_truncated": item["chosen_truncated"],
                    "rejected_truncated": item["rejected_truncated"],
                    "policy_chosen_logp": float(pc.item()), "policy_rejected_logp": float(pr.item()),
                    "reference_chosen_logp": float(rc.item()), "reference_rejected_logp": float(rr.item()),
                    "reference_adjusted_margin": float(margin.item()),
                    "preference_correct": bool((margin > 0).item()),
                    "dpo_loss": float(loss.item()),
                    "beta": trained_beta,
                }
                if not all(math.isfinite(record[key]) for key in (
                    "policy_chosen_logp", "policy_rejected_logp",
                    "reference_chosen_logp", "reference_rejected_logp",
                    "reference_adjusted_margin", "dpo_loss"
                )):
                    raise FloatingPointError(f"Nonfinite pair score for {item['prompt_id']}")
                append_jsonl(pair_path, record)
                pair_records[pos] = record
                del chosen, rejected, rc, rr, pc, pr, loss, diag, margin
        pair_seconds = time.perf_counter() - pair_start

        # Use complete fixed prompts for generation; the 768-token cap applies to
        # teacher-forced preference pairs. The generated prompt cap is recorded.
        max_prompt_tokens = max(
            len(chat_template_token_ids(tokenizer, prompt_messages_from_preference(row)))
            for row in rows
        ) + 16
        context_limit = getattr(policy.config, "max_position_embeddings", None)
        if context_limit and max_prompt_tokens + int(cfg["max_generation_tokens"]) > int(context_limit):
            raise ValueError("A fixed generation prompt plus response cap exceeds model context")
        reward_start = time.perf_counter()
        reward, reward_tokenizer = load_reward_model(cfg)
        reward_dtypes = dict(Counter(str(param.dtype) for param in reward.parameters()))
        reward_devices = dict(Counter(str(param.device) for param in reward.parameters()))
        reward_8bit = bool(getattr(reward, "is_loaded_in_8bit", False))
        reward_load_seconds = time.perf_counter() - reward_start
        generation_start = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            for pos, row in enumerate(tqdm(rows, desc=f"DPO {name} {dataset} generation", unit="prompt")):
                if pos in generation_records:
                    continue
                messages = prompt_messages_from_preference(row)
                generated = _generated(policy, tokenizer, messages, cfg, row["prompt_id"], max_prompt_tokens)
                policy_logp = _generated_logps(policy, generated)
                with reference_mode(policy):
                    ref_logp = _generated_logps(policy, generated)
                mask = generated["response_mask"]
                usable = int(mask.sum().item())
                if usable == 0:
                    raise ValueError(f"No usable response tokens for prompt {row['prompt_id']}")
                kl = sampled_kl(policy_logp, ref_logp, mask)
                reward_score = score_reward_pairs(
                    reward, reward_tokenizer, [messages], generated["responses"], max_length=1024
                )
                reward_text = reward_tokenizer.apply_chat_template(
                    list(messages) + [{"role": "assistant", "content": generated["responses"][0]}],
                    tokenize=False, add_generation_prompt=False,
                )
                reward_tokens = len(reward_tokenizer(reward_text, truncation=False)["input_ids"])
                record = {
                    "file_position": pos, "source_index": row["source_index"],
                    "prompt_id": row["prompt_id"], "seed": generated["seed"],
                    "prompt_tokens": generated["prompt_tokens"],
                    "response": generated["responses"][0],
                    "response_tokens": int(generated["response_lengths"][0]),
                    "usable_response_tokens": usable,
                    "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
                    "truncated_at_generation_cap": bool(generated["truncated"][0]),
                    "policy_sampled_logp_sum": float((policy_logp * mask).sum().item()),
                    "reference_sampled_logp_sum": float((ref_logp * mask).sum().item()),
                    "sampled_kl_token_mean": float(kl.item()),
                    "sampled_kl_sum": float(kl.item()) * usable,
                    "reward_model_score": float(reward_score[0].item()),
                    "reward_input_tokens": reward_tokens,
                    "reward_input_truncated": reward_tokens > 1024,
                }
                if not all(math.isfinite(record[key]) for key in (
                    "policy_sampled_logp_sum", "reference_sampled_logp_sum",
                    "sampled_kl_token_mean", "sampled_kl_sum", "reward_model_score"
                )):
                    raise FloatingPointError(f"Nonfinite generated score for {row['prompt_id']}")
                append_jsonl(generation_path, record)
                generation_records[pos] = record
                del generated, policy_logp, ref_logp, mask, kl, reward_score
        generation_seconds = time.perf_counter() - generation_start

        word_seconds = 0.0
        if word_limit:
            word_start = time.perf_counter()
            word_prompt_cap = max(len(chat_template_token_ids(tokenizer, prompt_messages(row))) for row in word_rows) + 16
            with torch.inference_mode():
                for pos, row in enumerate(tqdm(word_rows, desc=f"DPO {name} word limits", unit="prompt")):
                    if pos in word_records:
                        continue
                    messages = prompt_messages(row)
                    prompt_text = " ".join(str(message.get("content", "")) for message in messages)
                    limit = parse_word_limit(prompt_text)
                    if limit is None:
                        raise ValueError(f"Cannot parse word limit for {row['prompt_id']}")
                    generated = _generated(policy, tokenizer, messages, cfg, row["prompt_id"], word_prompt_cap)
                    response = generated["responses"][0]
                    compliant = word_limit_compliance(prompt_text, response)
                    if compliant is None:
                        raise ValueError(f"No word-limit result for {row['prompt_id']}")
                    record = {
                        "file_position": pos, "prompt_id": row["prompt_id"], "seed": generated["seed"],
                        "word_limit": limit, "response": response,
                        "response_tokens": int(generated["response_lengths"][0]),
                        "response_words": word_count(response), "compliant": float(compliant),
                    }
                    append_jsonl(word_path, record)
                    word_records[pos] = record
                    del generated
            word_seconds = time.perf_counter() - word_start

        if len(pair_records) != len(prepared) or len(generation_records) != len(rows):
            raise RuntimeError("Evaluation did not complete all eligible pairs and fixed generation prompts")
        if word_limit and len(word_records) != len(word_rows):
            raise RuntimeError("Word-limit evaluation did not complete all fixed prompts")
        torch.cuda.synchronize()
        summary = {
            "name": name, "dataset": dataset, "adapter": str(repo_path(adapter)),
            "adapter_sha256": adapter_hash, "beta": trained_beta,
            "pair_selection": selection,
            "pair_metrics": _pair_summary(list(pair_records.values())),
            "generation_metrics": _generation_summary(list(generation_records.values())),
            "generation_max_prompt_tokens": max_prompt_tokens,
            "generation_max_new_tokens": int(cfg["max_generation_tokens"]),
            "word_limit_metrics": _word_summary(list(word_records.values())) if word_limit else None,
            "loaded_models": {
                "policy": {"parameter_dtypes": policy_dtypes, "devices": policy_devices,
                           "loaded_8bit": bool(getattr(policy, "is_loaded_in_8bit", False)),
                           "loaded_4bit": bool(getattr(policy, "is_loaded_in_4bit", False))},
                "reward": {"parameter_dtypes": reward_dtypes, "devices": reward_devices,
                           "loaded_8bit": reward_8bit,
                           "loaded_4bit": bool(getattr(reward, "is_loaded_in_4bit", False))},
            },
            "autocast_dtype": "torch.float16",
            "model_loading_seconds": round(model_load_seconds + reward_load_seconds, 3),
            "pair_scoring_seconds": round(pair_seconds, 3),
            "generation_and_reward_seconds": round(generation_seconds, 3),
            "word_limit_seconds": round(word_seconds, 3),
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        save_json(summary_path, summary)
        print(f"Completed {name} {dataset} evaluation: {len(pair_records)} pairs, "
              f"{len(generation_records)} generations; summary={summary_path}", flush=True)
        return summary
    finally:
        del policy
        if "reward" in locals():
            del reward
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", default="standard")
    parser.add_argument("--dataset", choices=("standard", "length"), default="standard")
    parser.add_argument("--word-limit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.config, args.adapter, args.name, args.dataset, args.word_limit, args.resume)


if __name__ == "__main__":
    main()
