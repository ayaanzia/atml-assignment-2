"""Generate one fixed greedy XSTest response per prompt and frozen policy."""

from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, print_gpu_info, save_json
from common.models import load_policy, load_tokenizer
from task4_safety.runtime import (
    POLICIES, adapter_digest, check_policy_rows, dataset_digest, file_digest, gpu_memory,
    load_xstest, model_audit, signature, sync_time,
)


def policy_specs(cfg: dict) -> dict[str, str | None]:
    fixed = {
        "sft": None,
        "dpo": "outputs/task1_dpo/standard",
        "ppo": "outputs/task2_ppo/standard",
        "grpo": "outputs/task3_grpo/standard",
    }
    selected = {name: cfg["policies"].get(name) for name in POLICIES}
    if selected != fixed:
        raise ValueError("Task 4 must use the four released SFT/standard-adapter policy paths")
    return selected


def _prompt_lengths(tokenizer, frame, cap: int) -> list[int]:
    lengths = []
    for prompt in frame["prompt"]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            tokenize=False, add_generation_prompt=True,
        )
        count = len(tokenizer(rendered, truncation=False)["input_ids"])
        if count > cap:
            raise ValueError(f"XSTest chat prompt has {count} tokens, exceeding the {cap}-token cap")
        lengths.append(count)
    return lengths


def _existing_records(path: Path, frame, policy: str) -> dict[int, dict]:
    if not path.exists():
        return {}
    records = read_jsonl(path)
    indexed = {int(row["row_index"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate saved {policy} XSTest row indices")
    source = frame.to_dict("records")
    for position, record in indexed.items():
        if position < 0 or position >= len(source):
            raise ValueError(f"Saved {policy} row index is outside XSTest")
        expected = source[position]
        if record["policy"] != policy or int(record["xstest_id"]) != int(expected["xstest_id"]) or \
                record["prompt"] != expected["prompt"] or \
                record["benchmark_class"] != expected["benchmark_class"] or \
                record["type"] != expected["type"]:
            raise ValueError(f"Saved {policy} response differs from fixed XSTest row {position}")
    return indexed


def generate_for_policy(cfg: dict, policy_name: str, batch_size: int = 4,
                        resume: bool = False) -> dict:
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 4 generation requires a CUDA GPU")
    frame = load_xstest(cfg)
    adapter = specs[policy_name]
    adapter_sha = adapter_digest(adapter)
    root = repo_path(cfg["results_dir"]) / "task4_safety"
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / f"generated_{policy_name}.jsonl"
    manifest_path = root / f"generation_manifest_{policy_name}.json"
    summary_path = root / f"generation_summary_{policy_name}.json"
    protocol = {
        "policy": policy_name, "base_model": cfg["base_model"],
        "adapter": adapter, "adapter_sha256": adapter_sha,
        "xstest_sha256": dataset_digest(frame),
        "batch_size": batch_size, "prompt_cap": 256,
        "response_cap": int(cfg["safety_max_new_tokens"]),
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
        "storage_dtype": cfg.get("dtype", "float16"),
    }
    identity = signature(protocol)
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"Completed {policy_name} generation exists; use --resume")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") != identity:
            raise ValueError(f"Completed {policy_name} generation used a different protocol")
        rows = read_jsonl(record_path)
        check_policy_rows(rows, frame, policy_name)
        if summary.get("generated_sha256") != file_digest(record_path):
            raise ValueError(f"Completed {policy_name} generated responses changed")
        return summary
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Partial {policy_name} generation exists; use --resume")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("signature") != identity:
            raise ValueError(f"Resumed {policy_name} generation changed model, batch, or prompt protocol")
    else:
        if record_path.exists():
            raise FileExistsError(f"{record_path} has no matching manifest")
        save_json(manifest_path, {"signature": identity, "protocol": protocol})
    existing = _existing_records(record_path, frame, policy_name)
    previous = len(existing)
    memory_before = gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    model = None
    summary = None
    load_start = sync_time()
    try:
        tokenizer = load_tokenizer(cfg["base_model"])
        lengths = _prompt_lengths(tokenizer, frame, 256)
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        model.eval()
        model_loading_seconds = sync_time() - load_start
        audit = model_audit(model, f"{policy_name} frozen policy")
        generation_seconds = 0.0
        batches = math.ceil(len(frame) / batch_size)
        for start in tqdm(range(0, len(frame), batch_size), total=batches,
                          desc=f"XSTest {policy_name} generation", unit="batch"):
            positions = list(range(start, min(start + batch_size, len(frame))))
            if all(position in existing for position in positions):
                continue
            chunk = frame.iloc[positions]
            prompts = [[{"role": "user", "content": str(prompt)}]
                       for prompt in chunk["prompt"].tolist()]
            began = sync_time()
            generated = batch_generate(
                model, tokenizer, prompts,
                max_prompt_length=256,
                max_new_tokens=int(cfg["safety_max_new_tokens"]),
                temperature=0.0, top_p=1.0, do_sample=False,
            )
            generation_seconds += sync_time() - began
            for local, position in enumerate(positions):
                source = frame.iloc[position]
                record = {
                    "row_index": position, "xstest_id": int(source["xstest_id"]),
                    "policy": policy_name, "prompt": str(source["prompt"]),
                    "benchmark_class": str(source["benchmark_class"]),
                    "type": str(source["type"]),
                    "prompt_tokens": lengths[position],
                    "response": generated["responses"][local],
                    "response_tokens": int(generated["response_lengths"][local]),
                    "terminated_with_eos": bool(generated["terminated_with_eos"][local]),
                    "truncated_at_generation_cap": bool(generated["truncated"][local]),
                    "generation_batch_start": start,
                }
                if record["response_tokens"] < 1:
                    raise ValueError(f"Empty {policy_name} response for XSTest ID {record['xstest_id']}")
                if position in existing:
                    if record != existing[position]:
                        raise ValueError(f"Replayed {policy_name} batch changed response for ID {record['xstest_id']}")
                else:
                    append_jsonl(record_path, record)
                    existing[position] = record
            del generated
        if len(existing) != len(frame):
            raise RuntimeError(f"{policy_name} generation ended with {len(existing)} of {len(frame)} prompts")
        rows = read_jsonl(record_path)
        check_policy_rows(rows, frame, policy_name)
        summary = {
            "policy": policy_name, "signature": identity, "status": "PASS",
            "records": len(rows), "new_records_this_invocation": len(rows) - previous,
            "generated_sha256": file_digest(record_path),
            "adapter_sha256": adapter_sha, "model": audit,
            "model_loading_seconds": round(model_loading_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "timing_note": "generation seconds cover only batches generated in this invocation",
            "mean_response_tokens": sum(row["response_tokens"] for row in rows) / len(rows),
            "max_prompt_tokens": max(lengths),
            "memory_before": memory_before,
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    if summary is None:
        raise RuntimeError("Generation ended without a summary")
    summary["memory_after_release"] = gpu_memory()
    save_json(summary_path, summary)
    return summary


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    names = [args.policy] if args.policy else list(POLICIES)
    root = repo_path(cfg["results_dir"]) / "task4_safety"
    root.mkdir(parents=True, exist_ok=True)
    statuses = []
    for name in names:
        adapter = policy_specs(cfg)[name]
        try:
            if adapter is not None:
                try:
                    adapter_digest(adapter)
                except FileNotFoundError as error:
                    statuses.append({"policy": name, "status": "SKIPPED", "reason": str(error)})
                    continue
            result = generate_for_policy(cfg, name, args.batch_size, args.resume)
            statuses.append({"policy": name, "status": "PASS", "records": result["records"]})
        except Exception as error:
            statuses.append({"policy": name, "status": "FAIL",
                             "error_type": type(error).__name__, "error_message": str(error),
                             "traceback": traceback.format_exc()})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_json(root / "generation_status.json", {
        "policies": statuses,
        "four_policy_generation_complete": len(names) == 4 and all(
            row["status"] == "PASS" for row in statuses),
    })
    print("Task 4 generation status:", statuses, flush=True)
    if any(row["status"] == "FAIL" for row in statuses):
        raise RuntimeError("One or more Task 4 policy generations failed; see generation_status.json")


if __name__ == "__main__":
    main()
