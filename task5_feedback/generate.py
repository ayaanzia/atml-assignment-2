"""Resumable, matched generation from the three frozen Task 5 policies."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json
from common.models import load_policy, load_tokenizer
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.runtime import (
    adapter_digest, check_generated_rows, check_released_generation,
    check_released_precision,
    dataset_digest, file_digest,
    generation_seed, gpu_memory, load_fixed_rows, model_audit, policy_paths,
    signature, source_id, sync_time,
)


def result_dir(cfg: dict, dataset: str) -> Path:
    return repo_path(cfg["results_dir"]) / "task5_feedback" / dataset


def generation_protocol(cfg: dict, dataset: str, policy: str,
                        sources: list[dict], adapter_hash: str | None) -> dict:
    check_released_generation(cfg)
    return {
        "dataset": dataset, "policy": policy, "dataset_sha256": dataset_digest(sources),
        "verifier_source_sha256": file_digest("task5_feedback/rlvr.py"),
        "base_model": cfg["base_model"], "adapter": policy_paths(cfg)[policy],
        "adapter_sha256": adapter_hash, "storage_dtype": cfg.get("dtype", "float16"),
        "seed": int(cfg["seed"]), "generation": cfg["generation"],
        "max_new_tokens": int(cfg["math_max_new_tokens"]),
        "prompt_handling": "render fixed messages without truncation",
    }


def prompt_token_count(tokenizer, messages: list[dict]) -> int:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    count = len(tokenizer(rendered, truncation=False)["input_ids"])
    if count < 1:
        raise ValueError("Empty rendered Task 5 prompt")
    return count


def saved_generation(path: Path, sources: list[dict], dataset: str,
                     policy: str) -> list[dict]:
    records = read_jsonl(path) if path.exists() else []
    if len(records) > len(sources):
        raise ValueError(f"{dataset}/{policy} has too many saved responses")
    for position, record in enumerate(records):
        source = sources[position]
        if record.get("position") != position or record.get("dataset") != dataset or \
                record.get("policy") != policy or record.get("source_id") != source_id(source) or \
                record.get("question") != source["question"] or \
                str(record.get("gold_final")) != str(source["gold_final"]):
            raise ValueError(f"{dataset}/{policy} saved row {position} differs from the fixed source")
    return records


def complete_generation(cfg: dict, dataset: str, policy: str,
                        sources: list[dict]) -> tuple[list[dict], dict]:
    root = result_dir(cfg, dataset)
    path = root / f"generated_{policy}.jsonl"
    summary_path = root / f"generation_summary_{policy}.json"
    manifest_path = root / f"generation_manifest_{policy}.json"
    if not all(item.exists() for item in (path, summary_path, manifest_path)):
        raise FileNotFoundError(f"Complete {dataset}/{policy} generation is unavailable")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = signature(generation_protocol(
        cfg, dataset, policy, sources, adapter_digest(policy_paths(cfg)[policy])))
    rows = read_jsonl(path)
    check_generated_rows(rows, sources, dataset, policy)
    if summary.get("status") != "PASS" or summary.get("generated_sha256") != file_digest(path) or \
            summary.get("signature") != identity or manifest.get("signature") != identity:
        raise ValueError(f"{dataset}/{policy} generation is incomplete or responses changed")
    return rows, summary


def generate_policy(cfg: dict, dataset: str, policy: str,
                    resume: bool = False) -> dict:
    check_released_precision(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 5 model generation requires a CUDA GPU")
    sources = load_fixed_rows(cfg, dataset)
    adapter = policy_paths(cfg)[policy]
    adapter_hash = adapter_digest(adapter)
    root = result_dir(cfg, dataset)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / f"generated_{policy}.jsonl"
    manifest_path = root / f"generation_manifest_{policy}.json"
    summary_path = root / f"generation_summary_{policy}.json"
    protocol = generation_protocol(cfg, dataset, policy, sources, adapter_hash)
    identity = signature(protocol)
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"Completed {dataset}/{policy} generation exists; use --resume")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") != identity:
            raise ValueError(f"Completed {dataset}/{policy} generation used different settings")
        complete_generation(cfg, dataset, policy, sources)
        return summary
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Partial {dataset}/{policy} generation exists; use --resume")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("signature") != identity:
            raise ValueError(f"Resumed {dataset}/{policy} generation changed inputs or settings")
    else:
        if record_path.exists():
            raise FileExistsError(f"Saved {dataset}/{policy} responses lack a manifest")
        save_json(manifest_path, {"signature": identity, "protocol": protocol})
    saved = saved_generation(record_path, sources, dataset, policy)
    previous = len(saved)
    before = gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    model = None
    summary = None
    began_load = sync_time()
    try:
        tokenizer = load_tokenizer(cfg["base_model"])
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        model.eval()
        loading_seconds = sync_time() - began_load
        audit = model_audit(model, f"{dataset}/{policy} frozen policy")
        generation_seconds = 0.0
        for position in tqdm(range(len(sources)), total=len(sources),
                             desc=f"{dataset} {policy} generation", unit="prompt"):
            if position < previous:
                continue
            source = sources[position]
            messages = source["messages"]
            prompt_tokens = prompt_token_count(tokenizer, messages)
            seed = generation_seed(int(cfg["seed"]), dataset, source_id(source))
            began = sync_time()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                generated = batch_generate(
                    model, tokenizer, [messages],
                    max_prompt_length=prompt_tokens,
                    max_new_tokens=int(cfg["math_max_new_tokens"]),
                    **cfg["generation"],
                )
            generation_seconds += sync_time() - began
            if generated["prompt_width"] != prompt_tokens:
                raise ValueError(f"{dataset}/{policy} prompt {position} was unexpectedly truncated")
            response = generated["responses"][0]
            final = extract_designated_final(response)
            record = {
                "position": position, "dataset": dataset, "policy": policy,
                "source_id": source_id(source), "question": source["question"],
                "gold_final": source["gold_final"], "seed": seed,
                "prompt_tokens": prompt_tokens, "response": response,
                "response_tokens": int(generated["response_lengths"][0]),
                "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
                "truncated_at_generation_cap": bool(generated["truncated"][0]),
                "designated_final": final, "format_compliant": final is not None,
                "exact_reward": exact_reward(response, source["gold_final"]),
            }
            if record["response_tokens"] < 1:
                raise ValueError(f"{dataset}/{policy} generated no tokens for {record['source_id']}")
            append_jsonl(record_path, record)
            del generated
        rows = read_jsonl(record_path)
        check_generated_rows(rows, sources, dataset, policy)
        summary = {
            "status": "PASS", "dataset": dataset, "policy": policy,
            "signature": identity, "records": len(rows),
            "new_records_this_invocation": len(rows) - previous,
            "generated_sha256": file_digest(record_path),
            "adapter_sha256": adapter_hash, "model": audit,
            "model_loading_seconds": round(loading_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "timing_note": "generation seconds cover only new prompts in this invocation",
            "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
            "max_response_tokens": max(row["response_tokens"] for row in rows),
            "memory_before": before,
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    if summary is None:
        raise RuntimeError(f"{dataset}/{policy} generation did not complete")
    summary["memory_after_release"] = gpu_memory()
    save_json(summary_path, summary)
    return summary
