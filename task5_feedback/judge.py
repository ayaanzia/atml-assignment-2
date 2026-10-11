"""Fixed pairwise judge orchestration for Task 5 policy comparisons."""

from __future__ import annotations

import gc
import hashlib
import json
import traceback
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import read_jsonl
from common.logging_utils import append_jsonl, save_json
from task5_feedback.generate import complete_generation, result_dir
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.runtime import (
    COMPETITORS, check_released_precision, compare_binary_scores,
    file_digest, gpu_memory,
    load_fixed_rows, model_audit, signature, source_id, sync_time,
)


def prepare_judge_cache(cfg: dict, root: Path, scope: str) -> Path:
    """Keep a study-specific cache tied to the fixed model and rubric."""
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / "pairwise_judge_cache.json"
    manifest_path = root / "pairwise_judge_cache_manifest.json"
    protocol = {
        "judge_model": cfg["ai_judge_model"],
        "rubric_sha256": hashlib.sha256(PAIRWISE_RUBRIC.encode()).hexdigest(),
        "judge_helper_sha256": file_digest("task5_feedback/rlaif.py"),
        "compute_dtype": cfg["dtype"], "quantize_4bit": cfg["quantize_frozen_models"],
        "scope": scope,
    }
    identity = signature(protocol)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("signature") != identity:
            raise ValueError(f"{scope} judge cache belongs to a different rubric/model")
    else:
        if cache_path.exists():
            raise FileExistsError(f"{scope} judge cache exists without its manifest")
        save_json(manifest_path, {"signature": identity, "protocol": protocol})
    return cache_path


def judge_protocol(cfg: dict, dataset: str, competitor: str,
                   generated_hashes: dict) -> dict:
    return {
        "dataset": dataset, "competitor": competitor,
        "generated_sha256": generated_hashes,
        "judge_model": cfg["ai_judge_model"],
        "judge_rubric_sha256": hashlib.sha256(PAIRWISE_RUBRIC.encode()).hexdigest(),
        "judge_helper_sha256": file_digest("task5_feedback/rlaif.py"),
        "compute_dtype": cfg["dtype"], "quantize_4bit": cfg["quantize_frozen_models"],
        "comparison_order": "SFT first, competitor second",
        "tie_rule": "one-half win",
    }


def saved_judgments(path: Path, sources: list[dict], dataset: str,
                    competitor: str) -> list[dict]:
    rows = read_jsonl(path) if path.exists() else []
    if len(rows) > len(sources):
        raise ValueError(f"Too many {dataset}/{competitor} judge records")
    for position, record in enumerate(rows):
        if record.get("position") != position or record.get("source_id") != source_id(sources[position]) or \
                record.get("dataset") != dataset or record.get("competitor") != competitor or \
                record.get("ai_preference") not in {"A", "B", "TIE"}:
            raise ValueError(f"Invalid saved {dataset}/{competitor} judge row {position}")
    return rows


def complete_judgments(cfg: dict, dataset: str, competitor: str,
                       sources: list[dict], generated_hashes: dict) -> list[dict]:
    root = result_dir(cfg, dataset)
    record_path = root / f"judged_{competitor}.jsonl"
    summary_path = root / f"judge_summary_{competitor}.json"
    manifest_path = root / f"judge_manifest_{competitor}.json"
    if not all(path.exists() for path in (record_path, summary_path, manifest_path)):
        raise FileNotFoundError(f"Complete {dataset}/{competitor} judging is unavailable")
    identity = signature(judge_protocol(cfg, dataset, competitor, generated_hashes))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = saved_judgments(record_path, sources, dataset, competitor)
    if len(rows) != len(sources) or summary.get("status") != "PASS" or \
            summary.get("signature") != identity or manifest.get("signature") != identity or \
            summary.get("judged_sha256") != file_digest(record_path):
        raise ValueError(f"{dataset}/{competitor} judge records changed or did not complete")
    return rows


def prepare_comparison(cfg: dict, dataset: str, competitor: str,
                       sources: list[dict], resume: bool) -> dict:
    sft, _ = complete_generation(cfg, dataset, "sft", sources)
    other, _ = complete_generation(cfg, dataset, competitor, sources)
    root = result_dir(cfg, dataset)
    hashes = {
        "sft": file_digest(root / "generated_sft.jsonl"),
        competitor: file_digest(root / f"generated_{competitor}.jsonl"),
    }
    protocol = judge_protocol(cfg, dataset, competitor, hashes)
    identity = signature(protocol)
    record_path = root / f"judged_{competitor}.jsonl"
    manifest_path = root / f"judge_manifest_{competitor}.json"
    summary_path = root / f"judge_summary_{competitor}.json"
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"Completed {dataset}/{competitor} judging exists; use --resume")
        rows = complete_judgments(cfg, dataset, competitor, sources, hashes)
        return {"status": "ALREADY_COMPLETE", "records": len(rows)}
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Partial {dataset}/{competitor} judging exists; use --resume")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("signature") != identity:
            raise ValueError(f"Resumed {dataset}/{competitor} judging changed inputs or settings")
    else:
        if record_path.exists():
            raise FileExistsError(f"Saved {dataset}/{competitor} judge records lack a manifest")
        save_json(manifest_path, {"signature": identity, "protocol": protocol})
    existing = saved_judgments(record_path, sources, dataset, competitor)
    return {"status": "PENDING", "competitor": competitor, "sft": sft,
            "other": other, "existing": existing, "identity": identity,
            "record_path": record_path, "summary_path": summary_path}


def judge_competitor(cfg: dict, dataset: str, sources: list[dict],
                     prepared: dict, judge: PairwiseAIJudge,
                     loading_seconds: float, audit: dict) -> dict:
    competitor = prepared["competitor"]
    sft, other = prepared["sft"], prepared["other"]
    existing = prepared["existing"]
    previous = len(existing)
    judging_seconds = 0.0
    for position in tqdm(range(len(sources)), total=len(sources),
                         desc=f"{dataset} {competitor} vs SFT judge", unit="pair"):
        if position < previous:
            continue
        source = sources[position]
        baseline, candidate = sft[position], other[position]
        key = judge._key(source["question"], baseline["response"], candidate["response"])
        cache_hit = key in judge.cache
        began = sync_time()
        ai = judge.compare(source["question"], baseline["response"], candidate["response"])
        judging_seconds += sync_time() - began
        if ai not in {"A", "B", "TIE"}:
            raise ValueError(f"Fixed judge returned invalid preference {ai!r}")
        exact = compare_binary_scores(float(baseline["exact_reward"]),
                                      float(candidate["exact_reward"]))
        record = {
            "position": position, "dataset": dataset, "source_id": source_id(source),
            "competitor": competitor, "question": source["question"],
            "gold_final": source["gold_final"],
            "sft_response": baseline["response"],
            "competitor_response": candidate["response"],
            "sft_exact_reward": baseline["exact_reward"],
            "competitor_exact_reward": candidate["exact_reward"],
            "exact_preference": exact, "ai_preference": ai,
            "agrees_with_exact": ai == exact, "judge_cache_hit": cache_hit,
        }
        append_jsonl(prepared["record_path"], record)
        existing.append(record)
    if len(existing) != len(sources):
        raise RuntimeError(f"{dataset}/{competitor} judging ended early")
    summary = {
        "status": "PASS", "dataset": dataset, "competitor": competitor,
        "signature": prepared["identity"], "records": len(existing),
        "new_records_this_invocation": len(existing) - previous,
        "judged_sha256": file_digest(prepared["record_path"]),
        "judge_model": audit,
        "judge_model_loading_seconds": round(loading_seconds, 3),
        "judging_seconds": round(judging_seconds, 3),
        "timing_note": "judging seconds cover only new comparisons in this invocation",
        "judge_cache_hits": sum(row["judge_cache_hit"] for row in existing),
        "new_judge_inferences": sum(not row["judge_cache_hit"] for row in existing),
        "ties_include_possible_parse_failures": True,
        "peak_cuda_allocated_gib_so_far": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_cuda_reserved_gib_so_far": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    save_json(prepared["summary_path"], summary)
    return summary


def judge_dataset(cfg: dict, dataset: str, resume: bool = False) -> list[dict]:
    check_released_precision(cfg)
    sources = load_fixed_rows(cfg, dataset)
    root = result_dir(cfg, dataset)
    root.mkdir(parents=True, exist_ok=True)
    statuses = []
    pending = []
    for competitor in COMPETITORS:
        try:
            prepared = prepare_comparison(cfg, dataset, competitor, sources, resume)
            if prepared["status"] == "ALREADY_COMPLETE":
                statuses.append({"competitor": competitor, "status": "PASS",
                                 "records": prepared["records"], "reused": True})
            else:
                pending.append(prepared)
        except FileNotFoundError as error:
            statuses.append({"competitor": competitor, "status": "SKIPPED", "reason": str(error)})
        except Exception as error:
            statuses.append({"competitor": competitor, "status": "FAIL",
                             "error_type": type(error).__name__, "error_message": str(error),
                             "traceback": traceback.format_exc()})
    judge = None
    before = gpu_memory()
    loading_seconds = 0.0
    try:
        if pending:
            if not torch.cuda.is_available():
                raise RuntimeError("Task 5 fixed AI judging requires a CUDA GPU")
            cache_path = prepare_judge_cache(cfg, root, dataset)
            torch.cuda.reset_peak_memory_stats()
            began = sync_time()
            judge = PairwiseAIJudge(cfg, cache_path)
            loading_seconds = sync_time() - began
            audit = model_audit(judge.model, "fixed Task 5 pairwise AI judge")
            for prepared in pending:
                competitor = prepared["competitor"]
                try:
                    summary = judge_competitor(
                        cfg, dataset, sources, prepared, judge, loading_seconds, audit)
                    statuses.append({"competitor": competitor, "status": "PASS",
                                     "records": summary["records"]})
                except Exception as error:
                    statuses.append({"competitor": competitor, "status": "FAIL",
                                     "error_type": type(error).__name__, "error_message": str(error),
                                     "traceback": traceback.format_exc()})
    except Exception as error:
        for prepared in pending:
            if not any(row["competitor"] == prepared["competitor"] for row in statuses):
                statuses.append({"competitor": prepared["competitor"], "status": "FAIL",
                                 "error_type": type(error).__name__, "error_message": str(error),
                                 "traceback": traceback.format_exc()})
    finally:
        if judge is not None:
            del judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(root / "judge_status.json", {
        "dataset": dataset, "competitors": statuses,
        "complete": all(any(row["competitor"] == name and row["status"] == "PASS"
                            for row in statuses) for name in COMPETITORS),
        "model_loading_seconds": round(loading_seconds, 3),
        "memory_before": before, "memory_after_release": gpu_memory(),
    })
    return statuses
