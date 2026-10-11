"""Controlled clean-versus-perturbation verifier and judge study."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import statistics
import traceback
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common.data import read_jsonl, repo_path, write_jsonl
from common.logging_utils import append_jsonl, save_json
from task5_feedback.judge import prepare_judge_cache
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.rlvr import exact_reward
from task5_feedback.runtime import (
    CONTRASTS, VARIANTS, check_released_precision, compare_binary_scores,
    dataset_digest, file_digest,
    gpu_memory, model_audit, preference_rates, signature, sync_time,
)


def diagnostic_dir(cfg: dict) -> Path:
    return repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"


def fixed_diagnostic_groups(cfg: dict) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
    if len(rows) != 100:
        raise ValueError(f"Expected 100 fixed Task 5 diagnostic responses, found {len(rows)}")
    required = {"problem_id", "question", "gold_final", "variant_type", "response",
                "expected_exact_reward", "manual_validation"}
    grouped = {}
    for row in rows:
        if not required.issubset(row) or row["variant_type"] not in VARIANTS or \
                row["manual_validation"] is not True:
            raise ValueError("Diagnostic row lacks a required field, variant, or manual validation")
        problem_id = str(row["problem_id"])
        entries = grouped.setdefault(problem_id, {})
        if row["variant_type"] in entries:
            raise ValueError(f"Duplicate {row['variant_type']} variant for problem {problem_id}")
        entries[row["variant_type"]] = row
    if len(grouped) != 20:
        raise ValueError(f"Expected 20 fixed diagnostic problems, found {len(grouped)}")
    pairs = []
    for problem_id in sorted(grouped):
        group = grouped[problem_id]
        if set(group) != set(VARIANTS) or \
                len({row["question"] for row in group.values()}) != 1 or \
                len({str(row["gold_final"]) for row in group.values()}) != 1:
            raise ValueError(f"Problem {problem_id} has incomplete or inconsistent variants")
        clean = group["clean_correct"]
        for variant in CONTRASTS:
            pairs.append({"problem_id": problem_id, "variant_type": variant,
                          "question": clean["question"],
                          "gold_final": clean["gold_final"],
                          "clean_response": clean["response"],
                          "variant_response": group[variant]["response"]})
    if len(pairs) != 80:
        raise ValueError("Expected 80 clean-versus-variant diagnostic pairs")
    return rows, pairs


def _exact_protocol(cfg: dict, rows: list[dict]) -> dict:
    return {
        "diagnostic_data_sha256": dataset_digest(rows),
        "verifier_source_sha256": file_digest("task5_feedback/rlvr.py"),
        "pairing": "clean_correct versus each of four variants, clean first",
    }


def complete_exact(cfg: dict) -> tuple[list[dict], list[dict], dict]:
    source, expected_pairs = fixed_diagnostic_groups(cfg)
    root = diagnostic_dir(cfg)
    manifest_path = root / "exact_manifest.json"
    variant_path = root / "variant_scores.jsonl"
    pair_path = root / "exact_pairs.jsonl"
    summary_path = root / "exact_summary.json"
    if not all(path.exists() for path in (manifest_path, variant_path, pair_path, summary_path)):
        raise FileNotFoundError("Complete fixed diagnostic exact scoring is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_identity = signature(_exact_protocol(cfg, source))
    variants = read_jsonl(variant_path)
    pairs = read_jsonl(pair_path)
    if len(variants) != 100 or len(pairs) != 80 or \
            manifest.get("signature") != expected_identity or \
            summary.get("signature") != expected_identity or \
            summary.get("variant_scores_sha256") != file_digest(variant_path) or \
            summary.get("exact_pairs_sha256") != file_digest(pair_path):
        raise ValueError("Fixed diagnostic exact scores changed or did not complete")
    for position, (record, source_row) in enumerate(zip(variants, source)):
        if record["position"] != position or record["problem_id"] != str(source_row["problem_id"]) or \
                record["variant_type"] != source_row["variant_type"] or \
                float(record["exact_reward"]) != exact_reward(source_row["response"],
                                                               source_row["gold_final"]):
            raise ValueError(f"Diagnostic variant score {position} differs from fixed data")
    scores = {(row["problem_id"], row["variant_type"]): row["exact_reward"]
              for row in variants}
    for position, (record, source_pair) in enumerate(zip(pairs, expected_pairs)):
        if record["position"] != position or record["problem_id"] != source_pair["problem_id"] or \
                record["variant_type"] != source_pair["variant_type"] or \
                record["question"] != source_pair["question"] or \
                record["clean_response"] != source_pair["clean_response"] or \
                record["variant_response"] != source_pair["variant_response"] or \
                float(record["clean_exact_reward"]) != scores[
                    (source_pair["problem_id"], "clean_correct")] or \
                float(record["variant_exact_reward"]) != scores[
                    (source_pair["problem_id"], source_pair["variant_type"])] or \
                record["exact_preference"] != compare_binary_scores(
                    float(record["clean_exact_reward"]),
                    float(record["variant_exact_reward"])):
            raise ValueError(f"Diagnostic exact pair {position} differs from fixed pairing")
    return variants, pairs, summary


def score_exact(cfg: dict, resume: bool = False) -> dict:
    rows, source_pairs = fixed_diagnostic_groups(cfg)
    root = diagnostic_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    protocol = _exact_protocol(cfg, rows)
    identity = signature(protocol)
    manifest_path = root / "exact_manifest.json"
    summary_path = root / "exact_summary.json"
    variant_path = root / "variant_scores.jsonl"
    pair_path = root / "exact_pairs.jsonl"
    if summary_path.exists():
        if not resume:
            raise FileExistsError("Completed diagnostic exact scoring exists; use --resume")
        return complete_exact(cfg)[2]
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Partial diagnostic exact scoring exists; use --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != identity:
            raise ValueError("Resumed diagnostic exact scoring changed data or verifier")
    else:
        if variant_path.exists() or pair_path.exists():
            raise FileExistsError("Diagnostic exact records exist without a manifest")
        save_json(manifest_path, {"signature": identity, "protocol": protocol})
    began = sync_time()
    variants = []
    by_key = {}
    for position, row in enumerate(rows):
        score = exact_reward(row["response"], row["gold_final"])
        if score != float(row["expected_exact_reward"]):
            raise ValueError(f"Released verifier disagrees with validated problem {row['problem_id']}"
                             f" variant {row['variant_type']}")
        record = {"position": position, "problem_id": str(row["problem_id"]),
                  "variant_type": row["variant_type"], "question": row["question"],
                  "gold_final": row["gold_final"], "response": row["response"],
                  "exact_reward": score}
        variants.append(record)
        by_key[(str(row["problem_id"]), row["variant_type"])] = record
    pairs = []
    for position, pair in enumerate(source_pairs):
        clean_score = by_key[(pair["problem_id"], "clean_correct")]["exact_reward"]
        variant_score = by_key[(pair["problem_id"], pair["variant_type"])]["exact_reward"]
        pairs.append({"position": position, **pair,
                      "clean_exact_reward": clean_score,
                      "variant_exact_reward": variant_score,
                      "exact_preference": compare_binary_scores(clean_score, variant_score)})
    write_jsonl(variant_path, variants)
    write_jsonl(pair_path, pairs)
    summary = {"status": "PASS", "signature": identity,
               "variants": len(variants), "pairs": len(pairs),
               "variant_scores_sha256": file_digest(variant_path),
               "exact_pairs_sha256": file_digest(pair_path),
               "seconds": round(sync_time() - began, 3)}
    save_json(summary_path, summary)
    return summary


def _judge_protocol(cfg: dict, exact_pairs_hash: str) -> dict:
    return {
        "exact_pairs_sha256": exact_pairs_hash,
        "judge_model": cfg["ai_judge_model"],
        "rubric_sha256": hashlib.sha256(PAIRWISE_RUBRIC.encode()).hexdigest(),
        "judge_helper_sha256": file_digest("task5_feedback/rlaif.py"),
        "compute_dtype": cfg["dtype"], "quantize_4bit": cfg["quantize_frozen_models"],
        "pair_order": "clean first, variant second",
    }


def complete_diagnostic_judgments(cfg: dict) -> tuple[list[dict], dict]:
    _, exact_pairs, exact_state = complete_exact(cfg)
    root = diagnostic_dir(cfg)
    record_path = root / "judged_pairs.jsonl"
    manifest_path = root / "judge_manifest.json"
    summary_path = root / "judge_summary.json"
    if not all(path.exists() for path in (record_path, manifest_path, summary_path)):
        raise FileNotFoundError("Complete fixed diagnostic AI judgments are unavailable")
    identity = signature(_judge_protocol(cfg, exact_state["exact_pairs_sha256"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    judgments = read_jsonl(record_path)
    if len(judgments) != len(exact_pairs) or summary.get("status") != "PASS" or \
            manifest.get("signature") != identity or \
            summary.get("signature") != identity or \
            summary.get("judged_pairs_sha256") != file_digest(record_path):
        raise ValueError("Diagnostic AI judgments changed or did not complete")
    for position, (record, exact_pair) in enumerate(zip(judgments, exact_pairs)):
        if record["position"] != position or record["problem_id"] != exact_pair["problem_id"] or \
                record["variant_type"] != exact_pair["variant_type"] or \
                record["ai_preference"] not in {"A", "B", "TIE"}:
            raise ValueError(f"Diagnostic AI judgment {position} differs from fixed pairing")
    return judgments, summary


def judge_diagnostics(cfg: dict, resume: bool = False) -> dict:
    check_released_precision(cfg)
    _, pairs, exact_state = complete_exact(cfg)
    root = diagnostic_dir(cfg)
    record_path = root / "judged_pairs.jsonl"
    manifest_path = root / "judge_manifest.json"
    summary_path = root / "judge_summary.json"
    protocol = _judge_protocol(cfg, exact_state["exact_pairs_sha256"])
    identity = signature(protocol)
    if summary_path.exists():
        if not resume:
            raise FileExistsError("Completed diagnostic AI judging exists; use --resume")
        return complete_diagnostic_judgments(cfg)[1]
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Partial diagnostic AI judging exists; use --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != identity:
            raise ValueError("Resumed diagnostic judging changed exact pairs or rubric")
    else:
        if record_path.exists():
            raise FileExistsError("Diagnostic AI judgment records exist without manifest")
        save_json(manifest_path, {"signature": identity, "protocol": protocol})
    existing = read_jsonl(record_path) if record_path.exists() else []
    if len(existing) > len(pairs):
        raise ValueError("More saved diagnostic judgments than fixed pairs")
    for position, record in enumerate(existing):
        if record.get("position") != position or record.get("problem_id") != pairs[position]["problem_id"] or \
                record.get("variant_type") != pairs[position]["variant_type"] or \
                record.get("ai_preference") not in {"A", "B", "TIE"}:
            raise ValueError(f"Saved diagnostic judge row {position} differs from fixed pair")
    if not torch.cuda.is_available():
        raise RuntimeError("Fixed Task 5 AI judging requires a CUDA GPU")
    cache_path = prepare_judge_cache(cfg, root, "task5_diagnostics")
    previous = len(existing)
    before = gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    began = sync_time()
    judge = None
    summary = None
    try:
        judge = PairwiseAIJudge(cfg, cache_path)
        loading_seconds = sync_time() - began
        audit = model_audit(judge.model, "fixed Task 5 diagnostic pairwise judge")
        judging_seconds = 0.0
        for position in tqdm(range(len(pairs)), total=len(pairs),
                             desc="Task 5 diagnostic AI judging", unit="pair"):
            if position < previous:
                continue
            pair = pairs[position]
            key = judge._key(pair["question"], pair["clean_response"],
                             pair["variant_response"])
            cache_hit = key in judge.cache
            start = sync_time()
            verdict = judge.compare(pair["question"], pair["clean_response"],
                                    pair["variant_response"])
            judging_seconds += sync_time() - start
            if verdict not in {"A", "B", "TIE"}:
                raise ValueError(f"Fixed judge returned invalid preference {verdict!r}")
            record = {"position": position, "problem_id": pair["problem_id"],
                      "variant_type": pair["variant_type"], "question": pair["question"],
                      "clean_response": pair["clean_response"],
                      "variant_response": pair["variant_response"],
                      "exact_preference": pair["exact_preference"],
                      "ai_preference": verdict, "judge_cache_hit": cache_hit}
            append_jsonl(record_path, record)
            existing.append(record)
        if len(existing) != len(pairs):
            raise RuntimeError("Diagnostic AI judging ended early")
        summary = {"status": "PASS", "signature": identity,
                   "pairs": len(existing), "new_pairs_this_invocation": len(existing) - previous,
                   "judged_pairs_sha256": file_digest(record_path),
                   "judge_model": audit, "model_loading_seconds": round(loading_seconds, 3),
                   "judging_seconds": round(judging_seconds, 3),
                   "timing_note": "judging seconds cover only new pairs in this invocation",
                   "judge_cache_hits": sum(row["judge_cache_hit"] for row in existing),
                   "new_judge_inferences": sum(not row["judge_cache_hit"] for row in existing),
                   "ties_include_possible_parse_failures": True,
                   "memory_before": before,
                   "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
    finally:
        if judge is not None:
            del judge
        gc.collect()
        torch.cuda.empty_cache()
    if summary is None:
        raise RuntimeError("Diagnostic AI judging did not complete")
    summary["memory_after_release"] = gpu_memory()
    save_json(summary_path, summary)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_diagnostics(cfg: dict) -> dict:
    root = diagnostic_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    variants, exact_pairs, exact_state = complete_exact(cfg)
    try:
        judged, judge_state = complete_diagnostic_judgments(cfg)
        judged_status = "PASS"
    except FileNotFoundError as error:
        judged, judge_state, judged_status = [], {"reason": str(error)}, "SKIPPED"
    except Exception as error:
        judged, judged_status = [], "FAIL"
        judge_state = {"error_type": type(error).__name__, "error_message": str(error),
                       "traceback": traceback.format_exc()}
    by_variant = []
    contrast_rows = []
    for variant in VARIANTS:
        selected = [row for row in variants if row["variant_type"] == variant]
        exact_selected = [row for row in exact_pairs if row["variant_type"] == variant]
        ai_selected = [row for row in judged if row["variant_type"] == variant]
        base = {"variant_type": variant, "responses": len(selected),
                "exact_positive_count": sum(row["exact_reward"] for row in selected),
                "exact_positive_rate": statistics.fmean(row["exact_reward"] for row in selected),
                "reference_role": "clean baseline" if variant == "clean_correct" else "contrast to clean"}
        by_variant.append(base)
        if variant == "clean_correct":
            continue
        for mechanism, verdicts in (
            ("RLVR_exact", [row["exact_preference"] for row in exact_selected]),
            ("RLAIF_pairwise", [row["ai_preference"] for row in ai_selected]),
        ):
            if not verdicts:
                continue
            rates = preference_rates(verdicts, "A")
            contrast_rows.append({"variant_type": variant, "mechanism": mechanism,
                                  **rates})
    ai_by_key = {(row["problem_id"], row["variant_type"]): row for row in judged}
    disagreements = [{**pair, "ai_preference": ai_by_key[(pair["problem_id"],
                                                           pair["variant_type"])]["ai_preference"]}
                     for pair in exact_pairs if (pair["problem_id"], pair["variant_type"]) in ai_by_key
                     and pair["exact_preference"] != ai_by_key[(pair["problem_id"],
                                                                pair["variant_type"])]["ai_preference"]]
    by_key = {(row["variant_type"], row["mechanism"]): row for row in contrast_rows}
    sensitivity = {}
    for mechanism in ("RLVR_exact", "RLAIF_pairwise"):
        reason = by_key.get(("corrupt_reasoning_correct_final", mechanism))
        outcome = by_key.get(("good_reasoning_wrong_final", mechanism))
        sensitivity[mechanism] = {
            "Sreason": reason["better_rate"] if reason else None,
            "Soutcome": outcome["better_rate"] if outcome else None,
        }
    summary = {
        "status": "COMPLETE" if judged_status == "PASS" else "INCOMPLETE",
        "fixed_problems": 20, "fixed_variants": len(variants),
        "fixed_contrast_pairs": len(exact_pairs),
        "exact_summary_sha256": file_digest(root / "exact_summary.json"),
        "judge_status": judged_status, "judge_detail": judge_state,
        "variant_baselines": by_variant, "contrasts": contrast_rows,
        "sensitivity": sensitivity,
        "clean_reference_note": "Clean is the reference; no better-response pair exists for clean alone",
        "ai_tie_limitation": "The fixed judge maps unparseable answers to TIE",
        "qualitative_note": "Review the saved disagreements for reasoning, filler, and distractor cases",
    }
    save_json(root / "summary.json", summary)
    _write_csv(root / "variant_baselines.csv", by_variant)
    _write_csv(root / "contrast_metrics.csv", contrast_rows)
    write_jsonl(root / "verifier_judge_disagreements.jsonl", disagreements)
    from task5_feedback.plots import plot_diagnostics
    if judged_status == "PASS":
        plot_diagnostics(contrast_rows, root / "diagnostic_preferences.png")
    else:
        (root / "diagnostic_preferences.png").unlink(missing_ok=True)
    print(f"Task 5 controlled diagnostics: {summary['status']}", flush=True)
    return summary
