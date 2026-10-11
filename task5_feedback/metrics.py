"""Aggregate complete Task 5 math generations and fixed judge comparisons."""

from __future__ import annotations

import csv
import statistics
import traceback

from common.data import write_jsonl
from common.logging_utils import save_json
from task5_feedback.generate import complete_generation, result_dir
from task5_feedback.judge import complete_judgments
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.runtime import (
    COMPETITORS, POLICIES, dataset_digest, load_fixed_rows,
)


def policy_metrics(rows: list[dict], expected_count: int) -> dict:
    if len(rows) != expected_count:
        raise ValueError("Cannot compute fixed-policy metrics from partial generations")
    for row in rows:
        final = extract_designated_final(row["response"])
        if row["designated_final"] != final or row["format_compliant"] != (final is not None) or \
                float(row["exact_reward"]) != exact_reward(row["response"], row["gold_final"]):
            raise ValueError(f"Saved exact score or format flag is wrong for {row['source_id']}")
    lengths = [int(row["response_tokens"]) for row in rows]
    correct = sum(float(row["exact_reward"]) == 1.0 for row in rows)
    format_count = sum(bool(row["format_compliant"]) for row in rows)
    return {
        "prompts": expected_count, "exact_correct": correct,
        "exact_accuracy": correct / expected_count,
        "format_compliant_count": format_count,
        "format_compliance_rate": format_count / expected_count,
        "mean_response_tokens": statistics.fmean(lengths),
        "std_response_tokens": statistics.pstdev(lengths),
        "generated_response_tokens": sum(lengths),
        "eos_rate": sum(row["terminated_with_eos"] for row in rows) / expected_count,
        "generation_cap_rate": sum(row["truncated_at_generation_cap"] for row in rows) / expected_count,
        "failure_counts": {
            "correct": correct,
            "wrong_designated_final": format_count - correct,
            "missing_designated_final": expected_count - format_count,
        },
    }


def comparison_metrics(rows: list[dict], expected_count: int) -> dict:
    if len(rows) != expected_count:
        raise ValueError("Cannot compute fixed comparison metrics from partial judge records")
    valid = {"A", "B", "TIE"}
    if any(row["ai_preference"] not in valid or row["exact_preference"] not in valid
           for row in rows):
        raise ValueError("Invalid AI or verifier preference")
    wins = sum(row["ai_preference"] == "B" for row in rows)
    ties = sum(row["ai_preference"] == "TIE" for row in rows)
    losses = expected_count - wins - ties
    decisive = [row for row in rows if row["exact_preference"] != "TIE"]
    exact_ties = [row for row in rows if row["exact_preference"] == "TIE"]
    return {
        "prompts": expected_count,
        "ai_wins_vs_sft": wins, "ai_ties_vs_sft": ties, "ai_losses_vs_sft": losses,
        "ai_win_rate_vs_sft": (wins + 0.5 * ties) / expected_count,
        "ai_strict_win_rate_vs_sft": wins / expected_count,
        "ai_tie_rate_vs_sft": ties / expected_count,
        "verifier_judge_three_way_agreement": sum(
            row["ai_preference"] == row["exact_preference"] for row in rows) / expected_count,
        "verifier_decisive_pairs": len(decisive),
        "judge_agreement_on_decisive_pairs": (
            sum(row["ai_preference"] == row["exact_preference"] for row in decisive) / len(decisive)
            if decisive else None),
        "verifier_tie_pairs": len(exact_ties),
        "judge_prefers_competitor_on_verifier_tie": sum(
            row["ai_preference"] == "B" for row in exact_ties),
        "judge_prefers_sft_on_verifier_tie": sum(
            row["ai_preference"] == "A" for row in exact_ties),
        "judge_ties_on_verifier_tie": sum(
            row["ai_preference"] == "TIE" for row in exact_ties),
        "judge_cache_hits": sum(row["judge_cache_hit"] for row in rows),
        "new_judge_inferences": sum(not row["judge_cache_hit"] for row in rows),
        "tie_limitation": "fixed judge maps unparseable output to TIE",
    }


def _write_csv(path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_dataset(cfg: dict, dataset: str) -> dict:
    sources = load_fixed_rows(cfg, dataset)
    root = result_dir(cfg, dataset)
    root.mkdir(parents=True, exist_ok=True)
    policy_status = []
    comparison_status = []
    hashes = {}
    complete_policies = set()
    for policy in POLICIES:
        try:
            rows, state = complete_generation(cfg, dataset, policy, sources)
            metrics = policy_metrics(rows, len(sources))
            hashes[policy] = state["generated_sha256"]
            complete_policies.add(policy)
            policy_status.append({"policy": policy, "status": "PASS", "metrics": metrics})
        except FileNotFoundError as error:
            policy_status.append({"policy": policy, "status": "SKIPPED", "reason": str(error)})
        except Exception as error:
            policy_status.append({"policy": policy, "status": "FAIL",
                                  "error_type": type(error).__name__, "error_message": str(error),
                                  "traceback": traceback.format_exc()})
    candidates = []
    for competitor in COMPETITORS:
        if not {"sft", competitor}.issubset(complete_policies):
            comparison_status.append({"competitor": competitor, "status": "SKIPPED",
                                      "reason": "SFT and competitor generations are required"})
            continue
        try:
            pair_hashes = {name: hashes[name] for name in ("sft", competitor)}
            rows = complete_judgments(cfg, dataset, competitor, sources, pair_hashes)
            comparison_status.append({"competitor": competitor, "status": "PASS",
                                      "metrics": comparison_metrics(rows, len(sources))})
            candidates.extend(row for row in rows
                              if row["ai_preference"] != row["exact_preference"])
        except FileNotFoundError as error:
            comparison_status.append({"competitor": competitor, "status": "SKIPPED",
                                      "reason": str(error)})
        except Exception as error:
            comparison_status.append({"competitor": competitor, "status": "FAIL",
                                      "error_type": type(error).__name__, "error_message": str(error),
                                      "traceback": traceback.format_exc()})
    complete = all(row["status"] == "PASS" for row in policy_status + comparison_status)
    summary = {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "dataset": dataset, "fixed_prompts": len(sources),
        "dataset_sha256": dataset_digest(sources),
        "generation_protocol": {
            "seed": int(cfg["seed"]), "decoding": cfg["generation"],
            "max_new_tokens": int(cfg["math_max_new_tokens"]),
            "prompt_handling": "full fixed messages; no truncation",
        },
        "policies": policy_status, "comparisons": comparison_status,
        "qualitative_note": "Review verifier/judge disagreements; AI judgments are not ground truth",
    }
    save_json(root / "summary.json", summary)
    write_jsonl(root / "verifier_judge_disagreements.jsonl", candidates)
    _write_csv(root / "policy_metrics.csv", [
        {"policy": item["policy"], **{key: value for key, value in item["metrics"].items()
                                   if not isinstance(value, dict)}}
        for item in policy_status if item["status"] == "PASS"])
    _write_csv(root / "pairwise_metrics.csv", [
        {"competitor": item["competitor"], **item["metrics"]}
        for item in comparison_status if item["status"] == "PASS"])
    from task5_feedback.plots import plot_math_comparison
    if any(item["status"] == "PASS" for item in policy_status):
        plot_math_comparison(policy_status, comparison_status, root / "comparison.png")
    else:
        (root / "comparison.png").unlink(missing_ok=True)
    print(f"Task 5 {dataset} evaluation: {summary['status']}", flush=True)
    return summary
