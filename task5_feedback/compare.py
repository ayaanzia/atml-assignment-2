"""Validate completed Task 5 artifacts and compare GSM8K with SVAMP transfer."""

from __future__ import annotations

import csv
import json
import traceback

from common.logging_utils import save_json
from task5_feedback.diagnostics import (
    complete_diagnostic_judgments, complete_exact, diagnostic_dir,
)
from task5_feedback.generate import complete_generation, result_dir
from task5_feedback.judge import complete_judgments
from task5_feedback.metrics import comparison_metrics, policy_metrics
from task5_feedback.runtime import (
    COMPETITORS, POLICIES, dataset_digest, file_digest, load_fixed_rows,
    preference_rates,
)


def verified_domain(cfg: dict, dataset: str) -> dict:
    sources = load_fixed_rows(cfg, dataset)
    root = result_dir(cfg, dataset)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {dataset} summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or \
            summary.get("dataset_sha256") != dataset_digest(sources) or \
            summary.get("fixed_prompts") != len(sources) or \
            summary.get("generation_protocol") != {
                "seed": int(cfg["seed"]), "decoding": cfg["generation"],
                "max_new_tokens": int(cfg["math_max_new_tokens"]),
                "prompt_handling": "full fixed messages; no truncation",
            }:
        raise ValueError(f"{dataset} summary is incomplete or uses different fixed settings")
    policy_rows = {}
    hashes = {}
    saved_policy = {row["policy"]: row for row in summary["policies"]}
    saved_pairs = {row["competitor"]: row for row in summary["comparisons"]}
    if set(saved_policy) != set(POLICIES) or set(saved_pairs) != set(COMPETITORS):
        raise ValueError(f"{dataset} summary lacks required policies/comparisons")
    for policy in POLICIES:
        rows, state = complete_generation(cfg, dataset, policy, sources)
        metrics = policy_metrics(rows, len(sources))
        if saved_policy[policy].get("status") != "PASS" or saved_policy[policy].get("metrics") != metrics:
            raise ValueError(f"{dataset}/{policy} saved metrics differ from raw generations")
        policy_rows[policy] = metrics
        hashes[policy] = state["generated_sha256"]
    pair_rows = {}
    for competitor in COMPETITORS:
        selected_hashes = {name: hashes[name] for name in ("sft", competitor)}
        rows = complete_judgments(cfg, dataset, competitor, sources, selected_hashes)
        metrics = comparison_metrics(rows, len(sources))
        if saved_pairs[competitor].get("status") != "PASS" or \
                saved_pairs[competitor].get("metrics") != metrics:
            raise ValueError(f"{dataset}/{competitor} saved judge metrics differ from raw pairs")
        pair_rows[competitor] = metrics
    return {"dataset": dataset, "fixed_prompts": len(sources),
            "dataset_sha256": summary["dataset_sha256"],
            "policies": policy_rows, "comparisons": pair_rows}


def verified_diagnostics(cfg: dict) -> dict:
    root = diagnostic_dir(cfg)
    path = root / "summary.json"
    if not path.exists():
        raise FileNotFoundError("Missing controlled-diagnostic summary")
    saved = json.loads(path.read_text(encoding="utf-8"))
    variants, exact_pairs, _ = complete_exact(cfg)
    judged, _ = complete_diagnostic_judgments(cfg)
    if saved.get("status") != "COMPLETE" or saved.get("fixed_variants") != 100 or \
            saved.get("fixed_contrast_pairs") != 80 or saved.get("judge_status") != "PASS" or \
            saved.get("exact_summary_sha256") != file_digest(root / "exact_summary.json") or \
            len(variants) != 100 or len(exact_pairs) != 80 or len(judged) != 80:
        raise ValueError("Controlled diagnostics are incomplete or changed")
    indexed = {(row["variant_type"], row["mechanism"]): row
               for row in saved["contrasts"]}
    from task5_feedback.runtime import CONTRASTS
    if set(indexed) != {(variant, mechanism) for variant in CONTRASTS
                       for mechanism in ("RLVR_exact", "RLAIF_pairwise")}:
        raise ValueError("Controlled diagnostics lack a contrast or feedback mechanism")
    for variant in CONTRASTS:
        for mechanism, source, label in (
            ("RLVR_exact", exact_pairs, "exact_preference"),
            ("RLAIF_pairwise", judged, "ai_preference"),
        ):
            rows = [row for row in source if row["variant_type"] == variant]
            expected = preference_rates([row[label] for row in rows], "A")
            actual = {key: indexed[(variant, mechanism)][key] for key in expected}
            if actual != expected:
                raise ValueError(f"Diagnostic {variant}/{mechanism} metrics differ from raw pairs")
    for mechanism in ("RLVR_exact", "RLAIF_pairwise"):
        expected_sensitivity = {
            "Sreason": indexed[("corrupt_reasoning_correct_final", mechanism)]["better_rate"],
            "Soutcome": indexed[("good_reasoning_wrong_final", mechanism)]["better_rate"],
        }
        if saved.get("sensitivity", {}).get(mechanism) != expected_sensitivity:
            raise ValueError(f"Diagnostic {mechanism} sensitivity differs from raw pairs")
    return {"fixed_problems": 20, "variants": 100, "pairs": 80,
            "sensitivity": saved["sensitivity"],
            "contrasts": saved["contrasts"]}


def compare_feedback(cfg: dict) -> dict:
    root = result_dir(cfg, "gsm").parent
    root.mkdir(parents=True, exist_ok=True)
    domains = {}
    statuses = []
    for dataset in ("gsm", "transfer"):
        try:
            domains[dataset] = verified_domain(cfg, dataset)
            statuses.append({"stage": dataset, "status": "PASS"})
        except FileNotFoundError as error:
            statuses.append({"stage": dataset, "status": "SKIPPED", "reason": str(error)})
        except Exception as error:
            statuses.append({"stage": dataset, "status": "FAIL",
                             "error_type": type(error).__name__, "error_message": str(error),
                             "traceback": traceback.format_exc()})
    try:
        diagnostic = verified_diagnostics(cfg)
        statuses.append({"stage": "diagnostics", "status": "PASS"})
    except FileNotFoundError as error:
        diagnostic = None
        statuses.append({"stage": "diagnostics", "status": "SKIPPED", "reason": str(error)})
    except Exception as error:
        diagnostic = None
        statuses.append({"stage": "diagnostics", "status": "FAIL",
                         "error_type": type(error).__name__, "error_message": str(error),
                         "traceback": traceback.format_exc()})
    transfer_rows = []
    if set(domains) == {"gsm", "transfer"}:
        for policy in POLICIES:
            in_domain = domains["gsm"]["policies"][policy]
            out_domain = domains["transfer"]["policies"][policy]
            row = {
                "policy": policy,
                "gsm_exact_accuracy": in_domain["exact_accuracy"],
                "transfer_exact_accuracy": out_domain["exact_accuracy"],
                "accuracy_drop_gsm_minus_transfer": (
                    in_domain["exact_accuracy"] - out_domain["exact_accuracy"]),
                "gsm_format_compliance": in_domain["format_compliance_rate"],
                "transfer_format_compliance": out_domain["format_compliance_rate"],
                "format_drop_gsm_minus_transfer": (
                    in_domain["format_compliance_rate"] - out_domain["format_compliance_rate"]),
                "gsm_mean_response_tokens": in_domain["mean_response_tokens"],
                "transfer_mean_response_tokens": out_domain["mean_response_tokens"],
                "length_shift_transfer_minus_gsm": (
                    out_domain["mean_response_tokens"] - in_domain["mean_response_tokens"]),
                "gsm_failure_counts": in_domain["failure_counts"],
                "transfer_failure_counts": out_domain["failure_counts"],
            }
            if policy in COMPETITORS:
                gsm_win = domains["gsm"]["comparisons"][policy]["ai_win_rate_vs_sft"]
                transfer_win = domains["transfer"]["comparisons"][policy]["ai_win_rate_vs_sft"]
                row.update({"gsm_ai_win_rate_vs_sft": gsm_win,
                            "transfer_ai_win_rate_vs_sft": transfer_win,
                            "ai_win_rate_drop_gsm_minus_transfer": gsm_win - transfer_win})
            transfer_rows.append(row)
    complete = all(row["status"] == "PASS" for row in statuses)
    summary = {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "stages": statuses, "domains": domains,
        "diagnostics": diagnostic, "transfer_comparison": transfer_rows,
        "scale_warning": "Binary exact accuracy and pairwise AI win rate are different scales",
        "judge_tie_limitation": "The fixed judge maps unparseable output to TIE",
        "qualitative_sources": {
            "gsm": str(result_dir(cfg, "gsm") / "verifier_judge_disagreements.jsonl"),
            "transfer": str(result_dir(cfg, "transfer") / "verifier_judge_disagreements.jsonl"),
            "diagnostics": str(diagnostic_dir(cfg) / "verifier_judge_disagreements.jsonl"),
        },
    }
    save_json(root / "comparison_summary.json", summary)
    csv_path = root / "transfer_comparison.csv"
    if transfer_rows:
        fields = list(dict.fromkeys(key for row in transfer_rows for key, value in row.items()
                                   if not isinstance(value, dict)))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row.get(key) for key in fields} for row in transfer_rows])
        from task5_feedback.plots import plot_transfer
        plot_transfer(transfer_rows, root / "transfer_comparison.png")
    else:
        csv_path.unlink(missing_ok=True)
        (root / "transfer_comparison.png").unlink(missing_ok=True)
    print(f"Task 5 feedback comparison: {summary['status']}", flush=True)
    return summary
