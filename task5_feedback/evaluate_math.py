"""Entry point for fixed GSM8K/SVAMP generation, pairwise judging, and metrics."""

from __future__ import annotations

import argparse
import gc
import traceback

import torch

from common.data import load_yaml
from common.logging_utils import print_gpu_info, save_json, set_seed
from task5_feedback.generate import generate_policy, result_dir
from task5_feedback.judge import judge_dataset
from task5_feedback.metrics import aggregate_dataset
from task5_feedback.runtime import DATASETS, POLICIES, adapter_digest, policy_paths


def run_math_evaluation(config_path: str, dataset: str, stage: str = "all",
                        policy: str | None = None, resume: bool = False) -> dict:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    root = result_dir(cfg, dataset)
    root.mkdir(parents=True, exist_ok=True)
    results = {"dataset": dataset, "requested_stage": stage, "requested_policy": policy,
               "completion_scope": "requested stage and policy only; full Task 5 requires both domains and diagnostics",
               "generation": None,
               "judging": None, "aggregation": None}
    incomplete = []
    if stage in {"all", "generate"}:
        statuses = []
        names = (policy,) if policy else POLICIES
        for name in names:
            try:
                adapter_digest(policy_paths(cfg)[name])
            except FileNotFoundError as error:
                statuses.append({"policy": name, "status": "SKIPPED", "reason": str(error)})
                continue
            try:
                summary = generate_policy(cfg, dataset, name, resume)
                statuses.append({"policy": name, "status": "PASS", "records": summary["records"]})
            except Exception as error:
                statuses.append({"policy": name, "status": "FAIL",
                                 "error_type": type(error).__name__, "error_message": str(error),
                                 "traceback": traceback.format_exc()})
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        complete = all(row["status"] == "PASS" for row in statuses)
        save_json(root / "generation_status.json", {"dataset": dataset,
                  "policies": statuses, "requested_generation_complete": complete})
        results["generation"] = statuses
        if not complete:
            incomplete.append("generation")
    if stage in {"all", "judge"}:
        try:
            statuses = judge_dataset(cfg, dataset, resume)
        except Exception as error:
            statuses = [{"competitor": name, "status": "FAIL",
                         "error_type": type(error).__name__, "error_message": str(error),
                         "traceback": traceback.format_exc()} for name in ("rlvr", "rlaif")]
        results["judging"] = statuses
        if not all(row["status"] == "PASS" for row in statuses):
            incomplete.append("judging")
    if stage in {"all", "aggregate"}:
        try:
            summary = aggregate_dataset(cfg, dataset)
            results["aggregation"] = summary["status"]
        except Exception as error:
            results["aggregation"] = {"status": "FAIL",
                                      "error_type": type(error).__name__,
                                      "error_message": str(error),
                                      "traceback": traceback.format_exc()}
        if results["aggregation"] != "COMPLETE":
            incomplete.append("aggregation")
    results["status"] = "COMPLETE" if not incomplete else "INCOMPLETE"
    results["incomplete_stages"] = incomplete
    save_json(root / "execution_status.json", results)
    return results


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--dataset", choices=DATASETS, default="gsm")
    parser.add_argument("--stage", choices=("all", "generate", "judge", "aggregate"), default="all")
    parser.add_argument("--policy", choices=POLICIES,
                        help="Generate only this frozen policy; requires --stage generate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.policy and args.stage != "generate":
        parser.error("--policy requires --stage generate")
    result = run_math_evaluation(args.config, args.dataset, args.stage,
                                 args.policy, args.resume)
    if result["status"] != "COMPLETE":
        raise RuntimeError(f"Task 5 {args.dataset} stages incomplete; inspect execution_status.json")


if __name__ == "__main__":
    main()
