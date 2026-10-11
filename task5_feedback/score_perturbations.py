"""Entry point for the fixed 100-response controlled feedback diagnostic."""

from __future__ import annotations

import argparse
import traceback

from common.data import load_yaml
from common.logging_utils import print_gpu_info, save_json
from task5_feedback.diagnostics import (
    aggregate_diagnostics, diagnostic_dir, judge_diagnostics, score_exact,
)


def run_diagnostics(config_path: str, stage: str = "all",
                    resume: bool = False) -> dict:
    cfg = load_yaml(config_path)
    root = diagnostic_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    statuses = []
    if stage in {"all", "exact"}:
        try:
            result = score_exact(cfg, resume)
            statuses.append({"stage": "exact", "status": "PASS",
                             "pairs": result["pairs"]})
        except Exception as error:
            statuses.append({"stage": "exact", "status": "FAIL",
                             "error_type": type(error).__name__, "error_message": str(error),
                             "traceback": traceback.format_exc()})
    if stage in {"all", "judge"}:
        try:
            result = judge_diagnostics(cfg, resume)
            statuses.append({"stage": "judge", "status": "PASS",
                             "pairs": result["pairs"]})
        except FileNotFoundError as error:
            statuses.append({"stage": "judge", "status": "SKIPPED", "reason": str(error)})
        except Exception as error:
            statuses.append({"stage": "judge", "status": "FAIL",
                             "error_type": type(error).__name__, "error_message": str(error),
                             "traceback": traceback.format_exc()})
    if stage in {"all", "aggregate"}:
        try:
            result = aggregate_diagnostics(cfg)
            statuses.append({"stage": "aggregate", "status": result["status"]})
        except FileNotFoundError as error:
            statuses.append({"stage": "aggregate", "status": "SKIPPED", "reason": str(error)})
        except Exception as error:
            statuses.append({"stage": "aggregate", "status": "FAIL",
                             "error_type": type(error).__name__, "error_message": str(error),
                             "traceback": traceback.format_exc()})
    result = {"requested_stage": stage,
              "completion_scope": "requested stage only; full Task 5 requires diagnostics and both math domains",
              "stages": statuses,
              "status": "COMPLETE" if all(item["status"] in {"PASS", "COMPLETE"}
                                          for item in statuses) else "INCOMPLETE"}
    save_json(root / "execution_status.json", result)
    return result


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    parser.add_argument("--stage", choices=("all", "exact", "judge", "aggregate"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_diagnostics(args.config, args.stage, args.resume)
    if result["status"] != "COMPLETE":
        raise RuntimeError("Task 5 diagnostics incomplete; inspect execution_status.json")


if __name__ == "__main__":
    main()
