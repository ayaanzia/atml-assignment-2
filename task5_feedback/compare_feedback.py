"""Entry point for the fixed RLVR-versus-RLAIF feedback comparison."""

from __future__ import annotations

import argparse

from common.data import load_yaml
from common.logging_utils import print_gpu_info
from task5_feedback.compare import compare_feedback


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    args = parser.parse_args()
    result = compare_feedback(load_yaml(args.config))
    if result["status"] != "COMPLETE":
        raise RuntimeError("Task 5 comparison incomplete; inspect comparison_summary.json")


if __name__ == "__main__":
    main()
