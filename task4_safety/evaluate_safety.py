from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    load_yaml(args.config)
    raise NotImplementedError(
        "TODO(student): implement safe-answer, over-refusal, unsafe-compliance, justified-refusal, ambiguous-rate, category-level, length, and manual-vs-AI agreement analyses required by Task 4."
    )


if __name__ == "__main__":
    main()
