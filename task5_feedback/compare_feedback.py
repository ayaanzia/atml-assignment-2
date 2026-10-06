from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    load_yaml(args.config)
    raise NotImplementedError(
        "TODO(student): combine the in-domain, controlled-diagnostic, and transfer results into the final RLVR-vs-RLAIF comparison requested in Task 5."
    )


if __name__ == "__main__":
    main()
