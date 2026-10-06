from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    raise NotImplementedError(
        "TODO(student): train the length-balanced condition and implement the required per-stratum and word-limit analyses."
    )


if __name__ == "__main__":
    main()
