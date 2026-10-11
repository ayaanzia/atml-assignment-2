"""Train the supplied length-balanced condition and compare fixed evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import print_gpu_info, save_json


def _invoke(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _evaluation(config: str, cfg: dict, name: str, adapter: str, resume: bool) -> dict:
    result = repo_path(cfg["results_dir"]) / name / "length_eval"
    summary_path = result / "summary.json"
    if not summary_path.exists():
        manifest = result / "evaluation_manifest.json"
        if manifest.exists() and not resume:
            raise FileExistsError(f"Partial {name} length evaluation exists; rerun with --resume")
        args = ["--config", config, "--adapter", adapter, "--name", name,
                "--dataset", "length", "--word-limit"]
        if manifest.exists():
            args.append("--resume")
        _invoke("task1_dpo.evaluate", *args)
    else:
        print(f"Using completed evaluation: {summary_path}", flush=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_length_study(config: str, num_workers: int = 0, checkpoint_every: int = 25,
                     resume: bool = False) -> dict:
    cfg = load_yaml(config)
    standard = repo_path(cfg["standard_output"])
    if not (standard / "adapter_config.json").exists():
        raise FileNotFoundError("Standard DPO adapter is required before the length comparison")
    balanced = repo_path(cfg["length_output"])
    balanced_results = repo_path(cfg["results_dir"]) / "length_balanced"
    if not (balanced_results / "train_summary.json").exists():
        state = balanced / "latest_state.pt"
        partial = state.exists() or (balanced_results / "run_config.json").exists()
        if partial and not resume:
            raise FileExistsError("Partial balanced training exists; rerun with --resume")
        args = ["--config", config, "--run-name", "length_balanced", "--dataset",
                cfg["paths"]["dpo_length_train"], "--num-workers", str(num_workers),
                "--checkpoint-every", str(checkpoint_every)]
        if partial:
            args.append("--resume")
        _invoke("task1_dpo.train", *args)
    else:
        print(f"Using completed balanced training: {balanced_results / 'train_summary.json'}", flush=True)
    standard_eval = _evaluation(config, cfg, "standard", str(standard), resume)
    balanced_eval = _evaluation(config, cfg, "length_balanced", str(balanced), resume)

    root = repo_path(cfg["results_dir"])
    manifests = [json.loads((root / name / "length_eval" / "evaluation_manifest.json").read_text(encoding="utf-8"))
                 for name in ("standard", "length_balanced")]
    if (manifests[0]["prompt_ids"] != manifests[1]["prompt_ids"] or
            manifests[0]["word_limit_ids"] != manifests[1]["word_limit_ids"]):
        raise ValueError("Length comparison conditions used different fixed prompts")
    if any(manifest["generation"] != cfg["generation"] or int(
        manifest["max_generation_tokens"]
    ) != int(cfg["max_generation_tokens"]) for manifest in manifests):
        raise ValueError("Length comparison used inconsistent generation settings")
    for filename in ("generations.jsonl", "word_limit.jsonl"):
        keyed = [sorted((row["file_position"], row["prompt_id"], row["seed"])
                        for row in read_jsonl(root / name / "length_eval" / filename))
                 for name in ("standard", "length_balanced")]
        if keyed[0] != keyed[1]:
            raise ValueError(f"Length comparison used different IDs or seeds in {filename}")
    if standard_eval["word_limit_metrics"] is None or balanced_eval["word_limit_metrics"] is None:
        raise ValueError("Length comparison requires word-limit results for both policies")
    standard_strata = standard_eval["pair_metrics"].get("strata", {})
    balanced_strata = balanced_eval["pair_metrics"].get("strata", {})
    if not standard_strata or set(standard_strata) != set(balanced_strata):
        raise ValueError("Length strata are missing or differ across conditions")
    if any(standard_strata[key]["pairs"] != balanced_strata[key]["pairs"] for key in standard_strata):
        raise ValueError("Length-stratum pair denominators differ across conditions")
    table = []
    for name, evaluation in (("standard", standard_eval), ("length_balanced", balanced_eval)):
        for stratum, metrics in sorted(evaluation["pair_metrics"]["strata"].items()):
            table.append({
                "condition": name, "stratum": stratum, "pairs": metrics["pairs"],
                "mean_dpo_loss": metrics["mean_dpo_loss"],
                "preference_accuracy": metrics["preference_accuracy"],
            })
    summary = {
        "study": "Task 1 standard versus supplied length-balanced training",
        "strata": table,
        "conditions": {
            name: {
                "pair_metrics": evaluation["pair_metrics"],
                "generation_metrics": evaluation["generation_metrics"],
                "word_limit_metrics": evaluation["word_limit_metrics"],
                "evaluation_result": str(root / name / "length_eval" / "summary.json"),
            }
            for name, evaluation in (("standard", standard_eval), ("length_balanced", balanced_eval))
        },
        "generation_protocol": cfg["generation"],
    }
    save_json(root / "length_summary.json", summary)
    with (root / "length_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Length comparison: {root / 'length_summary.json'}", flush=True)
    return summary


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_length_study(args.config, args.num_workers, args.checkpoint_every, args.resume)


if __name__ == "__main__":
    main()
