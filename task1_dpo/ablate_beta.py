"""Run the three independent, matched Task 1 beta conditions."""

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


def _prepare_run(config: str, cfg: dict, name: str, beta: float,
                 num_workers: int, checkpoint_every: int, resume: bool) -> None:
    output = repo_path(f"outputs/task1_dpo/{name}")
    result = repo_path(cfg["results_dir"]) / name
    train_done = result / "train_summary.json"
    if not train_done.exists():
        state = output / "latest_state.pt"
        partial = state.exists() or (result / "run_config.json").exists()
        if partial and not resume:
            raise FileExistsError(f"Partial {name} training exists; rerun with --resume")
        args = ["--config", config, "--run-name", name, "--beta", str(beta),
                "--max-examples", str(cfg["short_ablation_examples"]),
                "--num-workers", str(num_workers), "--checkpoint-every", str(checkpoint_every)]
        if partial:
            args.append("--resume")
        _invoke("task1_dpo.train", *args)
    else:
        print(f"Using completed training: {train_done}", flush=True)

    summary = result / "standard_eval" / "summary.json"
    if not summary.exists():
        manifest = result / "standard_eval" / "evaluation_manifest.json"
        if manifest.exists() and not resume:
            raise FileExistsError(f"Partial {name} evaluation exists; rerun with --resume")
        args = ["--config", config, "--adapter", str(output), "--name", name]
        if manifest.exists():
            args.append("--resume")
        _invoke("task1_dpo.evaluate", *args)
    else:
        print(f"Using completed evaluation: {summary}", flush=True)


def run_beta_study(config: str, num_workers: int = 0, checkpoint_every: int = 25,
                   resume: bool = False) -> dict:
    cfg = load_yaml(config)
    results_root = repo_path(cfg["results_dir"])
    conditions = []
    for beta in cfg["betas"]:
        beta = float(beta)
        name = f"beta_{beta:.2f}".replace(".", "p")
        _prepare_run(config, cfg, name, beta, num_workers, checkpoint_every, resume)
        run_dir = results_root / name
        manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        train = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run_dir / "standard_eval" / "summary.json").read_text(encoding="utf-8"))
        if manifest["configuration"] != cfg:
            raise ValueError(f"Completed {name} training used a different configuration")
        if float(manifest["beta"]) != beta or float(evaluation["beta"]) != beta:
            raise ValueError(f"Beta mismatch in {name} artifacts")
        if manifest["selection"]["eligible_pairs"] != int(cfg["short_ablation_examples"]):
            raise ValueError(f"{name} did not train on the required short-run pair count")
        conditions.append({
            "run_name": name, "beta": beta,
            "selected_ids_sha256": manifest["selection"]["selected_ids_sha256"],
            "eligible_train_pairs": manifest["selection"]["eligible_pairs"],
            "optimizer_updates": train["optimizer_updates"],
            "heldout_pairs": evaluation["pair_metrics"]["pairs"],
            "heldout_dpo_loss": evaluation["pair_metrics"]["mean_dpo_loss"],
            "heldout_preference_accuracy": evaluation["pair_metrics"]["preference_accuracy"],
            "generation_prompts": evaluation["generation_metrics"]["prompts"],
            "mean_reward_model_score": evaluation["generation_metrics"]["mean_reward_model_score"],
            "sampled_kl_token_mean": evaluation["generation_metrics"]["sampled_kl_token_mean"],
            "mean_response_tokens": evaluation["generation_metrics"]["mean_response_tokens"],
            "std_response_tokens": evaluation["generation_metrics"]["std_response_tokens"],
            "train_result": str(run_dir / "train_summary.json"),
            "evaluation_result": str(run_dir / "standard_eval" / "summary.json"),
        })
    if len({row["selected_ids_sha256"] for row in conditions}) != 1:
        raise ValueError("Beta conditions selected different training pairs")
    prompt_lists, generation_keys = [], []
    for item in conditions:
        manifest_path = results_root / item["run_name"] / "standard_eval" / "evaluation_manifest.json"
        evaluation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if evaluation_manifest["generation"] != cfg["generation"] or int(
            evaluation_manifest["max_generation_tokens"]
        ) != int(cfg["max_generation_tokens"]):
            raise ValueError(f"Generation settings differ for {item['run_name']}")
        prompt_lists.append(evaluation_manifest["prompt_ids"])
        generation_rows = read_jsonl(results_root / item["run_name"] / "standard_eval" / "generations.jsonl")
        generation_keys.append(sorted((row["file_position"], row["prompt_id"], row["seed"])
                                      for row in generation_rows))
    if any(ids != prompt_lists[0] for ids in prompt_lists[1:]):
        raise ValueError("Beta conditions did not use identical held-out prompt IDs")
    if any(keys != generation_keys[0] for keys in generation_keys[1:]):
        raise ValueError("Beta conditions did not use identical generated prompt IDs and seeds")
    summary = {
        "study": "Task 1 short beta forks from fresh initialization",
        "short_run_examples": int(cfg["short_ablation_examples"]),
        "selected_ids_sha256": conditions[0]["selected_ids_sha256"],
        "generation_protocol": cfg["generation"],
        "conditions": conditions,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    save_json(results_root / "beta_summary.json", summary)
    table_path = results_root / "beta_summary.csv"
    fields = [key for key in conditions[0] if key not in {"train_result", "evaluation_result"}]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in conditions])
    print(f"Beta comparison: {results_root / 'beta_summary.json'}", flush=True)
    return summary


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_beta_study(args.config, args.num_workers, args.checkpoint_every, args.resume)


if __name__ == "__main__":
    main()
