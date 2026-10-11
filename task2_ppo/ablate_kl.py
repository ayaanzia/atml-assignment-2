"""Matched short PPO forks for the released KL-pressure study."""

from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import print_gpu_info, save_json
from task2_ppo.workflow import evaluate_midpoint, run_condition, verify_matched_conditions


def run_kl_study(config_path: str, num_workers: int = 0, resume: bool = False) -> dict:
    cfg = load_yaml(config_path)
    midpoint = evaluate_midpoint(config_path, cfg, resume)
    conditions = []
    for beta in cfg["kl_values"]:
        beta = float(beta)
        name = f"kl_{beta:.2f}".replace(".", "p")
        manifest, train, evaluation = run_condition(
            config_path, cfg, name, int(cfg["fork_updates"]),
            float(cfg["clip_epsilon"]), beta, num_workers, resume,
        )
        updates = read_jsonl(repo_path(cfg["results_dir"]) / name / "train_updates.jsonl")
        metrics = evaluation["metrics"]
        conditions.append({
            "run_name": name, "kl_beta": beta,
            "training_prompt_ids": manifest["prompt_ids"],
            "completed_updates": train["counters"]["completed_updates"],
            "actual_generated_response_tokens": train["counters"]["generated_response_tokens"],
            "training_reward_first": updates[0]["raw_reward"],
            "training_reward_last": updates[-1]["raw_reward"],
            "training_kl_first": updates[0]["sampled_kl_token_mean"],
            "training_kl_last": updates[-1]["sampled_kl_token_mean"],
            "training_entropy_first": updates[0]["token_entropy"],
            "training_entropy_last": updates[-1]["token_entropy"],
            "training_length_first": updates[0]["response_tokens"],
            "training_length_last": updates[-1]["response_tokens"],
            "heldout_reward": metrics["mean_raw_reward"],
            "heldout_sampled_kl_token": metrics["sampled_kl_token_mean"],
            "heldout_entropy": metrics["token_entropy_mean"],
            "heldout_response_tokens": metrics["mean_response_tokens"],
            "heldout_response_tokens_sd": metrics["std_response_tokens"],
        })
    matched = verify_matched_conditions(cfg, [item["run_name"] for item in conditions])
    root = repo_path(cfg["results_dir"])
    midpoint_keys = sorted((row["file_position"], row["prompt_id"], row["seed"])
                           for row in read_jsonl(root / "midpoint" / "heldout_eval" / "generations.jsonl"))
    fork_keys = sorted((row["file_position"], row["prompt_id"], row["seed"])
                       for row in read_jsonl(root / conditions[0]["run_name"] /
                                             "heldout_eval" / "generations.jsonl"))
    if midpoint_keys != fork_keys:
        raise ValueError("Midpoint baseline used different held-out prompt IDs or seeds")
    result = {
        "study": "Task 2 PPO KL pressure",
        "conditions": conditions, "matched": matched,
        "midpoint_heldout_metrics": midpoint["metrics"],
        "qualitative_review": "Inspect the saved held-out response texts; no automated quality labels are assigned.",
    }
    save_json(root / "kl_summary.json", result)
    fields = [key for key in conditions[0] if key != "training_prompt_ids"]
    with (root / "kl_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in conditions])
    from task2_ppo.plots import plot_condition_comparison, plot_training_trajectories
    plot_condition_comparison(conditions, "kl_beta", root / "kl_comparison.png")
    plot_training_trajectories(cfg, [item["run_name"] for item in conditions], root / "kl_trajectories.png")
    print(f"PPO KL comparison: {root / 'kl_summary.json'}", flush=True)
    return result


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_kl_study(args.config, args.num_workers, args.resume)


if __name__ == "__main__":
    main()
