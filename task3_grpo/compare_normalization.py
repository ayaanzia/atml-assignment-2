"""Matched canonical and Dr. GRPO short continuations from one midpoint."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import print_gpu_info, save_json
from task3_grpo.runtime import LENGTH_BINS, scheduled_rows


CONDITIONS = (("norm_grpo", "grpo"), ("norm_dr_grpo", "dr_grpo"))


def _invoke(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _evaluate(config_path: str, cfg: dict, name: str, adapter: str, resume: bool) -> dict:
    folder = repo_path(cfg["results_dir"]) / name / "heldout_eval"
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        partial = (folder / "evaluation_manifest.json").exists()
        if partial and not resume:
            raise FileExistsError(f"Partial {name} GRPO evaluation exists; rerun with --resume")
        args = ["--config", config_path, "--adapter", adapter, "--name", name]
        if partial:
            args.append("--resume")
        _invoke("task3_grpo.evaluate", *args)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _train_fork(config_path: str, cfg: dict, name: str, loss_type: str,
                num_workers: int, resume: bool) -> tuple[dict, dict]:
    result = repo_path(cfg["results_dir"]) / name
    output = repo_path(f"outputs/task3_grpo/{name}")
    summary_path = result / "train_summary.json"
    if not summary_path.exists():
        partial = (result / "run_config.json").exists() or (output / "latest_state.pt").exists()
        if partial and not resume:
            raise FileExistsError(f"Partial {name} GRPO training exists; rerun with --resume")
        args = ["--config", config_path, "--run-name", name,
                "--updates", str(cfg["fork_updates"]), "--loss-type", loss_type,
                "--num-workers", str(num_workers)]
        if partial:
            args.append("--resume")
        _invoke("task3_grpo.continue_train", *args)
    manifest = json.loads((result / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = dict(cfg)
    expected["updates"] = int(cfg["fork_updates"])
    if manifest["config"] != expected or manifest["loss_type"] != loss_type or \
            summary["signature"] != manifest["signature"] or \
            summary["counters"]["completed_updates"] != int(cfg["fork_updates"]) or \
            summary["counters"]["optimizer_steps"] != int(cfg["fork_updates"]) or \
            summary["counters"]["generated_completions"] != int(cfg["fork_updates"]) * int(cfg["num_generations"]):
        raise ValueError(f"Completed {name} run does not match its requested normalization fork")
    return manifest, summary


def _length_rows(cfg: dict, name: str, loss_type: str) -> list[dict]:
    updates = read_jsonl(repo_path(cfg["results_dir"]) / name / "train_updates.jsonl")
    rows = []
    for update in updates:
        for completion in update["completions"]:
            rows.append({
                "run_name": name, "loss_type": loss_type,
                "update": update["update"], "prompt_id": update["prompt_id"],
                "generation_index": completion["generation_index"],
                "seed": completion["seed"], "response_tokens": completion["response_tokens"],
                "length_bin": completion["length_bin"],
                "usable_response_tokens": completion["usable_response_tokens"],
                "masked_at_generation_cap": completion["truncated_at_generation_cap"],
                "advantage": completion["advantage"],
                "abs_policy_objective_contribution": completion["policy_objective_contribution_abs"],
            })
    return rows


def _verify_matched(cfg: dict, names: list[str]) -> dict:
    root = repo_path(cfg["results_dir"])
    schedules, training_keys, heldout_ids, heldout_keys, heldout_protocols = [], [], [], [], []
    for name in names:
        manifest = json.loads((root / name / "run_config.json").read_text(encoding="utf-8"))
        schedules.append(manifest["prompt_ids"])
        updates = read_jsonl(root / name / "train_updates.jsonl")
        training_keys.append([(row["update"], row["prompt_id"], completion["generation_index"],
                               completion["seed"])
                              for row in updates for completion in row["completions"]])
        expected_keys = [(index + 1, prompt_id, completion_index, seed)
                         for index, (prompt_id, seeds) in enumerate(
                             zip(manifest["prompt_ids"], manifest["training_seeds"]))
                         for completion_index, seed in enumerate(seeds)]
        if training_keys[-1] != expected_keys:
            raise ValueError(f"Saved {name} completions differ from its planned prompt/seed schedule")
        evaluation_manifest = json.loads(
            (root / name / "heldout_eval" / "evaluation_manifest.json").read_text(encoding="utf-8"))
        heldout_ids.append(evaluation_manifest["prompt_ids"])
        heldout_protocols.append((evaluation_manifest["generation"],
                                  evaluation_manifest["prompt_cap"],
                                  evaluation_manifest["response_cap"],
                                  evaluation_manifest["reward_input_cap"]))
        heldout_keys.append([(row["file_position"], row["prompt_id"], row["seed"])
                             for row in read_jsonl(root / name / "heldout_eval" / "generations.jsonl")])
    if len(set(map(tuple, schedules))) != 1 or training_keys[0] != training_keys[1]:
        raise ValueError("Normalization forks did not match training prompts and completion seeds")
    if heldout_ids[0] != heldout_ids[1] or heldout_keys[0] != heldout_keys[1] or \
            heldout_protocols[0] != heldout_protocols[1]:
        raise ValueError("Normalization forks used different held-out prompts, seeds, or caps")
    selected = scheduled_rows(read_jsonl(cfg["paths"]["rl_prompt_train"]),
                              int(cfg["seed"]), int(cfg["fork_updates"]))
    if schedules[0] != [row["prompt_id"] for row in selected]:
        raise ValueError("Short forks did not use the first eight scheduled prompts")
    return {"training_prompt_ids": schedules[0], "heldout_prompt_ids": heldout_ids[0],
            "planned_completion_attempts_per_fork": int(cfg["fork_updates"]) * int(cfg["num_generations"]),
            "heldout_protocol": {
                "generation": heldout_protocols[0][0],
                "prompt_cap": heldout_protocols[0][1],
                "response_cap": heldout_protocols[0][2],
                "reward_input_cap": heldout_protocols[0][3],
            }}


def compare_normalization(config_path: str, num_workers: int = 0,
                          resume: bool = False) -> dict:
    cfg = load_yaml(config_path)
    midpoint = _evaluate(config_path, cfg, "midpoint", cfg["paths"]["grpo_midpoint_policy"], resume)
    conditions = []
    all_lengths = []
    for name, loss_type in CONDITIONS:
        manifest, train = _train_fork(config_path, cfg, name, loss_type, num_workers, resume)
        evaluation = _evaluate(config_path, cfg, name, f"outputs/task3_grpo/{name}", resume)
        metrics = evaluation["metrics"]
        conditions.append({
            "run_name": name, "loss_type": loss_type,
            "completed_updates": train["counters"]["completed_updates"],
            "planned_completion_attempts": manifest["planned_completion_attempts"],
            "actual_generated_response_tokens": train["counters"]["generated_response_tokens"],
            "masked_completions": train["counters"]["masked_completions"],
            "uninformative_groups": train["counters"]["uninformative_groups"],
            "heldout_reward": metrics["mean_raw_reward"],
            "heldout_sampled_kl_token": metrics["sampled_kl_token_mean"],
            "heldout_response_tokens": metrics["mean_response_tokens"],
            "heldout_response_tokens_sd": metrics["std_response_tokens"],
            "heldout_entropy": metrics["token_entropy_mean"],
        })
        all_lengths.extend(_length_rows(cfg, name, loss_type))
    matched = _verify_matched(cfg, [name for name, _ in CONDITIONS])
    root = repo_path(cfg["results_dir"])
    midpoint_records = read_jsonl(root / "midpoint" / "heldout_eval" / "generations.jsonl")
    fork_records = read_jsonl(root / CONDITIONS[0][0] / "heldout_eval" / "generations.jsonl")
    if [(r["file_position"], r["prompt_id"], r["seed"]) for r in midpoint_records] != \
            [(r["file_position"], r["prompt_id"], r["seed"]) for r in fork_records]:
        raise ValueError("Midpoint held-out baseline used different prompts or seeds")
    length_table = []
    for name, loss_type in CONDITIONS:
        for label in LENGTH_BINS:
            selected = [row for row in all_lengths if row["run_name"] == name and
                        row["length_bin"] == label and row["usable_response_tokens"] > 0]
            length_table.append({
                "run_name": name, "loss_type": loss_type, "length_bin": label,
                "usable_completions": len(selected),
                "mean_abs_policy_contribution": (
                    statistics.fmean(row["abs_policy_objective_contribution"] for row in selected)
                    if selected else None),
                "mean_response_tokens": (
                    statistics.fmean(row["response_tokens"] for row in selected)
                    if selected else None),
            })
    write_jsonl(root / "normalization_length_contributions.jsonl", all_lengths)
    with (root / "normalization_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    with (root / "normalization_length_bins.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(length_table[0]))
        writer.writeheader()
        writer.writerows(length_table)
    result = {
        "study": "Matched canonical versus Dr. GRPO normalization",
        "midpoint_heldout_metrics": midpoint["metrics"],
        "conditions": conditions, "matched": matched,
        "length_bins": length_table,
        "length_statistic": "absolute per-completion policy objective contribution after 1/K group weighting; masked completions excluded from bin means",
        "token_budget_note": "Matched planned completion attempts/cap/seeds; EOS can change actual generated-token totals",
        "qualitative_review": "Inspect saved held-out texts and group logs; no automated quality labels are assigned",
    }
    save_json(root / "normalization_summary.json", result)
    from task3_grpo.plots import plot_length_contributions, plot_normalization, plot_training_trajectories
    plot_normalization(conditions, root / "normalization_heldout.png")
    plot_length_contributions(length_table, root / "normalization_length_contributions.png")
    plot_training_trajectories(cfg, [name for name, _ in CONDITIONS],
                               root / "normalization_trajectories.png")
    print(f"GRPO normalization comparison: {root / 'normalization_summary.json'}", flush=True)
    return result


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    compare_normalization(args.config, args.num_workers, args.resume)


if __name__ == "__main__":
    main()
