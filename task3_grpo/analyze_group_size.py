"""Equal-generation K=2/4/8 diagnostic from the fixed GRPO cache."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import print_gpu_info, save_json
from task3_grpo.grpo import group_relative_advantages


TOLERANCE = 1e-6


def load_k8_cache(path) -> dict[str, list[dict]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("The fixed GRPO group cache is empty")
    required = {"source_index", "prompt_id", "generation_index", "completion",
                "completion_tokens", "reward", "terminated_with_eos", "clipped_at_max"}
    by_prompt = defaultdict(list)
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"GRPO cache row lacks {sorted(required - set(row))}")
        if not math.isfinite(float(row["reward"])) or int(row["completion_tokens"]) < 1:
            raise ValueError("GRPO cache has nonfinite reward or empty completion")
        by_prompt[str(row["source_index"])].append(row)
    for source_index, group in by_prompt.items():
        if len(group) != 8 or len({row["prompt_id"] for row in group}) != 1 or \
                {int(row["generation_index"]) for row in group} != set(range(8)):
            raise ValueError(f"Expected eight uniquely indexed completions for prompt {source_index}")
        group.sort(key=lambda row: int(row["generation_index"]))
    if len(by_prompt) != 24:
        raise ValueError(f"Expected 24 prompts in the pinned GRPO cache, got {len(by_prompt)}")
    return dict(by_prompt)


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[dict]:
    """Partition each prompt's eight completions into consecutive nonoverlapping groups."""
    if k not in {2, 4, 8}:
        raise ValueError("Group size must be 2, 4, or 8")
    groups = []
    for source_index in sorted(by_prompt, key=lambda value: int(value)):
        rows = by_prompt[source_index]
        if len(rows) != 8:
            raise ValueError(f"Prompt {source_index} does not have eight cached completions")
        ordered = sorted(rows, key=lambda row: int(row["generation_index"]))
        for start in range(0, 8, k):
            selected = ordered[start:start + k]
            groups.append({
                "source_index": source_index,
                "prompt_id": selected[0]["prompt_id"],
                "group_index": start // k,
                "k": k,
                "completion_rows": selected,
            })
    if sum(len(group["completion_rows"]) for group in groups) != sum(map(len, by_prompt.values())):
        raise RuntimeError("Regrouping changed the cached generation budget")
    return groups


def _difficulty_bins(by_prompt: dict[str, list[dict]]) -> tuple[dict[str, str], dict]:
    means = sorted((statistics.fmean(float(row["reward"]) for row in group), source_index)
                   for source_index, group in by_prompt.items())
    if len(means) < 2:
        raise ValueError("Need at least two cached prompts for difficulty bins")
    midpoint = len(means) // 2
    labels = {source_index: ("lower_cached_reward" if index < midpoint else "higher_cached_reward")
              for index, (_, source_index) in enumerate(means)}
    threshold = (means[midpoint - 1][0] + means[midpoint][0]) / 2
    return labels, {
        "rule": "rank prompts by mean reward across all eight cache completions; lower and higher halves; ties by source_index",
        "median_threshold": threshold,
        "lower_prompt_ids": [source_index for _, source_index in means[:midpoint]],
        "higher_prompt_ids": [source_index for _, source_index in means[midpoint:]],
        "interpretation": "Cached reward is a difficulty proxy, not gold task accuracy",
    }


def _metrics(groups: list[dict]) -> dict:
    if not groups:
        raise ValueError("Cannot summarize an empty GRPO group set")
    advantages = [value for group in groups for value in group["advantages"]]
    return {
        "groups": len(groups),
        "generations": sum(group["k"] for group in groups),
        "informative_group_rate": sum(group["informative"] for group in groups) / len(groups),
        "uninformative_group_fraction": sum(not group["informative"] for group in groups) / len(groups),
        "mean_within_group_reward_std": statistics.fmean(group["reward_std"] for group in groups),
        "relative_signal_variance": statistics.pvariance(advantages),
        "mean_completion_tokens": statistics.fmean(
            tokens for group in groups for tokens in group["completion_tokens"]),
    }


def analyze_group_size(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    by_prompt = load_k8_cache(cfg["group_cache"])
    labels, difficulty = _difficulty_bins(by_prompt)
    expected_generations = sum(len(group) for group in by_prompt.values())
    all_groups = []
    table = []
    for k in cfg["group_sizes"]:
        k = int(k)
        groups = regroup_equal_generation_budget(by_prompt, k)
        for group in groups:
            rows = group.pop("completion_rows")
            rewards = [float(row["reward"]) for row in rows]
            values = torch.tensor(rewards, dtype=torch.float32)
            advantages = group_relative_advantages(values, torch.zeros(k, dtype=torch.long)).tolist()
            deviation = float(values.std(unbiased=False).item())
            group.update({
                "difficulty_bin": labels[group["source_index"]],
                "generation_indices": [int(row["generation_index"]) for row in rows],
                "rewards": rewards, "advantages": advantages,
                "reward_std": deviation, "informative": deviation > TOLERANCE,
                "completion_tokens": [int(row["completion_tokens"]) for row in rows],
                "clipped_at_max": [bool(row["clipped_at_max"]) for row in rows],
            })
        if sum(group["k"] for group in groups) != expected_generations:
            raise RuntimeError(f"K={k} did not preserve the fixed generation budget")
        all_groups.extend(groups)
        for label in ("all", "lower_cached_reward", "higher_cached_reward"):
            selected = groups if label == "all" else [g for g in groups if g["difficulty_bin"] == label]
            table.append({"k": k, "difficulty_bin": label, **_metrics(selected)})
    output = repo_path(cfg["results_dir"]) / "group_size"
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "groups.jsonl", all_groups)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {
        "cache": str(repo_path(cfg["group_cache"])),
        "cached_prompts": len(by_prompt), "fixed_generations_per_condition": expected_generations,
        "cached_completions_clipped_at_generation_cap": sum(
            bool(row["clipped_at_max"]) for group in by_prompt.values() for row in group),
        "partition": "consecutive nonoverlapping generation indices within each cached prompt",
        "reward_std_tolerance": TOLERANCE,
        "difficulty": difficulty, "conditions": table,
    }
    save_json(output / "summary.json", summary)
    from task3_grpo.plots import plot_group_size
    plot_group_size(table, output / "group_size.png")
    print(f"Analyzed GRPO K={list(map(int, cfg['group_sizes']))} using {expected_generations} cached generations each", flush=True)
    return summary


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()
    analyze_group_size(args.config)


if __name__ == "__main__":
    main()
