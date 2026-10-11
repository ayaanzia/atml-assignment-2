"""Small reproducible Task 3 plots made only from saved numerical records."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import read_jsonl, repo_path
from task3_grpo.runtime import LENGTH_BINS


def plot_training_trajectories(cfg: dict, names: list[str], path) -> None:
    specifications = [
        ("mean_raw_reward", "Mean learned reward"),
        ("sampled_kl_token_mean", "Sampled KL/token"),
        ("within_group_reward_std", "Within-group reward SD"),
        ("informative_group", "Informative group"),
        ("policy_loss", "Policy loss"),
        ("policy_gradient_norm_preclip", "Preclip gradient norm"),
        ("token_entropy_mean", "Token entropy"),
        ("mean_response_tokens", "Mean response tokens"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for name in names:
        records = read_jsonl(repo_path(cfg["results_dir"]) / name / "train_updates.jsonl")
        if not records:
            raise ValueError(f"No GRPO update trajectory for {name}")
        for axis, (key, label) in zip(axes.flat, specifications):
            axis.plot([row["update"] for row in records],
                      [float(row[key]) for row in records], marker=".", label=name)
            axis.set_xlabel("GRPO update")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_group_size(rows: list[dict], path) -> None:
    specs = [
        ("informative_group_rate", "Informative-group rate"),
        ("mean_within_group_reward_std", "Mean reward SD"),
        ("relative_signal_variance", "Relative-signal variance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for label in ("all", "lower_cached_reward", "higher_cached_reward"):
        selected = sorted((row for row in rows if row["difficulty_bin"] == label),
                          key=lambda row: row["k"])
        for axis, (key, title) in zip(axes, specs):
            axis.plot([row["k"] for row in selected], [row[key] for row in selected],
                      marker="o", label=label)
            axis.set_xticks([2, 4, 8])
            axis.set_xlabel("Completions per prompt group")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_normalization(conditions: list[dict], path) -> None:
    specs = [
        ("heldout_reward", "Held-out learned reward"),
        ("heldout_sampled_kl_token", "Held-out sampled KL/token"),
        ("heldout_response_tokens", "Held-out response tokens"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    names = [row["run_name"] for row in conditions]
    for axis, (key, label) in zip(axes, specs):
        axis.bar(names, [row[key] for row in conditions])
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_length_contributions(rows: list[dict], path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for loss_type in ("grpo", "dr_grpo"):
        selected = {row["length_bin"]: row for row in rows if row["loss_type"] == loss_type}
        axis.plot(range(len(LENGTH_BINS)),
                  [selected.get(label, {}).get("mean_abs_policy_contribution", float("nan"))
                   for label in LENGTH_BINS],
                  marker="o", label=loss_type)
    axis.set_xticks(range(len(LENGTH_BINS)), LENGTH_BINS)
    axis.set_xlabel("Generated response tokens")
    axis.set_ylabel("Mean absolute policy objective contribution")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
