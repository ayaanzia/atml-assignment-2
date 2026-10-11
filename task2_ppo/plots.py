"""Reproducible Task 2 diagnostics; all values come from saved logs."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import read_jsonl, repo_path


def plot_condition_comparison(conditions: list[dict], x_key: str, path) -> None:
    specifications = [
        ("heldout_reward", "Held-out learned reward"),
        ("heldout_sampled_kl_token", "Held-out sampled KL/token"),
        ("heldout_entropy", "Held-out token entropy"),
        ("heldout_response_tokens", "Held-out response tokens"),
    ]
    conditions = sorted(conditions, key=lambda row: row[x_key])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, specifications):
        axis.plot([row[x_key] for row in conditions], [row[key] for row in conditions], marker="o")
        axis.set_xlabel(x_key)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_trajectories(cfg: dict, names: list[str], path) -> None:
    specifications = [
        ("raw_reward", "Learned reward"),
        ("sampled_kl_token_mean", "Sampled KL/token"),
        ("token_entropy", "Token entropy"),
        ("response_tokens", "Response tokens"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("clip_fraction", "Clip fraction"),
        ("policy_gradient_norm_preclip", "Policy grad norm before clipping"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for name in names:
        rows = read_jsonl(repo_path(cfg["results_dir"]) / name / "train_updates.jsonl")
        if not rows:
            raise ValueError(f"No update trajectory for {name}")
        for axis, (key, label) in zip(axes.flat, specifications):
            values = [
                row[key] if key in row else sum(p[key] for p in row["ppo_passes"]) / len(row["ppo_passes"])
                for row in rows
            ]
            axis.plot([row["update"] for row in rows], values, marker=".", label=name)
            axis.set_xlabel("PPO update")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
