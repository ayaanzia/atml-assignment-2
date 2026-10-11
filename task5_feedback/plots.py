"""Compact plots from saved Task 5 numerical summaries."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task5_feedback.runtime import COMPETITORS, CONTRASTS, POLICIES


def plot_math_comparison(policies: list[dict], comparisons: list[dict], path) -> None:
    by_policy = {row["policy"]: row["metrics"] for row in policies if row["status"] == "PASS"}
    by_competitor = {row["competitor"]: row["metrics"] for row in comparisons
                     if row["status"] == "PASS"}
    names = [name for name in POLICIES if name in by_policy]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].bar(x - 0.16, [by_policy[name]["exact_accuracy"] for name in names],
                width=0.32, label="Exact accuracy")
    axes[0].bar(x + 0.16, [by_policy[name]["format_compliance_rate"] for name in names],
                width=0.32, label="Format compliance")
    axes[0].set_xticks(x, [name.upper() for name in names])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Fixed prompt set")
    axes[0].legend(fontsize=8)
    axes[1].bar(names, [by_policy[name]["mean_response_tokens"] for name in names])
    axes[1].set_title("Mean generated tokens")
    axes[1].set_ylabel("Tokens")
    compared = [name for name in COMPETITORS if name in by_competitor]
    if compared:
        axes[2].bar(compared, [by_competitor[name]["ai_win_rate_vs_sft"] for name in compared])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("AI win rate vs SFT\n(ties count one-half)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_diagnostics(contrasts: list[dict], path) -> None:
    by_key = {(row["variant_type"], row["mechanism"]): row for row in contrasts}
    names = [name for name in CONTRASTS if (name, "RLVR_exact") in by_key]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for axis, mechanism in zip(axes, ("RLVR_exact", "RLAIF_pairwise")):
        better = [by_key[(name, mechanism)]["better_rate"] for name in names]
        tie = [by_key[(name, mechanism)]["tie_rate"] for name in names]
        wrong = [by_key[(name, mechanism)]["wrong_rate"] for name in names]
        x = np.arange(len(names))
        axis.bar(x, better, label="Clean preferred")
        axis.bar(x, tie, bottom=better, label="Tie")
        axis.bar(x, wrong, bottom=np.array(better) + np.array(tie), label="Variant preferred")
        axis.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=7)
        axis.set_ylim(0, 1)
        axis.set_title(mechanism)
    axes[0].set_ylabel("Fraction of 20 fixed problems")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_transfer(rows: list[dict], path) -> None:
    names = [row["policy"].upper() for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, key, label in (
        (axes[0], "exact_accuracy", "Exact accuracy"),
        (axes[1], "format_compliance", "Format compliance"),
        (axes[2], "mean_response_tokens", "Mean response tokens"),
    ):
        axis.bar(x - 0.18, [row[f"gsm_{key}"] for row in rows],
                 width=0.36, label="GSM8K")
        axis.bar(x + 0.18, [row[f"transfer_{key}"] for row in rows],
                 width=0.36, label="SVAMP")
        axis.set_xticks(x, names)
        axis.set_title(label)
        if key != "mean_response_tokens":
            axis.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
