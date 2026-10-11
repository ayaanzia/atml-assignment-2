"""Small diagnostic plots for the fixed Task 4 safety comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task4_safety.runtime import POLICIES


def plot_safety_rates(rows: list[dict], path: str | Path) -> None:
    """Plot the four required rates with their fixed SAFE/UNSAFE denominators."""
    by_policy = {row["policy"]: row for row in rows}
    policies = [name for name in POLICIES if name in by_policy]
    metrics = (
        ("safe_answer_rate", "Safe answer / 250 SAFE"),
        ("safe_over_refusal_rate", "Over-refusal / 250 SAFE"),
        ("unsafe_compliance_rate", "Unsafe compliance / 200 UNSAFE"),
        ("unsafe_justified_refusal_rate", "Justified refusal / 200 UNSAFE"),
        ("ambiguous_judge_rate", "Ambiguous / 450 all"),
    )
    x = np.arange(len(policies))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (metric, label) in enumerate(metrics):
        values = [by_policy[name][metric] for name in policies]
        ax.bar(x + (index - 2) * width, values, width, label=label)
    ax.set_xticks(x, [name.upper() for name in policies])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of fixed prompts")
    ax.set_title("XSTest judge labels by frozen policy")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_category_failures(rows: list[dict], path: str | Path) -> None:
    """Show category rates for the two calibration failures, with missing cells blank."""
    categories = sorted({row["type"] for row in rows})
    policies = [name for name in POLICIES if any(row["policy"] == name for row in rows)]
    indexed = {(row["policy"], row["type"]): row for row in rows}
    values = np.full((len(categories), len(policies)), np.nan)
    for row_index, category in enumerate(categories):
        for column_index, policy in enumerate(policies):
            row = indexed.get((policy, category))
            if row is None:
                continue
            key = "OVER_REFUSAL_rate" if row["benchmark_class"] == "SAFE" else "UNSAFE_COMPLIANCE_rate"
            values[row_index, column_index] = row[key]
    fig, ax = plt.subplots(figsize=(7, 8))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(policies)), [name.upper() for name in policies])
    ax.set_yticks(range(len(categories)), categories, fontsize=8)
    ax.set_title("Failure label rate by XSTest category\nSAFE: over-refusal; UNSAFE: unsafe compliance")
    fig.colorbar(image, ax=ax, label="Fraction of 25 prompts")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
