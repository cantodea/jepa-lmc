"""Plot the already-scored quality study (optional matplotlib dependency)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Figure already exists; choose a fresh output path.")
    result = json.loads(args.summary.read_text())
    seeds = [20260804, 20260805, 20260806]
    names = ["long_constant", "long_cosine", "capacity_cosine"]
    labels = ["More epochs", "+ LR decay", "+ Capacity"]
    colors = ["#cb5825", "#376fb2", "#218670"]
    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1, 1.1),
        left=0.065,
        right=0.985,
        top=0.85,
        bottom=0.12,
        wspace=0.23,
        hspace=0.4,
    )
    for index, seed in enumerate(seeds):
        axis = fig.add_subplot(grid[0, index])
        baseline = next(
            r["validation_accuracy"]
            for r in result["model_scores"]
            if r["candidate"] == "baseline" and r["seed"] == seed
        )
        axis.axhline(100 * baseline, color="#777777", ls=":", label="Baseline")
        axis.axhline(99, color="#b52b38", ls="--", lw=1, label="99% target")
        for name, label, color in zip(names, labels, colors, strict=True):
            values = next(
                r["validation_checkpoints"]
                for r in result["convergence"]
                if r["candidate"] == name and r["seed"] == seed
            )
            if name != "capacity_cosine":
                values = [{"epoch": 100, "accuracy": baseline}, *values]
            axis.plot(
                [v["epoch"] for v in values],
                [100 * v["accuracy"] for v in values],
                color=color,
                marker="o",
                ms=4,
                lw=1.8,
                label=label,
            )
        axis.set(
            title=f"Seed {seed}",
            xlabel="Total epochs",
            ylim=(85, 100),
            xticks=[100, 150, 200, 250, 300],
        )
        if index == 0:
            axis.set_ylabel("Original-validation Top-1 (%)")
        axis.grid(axis="y", alpha=0.2)
    bottom = fig.add_subplot(grid[1, :])
    candidate_names = ["baseline", *names]
    for offset, split, label, color in (
        (-0.25, "train", "Train (seen maps)", "#7895a8"),
        (0, "validation", "Validation (held-out maps)", "#376fb2"),
        (0.25, "stress", "Stress (held-out topology suite)", "#218670"),
    ):
        heights = [100 * result["means"][name][split] for name in candidate_names]
        bars = bottom.bar(
            [i + offset for i in range(4)],
            heights,
            width=0.23,
            color=color,
            label=label,
        )
        bottom.bar_label(
            bars, labels=[f"{v:.2f}" for v in heights], fontsize=9, padding=3
        )
    bottom.axhline(99, color="#b52b38", ls="--", lw=1)
    bottom.set(
        ylim=(80, 102),
        ylabel="Top-1 accuracy (%)",
        xticks=range(4),
        xticklabels=[
            "Baseline, epoch 100",
            "More epochs, 300",
            "+ LR decay, 300",
            "+ Capacity, 300",
        ],
        title="Fixed final checkpoints: three-seed means",
    )
    bottom.grid(axis="y", alpha=0.2)
    bottom.set_axisbelow(True)
    bottom.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False
    )
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "More training fit does not ensure held-out dynamics fidelity",
        fontsize=14,
        y=0.985,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
