"""Plot validation selection and final held-out scores from completed round 2."""

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
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.preview and args.preview.exists()):
        raise ValueError("Choose fresh figure output paths.")
    data = json.loads(args.summary.read_text())
    names = ["earlystop_control", "ranking", "ranking_depth"]
    labels = ["Original loss + early stop", "+ Ranking", "+ Ranking + depth"]
    colors = ["#a3663a", "#206aa5", "#188270"]
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(
        2, 3, left=0.07, right=0.98, top=0.84, bottom=0.10, hspace=0.42, wspace=0.22
    )
    for i, seed in enumerate((20260804, 20260805, 20260806)):
        axis = fig.add_subplot(grid[0, i])
        baseline = next(
            r["validation_accuracy"]
            for r in data["model_scores"]
            if r["candidate"] == "baseline" and r["seed"] == seed
        )
        axis.axhline(100 * baseline, color="#777777", ls=":", label="Frozen baseline")
        axis.axhline(99, color="#aa3040", ls="--", lw=1, label="99% target")
        for name, label, color in zip(names, labels, colors, strict=True):
            record = next(
                r
                for r in data["convergence"]
                if r["candidate"] == name and r["seed"] == seed
            )
            values = record["validation_checkpoints"]
            axis.plot(
                [r["epoch"] for r in values],
                [100 * r["accuracy"] for r in values],
                color=color,
                lw=1.7,
                marker="o",
                ms=2.5,
                label=label,
            )
            best = next(
                r for r in values if r["epoch"] == record["selected_additional_epoch"]
            )
            axis.scatter(
                [best["epoch"]],
                [100 * best["accuracy"]],
                color=color,
                marker="*",
                s=125,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
        axis.set(
            title=f"Seed {seed}",
            xlabel="Additional epochs after baseline",
            ylim=(90, 100),
            xlim=(-3, 153),
            xticks=[0, 50, 100, 150],
        )
        if i == 0:
            axis.set_ylabel("Original-validation Top-1 (%)")
        axis.grid(axis="y", alpha=0.2)
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.915),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    bottom = fig.add_subplot(grid[1, :])
    for offset, split, label, color in (
        (-0.24, "train", "Train (seen maps)", "#7d8fa1"),
        (0, "validation", "Validation (selection)", "#206aa5"),
        (0.24, "stress", "Stress (evaluation only)", "#188270"),
    ):
        values = [100 * data["means"][name][split] for name in ["baseline", *names]]
        bars = bottom.bar(
            [i + offset for i in range(4)], values, width=0.22, label=label, color=color
        )
        bottom.bar_label(
            bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9
        )
    bottom.set(
        xticks=list(range(4)),
        xticklabels=["Frozen baseline", *labels],
        ylabel="Three-seed mean Top-1 (%)",
        ylim=(85, 102),
    )
    bottom.axhline(99, color="#aa3040", ls="--", lw=1)
    bottom.legend(loc="lower left", frameon=False, ncol=3, fontsize=9)
    bottom.grid(axis="y", alpha=0.2)
    bottom.set_axisbelow(True)
    fig.suptitle("Ranking loss and modest predictor depth", fontsize=18, y=0.985)
    fig.text(
        0.5,
        0.945,
        "Nine fixed runs · Stars mark validation-selected checkpoints · "
        "Stress scored after selection",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.07,
        0.03,
        "Same 40 training maps and full-latent nearest-neighbour decoder. "
        "Stress is a previously inspected holdout.",
        fontsize=10,
        color="#444444",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})
    if args.preview:
        fig.savefig(args.preview, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
