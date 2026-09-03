#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = sorted(report["tasks"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharex=True, sharey=True)
    for axis, task in zip(axes, tasks, strict=True):
        for key, label in (("before_temperature", "before"), ("after_temperature", "after")):
            bins = report["tasks"][task]["classifier"][key]["reliability_bins"]
            points = [(row["mean_probability"], row["positive_rate"]) for row in bins if row["count"]]
            axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=label)
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed positive rate")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output / "reliability_diagrams.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    budgets = sorted(report["active_correction"], key=float)
    for task in tasks:
        accuracy_delta = [report["active_correction"][budget]["tasks"][task]["accuracy"]["targeted_minus_random"] for budget in budgets]
        ece_delta = [report["active_correction"][budget]["tasks"][task]["ece"]["targeted_minus_random"] for budget in budgets]
        axes[0].plot([float(x) for x in budgets], accuracy_delta, marker="o", label=task)
        axes[1].plot([float(x) for x in budgets], ece_delta, marker="o", label=task)
    for axis, title in zip(axes, ("Accuracy gain", "ECE reduction"), strict=True):
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xlabel("correction fraction")
        axis.set_ylabel("targeted minus random")
        axis.set_title(title)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "active_correction.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
