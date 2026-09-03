#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from vision_language_classification.schema import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("ensemble")
    parser.add_argument("stage3_result")
    parser.add_argument("output")
    args = parser.parse_args()
    manifest = {row["example_id"]: row for row in read_jsonl(args.manifest)}
    stage3 = json.loads(Path(args.stage3_result).read_text())
    cut = stage3["texture_cut"]
    ensemble = read_jsonl(args.ensemble)
    flagged = np.array([row["vote_entropy"] for row in ensemble if manifest[row["example_id"]]["blur_score"] < 40 and manifest[row["example_id"]]["texture_score"] <= cut])
    comparison = np.array([row["vote_entropy"] for row in ensemble if not (manifest[row["example_id"]]["blur_score"] < 40 and manifest[row["example_id"]]["texture_score"] <= cut)])
    effect = flagged.mean() - comparison.mean()
    minimum = int(np.ceil(((1.959963984540054 + 0.8416212335729143) * flagged.std(ddof=1) / abs(effect)) ** 2))
    report = {
        "method": "normal approximation, large comparison group",
        "alpha": 0.05, "target_power": 0.80,
        "pilot_flagged_n": len(flagged), "pilot_comparison_n": len(comparison),
        "pilot_effect": float(effect), "pilot_flagged_sd": float(flagged.std(ddof=1)),
        "minimum_flagged_n": minimum,
        "followup_target_flagged_n": 2 * minimum,
        "inflation_rationale": "2x minimum to reduce reliance on an optimistic 10-example pilot effect",
        "n_trials": 1,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
