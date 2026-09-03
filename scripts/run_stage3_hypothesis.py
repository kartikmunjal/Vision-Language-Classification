#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vision_language_classification.analysis import disagreement_difference, training_texture_cut
from vision_language_classification.schema import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("ensemble")
    parser.add_argument("output")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    metadata = {row["example_id"]: row for row in read_jsonl(args.manifest)}
    rows = [{**metadata[row["example_id"]], "vote_entropy": row["vote_entropy"]} for row in read_jsonl(args.ensemble) if row["example_id"] in metadata]
    cut = training_texture_cut(rows)
    report = {
        "preregistration": "PREREGISTRATION.md",
        "texture_cut_fitted_on": "train",
        "texture_cut": cut,
        "bootstrap_replicates": args.bootstrap_replicates,
        "result": disagreement_difference(rows, cut, replicates=args.bootstrap_replicates, seed=args.seed),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
