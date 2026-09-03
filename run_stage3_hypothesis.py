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
    parser.add_argument("--texture-cut", type=float)
    parser.add_argument("--preregistration", default="PREREGISTRATION.md")
    args = parser.parse_args()
    metadata = {row["example_id"]: row for row in read_jsonl(args.manifest)}
    rows = [{**metadata[row["example_id"]], "vote_entropy": row["vote_entropy"]} for row in read_jsonl(args.ensemble) if row["example_id"] in metadata]
    cut = args.texture_cut if args.texture_cut is not None else training_texture_cut(rows)
    report = {
        "preregistration": args.preregistration,
        "texture_cut_fitted_on": "provided_frozen_value" if args.texture_cut is not None else "train",
        "texture_cut": cut,
        "bootstrap_replicates": args.bootstrap_replicates,
        "result": disagreement_difference(rows, cut, replicates=args.bootstrap_replicates, seed=args.seed),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
