#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from vision_language_classification.human_labels import validate_human_labels
from vision_language_classification.metrics import accuracy, cohens_kappa, metric_with_ci
from vision_language_classification.schema import TASKS, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("human_csv")
    parser.add_argument("weak_jsonl")
    parser.add_argument("output")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with open(args.human_csv, newline="", encoding="utf-8") as handle:
        human = list(csv.DictReader(handle))
    validate_human_labels(human)
    weak = {row["example_id"]: row for row in read_jsonl(args.weak_jsonl)}
    report = {"bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed, "tasks": {}}
    for task in TASKS:
        selected = [row for row in human if row["example_id"] in weak]
        truth = [int(row[task]) for row in selected]
        pred = [int(weak[row["example_id"]]["labels"][task]["label"]) for row in selected]
        report["tasks"][task] = {
            "accuracy": metric_with_ci(truth, pred, accuracy, replicates=args.bootstrap_replicates, seed=args.seed),
            "cohens_kappa": metric_with_ci(truth, pred, cohens_kappa, replicates=args.bootstrap_replicates, seed=args.seed),
        }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
