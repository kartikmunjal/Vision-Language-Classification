#!/usr/bin/env python3
import argparse
import json
from itertools import combinations
from pathlib import Path

from vision_language_classification.metrics import cohens_kappa, metric_with_ci
from vision_language_classification.schema import TASKS, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs=3)
    parser.add_argument("output")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    sources = []
    for path in args.inputs:
        rows = read_jsonl(path)
        sources.append((rows[0]["source"], {row["example_id"]: row for row in rows}))
    common = sorted(set.intersection(*(set(rows) for _, rows in sources)))
    report = {"n_examples": len(common), "bootstrap_replicates": args.bootstrap_replicates, "tasks": {}}
    for task in TASKS:
        report["tasks"][task] = {}
        for (name_a, rows_a), (name_b, rows_b) in combinations(sources, 2):
            a = [rows_a[item]["labels"][task]["label"] for item in common]
            b = [rows_b[item]["labels"][task]["label"] for item in common]
            report["tasks"][task][f"{name_a}_vs_{name_b}"] = metric_with_ci(
                a, b, cohens_kappa, replicates=args.bootstrap_replicates, seed=args.seed
            )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
