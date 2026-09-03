#!/usr/bin/env python3
import argparse

from vision_language_classification.ensemble import confidence_weighted_vote, vote_entropy
from vision_language_classification.schema import TASKS, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs=3, help="rules, LLM, and CLIP JSONL files")
    parser.add_argument("output")
    args = parser.parse_args()
    sources = [{row["example_id"]: row for row in read_jsonl(path)} for path in args.inputs]
    common = sorted(set.intersection(*(set(source) for source in sources)))
    rows = []
    for example_id in common:
        labels, entropies = {}, []
        for task in TASKS:
            opinions = [source[example_id]["labels"][task] for source in sources]
            labels[task] = confidence_weighted_vote(opinions)
            labels[task]["vote_entropy"] = vote_entropy(opinions)
            entropies.append(labels[task]["vote_entropy"])
        rows.append({"example_id": example_id, "source": "ensemble", "labels": labels, "vote_entropy": sum(entropies) / len(entropies)})
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
