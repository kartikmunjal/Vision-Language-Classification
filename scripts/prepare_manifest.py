#!/usr/bin/env python3
import argparse

from vision_language_classification.data import from_video_curation
from vision_language_classification.schema import read_jsonl, write_jsonl
from vision_language_classification.splits import assert_no_group_leakage, assign_group_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Video-Curation enriched JSONL manifest")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    examples = [from_video_curation(row).__dict__ for row in read_jsonl(args.input)]
    examples = assign_group_splits(examples, args.seed)
    assert_no_group_leakage(examples)
    write_jsonl(examples, args.output)


if __name__ == "__main__":
    main()
