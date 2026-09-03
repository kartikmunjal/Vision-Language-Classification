#!/usr/bin/env python3
import argparse

from vision_language_classification.human_labels import stratified_sample, write_annotation_sheet
from vision_language_classification.schema import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    test_rows = [row for row in read_jsonl(args.manifest) if row.get("split") == "test"]
    write_annotation_sheet(stratified_sample(test_rows, args.size, args.seed), args.output)


if __name__ == "__main__":
    main()
