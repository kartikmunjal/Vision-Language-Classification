#!/usr/bin/env python3
import argparse

from vision_language_classification.rules import label_caption
from vision_language_classification.schema import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    rows = []
    for example in read_jsonl(args.manifest):
        rows.append({"example_id": example["example_id"], "source": "rules", "labels": label_caption(example["caption"])})
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
