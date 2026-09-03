#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict

from vision_language_classification.schema import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("instances_json")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.instances_json, encoding="utf-8") as handle:
        instances = json.load(handle)
    categories = {row["id"]: row for row in instances["categories"]}
    counts = defaultdict(Counter)
    for annotation in instances["annotations"]:
        category = categories[annotation["category_id"]]
        counts[annotation["image_id"]][category["supercategory"]] += 1
    rows = []
    for example in read_jsonl(args.manifest):
        image_counts = counts[example["coco_image_id"]]
        people = image_counts["person"]
        animals = image_counts["animal"]
        rows.append({
            "example_id": example["example_id"],
            "source": "coco_2017_human_instance_annotations",
            "labels": {
                "human_present": int(people > 0),
                "animal_present": int(animals > 0),
                "multiple_subjects": int(people + animals >= 2),
            },
            "support": {"person_instances": people, "animal_instances": animals},
        })
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
