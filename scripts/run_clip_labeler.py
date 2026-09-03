#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image

from vision_language_classification.labelers import ClipLabeler
from vision_language_classification.schema import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()
    labeler = ClipLabeler(args.model, args.pretrained, args.device)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    done = {row["example_id"] for row in read_jsonl(target)} if target.exists() else set()
    pending = [row for row in read_jsonl(args.manifest) if row["example_id"] not in done]
    with target.open("a", encoding="utf-8") as handle:
        for index, example in enumerate(pending, 1):
            with Image.open(example["image_path"]) as image:
                labels = labeler.label(image.convert("RGB"))
            row = {"example_id": example["example_id"], "source": "clip", "model_id": labeler.model_id, "labels": labels}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if index % 100 == 0:
                handle.flush()
                print(f"processed {index}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
