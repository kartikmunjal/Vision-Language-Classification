#!/usr/bin/env python3
import argparse

from PIL import Image

from vision_language_classification.labelers import ClipLabeler
from vision_language_classification.schema import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()
    labeler = ClipLabeler(args.model, args.pretrained, args.device)
    rows = []
    for example in read_jsonl(args.manifest):
        with Image.open(example["image_path"]) as image:
            labels = labeler.label(image.convert("RGB"))
        rows.append({"example_id": example["example_id"], "source": "clip", "model_id": labeler.model_id, "labels": labels})
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
