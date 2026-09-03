#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from vision_language_classification.schema import stable_id, write_jsonl
from vision_language_classification.splits import assert_no_group_leakage, assign_group_splits


def visual_quality(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((512, 512))
        pixels = np.asarray(gray, dtype=np.float32)
        laplacian = (
            -4 * pixels
            + np.roll(pixels, 1, 0)
            + np.roll(pixels, -1, 0)
            + np.roll(pixels, 1, 1)
            + np.roll(pixels, -1, 1)
        )
        blur_score = float(laplacian[1:-1, 1:-1].var())
        block = 8
        height, width = pixels.shape
        cropped = pixels[: height - height % block, : width - width % block]
        blocks = cropped.reshape(height // block, block, width // block, block)
        local_std = blocks.std(axis=(1, 3))
        return blur_score, float(local_std.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_root")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path(args.coco_root).resolve()
    annotations = root / "annotations"
    captions = json.loads((annotations / "captions_val2017.json").read_text())
    instances = json.loads((annotations / "instances_val2017.json").read_text())
    by_image = defaultdict(list)
    for annotation in sorted(captions["annotations"], key=lambda row: row["id"]):
        by_image[annotation["image_id"]].append(annotation["caption"])
    category_name = {row["id"]: row["name"] for row in instances["categories"]}
    categories = defaultdict(list)
    for annotation in instances["annotations"]:
        categories[annotation["image_id"]].append(category_name[annotation["category_id"]])
    images = sorted(captions["images"], key=lambda row: row["id"])
    if args.limit:
        images = images[: args.limit]
    rows = []
    for index, image in enumerate(images, 1):
        image_id = str(image["id"])
        path = (root / "val2017" / image["file_name"]).resolve()
        blur, texture = visual_quality(path)
        counts = Counter(categories[image["id"]])
        dominant = counts.most_common(1)[0][0] if counts else "no_thing_annotation"
        rows.append({
            "example_id": stable_id(image_id, image["file_name"]),
            "image_path": str(path),
            "caption": by_image[image["id"]][0],
            "all_captions": by_image[image["id"]],
            "caption_selection": "lowest_annotation_id",
            "source_video_id": image_id,
            "source_category": dominant,
            "blur_score": blur,
            "texture_score": texture,
            "coco_image_id": image["id"],
            "coco_object_counts": dict(sorted(counts.items())),
        })
        if index % 500 == 0:
            print(f"scored {index}/{len(images)}", flush=True)
    rows = assign_group_splits(rows, args.seed)
    assert_no_group_leakage(rows)
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
