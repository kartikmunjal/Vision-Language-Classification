#!/usr/bin/env python3
"""Extract frozen CLIP embeddings for Sequences 3–6 on a CUDA host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import open_clip


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def batch_encode_images(model, preprocess, images, device, batch_size=128):
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(x) for x in images[i:i + batch_size]]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            z = model.encode_image(batch)
            z = z / z.norm(dim=-1, keepdim=True)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def batch_encode_text(model, tokenizer, texts, device, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            z = model.encode_text(batch)
            z = z / z.norm(dim=-1, keepdim=True)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def frame(image, trajectory, step):
    if trajectory == "static":
        return image.copy()
    if trajectory == "translation":
        shift = int(round(image.width * 0.12 * step / 7))
        canvas = Image.new("RGB", image.size, (0, 0, 0))
        canvas.paste(image, (shift, 0))
        return canvas
    if trajectory == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=4.0 * step / 7))
    if trajectory == "darkening":
        return ImageEnhance.Brightness(image).enhance(1.0 - 0.75 * step / 7)
    if trajectory == "occlusion":
        out = image.copy()
        draw = ImageDraw.Draw(out)
        frac = 0.45 * step / 7
        w, h = out.size
        draw.rectangle((int(w * (0.5 - frac / 2)), int(h * (0.5 - frac / 2)),
                        int(w * (0.5 + frac / 2)), int(h * (0.5 + frac / 2))), fill=(0, 0, 0))
        return out
    raise ValueError(trajectory)


def perturb_caption(text, mode):
    if mode == "negation":
        return "This image does not show " + text.rstrip(".") + "."
    swaps = [(" a ", " several "), (" one ", " many "), (" two ", " one "),
             (" three ", " one "), (" people", " person"), (" men", " man"), (" women", " woman")]
    padded = " " + text.lower() + " "
    for old, new in swaps:
        if old in padded:
            return padded.replace(old, new, 1).strip()
    return "several subjects: " + text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--temporal-n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = read_jsonl(args.manifest)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(args.device)
    paths = []
    for r in rows:
        p = Path(r["image_path"])
        if args.image_root:
            p = args.image_root / p.name
        paths.append(p)
    images = [Image.open(p).convert("RGB") for p in paths]
    clean_image = batch_encode_images(model, preprocess, images, args.device)
    clean_text = batch_encode_text(model, tokenizer, [r["caption"] for r in rows], args.device)
    test_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    rng = np.random.default_rng(20260903)
    temporal_idx = np.sort(rng.choice(test_idx, min(args.temporal_n, len(test_idx)), replace=False))
    trajectories = ["static", "translation", "blur", "darkening", "occlusion"]
    temporal = np.empty((len(temporal_idx), len(trajectories), 8, clean_image.shape[1]), dtype=np.float32)
    for ti, trajectory in enumerate(trajectories):
        all_frames = [frame(images[i], trajectory, step) for i in temporal_idx for step in range(8)]
        temporal[:, ti] = batch_encode_images(model, preprocess, all_frames, args.device).reshape(len(temporal_idx), 8, -1)
    test_captions = [rows[i]["caption"] for i in temporal_idx]
    negative_text = batch_encode_text(model, tokenizer, [perturb_caption(x, "negation") for x in test_captions], args.device)
    count_text = batch_encode_text(model, tokenizer, [perturb_caption(x, "count") for x in test_captions], args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metadata = json.dumps({"model_id":"open_clip:ViT-B-32:laion2b_s34b_b79k","torch_version":torch.__version__,
                           "device":args.device,"cuda_device":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                           "manifest_sha256":manifest_hash,"temporal_n":len(temporal_idx),"seed":20260903})
    np.savez_compressed(args.output, ids=np.array([r["example_id"] for r in rows]), splits=np.array([r["split"] for r in rows]),
                        clean_image=clean_image, clean_text=clean_text, temporal_idx=temporal_idx,
                        trajectories=np.array(trajectories), temporal_image=temporal,
                        negative_text=negative_text, count_text=count_text, metadata=np.array(metadata))


if __name__ == "__main__":
    main()
