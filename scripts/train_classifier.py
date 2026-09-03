#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from vision_language_classification.model import build_resnet18, build_vit_tiny
from vision_language_classification.schema import TASKS, read_jsonl, write_jsonl


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("labels")
    parser.add_argument("output_dir")
    parser.add_argument("--architecture", choices=["resnet18", "vit_tiny"], default="resnet18")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import v2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    labels = {row["example_id"]: row["labels"] for row in read_jsonl(args.labels)}
    rows = [row for row in read_jsonl(args.manifest) if row["example_id"] in labels]
    transform = v2.Compose([v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class Images(Dataset):
        def __init__(self, subset): self.rows = subset
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows[idx]
            with Image.open(row["image_path"]) as image:
                pixels = transform(image.convert("RGB"))
            targets = torch.tensor([labels[row["example_id"]][task].get("probability", labels[row["example_id"]][task]["label"]) for task in TASKS])
            return pixels, targets, row["example_id"]

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] in {"calibration", "test"}]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Images(train_rows), batch_size=args.batch_size, shuffle=True, generator=generator)
    model = (build_resnet18 if args.architecture == "resnet18" else build_vit_tiny)(len(TASKS), not args.no_pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(args.epochs):
        for pixels, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(pixels.to(args.device)), targets.to(args.device))
            loss.backward()
            optimizer.step()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    model.eval()
    predictions = []
    with torch.no_grad():
        for pixels, _, ids in DataLoader(Images(eval_rows), batch_size=args.batch_size):
            logits = model(pixels.to(args.device)).cpu().numpy()
            predictions.extend({"example_id": item_id, "logits": dict(zip(TASKS, row, strict=True))} for item_id, row in zip(ids, logits, strict=True))
    write_jsonl(predictions, output / "logits.jsonl")
    metadata = {
        "architecture": args.architecture, "pretrained": not args.no_pretrained,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "n_trials": 1, "train_examples": len(train_rows),
        "manifest_sha256": sha256(args.manifest), "labels_sha256": sha256(args.labels),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
