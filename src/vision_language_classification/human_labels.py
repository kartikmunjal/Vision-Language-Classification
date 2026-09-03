from __future__ import annotations

import csv
import random
from pathlib import Path

from .schema import TASKS


def stratified_sample(rows: list[dict], n: int, seed: int = 2026) -> list[dict]:
    """Deterministic approximately proportional sample over source categories."""
    if n > len(rows):
        raise ValueError("requested sample exceeds available rows")
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("source_category") or "unknown"), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    keys = sorted(groups)
    while len(chosen) < n:
        progressed = False
        for key in keys:
            if groups[key] and len(chosen) < n:
                chosen.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return chosen


def write_annotation_sheet(rows: list[dict], path: str | Path) -> None:
    fields = ["example_id", "image_path", "caption", "source_category", *TASKS, "notes"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def validate_human_labels(rows: list[dict]) -> None:
    for row in rows:
        for task in TASKS:
            if str(row.get(task, "")) not in {"0", "1"}:
                raise ValueError(f"{row.get('example_id')}: {task} must be 0 or 1")
