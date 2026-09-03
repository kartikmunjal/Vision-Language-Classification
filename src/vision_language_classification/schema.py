from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

TASKS = (
    "multiple_subjects",
    "outdoor",
    "human_present",
    "animal_present",
    "dynamic_scene",
    "night",
)


@dataclass(frozen=True)
class Example:
    example_id: str
    image_path: str
    caption: str
    source_video_id: str
    source_category: str | None = None
    blur_score: float | None = None
    texture_score: float | None = None
    split: str | None = None


def stable_id(source_video_id: str, image_path: str) -> str:
    value = f"{source_video_id}\0{image_path}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict | Example], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
