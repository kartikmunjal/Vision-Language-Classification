from __future__ import annotations

import hashlib


def split_for_group(group_id: str, seed: int = 2026) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    if draw < 0.70:
        return "train"
    if draw < 0.85:
        return "calibration"
    return "test"


def assign_group_splits(rows: list[dict], seed: int = 2026) -> list[dict]:
    return [{**row, "split": split_for_group(str(row["source_video_id"]), seed)} for row in rows]


def assert_no_group_leakage(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        group, split = str(row["source_video_id"]), str(row["split"])
        if group in seen and seen[group] != split:
            raise ValueError(f"source_video_id {group!r} occurs in multiple splits")
        seen[group] = split
