from __future__ import annotations

import math

import numpy as np


def confidence_weighted_vote(source_outputs: list[dict], threshold: float = 0.5) -> dict:
    if not source_outputs:
        raise ValueError("at least one source is required")
    weights = np.asarray([float(item.get("confidence", 1.0)) for item in source_outputs])
    labels = np.asarray([int(item["label"]) for item in source_outputs])
    probability = float(np.average(labels, weights=weights))
    return {"label": int(probability >= threshold), "probability": probability}


def vote_entropy(source_outputs: list[dict]) -> float:
    probability = float(np.mean([int(item["label"]) for item in source_outputs]))
    if probability in (0.0, 1.0):
        return 0.0
    return float(-(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability)))


def rank_for_correction(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (-float(row["vote_entropy"]), row["example_id"]))
