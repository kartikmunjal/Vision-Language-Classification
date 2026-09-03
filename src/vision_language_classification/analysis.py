from __future__ import annotations

import numpy as np

from .metrics import expected_calibration_error


def blur_band(score: float | None) -> str:
    if score is None:
        return "missing"
    if score < 40:
        return "low"
    if score < 80:
        return "medium"
    return "high"


def caption_length_band(caption: str) -> str:
    length = len(caption.split())
    for upper, label in ((8, "0-7"), (16, "8-15"), (32, "16-31")):
        if length < upper:
            return label
    return "32+"


def flagged_slice(row: dict, texture_cut: float) -> bool:
    blur = row.get("blur_score")
    texture = row.get("texture_score")
    return blur is not None and texture is not None and float(blur) < 40 and float(texture) <= texture_cut


def training_texture_cut(rows: list[dict]) -> float:
    scores = [float(row["texture_score"]) for row in rows if row.get("split") == "train" and row.get("texture_score") is not None]
    if not scores:
        raise ValueError("training split has no texture scores")
    return float(np.quantile(scores, 0.25))


def disagreement_difference(rows: list[dict], texture_cut: float, *, replicates=2000, seed=2026) -> dict:
    inside = np.asarray([float(row["vote_entropy"]) for row in rows if flagged_slice(row, texture_cut)])
    outside = np.asarray([float(row["vote_entropy"]) for row in rows if not flagged_slice(row, texture_cut)])
    if not len(inside) or not len(outside):
        raise ValueError("both flagged and comparison slices need observations")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates)
    for idx in range(replicates):
        a = rng.choice(inside, len(inside), replace=True)
        b = rng.choice(outside, len(outside), replace=True)
        differences[idx] = a.mean() - b.mean()
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "estimate": float(inside.mean() - outside.mean()),
        "ci95": [float(low), float(high)],
        "n_flagged": len(inside),
        "n_comparison": len(outside),
        "n_trials": 1,
        "supported": bool(low > 0),
    }


def correction_indices(entropy, budget: float, *, strategy: str, seed: int) -> np.ndarray:
    values = np.asarray(entropy, float)
    count = max(1, round(len(values) * budget))
    if strategy == "targeted":
        return np.argsort(-values, kind="stable")[:count]
    if strategy == "random":
        return np.random.default_rng(seed).choice(len(values), count, replace=False)
    raise ValueError("strategy must be targeted or random")


def evaluate_slice(labels, probabilities, mask) -> dict:
    y, p, selected = np.asarray(labels), np.asarray(probabilities), np.asarray(mask, bool)
    if not selected.any():
        return {"n": 0, "accuracy": None, "ece": None}
    return {
        "n": int(selected.sum()),
        "accuracy": float(np.mean((p[selected] >= 0.5) == y[selected])),
        "ece": expected_calibration_error(y[selected], p[selected]),
    }
