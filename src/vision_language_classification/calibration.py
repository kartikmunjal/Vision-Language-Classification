from __future__ import annotations

import numpy as np


def sigmoid(logits):
    values = np.asarray(logits, dtype=float)
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit_temperature(logits, labels, grid=None) -> float:
    """Fit one temperature on a held-out calibration split by log loss."""
    z, y = np.asarray(logits, float), np.asarray(labels, int)
    candidates = np.geomspace(0.05, 10.0, 400) if grid is None else np.asarray(grid)
    losses = []
    for temperature in candidates:
        p = np.clip(sigmoid(z / temperature), 1e-8, 1 - 1e-8)
        losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(candidates[int(np.argmin(losses))])


def reliability_bins(labels, probabilities, bins: int = 10) -> list[dict]:
    y, p = np.asarray(labels, int), np.asarray(probabilities, float)
    edges = np.linspace(0, 1, bins + 1)
    output = []
    for idx in range(bins):
        mask = (p >= edges[idx]) & (p < edges[idx + 1] if idx < bins - 1 else p <= 1)
        output.append({
            "lower": float(edges[idx]), "upper": float(edges[idx + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(p[mask].mean()) if mask.any() else None,
            "positive_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return output
