from __future__ import annotations

from collections.abc import Callable

import numpy as np


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    truth, pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    return {
        "tn": int(np.sum((truth == 0) & (pred == 0))),
        "fp": int(np.sum((truth == 0) & (pred == 1))),
        "fn": int(np.sum((truth == 1) & (pred == 0))),
        "tp": int(np.sum((truth == 1) & (pred == 1))),
    }


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def cohens_kappa(y_a, y_b) -> float:
    a, b = np.asarray(y_a, int), np.asarray(y_b, int)
    if len(a) != len(b) or not len(a):
        raise ValueError("labels must be non-empty and equal length")
    observed = np.mean(a == b)
    pa, pb = np.mean(a == 1), np.mean(b == 1)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y, p = np.asarray(y_true, int), np.asarray(probabilities, float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece = 0.0
    for idx in range(bins):
        mask = (p >= edges[idx]) & (p < edges[idx + 1] if idx < bins - 1 else p <= 1)
        if mask.any():
            confidence = np.mean(np.where(p[mask] >= 0.5, p[mask], 1 - p[mask]))
            correctness = np.mean((p[mask] >= 0.5) == y[mask])
            ece += mask.sum() / total * abs(correctness - confidence)
    return float(ece)


def bootstrap_ci(
    values,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    array = np.asarray(values)
    if not len(array):
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    estimates = [statistic(array[rng.integers(0, len(array), len(array))]) for _ in range(replicates)]
    alpha = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(estimates, [alpha, 1 - alpha]))


def metric_with_ci(y_true, y_pred, metric, *, replicates=2000, seed=2026) -> dict:
    y, p = np.asarray(y_true), np.asarray(y_pred)
    indices = np.arange(len(y))
    low, high = bootstrap_ci(
        indices,
        lambda idx: metric(y[idx.astype(int)], p[idx.astype(int)]),
        replicates=replicates,
        seed=seed,
    )
    return {"estimate": metric(y, p), "ci95": [low, high], "n": len(y), "n_trials": 1}
