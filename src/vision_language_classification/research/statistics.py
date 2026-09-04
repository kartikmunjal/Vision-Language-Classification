from __future__ import annotations
import numpy as np


def bootstrap_ci(values, *, seed: int = 20260903, n_resamples: int = 10_000) -> list[float]:
    """Percentile interval for a mean over independent units."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x):
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_resamples, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def paired_trial_summary(treatment, control, *, seed: int = 20260903) -> dict:
    """Summarize paired seed-level deltas without pretending seeds are examples."""
    a, b = np.asarray(treatment, float), np.asarray(control, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired trial arrays must be one-dimensional and equal length")
    delta = a - b
    return {"estimate": float(delta.mean()), "ci95": bootstrap_ci(delta, seed=seed),
            "trial_deltas": delta.tolist(), "n_trials": int(len(delta))}
