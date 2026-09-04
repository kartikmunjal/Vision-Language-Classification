from __future__ import annotations
import numpy as np


def retrieval_metrics(scores: np.ndarray) -> dict[str, float]:
    """Diagonal-pair retrieval metrics for a query-by-candidate score matrix."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square with correct pairs on the diagonal")
    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)[np.arange(len(scores)), np.arange(len(scores))] + 1
    return {"recall_at_1": float(np.mean(ranks <= 1)), "recall_at_5": float(np.mean(ranks <= 5)),
            "median_rank": float(np.median(ranks))}


def object_disjoint_semihard(similarities: np.ndarray, category_sets: list[set[str]], *, top_fraction: float = 0.1) -> np.ndarray:
    """Choose the hardest candidate in the top similarity fraction with no shared category."""
    n = len(category_sets); out = np.empty(n, dtype=int); width = max(1, int(np.ceil(n * top_fraction)))
    for i in range(n):
        order = np.argsort(-similarities[i])
        eligible = [j for j in order if j != i and category_sets[i].isdisjoint(category_sets[j])]
        if not eligible:
            raise ValueError(f"no object-disjoint negative for row {i}")
        out[i] = eligible[min(width - 1, len(eligible) - 1)]
    return out
