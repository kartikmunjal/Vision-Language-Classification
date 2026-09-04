from __future__ import annotations
import numpy as np


def minmax(values) -> np.ndarray:
    x = np.asarray(values, float); span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


def greedy_diversity(embeddings: np.ndarray, n: int, initial_scores=None) -> np.ndarray:
    """Deterministic farthest-first traversal, optionally seeded by highest score."""
    x = np.asarray(embeddings, float); x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    first = int(np.argmax(initial_scores)) if initial_scores is not None else 0
    chosen = [first]; nearest = 1 - x @ x[first]
    while len(chosen) < min(n, len(x)):
        nearest[chosen] = -np.inf; nxt = int(np.argmax(nearest)); chosen.append(nxt)
        nearest = np.minimum(nearest, 1 - x @ x[nxt])
    return np.asarray(chosen)


def diversity_scores(embeddings: np.ndarray, anchors: int = 64) -> np.ndarray:
    """Distance to deterministic farthest-first anchors as a coverage priority."""
    x=np.asarray(embeddings,float); ids=greedy_diversity(x,min(anchors,len(x)))
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12); a=x[ids]
    return 1-np.max(x@a.T,axis=1)
