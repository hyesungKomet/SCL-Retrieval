"""Rank fusion (Sec. 3.2) and the weighted-sum comparison (Sec. 4.3).

Tie rule: ranks come from a stable descending sort, so among equal scores the document
that appears earlier in the corpus array ranks higher. The evaluation always puts the
relevant document last, so every tie is resolved against it (a conservative choice).
"""
from __future__ import annotations

import numpy as np

from .config import RRF_K, ZSCORE_EPS


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every document (1 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf(channel_scores: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Eq. (4): s_RRF(d) = sum_c 1 / (k + rank_c(d)). Unweighted, no score normalisation."""
    fused = np.zeros(len(channel_scores[0]), dtype=np.float64)
    for scores in channel_scores:
        fused += 1.0 / (k + ranks_from_scores(scores))
    return fused


def zscore(scores: np.ndarray) -> np.ndarray:
    """Standardise one score vector to mean 0 and standard deviation 1."""
    return (scores - scores.mean()) / (scores.std() + ZSCORE_EPS)


def weighted_zsum(channel_scores: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """sum_c w_c * z(s_c). Used only for the 2:2:1 comparison in Sec. 4.3."""
    fused = np.zeros(len(channel_scores[0]), dtype=np.float64)
    for scores, weight in zip(channel_scores, weights):
        fused += weight * zscore(scores)
    return fused
