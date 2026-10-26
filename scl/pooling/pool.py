"""MIRACL-style relevance pooling (Sec. 4.1).

For each query:
    1. BM25, mDPR and mColBERT each keep their top POOL_RETRIEVER_TOP_K documents.
    2. Each retriever's kept scores are min-max normalised within the query and summed
       (equal weights, so the ranking equals that of the mean).
    3. The POOL_SIZE best documents form the pool.
    4. A pooled document is relevant iff its category equals the query category.

None of the evaluated systems (Dense, CCD, TF-IDF) takes part in pooling.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import POOL_RETRIEVER_TOP_K, POOL_SIZE

MINMAX_EPS = 1e-12           # all kept scores equal -> avoid division by zero


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """(doc index, score) of the k best documents; ties keep corpus order."""
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def minmax_fuse(per_retriever: list[list[tuple[int, float]]], pool_size: int = POOL_SIZE
                ) -> list[int]:
    """Sum of per-retriever min-max normalised scores; returns the best pool_size doc indices."""
    fused: dict[int, float] = {}
    for results in per_retriever:
        values = [s for _, s in results]
        lo, hi = min(values), max(values)
        for doc_idx, score in results:
            fused[doc_idx] = fused.get(doc_idx, 0.0) + (score - lo) / max(hi - lo, MINMAX_EPS)
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [doc_idx for doc_idx, _ in ranked[:pool_size]]


def pool_and_judge(score_matrices: dict[str, np.ndarray], query_ids: list[str],
                   query_category: str, doc_ids: list[str], doc_categories: list[str]
                   ) -> tuple[dict[str, list[str]], dict]:
    """qrels and a summary from (n_queries, n_docs) score matrices of the pooling retrievers."""
    depth = min(POOL_RETRIEVER_TOP_K, len(doc_ids))
    qrels: dict[str, list[str]] = {}
    pairs_total = pairs_relevant = 0
    for qi, qid in enumerate(query_ids):
        per_retriever = [top_k(m[qi], depth) for m in score_matrices.values()]
        pool = minmax_fuse(per_retriever)
        pairs_total += len(pool)
        relevant = sorted(doc_ids[i] for i in pool if doc_categories[i] == query_category)
        pairs_relevant += len(relevant)
        qrels[qid] = relevant
    distribution = Counter(len(v) for v in qrels.values())
    summary = {
        "retrievers": list(score_matrices),
        "retriever_top_k": depth,
        "pool_size": POOL_SIZE,
        "queries_total": len(query_ids),
        "pairs_total": pairs_total,
        "pairs_relevant": pairs_relevant,
        "queries_with_relevant": sum(1 for v in qrels.values() if v),
        "queries_zero_relevant": sum(1 for v in qrels.values() if not v),
        "per_query_relevant_dist": {str(k): distribution[k] for k in sorted(distribution)},
    }
    return qrels, summary
