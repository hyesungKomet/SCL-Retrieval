"""The three retrieval signals of SCL-Retrieval (Sec. 3.1).

Every function returns one score per corpus document; a higher score means a better rank.

    dense_scores  (S) Eq. (1)  cosine between query and document embeddings
    ccd_scores    (C) Eq. (2)  distance from the normalised corpus centroid (query independent)
    tfidf_scores  (L) Eq. (3)  word (eojeol) TF-IDF cosine fitted on the corpus

Embeddings are expected to be L2-normalised, so cosine similarity is a dot product.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (CENTROID_EPS, DOC_PLACEHOLDER_PATTERN, PLACEHOLDER_REPLACEMENT,
                     QUERY_MASK_PATTERN, TFIDF_KWARGS)

_DOC_PLACEHOLDER = re.compile(DOC_PLACEHOLDER_PATTERN)
_QUERY_MASK = re.compile(QUERY_MASK_PATTERN)


def dense_scores(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Eq. (1): s_dense(q, d) = e(q)^T e(d), shape (n_queries, n_docs).

    Computed once for all queries and documents; instance corpora take columns of it.
    """
    return query_vecs @ doc_vecs.T


def ccd_scores(doc_vecs: np.ndarray) -> np.ndarray:
    """Eq. (2): s_CCD(d) = -e(d)^T mu_D, with mu_D the normalised mean of the corpus embeddings.

    Documents far from the dominant direction of the corpus rank higher. The score does
    not depend on the query, so it is computed once per corpus.
    """
    # The centroid is accumulated in float64; the dot products are taken in the precision
    # of the embeddings (float32), as in the experiments reported in the paper. With
    # rank fusion, rounding differences can reorder near-tied documents.
    centroid = doc_vecs.sum(axis=0, dtype=np.float64)
    centroid = centroid / (np.linalg.norm(centroid) + CENTROID_EPS)
    centroid = centroid.astype(doc_vecs.dtype)
    return -(doc_vecs @ centroid).astype(np.float64)


def strip_placeholders(texts: list[str], label: str) -> list[str]:
    """Replace de-identification placeholders with a space before TF-IDF.

    Documents use typed tags (NAME, ORG, ...) and queries use masks (OO, XXX, ...).
    Reports how many characters and texts were affected.
    """
    out = []
    removed_chars = 0
    affected = 0
    for text in texts:
        doc_tags = _DOC_PLACEHOLDER.findall(text)
        stage1 = _DOC_PLACEHOLDER.sub(PLACEHOLDER_REPLACEMENT, text)
        query_masks = _QUERY_MASK.findall(stage1)
        stage2 = _QUERY_MASK.sub(PLACEHOLDER_REPLACEMENT, stage1)
        removed = sum(map(len, doc_tags)) + sum(map(len, query_masks))
        removed_chars += removed
        affected += int(removed > 0)
        out.append(stage2)
    total_chars = sum(map(len, texts))
    pct = 100.0 * removed_chars / total_chars if total_chars else 0.0
    print(f"[placeholders] {label}: removed {removed_chars}/{total_chars} chars ({pct:.2f}%) "
          f"in {affected}/{len(texts)} texts")
    return out


def tfidf_scores(query_texts: list[str], corpus_texts: list[str]) -> np.ndarray:
    """Eq. (3): TF-IDF cosine, shape (n_queries, n_docs).

    The vectoriser is fitted on the corpus being searched: words with document frequency
    below min_df are dropped, tf' = 1 + log tf, idf = log((1 + |D|) / (1 + df)) + 1.
    Texts must already have placeholders stripped (see strip_placeholders).
    """
    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform(query_texts)
    return np.asarray((query_matrix @ doc_matrix.T).todense(), dtype=np.float64)
