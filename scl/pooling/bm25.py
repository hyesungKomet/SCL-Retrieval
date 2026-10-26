"""BM25 with a regex tokenizer (Hangul, Latin letters, digits, underscore)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..config import BM25_B, BM25_K1

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Collapse whitespace, lowercase, keep word tokens; punctuation is dropped."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return _TOKEN_RE.findall(text)


class BM25:
    """Okapi BM25. Repeated query tokens contribute once per occurrence."""

    def __init__(self, doc_texts: list[str], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.n_docs = len(doc_texts)
        term_freqs = [Counter(tokenize(t)) for t in doc_texts]
        lengths = np.array([sum(tf.values()) for tf in term_freqs], dtype=np.float64)
        avg_length = float(lengths.mean()) if self.n_docs else 0.0
        # Length normalisation so that long transcripts are not favoured.
        self.length_norm = k1 * (1.0 - b + b * lengths / max(avg_length, 1e-9))

        # Posting lists: term -> (doc indices, term frequencies).
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for doc_idx, tf in enumerate(term_freqs):
            for term, freq in tf.items():
                docs, freqs = postings.setdefault(term, ([], []))
                docs.append(doc_idx)
                freqs.append(freq)
        self.postings = {t: (np.array(d), np.array(f, dtype=np.float64))
                         for t, (d, f) in postings.items()}
        self.idf = {t: math.log(1.0 + (self.n_docs - len(d) + 0.5) / (len(d) + 0.5))
                    for t, (d, _) in self.postings.items()}

    def scores(self, query_text: str) -> np.ndarray:
        """BM25 score of every document for one query."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        for token in tokenize(query_text):
            if token not in self.postings:
                continue
            docs, freqs = self.postings[token]
            out[docs] += self.idf[token] * (freqs * (self.k1 + 1.0)) / (freqs + self.length_norm[docs])
        return out
