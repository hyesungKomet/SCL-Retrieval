"""Unit tests for the signals, the fusion and the tie rule.  Run: python -m pytest tests"""
import math

import numpy as np

from scl import channels, fusion
from scl.evaluate import needle_hit


def test_ties_are_resolved_against_the_last_document():
    scores = np.array([1.0, 0.5, 0.5])
    assert list(fusion.ranks_from_scores(scores)) == [1, 2, 3]


def test_needle_hit_counts_ties_against_the_needle():
    top_k = 20
    tied = np.full(top_k + 1, 0.3)         # top_k distractors tie with the needle
    assert not needle_hit(tied)
    one_fewer = np.concatenate([np.full(top_k - 1, 0.3), [0.0, 0.3]])
    assert needle_hit(one_fewer)


def test_rrf_matches_equation_4():
    a = np.array([3.0, 2.0, 1.0])          # ranks 1, 2, 3
    b = np.array([1.0, 3.0, 2.0])          # ranks 3, 1, 2
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert np.allclose(fusion.rrf([a, b], k=60), expected)


def test_ccd_prefers_documents_away_from_the_centroid():
    docs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = channels.ccd_scores(docs)
    assert scores[2] > scores[0] == scores[1]


def test_tfidf_matches_equation_3():
    corpus = ["a1 b1 b1", "a1 c1", "b1 c1 c1"]
    query = ["b1 c1"]
    got = channels.tfidf_scores(query, corpus)

    n = len(corpus)
    vocab = ["a1", "b1", "c1"]              # every word has df = 2 >= min_df
    df = {"a1": 2, "b1": 2, "c1": 2}
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1 for t in vocab}

    def vec(text):
        counts = {t: text.split().count(t) for t in vocab}
        v = np.array([(1 + math.log(counts[t])) * idf[t] if counts[t] else 0.0 for t in vocab])
        return v / np.linalg.norm(v)

    expected = [vec(query[0]) @ vec(d) for d in corpus]
    assert np.allclose(got[0], expected)


def test_placeholders_are_replaced_with_a_space():
    out = channels.strip_placeholders(["NAME 씨 안녕하세요", "OO은행 XXX 고객님"], "test")
    assert out == ["  씨 안녕하세요", " 은행   고객님"]
