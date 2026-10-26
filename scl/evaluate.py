"""Needle-in-distractors evaluation (Sec. 4.1).

    instance   = (query q, relevant document d) for every d in qrels[q]
    corpus     = all documents of the other category (distractors) + {d}
    Recall@20  = fraction of instances whose d is ranked in the top 20

Each relevant document is a separate instance, so every instance has exactly one
correct answer among 915 (fin_gov) or 2,101 (gov_fin) documents. The corpus array is
[distractors in gold_final.csv order..., d]: d is last, so every score tie is resolved
against it (see scl/fusion.py).

CCD and TF-IDF depend on the corpus, and the corpus depends only on d, so instances are
grouped by d and both signals are computed once per group.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from . import channels, fusion
from .config import TOP_K, WEIGHTED_SUM_WEIGHTS, Direction
from .data import Document, Query
from .encoders import Embeddings

# name -> (rule, channels). "single" ranks by one signal; "rrf" is Eq. (4);
# "weighted" is the 2:2:1 z-score sum of Sec. 4.3.
CONFIGS = {
    "tfidf": ("single", ("tfidf",)),
    "dense": ("single", ("dense",)),
    "ccd": ("single", ("ccd",)),
    "dense+ccd": ("rrf", ("dense", "ccd")),
    "dense+tfidf": ("rrf", ("dense", "tfidf")),
    "ccd+tfidf": ("rrf", ("ccd", "tfidf")),
    "scl": ("rrf", ("dense", "ccd", "tfidf")),
    "scl_weighted_221": ("weighted", ("dense", "ccd", "tfidf")),
}


def build_instances(qrels: dict[str, list[str]], queries: list[Query],
                    docs_by_id: dict[str, Document], direction: Direction
                    ) -> list[tuple[str, str]]:
    """(qid, needle doc_id) pairs, sorted by qid then doc_id."""
    query_ids = {q.qid for q in queries}
    unknown_q = sorted(set(qrels) - query_ids)
    if unknown_q:
        raise ValueError(f"qrels contains {len(unknown_q)} unknown qids, e.g. {unknown_q[:3]}")
    instances = []
    n_zero = 0
    for qid in sorted(qrels):
        if not qrels[qid]:
            n_zero += 1          # pooling found no relevant document for this query
            continue
        for doc_id in qrels[qid]:
            doc = docs_by_id.get(doc_id)
            if doc is None:
                raise ValueError(f"qrels[{qid}] points to unknown document {doc_id}")
            if doc.category != direction.query_category:
                raise ValueError(f"qrels[{qid}] document {doc_id} is {doc.category}")
            instances.append((qid, doc_id))
    print(f"[instances] {direction.key}: {len(qrels)} queries in qrels · "
          f"{n_zero} without a relevant document (excluded) · "
          f"{len(instances)} instances · {len({q for q, _ in instances})} queries · "
          f"{len({d for _, d in instances})} distinct relevant documents")
    return instances


def needle_hit(scores: np.ndarray) -> bool:
    """Is the last document (the relevant one) ranked within the top TOP_K?"""
    return int(fusion.ranks_from_scores(scores)[-1]) <= TOP_K


def fuse(rule: str, signal_names: tuple[str, ...], signals: dict[str, np.ndarray]) -> np.ndarray:
    selected = [signals[name] for name in signal_names]
    if rule == "single":
        return selected[0]
    if rule == "rrf":
        return fusion.rrf(selected)
    if rule == "weighted":
        return fusion.weighted_zsum(selected, [WEIGHTED_SUM_WEIGHTS[n] for n in signal_names])
    raise ValueError(f"Unknown fusion rule {rule!r}")


def evaluate_direction(direction: Direction, docs: list[Document], queries: list[Query],
                       qrels: dict[str, list[str]], embeddings: dict[str, Embeddings]) -> dict:
    """Recall@20 of every config in CONFIGS for every embedding model."""
    docs_by_id = {d.doc_id: d for d in docs}
    query_by_id = {q.qid: q for q in queries}
    instances = build_instances(qrels, queries, docs_by_id, direction)

    distractors = [d for d in docs if d.category == direction.distractor_category]
    print(f"[corpus] {direction.key}: {len(distractors)} {direction.distractor_category} "
          f"distractors + 1 relevant document = {len(distractors) + 1} per instance")

    # Placeholder stripping for the lexical channel, done once for every text involved.
    needle_ids = sorted({d for _, d in instances})
    query_ids = sorted({q for q, _ in instances})
    distractor_lex = channels.strip_placeholders([d.text for d in distractors], "distractors")
    needle_lex = dict(zip(needle_ids, channels.strip_placeholders(
        [docs_by_id[d].text for d in needle_ids], "relevant documents")))
    query_lex = dict(zip(query_ids, channels.strip_placeholders(
        [query_by_id[q].text for q in query_ids], "queries")))

    qids_by_needle: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in instances:
        qids_by_needle[doc_id].append(qid)

    dense_all = {m: channels.dense_scores(emb.query, emb.doc) for m, emb in embeddings.items()}
    hits = {m: {c: {} for c in CONFIGS} for m in embeddings}   # model -> config -> (qid, d) -> bool
    started = time.time()
    for n_done, (needle_id, qids) in enumerate(sorted(qids_by_needle.items()), start=1):
        corpus = distractors + [docs_by_id[needle_id]]
        tfidf = channels.tfidf_scores([query_lex[q] for q in qids],
                                      distractor_lex + [needle_lex[needle_id]])
        for model_key, emb in embeddings.items():
            rows = [emb.doc_row[d.doc_id] for d in corpus]
            ccd = channels.ccd_scores(emb.doc[rows])
            for i, qid in enumerate(qids):
                signals = {
                    "dense": dense_all[model_key][emb.query_row[qid], rows].astype(np.float64),
                    "ccd": ccd,
                    "tfidf": tfidf[i],
                }
                for name, (rule, signal_names) in CONFIGS.items():
                    hits[model_key][name][(qid, needle_id)] = needle_hit(
                        fuse(rule, signal_names, signals))
        if n_done % 200 == 0 or n_done == len(qids_by_needle):
            print(f"  [{direction.key}] {n_done}/{len(qids_by_needle)} relevant documents "
                  f"· {time.time() - started:.0f}s")

    recall = {m: {c: round(float(np.mean([h[i] for i in instances])), 4)
                  for c, h in by_config.items()}
              for m, by_config in hits.items()}
    return {
        "direction": direction.key,
        "n_instances": len(instances),
        "n_queries": len(query_ids),
        "n_relevant_documents": len(needle_ids),
        "corpus_size": len(distractors) + 1,
        "top_k": TOP_K,
        "configs": {name: {"rule": rule, "signals": list(sig)}
                    for name, (rule, sig) in CONFIGS.items()},
        "recall": recall,
    }
