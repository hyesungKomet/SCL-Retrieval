"""Rebuild the relevance judgements (qrels) by pooling BM25, mDPR and mColBERT (Sec. 4.1).

The qrels used in the paper are shipped in qrels/. This script regenerates them from the
data and, when the released files exist, reports how many queries differ.

    python build_qrels.py --device cuda
    python build_qrels.py --device mps --directions fin_gov

Pooling is run on the combined corpus of all documents (finance + government), not on
the evaluation corpus, so the relevant documents compete with every same-category document.
"""
from __future__ import annotations

import argparse

from scl.config import DIRECTIONS, POOLED_QRELS_DIR, RELEASED_QRELS_DIR
from scl.data import load_documents, load_qrels, load_queries, write_json
from scl.pooling.bm25 import BM25
from scl.pooling.mcolbert import MColBERT
from scl.pooling.mdpr import MDPR
from scl.pooling.pool import pool_and_judge


def compare_with_released(qrels: dict[str, list[str]], name: str) -> None:
    released_path = RELEASED_QRELS_DIR / name
    if not released_path.exists():
        print(f"[compare] {released_path} not found; skipped")
        return
    released = load_qrels(released_path)
    differing = [qid for qid in released if released[qid] != qrels.get(qid)]
    print(f"[compare] {name}: {len(differing)}/{len(released)} queries differ from the "
          f"released qrels" + (f", e.g. {differing[:5]}" if differing else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--directions", nargs="+", default=list(DIRECTIONS),
                        choices=list(DIRECTIONS))
    args = parser.parse_args()

    docs = load_documents()
    doc_ids = [d.doc_id for d in docs]
    doc_texts = [d.text for d in docs]
    doc_categories = [d.category for d in docs]
    print(f"[pool] combined corpus: {len(docs)} documents")

    bm25 = BM25(doc_texts)
    mdpr = MDPR(args.device)
    mcolbert = MColBERT(args.device)
    for key in args.directions:
        direction = DIRECTIONS[key]
        queries = load_queries(direction)
        query_texts = [q.text for q in queries]
        print(f"[pool] {key}: scoring {len(queries)} queries with BM25, mDPR, mColBERT")
        score_matrices = {
            "bm25": [bm25.scores(t) for t in query_texts],
            "mdpr": mdpr.score_matrix(query_texts, doc_texts),
            "mcolbert": mcolbert.score_matrix(query_texts, doc_texts),
        }
        qrels, summary = pool_and_judge(score_matrices, [q.qid for q in queries],
                                        direction.query_category, doc_ids, doc_categories)
        print(f"[pool] {key}: {summary['pairs_relevant']}/{summary['pairs_total']} pooled pairs "
              f"relevant, {summary['queries_zero_relevant']} queries without a relevant document")
        write_json(POOLED_QRELS_DIR / direction.qrels_name, qrels)
        write_json(POOLED_QRELS_DIR / f"{direction.query_category}_summary.json", summary)
        compare_with_released(qrels, direction.qrels_name)


if __name__ == "__main__":
    main()
