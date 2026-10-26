"""Recall@20 of SCL-Retrieval, its components and the baselines (Tables 1 and 2).

    python evaluate.py                                        # Table 1 (KURE-v1)
    python evaluate.py --models kure_v1 bge_m3 qwen3_0_6b qwen3_8b   # Tables 1 and 2

Reads the released qrels in qrels/ unless --qrels-dir points elsewhere (e.g. work/qrels
written by build_qrels.py). Results go to work/results/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scl.config import DEFAULT_MODEL, DIRECTIONS, EMBEDDING_MODELS, RELEASED_QRELS_DIR, RESULTS_DIR
from scl.data import load_all_queries, load_documents, load_qrels, write_json
from scl.encoders import load_embeddings
from scl.evaluate import evaluate_direction

# Rows of Table 1 (paper name, config name). The RPRS baseline is not part of this repo.
TABLE1_ROWS = (
    ("Word TF-IDF", "tfidf"),
    ("Dense", "dense"),
    ("Dense + CCD", "dense+ccd"),
    ("Dense + TF-IDF", "dense+tfidf"),
    ("SCL-Retrieval", "scl"),
    ("SCL-Retrieval, 2:2:1 weighted sum (Sec. 4.3)", "scl_weighted_221"),
)


def mean_of(results: dict, model: str, config: str) -> float:
    return sum(results[d]["recall"][model][config] for d in DIRECTIONS) / len(DIRECTIONS)


def print_tables(results: dict, models: list[str]) -> None:
    print(f"\nTable 1 · Recall@20 · {models[0]}")
    print(f"| {'Method':<46} | Finance | Government | Mean  |")
    print(f"|{'-' * 48}|---------|------------|-------|")
    for label, config in TABLE1_ROWS:
        fin = results["fin_gov"]["recall"][models[0]][config]
        gov = results["gov_fin"]["recall"][models[0]][config]
        print(f"| {label:<46} | {fin:.3f}   | {gov:.3f}      | "
              f"{mean_of(results, models[0], config):.3f} |")
    if len(models) > 1:
        print("\nTable 2 · Recall@20, mean of both query sets")
        print(f"| {'Embedding model':<16} | Dense | SCL-Retrieval | Gain   |")
        print(f"|{'-' * 18}|-------|---------------|--------|")
        for model in models:
            dense = mean_of(results, model, "dense")
            scl = mean_of(results, model, "scl")
            print(f"| {model:<16} | {dense:.3f} | {scl:.3f}         | "
                  f"{100 * (scl - dense) / dense:+.1f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        choices=list(EMBEDDING_MODELS))
    parser.add_argument("--qrels-dir", type=Path, default=RELEASED_QRELS_DIR)
    args = parser.parse_args()

    docs = load_documents()
    queries = load_all_queries()
    doc_ids = [d.doc_id for d in docs]
    qids = [q.qid for q in queries]
    embeddings = {m: load_embeddings(m, doc_ids, qids) for m in args.models}

    results = {}
    for key, direction in DIRECTIONS.items():
        qrels = load_qrels(args.qrels_dir / direction.qrels_name)
        direction_queries = [q for q in queries if q.qid.startswith(f"{key}_")]
        results[key] = evaluate_direction(direction, docs, direction_queries, qrels, embeddings)
        results[key]["qrels"] = str(args.qrels_dir / direction.qrels_name)
        write_json(RESULTS_DIR / f"{key}.json", results[key])
    print_tables(results, args.models)


if __name__ == "__main__":
    main()
