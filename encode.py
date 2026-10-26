"""Encode every document and query once per embedding model.

    python encode.py --models kure_v1 --device cuda
    python encode.py --models kure_v1 bge_m3 qwen3_0_6b qwen3_8b --device cuda

Vectors go to work/embeddings/<model>/ and are reused by evaluate.py.
"""
from __future__ import annotations

import argparse

from scl.config import DEFAULT_MODEL, EMBEDDING_MODELS
from scl.data import load_all_queries, load_documents
from scl.encoders import encode_texts, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        choices=list(EMBEDDING_MODELS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    docs = load_documents()
    queries = load_all_queries()
    for model_key in args.models:
        doc_vecs = encode_texts(model_key, [d.text for d in docs], args.device, is_query=False)
        query_vecs = encode_texts(model_key, [q.text for q in queries], args.device, is_query=True)
        save_embeddings(model_key, [d.doc_id for d in docs], doc_vecs,
                        [q.qid for q in queries], query_vecs)


if __name__ == "__main__":
    main()
