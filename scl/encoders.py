"""Document and query embeddings, stored once per model and reused by the evaluation.

Layout of work/embeddings/<model_key>/:
    doc.npy     (n_docs, dim)     float32, L2-normalised, gold_final.csv order
    query.npy   (n_queries, dim)  float32, L2-normalised, fin_gov then gov_fin queries
    meta.json   model name, truncation length, doc_ids and qids (used to verify the rows)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (EMBED_MAX_SEQ_LENGTH, EMBEDDING_DIR, EMBEDDING_MODELS,
                     QWEN3_QUERY_INSTRUCTION)
from .data import write_json

NORM_TOLERANCE = 1e-3


@dataclass
class Embeddings:
    model_key: str
    doc: np.ndarray
    query: np.ndarray
    doc_row: dict[str, int]      # doc_id -> row
    query_row: dict[str, int]    # qid -> row


def embedding_dir(model_key: str) -> Path:
    return EMBEDDING_DIR / model_key


def _encode_sentence_transformers(hf_name: str, texts: list[str], device: str,
                                  batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(hf_name, device=device)
    model.max_seq_length = EMBED_MAX_SEQ_LENGTH
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True)
    return np.asarray(vecs, dtype=np.float32)


def _encode_qwen3(hf_name: str, texts: list[str], device: str, batch_size: int) -> np.ndarray:
    """Last-token pooling as in the Qwen3-Embedding model card (left padding)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_name, padding_side="left")
    # float16 produces NaNs on Apple MPS; bfloat16 does not.
    dtype = torch.bfloat16 if device == "mps" else torch.float16
    model = AutoModel.from_pretrained(hf_name, dtype=dtype).to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=EMBED_MAX_SEQ_LENGTH, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            last = hidden[:, -1]                    # left padding: last position is the last token
            last = torch.nn.functional.normalize(last.float(), p=2, dim=-1)
            out.append(last.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def encode_texts(model_key: str, texts: list[str], device: str, is_query: bool) -> np.ndarray:
    spec = EMBEDDING_MODELS[model_key]
    started = time.time()
    if spec["kind"] == "sentence_transformers":
        vecs = _encode_sentence_transformers(spec["hf_name"], texts, device, spec["batch_size"])
    elif spec["kind"] == "qwen3":
        # The instruction is prepended to queries only; documents are encoded as is.
        if is_query:
            texts = [f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery:{t}" for t in texts]
        vecs = _encode_qwen3(spec["hf_name"], texts, device, spec["batch_size"])
    else:
        raise ValueError(f"Unknown encoder kind {spec['kind']!r}")
    print(f"[encode] {model_key}: {len(texts)} {'queries' if is_query else 'documents'} "
          f"-> {vecs.shape} in {time.time() - started:.0f}s")
    return vecs


def save_embeddings(model_key: str, doc_ids: list[str], doc_vecs: np.ndarray,
                    qids: list[str], query_vecs: np.ndarray) -> None:
    out = embedding_dir(model_key)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "doc.npy", doc_vecs)
    np.save(out / "query.npy", query_vecs)
    spec = EMBEDDING_MODELS[model_key]
    write_json(out / "meta.json", {
        "model_key": model_key, "hf_name": spec["hf_name"], "kind": spec["kind"],
        "max_seq_length": EMBED_MAX_SEQ_LENGTH,
        "query_instruction": QWEN3_QUERY_INSTRUCTION if spec["kind"] == "qwen3" else None,
        "doc_ids": doc_ids, "qids": qids,
    })


def load_embeddings(model_key: str, doc_ids: list[str], qids: list[str]) -> Embeddings:
    """Load the stored vectors and check that their rows match the current data exactly."""
    d = embedding_dir(model_key)
    if not (d / "meta.json").exists():
        raise SystemExit(f"No embeddings for {model_key} in {d}. "
                         f"Run: python scripts/encode.py --models {model_key}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta["doc_ids"] != doc_ids or meta["qids"] != qids:
        raise SystemExit(f"{d}/meta.json lists different doc_ids/qids than the data. "
                         f"Re-run scripts/encode.py --models {model_key}")
    doc = np.load(d / "doc.npy").astype(np.float32)
    query = np.load(d / "query.npy").astype(np.float32)
    for name, mat in (("doc", doc), ("query", query)):
        norms = np.linalg.norm(mat, axis=1)
        if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
            raise SystemExit(f"{model_key} {name} vectors are not L2-normalised "
                             f"(norm {norms.min():.4f}..{norms.max():.4f})")
    print(f"[embeddings] {model_key}: doc {doc.shape} · query {query.shape}")
    return Embeddings(model_key, doc, query,
                      {doc_id: i for i, doc_id in enumerate(doc_ids)},
                      {qid: i for i, qid in enumerate(qids)})
