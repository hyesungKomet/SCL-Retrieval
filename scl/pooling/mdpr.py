"""mDPR (MIRACL baseline): mBERT fine-tuned on MS MARCO, shared query/document encoder.

The [CLS] vector is the representation and dot product the score. Vectors are not
normalised; the pool is built from per-query min-max normalised scores anyway.
"""
from __future__ import annotations

import numpy as np

from ..config import MDPR_MAX_LENGTH, MDPR_MODEL


class MDPR:
    def __init__(self, device: str, batch_size: int = 16) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MDPR_MODEL)
        self.model = AutoModel.from_pretrained(MDPR_MODEL).to(device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                       truncation=True, max_length=MDPR_MAX_LENGTH,
                                       return_tensors="pt").to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0]
                out.append(cls.float().cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def score_matrix(self, query_texts: list[str], doc_texts: list[str]) -> np.ndarray:
        """(n_queries, n_docs) dot products."""
        return self.encode(query_texts) @ self.encode(doc_texts).T
