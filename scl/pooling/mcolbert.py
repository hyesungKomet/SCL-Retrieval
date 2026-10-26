"""mColBERT (MIRACL baseline), checkpoint vjeronymo2/mColBERT colbert-400000.dnn.

The forward pass of ColBERT-v1 is re-implemented here instead of depending on the old
`colbert` package:

    encoder     bert-base-multilingual-uncased
    projection  Linear(768 -> 2*dim, no bias); the output is split in two halves that are
                concatenated along the sequence axis, so each token yields two vectors
                (a property of this fork)
    query       [Q] marker, padded to query_maxlen with [MASK] (query augmentation)
    document    [D] marker; punctuation and padding vectors are zeroed
    score       MaxSim: for each query vector the max cosine over document vectors, summed

Training settings (dim, maxlen, similarity) are read from the checkpoint and checked.
"""
from __future__ import annotations

import string

import numpy as np

from ..config import MCOLBERT_BASE_MODEL, MCOLBERT_CHECKPOINT_FILE, MCOLBERT_REPO

EXPECTED_DIM = 32
EXPECTED_SIMILARITY = "cosine"


class MColBERT:
    def __init__(self, device: str, doc_batch_size: int = 16, query_batch_size: int = 32) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import BertModel, BertTokenizerFast

        self.torch = torch
        self.device = device
        self.doc_batch_size = doc_batch_size
        self.query_batch_size = query_batch_size

        checkpoint = torch.load(hf_hub_download(MCOLBERT_REPO, MCOLBERT_CHECKPOINT_FILE),
                                map_location="cpu", weights_only=False)
        args = checkpoint.get("arguments", {})
        self.dim = int(args["dim"])
        self.query_maxlen = int(args["query_maxlen"])
        self.doc_maxlen = int(args["doc_maxlen"])
        if self.dim != EXPECTED_DIM or args.get("similarity") != EXPECTED_SIMILARITY:
            raise RuntimeError(f"Unexpected mColBERT checkpoint settings: {args}")

        self.bert = BertModel.from_pretrained(MCOLBERT_BASE_MODEL)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 2 * self.dim, bias=False)
        state = checkpoint["model_state_dict"]
        bert_state = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
        missing, unexpected = self.bert.load_state_dict(bert_state, strict=False)
        leftover = [k for k in list(missing) + list(unexpected) if "position_ids" not in k]
        if leftover:
            raise RuntimeError(f"mColBERT state_dict mismatch: {leftover}")
        self.linear.load_state_dict({"weight": state["linear.weight"]})
        self.bert = self.bert.to(device).eval()
        self.linear = self.linear.to(device).eval()

        self.tok = BertTokenizerFast.from_pretrained(MCOLBERT_BASE_MODEL)
        self.q_marker = self.tok.convert_tokens_to_ids("[unused0]")
        self.d_marker = self.tok.convert_tokens_to_ids("[unused1]")
        self.skip_ids = {self.tok.convert_tokens_to_ids(s) for s in string.punctuation}
        self.skip_ids.discard(self.tok.unk_token_id)

    def _project(self, input_ids, attention_mask):
        hidden = self.bert(input_ids.to(self.device),
                           attention_mask=attention_mask.to(self.device))[0]
        halves = self.linear(hidden).split(self.dim, dim=2)
        return self.torch.cat(halves, dim=1)                       # (B, 2L, dim)

    def _encode_queries(self, texts: list[str]):
        out = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.query_batch_size):
                # The leading ". " reserves position 1 for the [Q] marker.
                enc = self.tok([". " + t for t in texts[start:start + self.query_batch_size]],
                               padding="max_length", truncation=True,
                               max_length=self.query_maxlen, return_tensors="pt")
                ids = enc["input_ids"]
                ids[:, 1] = self.q_marker
                ids[ids == self.tok.pad_token_id] = self.tok.mask_token_id
                q = self._project(ids, enc["attention_mask"])
                out.append(self.torch.nn.functional.normalize(q, p=2, dim=2).cpu())
        return self.torch.cat(out, dim=0)

    def _encode_docs(self, texts: list[str]):
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.doc_batch_size):
                enc = self.tok([". " + t for t in texts[start:start + self.doc_batch_size]],
                               padding="longest", truncation=True,
                               max_length=self.doc_maxlen, return_tensors="pt")
                ids = enc["input_ids"]
                ids[:, 1] = self.d_marker
                keep = self.torch.tensor(
                    [[(x not in self.skip_ids) and (x != self.tok.pad_token_id) for x in row]
                     for row in ids.tolist()], dtype=self.torch.bool)
                keep = self.torch.cat([keep, keep], dim=1)         # two vectors per token
                d = self._project(ids, enc["attention_mask"])
                d = d * keep.to(self.device).unsqueeze(2).float()
                d = self.torch.nn.functional.normalize(d, p=2, dim=2).cpu()
                for i in range(d.shape[0]):
                    vectors.append(d[i][keep[i]])
        return vectors

    def score_matrix(self, query_texts: list[str], doc_texts: list[str]) -> np.ndarray:
        """(n_queries, n_docs) MaxSim scores."""
        torch = self.torch
        doc_vecs = self._encode_docs(doc_texts)
        q_vecs = self._encode_queries(query_texts)
        max_len = max(v.shape[0] for v in doc_vecs)
        doc_matrix = torch.zeros(len(doc_vecs), max_len, self.dim)
        valid = torch.zeros(len(doc_vecs), max_len, dtype=torch.bool)
        for i, v in enumerate(doc_vecs):
            doc_matrix[i, :v.shape[0]] = v
            valid[i, :v.shape[0]] = True
        doc_matrix, valid = doc_matrix.to(self.device), valid.to(self.device)
        scores = np.zeros((len(query_texts), len(doc_texts)), dtype=np.float32)
        with torch.no_grad():
            for qi in range(len(query_texts)):
                sim = torch.einsum("qd,nld->qnl", q_vecs[qi].to(self.device), doc_matrix)
                sim = sim.masked_fill(~valid.unsqueeze(0), float("-inf"))
                max_sim = torch.nan_to_num(sim.max(dim=-1).values, neginf=0.0)
                scores[qi] = max_sim.sum(dim=0).cpu().numpy()
        return scores
