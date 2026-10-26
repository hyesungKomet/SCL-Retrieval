"""Paths, constants and model specs. Every number used by the pipeline lives here.

Two query directions are evaluated (cross-domain retrieval):

    fin_gov : finance-impersonation queries; distractors are all government documents
    gov_fin : government-impersonation queries; distractors are all finance documents
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Paths -------------------------------------------------------------------
# The data folder is only read, never written. Everything generated goes under WORK_ROOT.
DATA_ROOT = Path(os.environ.get("SCL_DATA_ROOT", REPO_ROOT / "data")).resolve()
WORK_ROOT = Path(os.environ.get("SCL_WORK_ROOT", REPO_ROOT / "work")).resolve()

GOLD_CSV = DATA_ROOT / "gold_final.csv"
EMBEDDING_DIR = WORK_ROOT / "embeddings"        # scripts/encode.py writes here
POOLED_QRELS_DIR = WORK_ROOT / "qrels"          # scripts/build_qrels.py writes here
RESULTS_DIR = WORK_ROOT / "results"             # scripts/evaluate.py writes here
RELEASED_QRELS_DIR = REPO_ROOT / "qrels"        # qrels used in the paper (shipped with the repo)

CATEGORIES = ("finance", "government")


@dataclass(frozen=True)
class Direction:
    key: str                 # fin_gov / gov_fin
    query_category: str      # category of the queries (and of their relevant documents)
    distractor_category: str
    query_csv: Path

    @property
    def qrels_name(self) -> str:
        """qrels files are named after the query category."""
        return f"{self.query_category}.json"


DIRECTIONS = {
    "fin_gov": Direction("fin_gov", "finance", "government",
                         DATA_ROOT / "finance_ood_challenge.csv"),
    "gov_fin": Direction("gov_fin", "government", "finance",
                         DATA_ROOT / "government_ood_challenge.csv"),
}

# --- Evaluation (Sec. 4.1) ----------------------------------------------------
TOP_K = 20                   # Recall@20

# --- Fusion (Sec. 3.2) --------------------------------------------------------
RRF_K = 60                   # standard RRF constant (Cormack et al., 2009); not tuned
WEIGHTED_SUM_WEIGHTS = {"dense": 2.0, "ccd": 2.0, "tfidf": 1.0}   # Sec. 4.3 comparison only
ZSCORE_EPS = 1e-12           # guards against a zero standard deviation
CENTROID_EPS = 1e-12         # guards against a zero-norm centroid

# --- Lexical channel (Sec. 3.1 (3)) ------------------------------------------
# Word (eojeol) TF-IDF fitted on each instance corpus. sklearn's default token pattern
# keeps tokens of two or more word characters; case is preserved.
TFIDF_KWARGS = dict(analyzer="word", ngram_range=(1, 1), lowercase=False,
                    min_df=2, sublinear_tf=True, smooth_idf=True, norm="l2")
# De-identification placeholders differ between documents (typed tags such as NAME)
# and queries (masks such as "OO" or "XXX"). Both are replaced with a space before
# TF-IDF so that the lexical channel does not match on masking conventions.
DOC_PLACEHOLDER_PATTERN = r"\b(?:NAME|TIME|ORG|ADDRESS|ACCOUNT|OTHERS|PHONE|PASSWORD)\b"
QUERY_MASK_PATTERN = r"[OoXx○●\*]{2,}"
PLACEHOLDER_REPLACEMENT = " "

# --- Embedding models (Sec. 4.3, Table 2) ------------------------------------
# kind "sentence_transformers": mean/CLS pooling as defined by the model card.
# kind "qwen3": last-token pooling, instruction prefix on queries only (model card).
EMBED_MAX_SEQ_LENGTH = 2048  # same truncation for every model
QWEN3_QUERY_INSTRUCTION = ("Given a web search query, "
                           "retrieve relevant passages that answer the query")
EMBEDDING_MODELS = {
    "kure_v1": dict(hf_name="nlpai-lab/KURE-v1", kind="sentence_transformers", batch_size=4),
    "bge_m3": dict(hf_name="BAAI/bge-m3", kind="sentence_transformers", batch_size=8),
    "qwen3_0_6b": dict(hf_name="Qwen/Qwen3-Embedding-0.6B", kind="qwen3", batch_size=4),
    "qwen3_8b": dict(hf_name="Qwen/Qwen3-Embedding-8B", kind="qwen3", batch_size=1),
}
DEFAULT_MODEL = "kure_v1"    # the encoder of SCL-Retrieval

# --- Relevance pooling (Sec. 4.1) --------------------------------------------
# MIRACL-style annotation pool (Zhang et al., TACL 2023): BM25, mDPR and mColBERT each
# retrieve their top POOL_RETRIEVER_TOP_K from the combined corpus of all 3,014
# documents; scores are min-max normalised per query and averaged, and the top
# POOL_SIZE documents whose category equals the query category become relevant.
# MIRACL used depth 1,000; with ~3k documents that would cover the whole corpus.
POOL_RETRIEVER_TOP_K = 100
POOL_SIZE = 10
BM25_K1 = 1.5
BM25_B = 0.75
MDPR_MODEL = "castorini/mdpr-tied-pft-msmarco"
MDPR_MAX_LENGTH = 512        # mBERT position limit
MCOLBERT_REPO = "vjeronymo2/mColBERT"
MCOLBERT_CHECKPOINT_FILE = (
    "experiments/MSMARCO-psg/train.py/msmarco.psg.cosine/checkpoints/colbert-400000.dnn")
MCOLBERT_BASE_MODEL = "bert-base-multilingual-uncased"
