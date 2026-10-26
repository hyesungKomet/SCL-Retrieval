"""Documents, queries and qrels.

Identifier rules (the released qrels depend on them, so they must not change):

    doc_id : gold_<category>_<5-digit sequence>, numbered in gold_final.csv file order
             within each category (gold_finance_00001, ...)
    qid    : <direction>_q<4-digit row number>, in query csv file order (fin_gov_q0001, ...)
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import CATEGORIES, DIRECTIONS, GOLD_CSV, Direction

# Call transcripts exceed the default csv field size limit.
csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class Document:
    doc_id: str
    review_id: str      # row key in gold_final.csv
    category: str       # finance / government (relabelled gold label)
    text: str


@dataclass(frozen=True)
class Query:
    qid: str
    category: str
    text: str


def _read_csv(path: Path, required: set[str]) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing data file: {path}\n"
                         "Set SCL_DATA_ROOT to the folder described in data/README.md.")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        return list(reader)


def load_documents(path: Path = GOLD_CSV) -> list[Document]:
    """All documents in gold_final.csv file order. No row is dropped."""
    rows = _read_csv(path, {"review_id", "gold_label", "text"})
    seq = {category: 0 for category in CATEGORIES}
    docs = []
    for row in rows:
        category = row["gold_label"].strip()
        if category not in seq:
            raise ValueError(f"Unknown gold_label {category!r} ({row['review_id']})")
        seq[category] += 1
        docs.append(Document(f"gold_{category}_{seq[category]:05d}",
                             row["review_id"], category, row["text"]))
    counts = ", ".join(f"{c} {n}" for c, n in seq.items())
    print(f"[data] {path.name}: {len(docs)} documents ({counts})")
    return docs


def load_queries(direction: Direction) -> list[Query]:
    """Queries of one direction in file order. Every query must have the expected category."""
    rows = _read_csv(direction.query_csv, {"text", "category"})
    queries = [Query(f"{direction.key}_q{i:04d}", row["category"].strip(), row["text"])
               for i, row in enumerate(rows, start=1)]
    wrong = [q.qid for q in queries if q.category != direction.query_category]
    if wrong:
        raise ValueError(f"{len(wrong)} queries in {direction.query_csv.name} are not "
                         f"{direction.query_category!r}, e.g. {wrong[:3]}")
    print(f"[data] {direction.query_csv.name}: {len(queries)} queries")
    return queries


def load_all_queries() -> list[Query]:
    """fin_gov queries followed by gov_fin queries (the order used for embedding files)."""
    return [q for d in DIRECTIONS.values() for q in load_queries(d)]


def load_qrels(path: Path) -> dict[str, list[str]]:
    """qrels[qid] = sorted list of relevant doc_ids (possibly empty)."""
    qrels = json.loads(path.read_text(encoding="utf-8"))
    return {qid: sorted(docs) for qid, docs in qrels.items()}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[write] {path}")
