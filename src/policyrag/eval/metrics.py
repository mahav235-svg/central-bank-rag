"""Retrieval and answer metrics.

Retrieval is scored against two kinds of ground truth per question:

* ``doc_ids``: documents that contain the answer (document-level relevance), and
* ``evidence``: short strings that must appear in a retrieved passage (passage-level relevance).

A passage counts as relevant if it comes from a gold document and, when evidence strings are
given, contains at least one of them (case-insensitive, whitespace-normalised). This avoids hand
labelling chunk ids, which would go stale every time the chunking changes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from policyrag.models import Hit

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower().replace("’", "'")).strip()


def is_relevant(hit: Hit, doc_ids: Sequence[str], evidence: Sequence[str]) -> bool:
    if doc_ids and hit.chunk.doc_id not in doc_ids:
        return False
    if not evidence:
        return True
    text = _norm(hit.chunk.text)
    return any(_norm(e) in text for e in evidence)


def relevance_vector(hits: Sequence[Hit], doc_ids: Sequence[str], evidence: Sequence[str]) -> list[bool]:
    return [is_relevant(h, doc_ids, evidence) for h in hits]


def hit_at_k(rel: Sequence[bool], k: int) -> float:
    return float(any(rel[:k]))


def reciprocal_rank(rel: Sequence[bool]) -> float:
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: Sequence[bool], k: int) -> float:
    """Binary-relevance nDCG. The ideal ranking puts every relevant passage we retrieved first."""
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rel[:k], start=1) if r)
    n_rel = sum(rel[:k])
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg else 0.0


def doc_recall_at_k(hits: Sequence[Hit], doc_ids: Sequence[str], k: int) -> float:
    """Share of gold documents represented in the top k. Matters for comparison questions."""
    if not doc_ids:
        return float("nan")
    found = {h.chunk.doc_id for h in hits[:k]} & set(doc_ids)
    return len(found) / len(set(doc_ids))


def contains_all(answer: str, must_include: Sequence[str]) -> float:
    """1.0 if every required fact string appears in the answer (normalised)."""
    a = _norm(answer)
    return float(all(_norm(m) in a for m in must_include)) if must_include else float("nan")
