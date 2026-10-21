"""Evaluation runner.

``retrieval``: compares retrieval configurations on the golden set without calling any LLM, so it
is free and deterministic and can run in CI against a built index.

``answer``: runs the full agent and scores the final answers. It needs an LLM provider and costs
money with hosted models; run it deliberately, not on every commit.
"""

from __future__ import annotations

import json
import math
import statistics
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policyrag.catalog import Catalog
from policyrag.eval import metrics
from policyrag.index.embeddings import BGE_QUERY_INSTRUCTION
from policyrag.retrieval.retriever import RetrievalConfig, Retriever


@dataclass
class GoldItem:
    id: str
    question: str
    category: str
    doc_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    answerable: bool = True


def load_golden(path: Path) -> list[GoldItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            items.append(GoldItem(**json.loads(line)))
    return items


RETRIEVAL_VARIANTS: dict[str, dict[str, Any]] = {
    "dense": {"mode": "dense", "rerank": False, "use_query_filters": False},
    "bm25": {"mode": "sparse", "rerank": False, "use_query_filters": False},
    "hybrid": {"mode": "hybrid", "rerank": False, "use_query_filters": False},
    "hybrid+filters": {"mode": "hybrid", "rerank": False, "use_query_filters": True},
    "hybrid+filters+rerank": {"mode": "hybrid", "rerank": True, "use_query_filters": True},
    "hybrid+filters+bge_instruction": {
        "mode": "hybrid",
        "rerank": False,
        "use_query_filters": True,
        "query_instruction": True,
    },
}


def evaluate_retrieval(
    retriever: Retriever, items: Sequence[GoldItem], k: int = 5
) -> dict[str, dict[str, float]]:
    """Return per-question metrics for one retriever configuration."""
    out: dict[str, dict[str, float]] = {}
    for item in items:
        if not item.answerable:
            continue
        hits = retriever.retrieve(item.question, top_k=max(k, 10))
        rel = metrics.relevance_vector(hits, item.doc_ids, item.evidence)
        out[item.id] = {
            "hit@1": metrics.hit_at_k(rel, 1),
            f"hit@{k}": metrics.hit_at_k(rel, k),
            "mrr@10": metrics.reciprocal_rank(rel[:10]),
            f"ndcg@{k}": metrics.ndcg_at_k(rel, k),
            f"doc_recall@{k}": metrics.doc_recall_at_k(hits, item.doc_ids, k),
        }
    return out


def run_retrieval_suite(
    make: Any,
    items: Sequence[GoldItem],
    catalog: Catalog,
    variants: Iterable[str] = RETRIEVAL_VARIANTS,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """``make(**overrides)`` must return a Retriever. Returns mean metrics per variant."""
    summary: dict[str, dict[str, float]] = {}
    for label in variants:
        overrides = dict(RETRIEVAL_VARIANTS[label])
        instruction = overrides.pop("query_instruction", False)
        retriever: Retriever = make(**overrides)
        encoder = retriever.store.encoder
        saved = encoder.query_instruction
        encoder.query_instruction = BGE_QUERY_INSTRUCTION if instruction else None
        try:
            per_q = evaluate_retrieval(retriever, items, k)
        finally:
            encoder.query_instruction = saved
        run_id = f"ret-{label}-{uuid.uuid4().hex[:8]}"
        rows = [(qid, m, v, None) for qid, ms in per_q.items() for m, v in ms.items()]
        catalog.record_eval(run_id, "retrieval", label, {**overrides, "k": k, **_cfg(retriever.config)}, rows)
        summary[label] = _mean(per_q)
    return summary


def run_answer_suite(agent: Any, items: Sequence[GoldItem], catalog: Catalog, label: str) -> dict[str, float]:
    rows: list[tuple[str, str, float | None, str | None]] = []
    per_q: dict[str, dict[str, float]] = {}
    for item in items:
        res = agent.ask(item.question)
        answered = res.status in ("answered", "approved", "edited")
        cited_docs = {s["doc_id"] for s in res.sources}
        m: dict[str, float] = {}
        if item.answerable:
            m["answered"] = float(answered)
            m["correct"] = metrics.contains_all(res.answer, item.must_include) if answered else 0.0
            m["citation_precision"] = (
                len(cited_docs & set(item.doc_ids)) / len(cited_docs)
                if cited_docs and item.doc_ids
                else float("nan")
            )
        else:
            m["abstained"] = float(not answered)
        m["tokens"] = float(res.tokens)
        per_q[item.id] = m
        rows += [
            (item.id, k, v, json.dumps({"status": res.status, "answer": res.answer[:2000]}))
            for k, v in m.items()
        ]
    catalog.record_eval(f"ans-{label}-{uuid.uuid4().hex[:8]}", "answer", label, {"label": label}, rows)
    return _mean(per_q)


def _cfg(cfg: RetrievalConfig) -> dict[str, Any]:
    return {"top_k": cfg.top_k, "candidates": cfg.candidates}


def _mean(per_q: dict[str, dict[str, float]]) -> dict[str, float]:
    cols: dict[str, list[float]] = defaultdict(list)
    for ms in per_q.values():
        for k, v in ms.items():
            if not (isinstance(v, float) and math.isnan(v)):
                cols[k].append(v)
    return {k: round(statistics.fmean(v), 4) for k, v in cols.items() if v} | {"n": float(len(per_q))}


def markdown_table(summary: dict[str, dict[str, float]], title: str) -> str:
    metrics_order = sorted({m for s in summary.values() for m in s if m != "n"})
    lines = [
        f"### {title}",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.",
        "",
        "| variant | n | " + " | ".join(metrics_order) + " |",
        "|---|---|" + "---|" * len(metrics_order),
    ]
    for label, s in summary.items():
        vals = " | ".join(f"{s.get(m, float('nan')):.3f}" for m in metrics_order)
        lines.append(f"| {label} | {int(s.get('n', 0))} | {vals} |")
    return "\n".join(lines) + "\n"
