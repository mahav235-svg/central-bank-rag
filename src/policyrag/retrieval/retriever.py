"""Retrieval pipeline: metadata filters, hybrid first stage, optional cross-encoder rerank."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from policyrag.index.embeddings import Reranker
from policyrag.index.store import Filters, SearchMode, VectorStore
from policyrag.models import Hit
from policyrag.retrieval.query_parser import parse

log = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    mode: SearchMode = "hybrid"
    top_k: int = 6
    candidates: int = 30
    rerank: bool = True
    use_query_filters: bool = True


class Retriever:
    def __init__(
        self, store: VectorStore, reranker: Reranker | None, config: RetrievalConfig | None = None
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def retrieve(self, query: str, filters: Filters | None = None, top_k: int | None = None) -> list[Hit]:
        cfg = self.config
        k = top_k or cfg.top_k
        if filters is None:
            filters = parse(query).filters if cfg.use_query_filters else Filters()

        first_stage = cfg.candidates if (cfg.rerank and self.reranker) else k
        hits = self.store.search(
            query, limit=first_stage, mode=cfg.mode, filters=filters, candidates=cfg.candidates
        )
        if not hits and not filters.is_empty():
            # a filter guessed from the wording can be wrong ("India" in a question about the Fed's
            # view of emerging markets); fall back to an unfiltered search rather than return nothing
            log.info("no hits with filters %s; retrying without filters", filters)
            hits = self.store.search(
                query, limit=first_stage, mode=cfg.mode, filters=None, candidates=cfg.candidates
            )

        if cfg.rerank and self.reranker and hits:
            # Score the provenance header with the text. Statements reuse near-identical wording from
            # meeting to meeting; without the bank and date the cross-encoder cannot tell them apart
            # and reranking made retrieval worse (see docs/evaluation.md).
            scores = self.reranker.scores(query, [f"{h.chunk.header()}\n{h.chunk.text}" for h in hits])
            for h, s in zip(hits, scores, strict=True):
                h.score = s
                h.sources.append("rerank")
            hits.sort(key=lambda h: h.score, reverse=True)

        hits = hits[:k]
        for rank, h in enumerate(hits, start=1):
            h.rank = rank
        return hits
