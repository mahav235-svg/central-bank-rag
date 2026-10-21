"""Ingest and index orchestration, shared by the CLI and the container entrypoint."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from policyrag.catalog import Catalog, load_documents, save_document
from policyrag.chunking import chunk_document
from policyrag.config import Settings
from policyrag.index.embeddings import BGE_QUERY_INSTRUCTION, Encoder, Reranker
from policyrag.index.store import VectorStore
from policyrag.retrieval.retriever import RetrievalConfig, Retriever
from policyrag.sources import SOURCES
from policyrag.sources.http import Fetcher

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    listed: int = 0
    fetched: int = 0
    skipped_existing: int = 0
    failed: list[str] = field(default_factory=list)


def ingest(
    settings: Settings, banks: Iterable[str] = ("fed", "rbi"), refresh: bool = False, offline: bool = False
) -> IngestReport:
    """Discover documents, download what is new, parse, chunk and store them in the catalog."""
    fetcher = Fetcher(settings.raw_dir, settings.user_agent, settings.request_interval, offline=offline)
    docs_dir = settings.processed_dir / "documents"
    report = IngestReport()
    with Catalog(settings.catalog_path) as catalog:
        known = catalog.known_doc_ids()
        for bank in banks:
            source = SOURCES[bank]
            refs = source.list_refs(fetcher, settings.start_year)
            report.listed += len(refs)
            todo = [r for r in refs if refresh or r.doc_id not in known]
            report.skipped_existing += len(refs) - len(todo)
            parsed = {
                d.doc_id
                for d in source.iter_documents(fetcher, todo)
                if _store(d, catalog, docs_dir, settings)
            }
            report.fetched += len(parsed)
            report.failed += [r.doc_id for r in todo if r.doc_id not in parsed]
    return report


def _store(doc, catalog: Catalog, docs_dir, settings: Settings) -> bool:  # type: ignore[no-untyped-def]
    chunks = chunk_document(doc, settings.chunk_words, settings.chunk_overlap_words)
    if not chunks:
        return False
    save_document(doc, docs_dir)
    catalog.upsert(doc, chunks)
    return True


def rechunk(settings: Settings) -> int:
    """Re-chunk every stored document with the current chunk settings (no network)."""
    n = 0
    with Catalog(settings.catalog_path) as catalog:
        for doc in load_documents(settings.processed_dir / "documents"):
            catalog.upsert(doc, chunk_document(doc, settings.chunk_words, settings.chunk_overlap_words))
            n += 1
    return n


def make_encoder(settings: Settings) -> Encoder:
    instruction = BGE_QUERY_INSTRUCTION if settings.bge_query_instruction else None
    return Encoder(settings.dense_model, settings.sparse_model, query_instruction=instruction)


def open_store(settings: Settings, encoder: Encoder | None = None) -> VectorStore:
    return VectorStore(
        encoder or make_encoder(settings),
        settings.collection,
        path=None if settings.qdrant_url else settings.qdrant_path,
        url=settings.qdrant_url,
    )


def build_index(settings: Settings, full: bool = False) -> tuple[int, int]:
    """Sync the vector collection with the catalog; ``full`` drops and rebuilds it.

    Returns (chunks embedded, points deleted).
    """
    with Catalog(settings.catalog_path) as catalog:
        chunks = catalog.chunks()
    store = open_store(settings)
    try:
        if full or not store.client.collection_exists(store.collection):
            store.recreate()
            return store.upsert(chunks), 0
        return store.sync(chunks)
    finally:
        store.close()


def make_retriever(settings: Settings, store: VectorStore | None = None, **overrides: object) -> Retriever:
    store = store or open_store(settings)
    cfg = RetrievalConfig(top_k=settings.top_k, candidates=settings.candidates, rerank=settings.use_reranker)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    reranker = Reranker(settings.rerank_model) if cfg.rerank else None
    return Retriever(store, reranker, cfg)
