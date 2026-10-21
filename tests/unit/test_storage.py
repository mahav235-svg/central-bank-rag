from __future__ import annotations

from datetime import date
from pathlib import Path

from policyrag.catalog import Catalog, load_documents, save_document
from policyrag.chunking import chunk_document
from policyrag.index.store import Filters, VectorStore
from policyrag.models import Bank, Chunk, DocType, Document, Section
from policyrag.retrieval.retriever import RetrievalConfig, Retriever
from tests.conftest import HashEncoder


def _doc() -> Document:
    return Document(
        doc_id="rbi-statement-20240208",
        bank=Bank.RBI,
        doc_type=DocType.STATEMENT,
        meeting_date=date(2024, 2, 8),
        title="Resolution",
        url="https://example.org",
        sections=[
            Section(text="The MPC kept the repo rate at 6.50 per cent."),
            Section(heading="Outlook", text="Inflation is projected at 4.5 per cent."),
        ],
    )


def test_catalog_roundtrip_and_upsert_replaces(tmp_path: Path) -> None:
    doc = _doc()
    with Catalog(tmp_path / "c.duckdb") as cat:
        cat.upsert(doc, chunk_document(doc))
        cat.upsert(doc, chunk_document(doc))  # idempotent
        chunks = cat.chunks()
        assert [c.chunk_id for c in chunks] == ["rbi-statement-20240208#0000", "rbi-statement-20240208#0001"]
        assert chunks[1].heading == "Outlook"
        assert cat.known_doc_ids() == {"rbi-statement-20240208"}
        (bank, dtype, docs, *_rest, n_chunks) = cat.summary()[0]
        assert (bank, dtype, docs, n_chunks) == ("rbi", "statement", 1, 2)
        assert cat.export_parquet(tmp_path / "out").exists()


def test_document_json_roundtrip(tmp_path: Path) -> None:
    save_document(_doc(), tmp_path)
    assert load_documents(tmp_path) == [_doc()]


def test_store_filters_and_modes(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    store = VectorStore(HashEncoder(), "test", path=tmp_path / "q")  # type: ignore[arg-type]
    store.recreate()
    assert store.upsert(sample_chunks) == len(sample_chunks)
    assert store.count() == len(sample_chunks)

    for mode in ("dense", "sparse", "hybrid"):
        hits = store.search("policy repo rate unchanged", limit=2, mode=mode)  # type: ignore[arg-type]
        assert hits[0].chunk.doc_id == "rbi-statement-20240208", mode

    only_fed = store.search("inflation", limit=5, filters=Filters(banks=[Bank.FED]))
    assert only_fed and all(h.chunk.bank is Bank.FED for h in only_fed)
    feb = store.search(
        "inflation", limit=5, filters=Filters(date_from=date(2024, 2, 1), date_to=date(2024, 2, 29))
    )
    assert feb and all(h.chunk.meeting_date == date(2024, 2, 8) for h in feb)
    store.close()


class ReverseReranker:
    """Scores passages by length so the test can see the reranker reorder results."""

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [float(len(p)) for p in passages]


def test_retriever_reranks_and_falls_back_without_filters(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    store = VectorStore(HashEncoder(), "test", path=tmp_path / "q")  # type: ignore[arg-type]
    store.recreate()
    store.upsert(sample_chunks)
    r = Retriever(store, ReverseReranker(), RetrievalConfig(top_k=2, candidates=5))  # type: ignore[arg-type]
    hits = r.retrieve("inflation")
    assert len(hits) == 2 and hits[0].score >= hits[1].score and "rerank" in hits[0].sources
    assert [h.rank for h in hits] == [1, 2]

    # a filter that matches nothing triggers the unfiltered retry
    none = Filters(date_from=date(2030, 1, 1))
    assert r.retrieve("inflation", filters=none)
    store.close()


def test_sync_embeds_only_new_or_changed_chunks(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    store = VectorStore(HashEncoder(), "test", path=tmp_path / "q")  # type: ignore[arg-type]
    store.recreate()
    assert store.sync(sample_chunks) == (5, 0)
    assert store.sync(sample_chunks) == (0, 0)
    edited = [sample_chunks[0].model_copy(update={"text": "Edited text."}), *sample_chunks[1:4]]
    assert store.sync(edited) == (1, 1)  # one re-embedded, the dropped fifth chunk removed
    assert store.count() == 4
    assert store.search("Edited text", limit=1)[0].chunk.text == "Edited text."
    store.close()
