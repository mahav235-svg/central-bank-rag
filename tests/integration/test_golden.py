"""Checks the golden set against the built catalog, so a re-chunk or parser change that silently
breaks a question's ground truth fails loudly. Skipped when no catalog has been built."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyrag.catalog import Catalog
from policyrag.eval.metrics import _norm
from policyrag.eval.run import load_golden

CATALOG = Path("data/catalog.duckdb")
GOLDEN = Path("eval/golden.jsonl")

pytestmark = pytest.mark.skipif(not CATALOG.exists(), reason="no catalog; run `cbrag ingest` first")


@pytest.fixture(scope="module")
def doc_texts() -> dict[str, str]:
    with Catalog(CATALOG, read_only=True) as cat:
        texts: dict[str, list[str]] = {}
        for c in cat.chunks():
            texts.setdefault(c.doc_id, []).append(_norm(c.text))
    return {k: " ".join(v) for k, v in texts.items()}


def test_golden_ids_unique() -> None:
    ids = [i.id for i in load_golden(GOLDEN)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("item", [i for i in load_golden(GOLDEN) if i.answerable], ids=lambda i: i.id)
def test_every_evidence_string_is_in_a_gold_document(item, doc_texts) -> None:  # type: ignore[no-untyped-def]
    missing_docs = [d for d in item.doc_ids if d not in doc_texts]
    assert not missing_docs, f"gold documents not in catalog: {missing_docs}"
    for ev in item.evidence:
        assert any(_norm(ev) in doc_texts[d] for d in item.doc_ids), f"{ev!r} not found in {item.doc_ids}"
