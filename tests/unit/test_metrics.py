import math

import pytest

from policyrag.eval import metrics
from tests.conftest import make_hit


def test_relevance_requires_doc_and_evidence() -> None:
    hits = [
        make_hit("rate unchanged at 6.50 per cent", doc_id="rbi-statement-20240208"),
        make_hit("rate unchanged at 6.50 per cent", doc_id="rbi-statement-20231208"),
        make_hit("food inflation", doc_id="rbi-statement-20240208"),
    ]
    rel = metrics.relevance_vector(hits, ["rbi-statement-20240208"], ["6.50 per cent"])
    assert rel == [True, False, False]


def test_rank_metrics() -> None:
    rel = [False, True, False, True]
    assert metrics.hit_at_k(rel, 1) == 0.0
    assert metrics.hit_at_k(rel, 2) == 1.0
    assert metrics.reciprocal_rank(rel) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(rel, 4) == pytest.approx(expected)
    assert metrics.ndcg_at_k([False, False], 2) == 0.0


def test_doc_recall_and_contains_all() -> None:
    hits = [make_hit("a", doc_id="d1"), make_hit("b", doc_id="d3")]
    assert metrics.doc_recall_at_k(hits, ["d1", "d2"], 2) == 0.5
    assert metrics.contains_all("Repo rate kept at 6.50 per cent.", ["6.50 per cent", "repo"]) == 1.0
    assert metrics.contains_all("Repo rate kept.", ["6.50"]) == 0.0
    assert math.isnan(metrics.contains_all("x", []))
