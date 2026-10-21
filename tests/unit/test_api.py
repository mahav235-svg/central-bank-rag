from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from policyrag.agent.service import AgentResult
from policyrag.api import app as api
from tests.conftest import make_hit


class StubAgent:
    def __init__(self) -> None:
        self.retriever = self
        self.store = self

    def count(self) -> int:
        return 42

    def retrieve(self, q: str, filters: Any = None, top_k: int = 5) -> list[Any]:
        self.last_filters = filters
        return [make_hit("repo rate unchanged")]

    def ask(self, question: str, thread_id: str | None = None) -> AgentResult:
        return AgentResult(
            thread_id="t1",
            status="answered",
            answer="Unchanged [1].",
            sources=[{"n": 1, "doc_id": "rbi-statement-20240208"}],
        )

    def resume(self, thread_id: str, action: str, answer: str | None = None) -> AgentResult:
        return AgentResult(
            thread_id=thread_id, status={"approve": "approved"}.get(action, action), answer=answer or ""
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    stub = StubAgent()
    monkeypatch.setattr(api, "PolicyAgent", lambda settings: stub)
    with TestClient(api.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["indexed_chunks"] == 42


def test_ask(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "What did the RBI decide?"}).json()
    assert body["status"] == "answered" and body["sources"][0]["n"] == 1


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "?"}).status_code == 422


def test_review_requires_text_for_edit(client: TestClient) -> None:
    assert client.post("/reviews/t1", json={"action": "edit"}).status_code == 422
    assert client.post("/reviews/t1", json={"action": "approve"}).json()["status"] == "approved"


def test_search_passes_filters(client: TestClient) -> None:
    rows = client.get("/search", params={"q": "repo", "bank": "rbi", "date_from": "2024-01-01"}).json()
    assert rows[0]["rank"] == 1 and rows[0]["doc_id"] == "fed-statement-20240131"
