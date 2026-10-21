from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from policyrag.agent import prompts
from policyrag.agent.graph import build_graph
from policyrag.agent.nodes import AgentConfig
from policyrag.index.store import Filters
from policyrag.llm.base import StopReason
from policyrag.llm.fake import FakeChat
from policyrag.models import Hit
from tests.conftest import make_hit


class StubRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def retrieve(self, query: str, filters: Filters | None = None, top_k: int | None = None) -> list[Hit]:
        self.queries.append(query)
        return self.hits


HITS = [
    make_hit(
        "The MPC kept the policy repo rate unchanged at 6.50 per cent.", 1, doc_id="rbi-statement-20240208"
    ),
    make_hit("Food inflation stayed high.", 2, doc_id="rbi-minutes-20240208"),
]


def responder(relevant: list[int], answer: str, supported: bool = True, rewrite: str = "better query") -> Any:
    def respond(system: str, user: str, schema: dict[str, Any] | None) -> str:
        if system == prompts.GRADE_SYSTEM:
            return json.dumps({"relevant": relevant})
        if system == prompts.REWRITE_SYSTEM:
            return json.dumps({"query": rewrite})
        if system == prompts.JUDGE_SYSTEM:
            return json.dumps(
                {"supported": supported, "unsupported_claims": [] if supported else ["made up"]}
            )
        return answer

    return respond


def _run(
    llm: FakeChat, retriever: StubRetriever, tmp_path: Path, mode: str = "queue"
) -> tuple[Any, dict[str, Any]]:
    cfg = AgentConfig(max_rewrites=1, review_mode=mode, review_queue=tmp_path / "queue.jsonl")  # type: ignore[arg-type]
    graph = build_graph(retriever, llm, cfg)  # type: ignore[arg-type]
    config = {"configurable": {"thread_id": "t1"}}
    return graph, graph.invoke(
        {"question": "What did the RBI decide on the repo rate in February 2024?"}, config
    )


def test_happy_path_answers_with_sources(tmp_path: Path) -> None:
    llm = FakeChat(responder([1], "The repo rate was kept at 6.50 per cent [1]."))
    _, out = _run(llm, StubRetriever(HITS), tmp_path)
    assert out["final"]["status"] == "answered"
    assert out["final"]["sources"][0]["doc_id"] == "rbi-statement-20240208"
    assert [t.split(":")[0] for t in out["trace"]] == [
        "analyze",
        "retrieve",
        "grade",
        "generate",
        "check",
        "finalize",
    ]
    assert out["filters"]["banks"] == ["rbi"]


def test_nothing_relevant_rewrites_once_then_gives_up(tmp_path: Path) -> None:
    retriever = StubRetriever(HITS)
    llm = FakeChat(responder([], "unused", rewrite="RBI policy repo rate February 2024 decision"))
    _, out = _run(llm, retriever, tmp_path)
    assert out["final"]["status"] == "no_answer"
    assert retriever.queries[1] == "RBI policy repo rate February 2024 decision"
    assert out["rewrites"] == 1


@pytest.mark.parametrize(
    "answer, stop, supported, issue",
    [
        ("The rate was 6.50 per cent.", StopReason.COMPLETE, True, "no citations"),
        ("The rate was 6.50 per cent [7].", StopReason.COMPLETE, True, "not provided"),
        (prompts.INSUFFICIENT, StopReason.COMPLETE, True, "do not answer"),
        ("The rate was 6.50 per cent [1].", StopReason.MAX_TOKENS, True, "cut off"),
        ("The rate was 6.50 per cent [1].", StopReason.REFUSAL, True, "declined"),
        ("The rate was 7 per cent [1].", StopReason.COMPLETE, False, "unsupported"),
    ],
)
def test_failed_checks_go_to_the_review_queue(
    tmp_path: Path, answer: str, stop: StopReason, supported: bool, issue: str
) -> None:
    llm = FakeChat(responder([1], answer, supported=supported), stop=stop)
    _, out = _run(llm, StubRetriever(HITS), tmp_path)
    assert out["final"]["status"] == "pending_review"
    assert any(issue in i for i in out["issues"])
    queued = [json.loads(x) for x in (tmp_path / "queue.jsonl").read_text().splitlines()]
    assert queued[0]["draft_answer"] == answer


@pytest.mark.parametrize(
    "decision, status",
    [
        ({"action": "approve"}, "approved"),
        ({"action": "edit", "answer": "Kept at 6.50 per cent [1]."}, "edited"),
        ({"action": "reject"}, "rejected"),
    ],
)
def test_interrupt_mode_pauses_and_resumes(tmp_path: Path, decision: dict[str, str], status: str) -> None:
    llm = FakeChat(responder([1], "No citation here."))
    graph, out = _run(llm, StubRetriever(HITS), tmp_path, mode="interrupt")
    assert "__interrupt__" in out and "final" not in out
    assert out["__interrupt__"][0].value["issues"] == ["answer has no citations"]
    resumed = graph.invoke(Command(resume=decision), {"configurable": {"thread_id": "t1"}})
    assert resumed["final"]["status"] == status
    if status == "edited":
        assert resumed["final"]["sources"][0]["n"] == 1


def test_grader_garbage_keeps_all_hits(tmp_path: Path) -> None:
    def respond(system: str, user: str, schema: dict[str, Any] | None) -> str:
        if system == prompts.GRADE_SYSTEM:
            return "not json"
        return responder([1], "Kept at 6.50 per cent [1][2].")(system, user, schema)

    _, out = _run(FakeChat(respond), StubRetriever(HITS), tmp_path)
    assert len(out["relevant"]) == 2 and out["final"]["status"] == "answered"
