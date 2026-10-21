"""A thin facade over the compiled graph, used by the CLI, the HTTP API and the MCP server."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from policyrag.agent.graph import build_graph, sqlite_checkpointer
from policyrag.agent.nodes import AgentConfig
from policyrag.config import Settings
from policyrag.llm import make_chat_model
from policyrag.llm.base import ChatModel
from policyrag.pipeline import make_retriever
from policyrag.retrieval.retriever import Retriever


@dataclass
class AgentResult:
    thread_id: str
    status: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] | None = None  # set when the run is paused for a human decision
    trace: list[str] = field(default_factory=list)
    tokens: int = 0


class PolicyAgent:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever | None = None,
        llm: ChatModel | None = None,
        durable: bool = True,
    ) -> None:
        self.settings = settings
        cfg = AgentConfig(
            max_rewrites=settings.max_rewrites,
            review_mode=settings.review_mode,
            review_queue=settings.review_queue_path,
            max_answer_tokens=settings.max_output_tokens,
        )
        checkpointer = sqlite_checkpointer(settings.checkpoint_path) if durable else None
        self.retriever = retriever or make_retriever(settings)
        self.graph = build_graph(self.retriever, llm or make_chat_model(settings), cfg, checkpointer)

    def ask(self, question: str, thread_id: str | None = None) -> AgentResult:
        thread_id = thread_id or uuid.uuid4().hex
        out = self.graph.invoke({"question": question}, self._config(thread_id))
        return self._result(thread_id, out)

    def resume(self, thread_id: str, action: str, answer: str | None = None) -> AgentResult:
        """Continue a run paused in ``human_review`` (interrupt mode) with approve, edit or reject."""
        decision = {"action": action, "answer": answer or ""}
        out = self.graph.invoke(Command(resume=decision), self._config(thread_id))
        return self._result(thread_id, out)

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _result(self, thread_id: str, out: dict[str, Any]) -> AgentResult:
        interrupts = out.get("__interrupt__") or []
        if interrupts:
            return AgentResult(
                thread_id=thread_id,
                status="awaiting_review",
                answer=out.get("answer", ""),
                review=interrupts[0].value,
                trace=out.get("trace", []),
                tokens=out.get("tokens", 0),
            )
        final = out.get("final") or {"status": "error", "answer": "", "sources": []}
        return AgentResult(
            thread_id=thread_id,
            status=final["status"],
            answer=final["answer"],
            sources=final["sources"],
            trace=out.get("trace", []),
            tokens=out.get("tokens", 0),
        )
