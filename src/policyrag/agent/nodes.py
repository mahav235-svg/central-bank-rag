"""The agent's steps. Each node reads the state it needs and returns only the keys it changes."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.types import interrupt

from policyrag.agent import prompts
from policyrag.agent.state import AgentState, Final
from policyrag.index.store import Filters
from policyrag.llm.base import ChatModel, LLMResponse, StopReason
from policyrag.models import Bank, DocType, Hit
from policyrag.retrieval.query_parser import parse
from policyrag.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")


def filters_to_dict(f: Filters) -> dict[str, Any]:
    return {
        "banks": [b.value for b in f.banks],
        "doc_types": [d.value for d in f.doc_types],
        "date_from": f.date_from.isoformat() if f.date_from else None,
        "date_to": f.date_to.isoformat() if f.date_to else None,
    }


def filters_from_dict(d: dict[str, Any]) -> Filters:
    return Filters(
        banks=[Bank(b) for b in d.get("banks", [])],
        doc_types=[DocType(t) for t in d.get("doc_types", [])],
        date_from=date.fromisoformat(d["date_from"]) if d.get("date_from") else None,
        date_to=date.fromisoformat(d["date_to"]) if d.get("date_to") else None,
    )


def citations(text: str) -> list[int]:
    return sorted({int(n) for n in _CITATION.findall(text)})


def source_list(hits: list[Hit], cited: list[int]) -> list[dict[str, Any]]:
    out = []
    for n in cited:
        if 1 <= n <= len(hits):
            c = hits[n - 1].chunk
            out.append(
                {
                    "n": n,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "date": c.meeting_date.isoformat(),
                    "url": c.url,
                    "speaker": c.speaker,
                    "heading": c.heading,
                    "chunk_id": c.chunk_id,
                }
            )
    return out


@dataclass
class AgentConfig:
    max_rewrites: int = 1
    review_mode: Literal["interrupt", "queue"] = "queue"
    review_queue: Path = Path("data/review_queue.jsonl")
    judge: bool = True
    max_answer_tokens: int = 2000


class Nodes:
    def __init__(self, retriever: Retriever, llm: ChatModel, config: AgentConfig) -> None:
        self.retriever = retriever
        self.llm = llm
        self.cfg = config

    # -- retrieval ------------------------------------------------------------------------------

    def analyze(self, state: AgentState) -> dict[str, Any]:
        parsed = parse(state["question"])
        return {
            "query": state["question"],
            "filters": filters_to_dict(parsed.filters),
            "rewrites": 0,
            "trace": [f"analyze: filters={filters_to_dict(parsed.filters)}"],
        }

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        hits = self.retriever.retrieve(state["query"], filters=filters_from_dict(state.get("filters", {})))
        return {"hits": hits, "trace": [f"retrieve: {len(hits)} hits for {state['query']!r}"]}

    def grade_documents(self, state: AgentState) -> dict[str, Any]:
        hits = state.get("hits", [])
        if not hits:
            return {"relevant": [], "trace": ["grade: nothing to grade"]}
        resp = self.llm.complete_json(
            prompts.GRADE_SYSTEM,
            prompts.grade_prompt(state["question"], hits),
            prompts.GRADE_SCHEMA,
            max_tokens=400,
        )
        keep = _parse_relevant(resp, len(hits))
        relevant = [hits[i - 1] for i in keep]
        return {
            "relevant": relevant,
            "tokens": _tokens(resp),
            "trace": [f"grade: {len(relevant)}/{len(hits)} relevant"],
        }

    def rewrite_query(self, state: AgentState) -> dict[str, Any]:
        resp = self.llm.complete_json(
            prompts.REWRITE_SYSTEM,
            prompts.rewrite_prompt(state["question"], state["query"]),
            prompts.REWRITE_SCHEMA,
            max_tokens=200,
        )
        try:
            query = str(resp.json()["query"]).strip() or state["query"]
        except (ValueError, KeyError, TypeError):
            query = state["query"]
        return {
            "query": query,
            "rewrites": state.get("rewrites", 0) + 1,
            "tokens": _tokens(resp),
            "trace": [f"rewrite: {query!r}"],
        }

    # -- generation and checks ------------------------------------------------------------------

    def generate(self, state: AgentState) -> dict[str, Any]:
        hits = state["relevant"]
        resp = self.llm.complete(
            prompts.ANSWER_SYSTEM,
            prompts.answer_prompt(state["question"], hits),
            max_tokens=self.cfg.max_answer_tokens,
        )
        return {
            "answer": resp.text.strip(),
            "stop": resp.stop.value,
            "cited": citations(resp.text),
            "tokens": _tokens(resp),
            "trace": [f"generate: stop={resp.stop.value}, model={resp.model}"],
        }

    def check_answer(self, state: AgentState) -> dict[str, Any]:
        """Cheap deterministic checks first; the model judge only runs if they all pass."""
        answer, hits, cited = state.get("answer", ""), state["relevant"], state.get("cited", [])
        issues: list[str] = []
        if state.get("stop") == StopReason.REFUSAL.value:
            issues.append("model declined to answer")
        elif state.get("stop") == StopReason.MAX_TOKENS.value:
            issues.append("answer was cut off at the token limit")
        if answer == prompts.INSUFFICIENT:
            issues.append("model reported the excerpts do not answer the question")
        elif not cited:
            issues.append("answer has no citations")
        invalid = [n for n in cited if not 1 <= n <= len(hits)]
        if invalid:
            issues.append(f"cites excerpts that were not provided: {invalid}")
        tokens = 0
        if not issues and self.cfg.judge:
            resp = self.llm.complete_json(
                prompts.JUDGE_SYSTEM, prompts.judge_prompt(answer, hits), prompts.JUDGE_SCHEMA, max_tokens=600
            )
            tokens = _tokens(resp)
            try:
                verdict = resp.json()
                if not verdict.get("supported", False):
                    claims = verdict.get("unsupported_claims") or ["(unspecified)"]
                    issues.append("unsupported claims: " + "; ".join(map(str, claims)))
            except ValueError:
                issues.append("groundedness judge returned invalid JSON")
        return {
            "issues": issues,
            "tokens": tokens,
            "trace": [f"check: {'ok' if not issues else '; '.join(issues)}"],
        }

    # -- endings --------------------------------------------------------------------------------

    def finalize(self, state: AgentState) -> dict[str, Any]:
        final: Final = {
            "status": "answered",
            "answer": state["answer"],
            "sources": source_list(state["relevant"], state.get("cited", [])),
        }
        return {"final": final, "trace": ["finalize: answered"]}

    def no_answer(self, state: AgentState) -> dict[str, Any]:
        final: Final = {
            "status": "no_answer",
            "answer": "I could not find this in the Fed or RBI documents I have indexed.",
            "sources": [],
        }
        return {"final": final, "trace": ["no_answer: nothing relevant after retries"]}

    def human_review(self, state: AgentState) -> dict[str, Any]:
        payload = {
            "question": state["question"],
            "draft_answer": state.get("answer", ""),
            "issues": state.get("issues", []),
            "sources": source_list(
                state.get("relevant", []), list(range(1, len(state.get("relevant", [])) + 1))
            ),
        }
        if self.cfg.review_mode == "queue":
            self.cfg.review_queue.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.review_queue.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**payload, "queued_at": datetime.now(UTC).isoformat()}) + "\n")
            final: Final = {
                "status": "pending_review",
                "answer": "This answer needs a human check before it can be shown.",
                "sources": [],
            }
            return {"final": final, "trace": ["human_review: queued"]}

        decision = interrupt(payload)  # execution pauses here until resumed with Command(resume=...)
        action = (decision or {}).get("action", "reject")
        if action == "approve":
            final = {
                "status": "approved",
                "answer": state.get("answer", ""),
                "sources": source_list(state["relevant"], state.get("cited", [])),
            }
        elif action == "edit":
            text = str(decision.get("answer", "")).strip()
            final = {
                "status": "edited",
                "answer": text,
                "sources": source_list(state["relevant"], citations(text)),
            }
        else:
            final = {"status": "rejected", "answer": "A reviewer rejected the draft answer.", "sources": []}
        return {"final": final, "trace": [f"human_review: {action}"]}

    # -- routing --------------------------------------------------------------------------------

    def route_after_grade(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("rewrites", 0) < self.cfg.max_rewrites:
            return "rewrite_query"
        return "no_answer"

    @staticmethod
    def route_after_check(state: AgentState) -> str:
        return "finalize" if not state.get("issues") else "human_review"


def _parse_relevant(resp: LLMResponse, n: int) -> list[int]:
    try:
        nums = resp.json().get("relevant", [])
    except (ValueError, AttributeError):
        log.warning("grader returned invalid JSON; keeping all hits")
        return list(range(1, n + 1))
    return sorted({int(x) for x in nums if isinstance(x, int | float) and 1 <= int(x) <= n})


def _tokens(resp: LLMResponse) -> int:
    return resp.input_tokens + resp.output_tokens
