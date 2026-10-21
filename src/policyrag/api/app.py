"""HTTP API (FastAPI).

POST /ask                      ask a question
POST /reviews/{thread_id}      approve, edit or reject an answer paused for review
GET  /search                   retrieval only, no LLM (useful for debugging)
GET  /health                   liveness and index size
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from policyrag import __version__
from policyrag.agent.service import PolicyAgent
from policyrag.config import get_settings
from policyrag.index.store import Filters
from policyrag.llm.base import LLMError
from policyrag.models import Bank, DocType


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    thread_id: str | None = None


class ReviewRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    answer: str | None = None


class AskResponse(BaseModel):
    thread_id: str
    status: str
    answer: str
    sources: list[dict[str, Any]]
    review: dict[str, Any] | None = None


_state: dict[str, PolicyAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _state["agent"] = PolicyAgent(get_settings())
    yield
    _state.clear()


app = FastAPI(title="central-bank-rag", version=__version__, lifespan=lifespan)


def _agent() -> PolicyAgent:
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, "agent not initialised")
    return agent


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "indexed_chunks": _agent().retriever.store.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        res = _agent().ask(req.question, req.thread_id)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return AskResponse(
        thread_id=res.thread_id, status=res.status, answer=res.answer, sources=res.sources, review=res.review
    )


@app.post("/reviews/{thread_id}", response_model=AskResponse)
def review(thread_id: str, req: ReviewRequest) -> AskResponse:
    if req.action == "edit" and not (req.answer or "").strip():
        raise HTTPException(422, "an edited answer is required for action 'edit'")
    res = _agent().resume(thread_id, req.action, req.answer)
    return AskResponse(thread_id=res.thread_id, status=res.status, answer=res.answer, sources=res.sources)


@app.get("/search")
def search(
    q: str = Query(min_length=2),
    k: int = Query(5, ge=1, le=20),
    bank: Bank | None = None,
    doc_type: DocType | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict[str, Any]]:
    filters = None
    if any((bank, doc_type, date_from, date_to)):
        filters = Filters(
            banks=[bank] if bank else [],
            doc_types=[doc_type] if doc_type else [],
            date_from=date_from,
            date_to=date_to,
        )
    hits = _agent().retriever.retrieve(q, filters=filters, top_k=k)
    return [{"rank": h.rank, "score": round(h.score, 4), **h.chunk.model_dump(mode="json")} for h in hits]
