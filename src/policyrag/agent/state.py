"""Graph state. Every field is serialisable so runs can be checkpointed and resumed after a human review."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from policyrag.models import Hit

Status = Literal["answered", "no_answer", "pending_review", "approved", "edited", "rejected"]


class Final(TypedDict):
    status: Status
    answer: str
    sources: list[dict[str, Any]]


class AgentState(TypedDict, total=False):
    question: str
    query: str
    filters: dict[str, Any]
    hits: list[Hit]
    relevant: list[Hit]
    rewrites: int
    answer: str
    stop: str
    cited: list[int]
    issues: list[str]
    final: Final
    # appended to by several nodes; the reducer concatenates instead of overwriting
    trace: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]
