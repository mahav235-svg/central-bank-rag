"""A deliberately small interface over chat models.

The agent needs two things from a model: free text (the answer) and JSON that follows a schema
(grading and query rewriting). Each provider maps these onto its own SDK, and normalises the
reason generation stopped so the agent can treat a truncated or refused answer the same way
whichever vendor produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class StopReason(StrEnum):
    COMPLETE = "complete"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass
class LLMResponse:
    text: str
    stop: StopReason
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw_stop_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.text)


class LLMError(RuntimeError):
    """Raised for provider failures the agent cannot recover from (auth, bad request, outage)."""


@runtime_checkable
class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse: ...

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 1000
    ) -> LLMResponse:
        """Return a response whose ``text`` is JSON valid against ``schema``."""
        ...
