"""A scripted chat model for tests and offline demos. No network, fully deterministic."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from policyrag.llm.base import LLMResponse, StopReason

Responder = Callable[[str, str, dict[str, Any] | None], str]


class FakeChat:
    """Answers with whatever ``responder(system, user, schema)`` returns.

    The default responder returns a schema-shaped JSON object for structured calls (every boolean
    true, every number 1, every string "ok") and, for free text, cites excerpt [1].
    """

    def __init__(self, responder: Responder | None = None, stop: StopReason = StopReason.COMPLETE) -> None:
        self.name = "fake"
        self.responder = responder or _default
        self.stop = stop
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        self.calls.append((system, user, None))
        return LLMResponse(text=self.responder(system, user, None), stop=self.stop, model=self.name)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 1000
    ) -> LLMResponse:
        self.calls.append((system, user, schema))
        return LLMResponse(text=self.responder(system, user, schema), stop=self.stop, model=self.name)


def _shape(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if kind == "object":
        return {k: _shape(v) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return [_shape(schema.get("items", {}))]
    if kind == "boolean":
        return True
    if kind in ("integer", "number"):
        return 1
    if "enum" in schema:
        return schema["enum"][0]
    return "ok"


def _default(system: str, user: str, schema: dict[str, Any] | None) -> str:
    if schema is not None:
        return json.dumps(_shape(schema))
    return "The excerpt answers the question [1]."
