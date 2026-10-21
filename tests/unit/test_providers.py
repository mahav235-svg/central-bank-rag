"""Provider adapters, exercised against stub clients so no network or API key is needed."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from policyrag.llm.anthropic_provider import AnthropicChat
from policyrag.llm.base import StopReason
from policyrag.llm.ollama_provider import OllamaChat
from policyrag.llm.openai_provider import OpenAIChat


class Recorder:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _claude_response(stop: str, text: str = "Answer [1].") -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("end_turn", StopReason.COMPLETE),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("pause_turn", StopReason.OTHER),
    ],
)
def test_anthropic_stop_mapping_and_text(raw: str, expected: StopReason) -> None:
    rec = Recorder(_claude_response(raw))
    client = SimpleNamespace(beta=SimpleNamespace(messages=rec), messages=rec)
    resp = AnthropicChat(client=client).complete("sys", "user")
    assert resp.stop is expected and resp.text == "Answer [1]." and resp.input_tokens == 100


def test_anthropic_request_shape() -> None:
    rec = Recorder(_claude_response("end_turn", '{"relevant": [1]}'))
    client = SimpleNamespace(beta=SimpleNamespace(messages=rec), messages=rec)
    schema = {"type": "object", "properties": {"relevant": {"type": "array"}}}
    AnthropicChat(client=client, effort="low").complete_json("sys", "user", schema)
    assert rec.kwargs["model"] == "claude-opus-5-5"
    assert rec.kwargs["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert rec.kwargs["fallbacks"] == "default" and rec.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in rec.kwargs  # thinking cannot be disabled on this model; effort controls depth


def test_anthropic_without_fallbacks_uses_plain_endpoint() -> None:
    plain = Recorder(_claude_response("end_turn"))
    client = SimpleNamespace(beta=SimpleNamespace(messages=None), messages=plain)
    AnthropicChat(client=client, fallbacks=False).complete("s", "u")
    assert "betas" not in plain.kwargs


def _openai_response(status: str = "completed", reason: str | None = None, refusal: bool = False) -> Any:
    content = [SimpleNamespace(type="refusal" if refusal else "output_text")]
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(content=content)],
        output_text="Answer [1].",
        model="gpt-5",
        usage=SimpleNamespace(input_tokens=50, output_tokens=10),
    )


@pytest.mark.parametrize(
    "resp, expected",
    [
        (_openai_response(), StopReason.COMPLETE),
        (_openai_response("incomplete", "max_output_tokens"), StopReason.MAX_TOKENS),
        (_openai_response("incomplete", "content_filter"), StopReason.REFUSAL),
        (_openai_response(refusal=True), StopReason.REFUSAL),
    ],
)
def test_openai_stop_mapping(resp: Any, expected: StopReason) -> None:
    rec = Recorder(resp)
    out = OpenAIChat(client=SimpleNamespace(responses=rec)).complete("sys", "user", max_tokens=99)
    assert out.stop is expected
    assert rec.kwargs["instructions"] == "sys" and rec.kwargs["max_output_tokens"] == 99


def test_openai_json_schema_is_strict() -> None:
    rec = Recorder(_openai_response())
    OpenAIChat(client=SimpleNamespace(responses=rec)).complete_json("s", "u", {"type": "object"})
    assert rec.kwargs["text"]["format"]["strict"] is True


def test_ollama_request_and_mapping() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b-instruct",
                "done_reason": "length",
                "message": {"content": "{}"},
                "prompt_eval_count": 7,
                "eval_count": 3,
            },
        )

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    out = OllamaChat(client=client).complete_json("s", "u", {"type": "object"}, max_tokens=50)
    assert seen["format"] == {"type": "object"} and seen["options"]["num_predict"] == 50
    assert out.stop is StopReason.MAX_TOKENS and out.input_tokens == 7
