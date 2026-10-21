"""OpenAI models through the official ``openai`` SDK (Responses API)."""

from __future__ import annotations

from typing import Any

from policyrag.llm.base import LLMError, LLMResponse, StopReason


class OpenAIChat:
    def __init__(self, model: str = "gpt-5", client: Any = None) -> None:
        if client is None:
            import openai

            client = openai.OpenAI()  # reads OPENAI_API_KEY
        self.client = client
        self.name = model

    def _create(
        self, system: str, user: str, max_tokens: int, text_format: dict[str, Any] | None
    ) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.name,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if text_format:
            kwargs["text"] = {"format": text_format}
        try:
            resp = self.client.responses.create(**kwargs)
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.BadRequestError,
            openai.NotFoundError,
        ) as exc:
            raise LLMError(f"OpenAI request rejected: {exc}") from exc
        except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as exc:
            raise LLMError(f"OpenAI request failed after retries: {exc}") from exc

        stop = StopReason.COMPLETE
        raw = resp.status
        if resp.status == "incomplete":
            reason = getattr(resp.incomplete_details, "reason", None)
            raw = f"incomplete:{reason}"
            stop = StopReason.MAX_TOKENS if reason == "max_output_tokens" else StopReason.REFUSAL
        if _has_refusal(resp):
            stop = StopReason.REFUSAL
        usage = resp.usage
        return LLMResponse(
            text=resp.output_text or "",
            stop=stop,
            model=resp.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            raw_stop_reason=raw,
        )

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        return self._create(system, user, max_tokens, None)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 1000
    ) -> LLMResponse:
        fmt = {"type": "json_schema", "name": "result", "schema": schema, "strict": True}
        return self._create(system, user, max_tokens, fmt)


def _has_refusal(resp: Any) -> bool:
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return True
    return False
