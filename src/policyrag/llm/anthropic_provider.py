"""Claude through the official ``anthropic`` SDK (Messages API)."""

from __future__ import annotations

from typing import Any

from policyrag.llm.base import LLMError, LLMResponse, StopReason

_STOP = {
    "end_turn": StopReason.COMPLETE,
    "stop_sequence": StopReason.COMPLETE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}


_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicChat:
    """Claude chat model.

    ``effort`` maps to ``output_config.effort``. Grounded question answering over a handful of
    excerpts is not a hard reasoning task, so ``low`` keeps latency and cost down; raise it if
    the evaluation shows answers missing nuance.

    ``fallbacks`` turns on server-side refusal fallbacks: if a safety classifier declines the
    request, the API re-runs it on a fallback model inside the same call. A refusal that survives
    the fallback still comes back as ``StopReason.REFUSAL`` and the agent sends it to review.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str | None = "low",
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = (
                anthropic.Anthropic()
            )  # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
        self.client = client
        self.name = model
        self.effort = effort
        self.fallbacks = fallbacks

    def _create(
        self, system: str, user: str, max_tokens: int, output_format: dict[str, Any] | None
    ) -> LLMResponse:
        import anthropic

        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if output_format:
            output_config["format"] = output_format
        kwargs: dict[str, Any] = {
            "model": self.name,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        try:
            if self.fallbacks:
                resp = self.client.beta.messages.create(**kwargs, betas=[_FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.BadRequestError,
            anthropic.NotFoundError,
        ) as exc:
            raise LLMError(f"Claude request rejected: {exc}") from exc
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            # the SDK has already retried transient failures (429, 5xx, network) with backoff
            raise LLMError(f"Claude request failed after retries: {exc}") from exc

        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            stop=_STOP.get(resp.stop_reason or "", StopReason.OTHER),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw_stop_reason=resp.stop_reason,
        )

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        return self._create(system, user, max_tokens, None)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 1000
    ) -> LLMResponse:
        return self._create(system, user, max_tokens, {"type": "json_schema", "schema": schema})
