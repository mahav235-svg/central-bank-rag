"""Local open-weight models through Ollama's REST API (no API key, runs offline).

Ollama's ``/api/chat`` accepts a JSON schema in ``format`` and constrains decoding to it, which
gives the same structured-output contract as the hosted providers.
"""

from __future__ import annotations

from typing import Any

import httpx

from policyrag.llm.base import LLMError, LLMResponse, StopReason


class OllamaChat:
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = model
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def _chat(self, system: str, user: str, max_tokens: int, fmt: dict[str, Any] | None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if fmt is not None:
            body["format"] = fmt
        try:
            resp = self.client.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama request failed (is `ollama serve` running and the model pulled?): {exc}"
            ) from exc
        data = resp.json()
        done = data.get("done_reason")
        stop = {"stop": StopReason.COMPLETE, "length": StopReason.MAX_TOKENS}.get(
            done or "", StopReason.OTHER
        )
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            stop=stop,
            model=data.get("model", self.name),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            raw_stop_reason=done,
        )

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        return self._chat(system, user, max_tokens, None)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 1000
    ) -> LLMResponse:
        return self._chat(system, user, max_tokens, schema)
