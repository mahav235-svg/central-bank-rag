"""Chat model providers behind one interface."""

from __future__ import annotations

from policyrag.config import Settings
from policyrag.llm.base import ChatModel, LLMError, LLMResponse, StopReason


def make_chat_model(settings: Settings) -> ChatModel:
    """Build the configured provider. SDKs are imported lazily so only the chosen one is required."""
    if settings.provider == "anthropic":
        from policyrag.llm.anthropic_provider import AnthropicChat

        return AnthropicChat(settings.anthropic_model)
    if settings.provider == "openai":
        from policyrag.llm.openai_provider import OpenAIChat

        return OpenAIChat(settings.openai_model)
    if settings.provider == "ollama":
        from policyrag.llm.ollama_provider import OllamaChat

        return OllamaChat(settings.ollama_model, settings.ollama_url)
    from policyrag.llm.fake import FakeChat

    return FakeChat()


__all__ = ["ChatModel", "LLMError", "LLMResponse", "StopReason", "make_chat_model"]
