"""Shared implementation for providers that expose an OpenAI-compatible
chat-completions endpoint (Fireworks AI, CloudRift AI) -- avoids duplicating
the request/response handling in ``openai_provider.py`` while keeping each
provider's own env-var names and default base URL/model explicit and
separate.
"""

from __future__ import annotations

from .base import LLMProvider
from .types import ModelConfig


class OpenAICompatibleProvider(LLMProvider):
    """Subclasses set ``name``, and implement ``_base_url()`` /
    ``_api_key()`` / ``_default_model()``."""

    def _base_url(self) -> str:
        raise NotImplementedError

    def _api_key(self) -> str:
        raise NotImplementedError

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key(), base_url=self._base_url())

    def _call(self, prompt: str, config: ModelConfig) -> dict:
        client = self._client()
        kwargs: dict = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        }
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
