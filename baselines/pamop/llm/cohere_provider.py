"""Cohere provider adapter (Cohere ClientV2 chat API)."""

from __future__ import annotations

from .base import LLMProvider, get_env_token
from .types import ModelConfig, ProviderAuthError


class CohereProvider(LLMProvider):
    name = "cohere"

    def _client(self):
        token = get_env_token("COHERE_API_KEY", "CO_API_KEY")
        if not token:
            raise ProviderAuthError("Neither COHERE_API_KEY nor CO_API_KEY is set.")
        import cohere

        return cohere.ClientV2(api_key=token)

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
            kwargs["p"] = config.top_p  # Cohere calls this "p", not "top_p"

        response = client.chat(**kwargs)
        content = response.message.content or []
        text = "".join(block.text for block in content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        billed = getattr(usage, "billed_units", None) if usage else None
        return {
            "text": text,
            "finish_reason": getattr(response, "finish_reason", None),
            "prompt_tokens": getattr(billed, "input_tokens", None) if billed else None,
            "completion_tokens": getattr(billed, "output_tokens", None) if billed else None,
            "total_tokens": None,
        }
