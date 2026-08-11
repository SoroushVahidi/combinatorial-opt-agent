"""OpenAI provider adapter.

Environment note (verified on this workstation during this milestone): the
``OPENAI_BASE_URL`` environment variable is currently set to
``https://inference.cloudrift.ai/v1`` -- i.e. it points at CloudRift, not
OpenAI, presumably to let other tools reuse an OpenAI-compatible client
against CloudRift's endpoint. The ``openai`` Python SDK reads
``OPENAI_BASE_URL`` automatically when no ``base_url`` is passed, so
constructing ``OpenAI()`` with no arguments here would silently send
"OpenAI" calls to CloudRift and could never reach a real GPT-4 model. This
adapter therefore *always* passes an explicit ``base_url`` for the real
OpenAI API, ignoring ``OPENAI_BASE_URL`` -- see ``CloudRiftProvider`` for
the (separate, correctly-named) adapter that intentionally targets that
endpoint.
"""

from __future__ import annotations

from .base import LLMProvider, get_env_token
from .types import ModelConfig, ProviderAuthError

REAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider):
    name = "openai"

    def _client(self):
        token = get_env_token("OPENAI_API_KEY")
        if not token:
            raise ProviderAuthError(
                "OPENAI_API_KEY is not set. (Note: OPENAI_BASE_URL in this "
                "environment points at CloudRift, not OpenAI -- this "
                "adapter ignores it and always targets the real OpenAI API.)"
            )
        from openai import OpenAI

        return OpenAI(api_key=token, base_url=REAL_OPENAI_BASE_URL)

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
