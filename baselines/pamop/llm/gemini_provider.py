"""Google Gemini provider adapter (google-genai SDK, Gemini Developer API).

This repository's other Gemini code (``tools/llm_baselines.py``) reads
``GEMINI_API_KEY``. On this workstation that variable is unset but
``GOOGLE_API_KEY`` is set instead (both are recognized by the google-genai
SDK itself) -- this adapter checks both, preferring ``GEMINI_API_KEY`` for
consistency with the rest of the repo, falling back to ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

from .base import LLMProvider, get_env_token
from .types import ModelConfig, ProviderAuthError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def _client(self):
        token = get_env_token("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not token:
            raise ProviderAuthError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set.")
        from google import genai

        return genai.Client(api_key=token)

    def _call(self, prompt: str, config: ModelConfig) -> dict:
        from google.genai import types as genai_types

        client = self._client()
        gen_config = genai_types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=gen_config,
        )
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = str(finish_reason) if finish_reason is not None else None
        return {
            "text": response.text or "",
            "finish_reason": finish_reason,
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
