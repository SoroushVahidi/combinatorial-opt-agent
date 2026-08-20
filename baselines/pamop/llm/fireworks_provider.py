"""Fireworks AI provider adapter.

Uses Fireworks's OpenAI-compatible REST endpoint via the ``openai`` SDK
(no separate ``fireworks-ai`` package is installed or required on this
workstation) -- see ``_openai_compatible.py``.
"""

from __future__ import annotations

from .base import get_env_token
from .types import ProviderAuthError
from ._openai_compatible import OpenAICompatibleProvider

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksProvider(OpenAICompatibleProvider):
    name = "fireworks"

    def _base_url(self) -> str:
        return FIREWORKS_BASE_URL

    def _api_key(self) -> str:
        token = get_env_token("FIREWORKS_API_KEY")
        if not token:
            raise ProviderAuthError("FIREWORKS_API_KEY is not set.")
        return token
