"""CloudRift AI provider adapter.

Uses CloudRift's OpenAI-compatible endpoint (``CLOUDRIFT_BASE_URL``, e.g.
``https://inference.cloudrift.ai/v1``) via the ``openai`` SDK -- this is
also what this workstation's ambient ``OPENAI_BASE_URL`` happens to point
at, which is exactly why ``openai_provider.py`` refuses to trust that
variable: "OpenAI" and "CloudRift" must stay two distinct, explicitly
configured providers even though they share client machinery.
"""

from __future__ import annotations

from .base import get_env_token
from .types import ProviderAuthError
from ._openai_compatible import OpenAICompatibleProvider

DEFAULT_CLOUDRIFT_BASE_URL = "https://inference.cloudrift.ai/v1"


class CloudRiftProvider(OpenAICompatibleProvider):
    name = "cloudrift"

    def _base_url(self) -> str:
        return get_env_token("CLOUDRIFT_BASE_URL") or DEFAULT_CLOUDRIFT_BASE_URL

    def _api_key(self) -> str:
        token = get_env_token("CLOUDRIFT_API_KEY")
        if not token:
            raise ProviderAuthError("CLOUDRIFT_API_KEY is not set.")
        return token
