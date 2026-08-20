"""Provider-agnostic request/response types for the PaMOP LLM stage.

Every provider adapter in this package returns an ``LLMResponse`` with the
same shape, so the extraction stage (and later, self-augmented modeling and
correction) never needs to know which provider or SDK actually served a
request. No field here ever holds an API key or token value.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """What to call and how -- mirrors baselines/pamop/config.py's LlmConfig
    but scoped to a single concrete call (a specific provider/model pair)."""

    provider: str
    model: str
    temperature: float
    max_tokens: int | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Everything the extraction/validation layer, and any future audit
    trail, needs about one LLM call -- reproducibility metadata, never
    secrets."""

    text: str
    provider: str
    model: str
    timestamp: str  # ISO-8601 UTC, e.g. "2026-08-11T12:00:00+00:00"
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    retry_count: int
    prompt_hash: str
    finish_reason: str | None = None
    # The exact served model snapshot, when the API echoes one back and it
    # can differ from ``model`` (e.g. an Azure deployment name like
    # "gpt-4.1-mini" resolving to "gpt-4.1-mini-2025-04-14"). None for
    # providers where ``model`` already is the exact snapshot.
    underlying_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderAuthError(RuntimeError):
    """Raised when a provider's required API key/token is not configured.

    This is a real, actionable configuration problem -- surface it clearly,
    but never include the (absent) key's value in the message.
    """


class ProviderCallError(RuntimeError):
    """Raised when a provider call fails for a reason other than missing auth
    (network error, rate limit, malformed response, etc.). The underlying
    exception is chained (``raise ... from exc``), never swallowed."""
