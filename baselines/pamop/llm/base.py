"""Base provider interface and shared helpers (prompt hashing, env lookup,
timing/retry scaffolding) used by every provider adapter in this package."""

from __future__ import annotations

import hashlib
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .types import LLMResponse, ModelConfig, ProviderAuthError


def prompt_hash(prompt: str) -> str:
    """Stable content hash for a prompt string -- used for
    ``LLMResponse.prompt_hash`` and for versioning files under
    ``baselines/pamop/prompts/``."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_env_token(*var_names: str) -> str:
    """Return the first non-empty value among ``var_names``, or "".

    Never log or print the return value. Mirrors
    ``baselines/pamop/data.py::_get_hf_token``'s pattern for consistency.
    """
    for var in var_names:
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return ""


class LLMProvider(ABC):
    """One provider adapter. Subclasses implement ``_call`` only; ``generate``
    handles timing, retries, and building the common ``LLMResponse``."""

    name: str

    def __init__(self, max_retries: int = 2, retry_backoff_seconds: float = 1.0):
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    @abstractmethod
    def _call(self, prompt: str, config: ModelConfig) -> dict:
        """Perform one raw provider call.

        Must return a dict with at least ``{"text": str}`` and may include
        ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
        ``finish_reason``. Raise ``ProviderAuthError`` for missing
        credentials and ``ProviderCallError`` (or let the underlying SDK
        exception propagate, which ``generate`` wraps) for any other
        failure.
        """
        raise NotImplementedError

    def generate(self, prompt: str, config: ModelConfig) -> LLMResponse:
        from .types import ProviderCallError

        retry_count = 0
        last_exc: Exception | None = None
        start = time.monotonic()
        while retry_count <= self.max_retries:
            try:
                raw = self._call(prompt, config)
                latency = time.monotonic() - start
                return LLMResponse(
                    text=raw["text"],
                    provider=self.name,
                    model=config.model,
                    timestamp=utc_timestamp(),
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    prompt_tokens=raw.get("prompt_tokens"),
                    completion_tokens=raw.get("completion_tokens"),
                    total_tokens=raw.get("total_tokens"),
                    latency_seconds=latency,
                    retry_count=retry_count,
                    prompt_hash=prompt_hash(prompt),
                    finish_reason=raw.get("finish_reason"),
                )
            except ProviderAuthError:
                raise  # never worth retrying -- the key is missing/invalid
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any SDK's own exception type
                last_exc = exc
                retry_count += 1
                if retry_count > self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * retry_count)

        raise ProviderCallError(
            f"{self.name} call failed after {retry_count} attempt(s)"
        ) from last_exc
