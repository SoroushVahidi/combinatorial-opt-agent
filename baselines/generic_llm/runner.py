"""Runner for the GENERAL_PURPOSE_LLM_BASELINE (API-backed).

Reuses the verified PaMOP Azure OpenAI provider adapter; every call records
the exact deployment name AND the served underlying model snapshot, token
counts, latency, retry count, and finish reason. Never contains secrets.
"""
from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from typing import Any

from baselines.generic_llm.config import GenericLLMConfig
from baselines.generic_llm.prompt import PromptBundle
from baselines.pamop.llm.azure_openai_provider import AzureOpenAIProvider
from baselines.pamop.llm.types import ModelConfig


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    status: str
    provider: str
    deployment: str
    underlying_model: str | None
    prompt_sha256: str
    runtime_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    retry_count: int | None = None
    error_category: str | None = None
    error_message: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_model_config(config: GenericLLMConfig) -> ModelConfig:
    return ModelConfig(
        provider=config.provider,
        model=config.deployment,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
    )


def generate(prompt: PromptBundle, config: GenericLLMConfig) -> GenerationResult:
    prompt_hash = hashlib.sha256(prompt.user.encode()).hexdigest()
    provider = AzureOpenAIProvider()
    try:
        response = provider.generate(prompt.user, build_model_config(config))
        return GenerationResult(
            raw_output=response.text,
            status="COMPLETED",
            provider=response.provider,
            deployment=response.model,
            underlying_model=response.underlying_model,
            prompt_sha256=prompt_hash,
            runtime_seconds=response.latency_seconds,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            finish_reason=response.finish_reason,
            retry_count=response.retry_count,
            environment={"python": platform.python_version(), "platform": platform.platform()},
        )
    except Exception as exc:  # noqa: BLE001 -- surface any provider failure as a FAILED row, never fabricate.
        return GenerationResult(
            raw_output="",
            status="FAILED",
            provider=config.provider,
            deployment=config.deployment,
            underlying_model=None,
            prompt_sha256=prompt_hash,
            runtime_seconds=0.0,
            error_category="api_call_failed",
            error_message=str(exc)[:500],
            environment={"python": platform.python_version(), "platform": platform.platform()},
        )