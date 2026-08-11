"""Tests for the provider-agnostic LLM interface (baselines/pamop/llm/).

All network-free: uses a fake in-process provider subclass and monkeypatched
environment variables, never a real SDK call.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.llm.base import LLMProvider, get_env_token, prompt_hash
from baselines.pamop.llm.registry import UnknownProviderError, get_provider, list_providers
from baselines.pamop.llm.types import LLMResponse, ModelConfig, ProviderAuthError, ProviderCallError


class _EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, *, fail_times: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.fail_times = fail_times
        self.calls = 0

    def _call(self, prompt, config):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated transient failure")
        return {
            "text": f"echo:{prompt}",
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
        }


class _AlwaysAuthFailProvider(LLMProvider):
    name = "always_auth_fail"

    def _call(self, prompt, config):
        raise ProviderAuthError("no key configured")


def _config() -> ModelConfig:
    return ModelConfig(provider="echo", model="echo-1", temperature=0.2, max_tokens=100, top_p=1.0)


def test_generate_returns_llm_response_with_expected_shape():
    provider = _EchoProvider()
    response = provider.generate("hello", _config())
    assert isinstance(response, LLMResponse)
    assert response.text == "echo:hello"
    assert response.provider == "echo"
    assert response.model == "echo-1"
    assert response.temperature == 0.2
    assert response.retry_count == 0
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 5
    assert response.total_tokens == 8
    assert response.latency_seconds >= 0
    assert response.prompt_hash == prompt_hash("hello")
    # ISO-8601-ish sanity check without over-specifying the exact format.
    assert "T" in response.timestamp


def test_generate_retries_on_transient_failure_and_records_retry_count():
    provider = _EchoProvider(fail_times=2, max_retries=3, retry_backoff_seconds=0.0)
    response = provider.generate("hi", _config())
    assert response.retry_count == 2
    assert provider.calls == 3


def test_generate_gives_up_after_max_retries():
    provider = _EchoProvider(fail_times=99, max_retries=1, retry_backoff_seconds=0.0)
    with pytest.raises(ProviderCallError):
        provider.generate("hi", _config())
    assert provider.calls == 2  # initial attempt + 1 retry


def test_auth_error_is_never_retried():
    provider = _AlwaysAuthFailProvider(max_retries=5, retry_backoff_seconds=0.0)
    with pytest.raises(ProviderAuthError):
        provider.generate("hi", _config())


def test_prompt_hash_is_deterministic_and_content_sensitive():
    assert prompt_hash("abc") == prompt_hash("abc")
    assert prompt_hash("abc") != prompt_hash("abd")


def test_get_env_token_prefers_first_nonempty(monkeypatch):
    monkeypatch.delenv("FAKE_VAR_A", raising=False)
    monkeypatch.setenv("FAKE_VAR_B", "value-b")
    assert get_env_token("FAKE_VAR_A", "FAKE_VAR_B") == "value-b"


def test_get_env_token_treats_empty_string_as_unset(monkeypatch):
    """Regression test for a real finding on this workstation: GOOGLE_API_KEY
    is present as an environment variable name but set to "", which must be
    treated as absent, not as a valid (empty) credential."""
    monkeypatch.setenv("FAKE_EMPTY_KEY", "")
    assert get_env_token("FAKE_EMPTY_KEY") == ""


def test_registry_lists_all_five_providers():
    assert list_providers() == sorted(
        ["openai", "gemini", "cohere", "fireworks", "cloudrift"]
    )


def test_registry_rejects_unknown_provider():
    with pytest.raises(UnknownProviderError):
        get_provider("not-a-real-provider")


@pytest.mark.parametrize("name", ["openai", "gemini", "cohere", "fireworks", "cloudrift"])
def test_registry_instantiates_every_provider(name):
    provider = get_provider(name)
    assert provider.name == name


def test_openai_provider_raises_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from baselines.pamop.llm.openai_provider import OpenAIProvider

    with pytest.raises(ProviderAuthError):
        OpenAIProvider()._client()


def test_openai_provider_ignores_openai_base_url_env_var(monkeypatch):
    """Regression test for a real finding on this workstation:
    OPENAI_BASE_URL is set to CloudRift's endpoint. The OpenAI provider must
    always target the real OpenAI API regardless of that variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://inference.cloudrift.ai/v1")
    from baselines.pamop.llm.openai_provider import OpenAIProvider, REAL_OPENAI_BASE_URL

    client = OpenAIProvider()._client()
    assert str(client.base_url).rstrip("/") == REAL_OPENAI_BASE_URL.rstrip("/")


def test_gemini_provider_raises_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from baselines.pamop.llm.gemini_provider import GeminiProvider

    with pytest.raises(ProviderAuthError):
        GeminiProvider()._client()


def test_cohere_provider_raises_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    from baselines.pamop.llm.cohere_provider import CohereProvider

    with pytest.raises(ProviderAuthError):
        CohereProvider()._client()


def test_fireworks_provider_raises_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    from baselines.pamop.llm.fireworks_provider import FireworksProvider

    with pytest.raises(ProviderAuthError):
        FireworksProvider()._api_key()


def test_cloudrift_provider_raises_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("CLOUDRIFT_API_KEY", raising=False)
    from baselines.pamop.llm.cloudrift_provider import CloudRiftProvider

    with pytest.raises(ProviderAuthError):
        CloudRiftProvider()._api_key()


def test_cloudrift_provider_default_base_url_when_env_unset(monkeypatch):
    monkeypatch.delenv("CLOUDRIFT_BASE_URL", raising=False)
    from baselines.pamop.llm.cloudrift_provider import CloudRiftProvider, DEFAULT_CLOUDRIFT_BASE_URL

    assert CloudRiftProvider()._base_url() == DEFAULT_CLOUDRIFT_BASE_URL


def test_llm_response_never_has_a_field_that_looks_like_a_secret():
    """Guardrail: LLMResponse's field set must never grow an api_key/secret
    field -- reproducibility metadata only. (``*_tokens`` fields are LLM
    token *counts*, e.g. ``prompt_tokens`` -- not credentials, and are
    intentionally excluded from this check.)"""
    from dataclasses import fields

    names = {f.name for f in fields(LLMResponse)}
    assert not any("api_key" in n or "secret" in n or "password" in n for n in names)
    assert not any(n.endswith("_token") for n in names)  # a lone "*_token" (not "*_tokens") would be suspicious
