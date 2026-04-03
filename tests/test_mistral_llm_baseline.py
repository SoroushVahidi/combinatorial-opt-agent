"""Mistral LLM baseline wiring (no live API calls)."""

from __future__ import annotations

from tools.llm_baselines import (
    LLMTwoStageBaseline,
    classify_mistral_api_failure,
    resolve_mistral_model_from_config,
)


def test_resolve_mistral_model_from_config() -> None:
    m = resolve_mistral_model_from_config()
    assert m and isinstance(m, str)


def test_mistral_baseline_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    cat = [{"id": "nlp4lp_test_x", "description": "test"}]
    try:
        LLMTwoStageBaseline("mistral", catalog=cat, expected_type_fn=lambda _p: "float")
    except RuntimeError as e:
        assert "MISTRAL_API_KEY" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError when MISTRAL_API_KEY is missing")


def test_classify_mistral_api_failure_unknown() -> None:
    assert classify_mistral_api_failure(RuntimeError("oops")) == "other"
