"""Unit tests for Gemini hard-quota detection (no live API calls)."""

from __future__ import annotations

import pytest


def test_hard_zero_quota_google_style():
    from tools.llm_baselines import GeminiHardQuotaError, is_gemini_hard_zero_quota_error

    msg = (
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 0"
    )
    exc = RuntimeError(msg)
    assert is_gemini_hard_zero_quota_error(exc)
    assert is_gemini_hard_zero_quota_error(GeminiHardQuotaError(msg))


def test_transient_429_without_limit_zero_not_hard():
    from tools.llm_baselines import is_gemini_hard_zero_quota_error

    exc = RuntimeError("Resource exhausted; Please retry in 45s. quota exceeded for requests per minute")
    assert not is_gemini_hard_zero_quota_error(exc)


def test_limit_zero_without_quota_context_not_flagged():
    from tools.llm_baselines import is_gemini_hard_zero_quota_error

    exc = RuntimeError("some other limit: 0 edge case")
    assert not is_gemini_hard_zero_quota_error(exc)


def test_resource_exhausted_with_limit_zero():
    from tools.llm_baselines import is_gemini_hard_zero_quota_error

    pytest.importorskip("google.api_core.exceptions")
    from google.api_core import exceptions as gexc

    exc = gexc.ResourceExhausted("limit: 0")
    assert is_gemini_hard_zero_quota_error(exc)


def test_write_gemini_selection_artifact_failure_schema(tmp_path):
    from tools.llm_baselines import write_gemini_selection_artifact
    import json

    p = tmp_path / "gemini_selected_model.json"
    write_gemini_selection_artifact(
        p,
        ok=False,
        model=None,
        candidates_ordered=["a", "b"],
        failures=[("a", "limit: 0"), ("b", "404")],
        source="pick-model",
    )
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert data["model"] is None
    assert data["reason"] == "all_candidates_failed"
    assert "summary" in data
    assert len(data["failures"]) == 2


def test_default_gemini_model_constant():
    from tools.llm_baselines import DEFAULT_GEMINI_MODEL

    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash-lite"


def test_resolve_gemini_model_from_config_uses_default(tmp_path, monkeypatch):
    from tools.llm_baselines import DEFAULT_GEMINI_MODEL, resolve_gemini_model_from_config

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = tmp_path / "llm_baselines.yaml"
    cfg.write_text("gemini:\n  model: gemini-2.5-flash-lite\n", encoding="utf-8")
    assert resolve_gemini_model_from_config(cfg) == "gemini-2.5-flash-lite"
    empty = tmp_path / "empty.yaml"
    empty.write_text("gemini: {}\n", encoding="utf-8")
    assert resolve_gemini_model_from_config(empty) == DEFAULT_GEMINI_MODEL


def test_gemini_fallback_candidates_order(tmp_path, monkeypatch):
    from tools.llm_baselines import gemini_fallback_candidates

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_FALLBACKS", raising=False)
    cfg = tmp_path / "llm_baselines.yaml"
    cfg.write_text(
        "gemini:\n"
        "  model: gemini-2.5-flash-lite\n"
        "  fallback_models:\n"
        "    - gemini-2.0-flash-lite\n"
        "    - gemini-2.0-flash\n",
        encoding="utf-8",
    )
    c = gemini_fallback_candidates(cfg)
    assert c == ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"]


def test_apply_optional_gemini_thread_env_sets_omp(monkeypatch):
    import os

    from tools import llm_baselines as lb

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_MAX_THREADS", raising=False)
    monkeypatch.setenv("GEMINI_LIMIT_RUNTIME_THREADS", "1")
    lb._apply_optional_gemini_thread_env()
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("NUMEXPR_MAX_THREADS") == "1"


def test_apply_optional_gemini_thread_env_off(monkeypatch):
    import os

    from tools import llm_baselines as lb

    monkeypatch.setenv("GEMINI_LIMIT_RUNTIME_THREADS", "0")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    lb._apply_optional_gemini_thread_env()
    assert os.environ.get("OMP_NUM_THREADS") is None
