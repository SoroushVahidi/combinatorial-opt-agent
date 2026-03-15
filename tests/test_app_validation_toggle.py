"""
Tests for answer() input validation and error-handling behavior.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_answer_accepts_validate_arg():
    """answer() can be called with three args (query, top_k, validate)."""
    import inspect
    from app import answer

    sig = inspect.signature(answer)
    params = list(sig.parameters)
    assert "query" in params and "top_k" in params and "validate" in params


def test_answer_empty_query_returns_placeholder():
    """answer() returns the empty-state placeholder for blank input."""
    from app import answer

    result = asyncio.run(answer("", 3))
    assert "coa-empty-state" in result
    assert "coa-empty-warn" not in result  # not a warning, just a hint


def test_answer_whitespace_only_returns_placeholder():
    """answer() treats whitespace-only input as empty."""
    from app import answer

    result = asyncio.run(answer("   \t\n", 3))
    assert "coa-empty-state" in result
    assert "coa-empty-warn" not in result


def test_answer_query_too_long_returns_warning():
    """answer() returns a warning and does NOT attempt the pipeline for oversized queries."""
    from app import answer, _MAX_QUERY_LEN

    long_query = "minimize cost " + ("x " * (_MAX_QUERY_LEN + 100))
    result = asyncio.run(answer(long_query, 3))
    assert "coa-empty-warn" in result
    assert "too long" in result.lower()
    # The character count should appear somewhere in the message (any numeric representation).
    assert any(c.isdigit() for c in result)


def test_answer_pipeline_error_returns_warning():
    """answer() catches unexpected exceptions and returns a user-friendly error card."""
    from unittest.mock import patch
    from app import answer

    def _boom(*args, **kwargs):
        raise RuntimeError("injected test failure")

    with patch("app.get_model", side_effect=_boom):
        result = asyncio.run(answer("minimize cost", 3))

    result_lower = result.lower()
    assert "coa-empty-warn" in result
    assert "unexpected error" in result_lower or "injected test failure" in result_lower


def test_max_query_len_is_positive_int():
    """_MAX_QUERY_LEN is a positive integer constant."""
    from app import _MAX_QUERY_LEN

    assert isinstance(_MAX_QUERY_LEN, int)
    assert _MAX_QUERY_LEN > 0
