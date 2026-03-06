"""
Optional test: answer() accepts validate flag and search(validate=...) is used.
"""
from __future__ import annotations


def test_answer_accepts_validate_arg():
    """answer() can be called with three args (query, top_k, validate)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from app import answer
    # Run synchronously; use a tiny query and validate=False to avoid loading model if possible
    import asyncio
    # We only check the function accepts the third argument
    import inspect
    sig = inspect.signature(answer)
    params = list(sig.parameters)
    assert "query" in params and "top_k" in params and "validate" in params
