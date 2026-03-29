from __future__ import annotations

import pytest

from retrieval.baselines import (
    BGEBaseline,
    BM25Baseline,
    E5Baseline,
    LSABaseline,
    SBERTBaseline,
    TfidfBaseline,
)
from retrieval.search import search


def test_search_returns_empty_for_non_positive_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    def _should_not_load_catalog() -> list[dict]:
        raise AssertionError("search() should short-circuit before loading catalog")

    monkeypatch.setattr("retrieval.search._load_catalog", _should_not_load_catalog)
    assert search("knapsack", top_k=0) == []
    assert search("knapsack", top_k=-5) == []


def test_search_returns_empty_for_empty_catalog() -> None:
    class _FailIfUsedModel:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("model.encode should not be called for empty catalogs")

    assert search("knapsack", catalog=[], model=_FailIfUsedModel(), top_k=3) == []


@pytest.mark.parametrize(
    "baseline",
    [
        BM25Baseline(),
        TfidfBaseline(),
        LSABaseline(),
        SBERTBaseline(),
        E5Baseline(),
        BGEBaseline(),
    ],
)
def test_baselines_short_circuit_for_non_positive_top_k(baseline) -> None:
    assert baseline.rank("knapsack", top_k=0) == []
    assert baseline.rank("knapsack", top_k=-3) == []


@pytest.mark.parametrize(
    "baseline",
    [
        BM25Baseline(),
        TfidfBaseline(),
        LSABaseline(),
        SBERTBaseline(),
        E5Baseline(),
        BGEBaseline(),
    ],
)
def test_baselines_still_require_fit_for_positive_top_k(baseline) -> None:
    with pytest.raises(RuntimeError, match="Call fit\\(catalog\\) first"):
        baseline.rank("knapsack", top_k=1)
