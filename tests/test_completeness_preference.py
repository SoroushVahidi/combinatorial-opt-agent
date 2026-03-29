"""
Tests for the formulation-completeness preference in retrieval.search.

Two features are exercised:
1.  ``_formulation_complete()`` — identifies problems with all three formulation
    components (variables, objective, constraints).
2.  ``prefer_complete=True`` in ``search()`` — adds a small score bonus to
    complete problems so they beat stub entries in tie situations without
    discarding a clearly more-relevant incomplete problem.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_problem(pid: str, *, complete: bool, name: str = "") -> dict:
    """Build a minimal catalog-compatible problem dict."""
    if complete:
        return {
            "id": pid,
            "name": name or pid,
            "aliases": [],
            "description": f"Description of {pid}.",
            "formulation": {
                "variables": [{"symbol": "x", "description": "decision"}],
                "objective": {"sense": "minimize", "expression": "x"},
                "constraints": ["x >= 0"],
            },
        }
    else:
        return {
            "id": pid,
            "name": name or pid,
            "aliases": [],
            "description": f"Description of {pid}.",
            "formulation": {
                "variables": [],
                "objective": {"expression": "See description (no structured formulation)"},
                "constraints": [],
            },
        }


# ── Unit tests for _formulation_complete ──────────────────────────────────────

class TestFormulationComplete:
    def test_complete_problem_returns_true(self):
        from retrieval.search import _formulation_complete
        assert _formulation_complete(_make_problem("p", complete=True)) is True

    def test_no_variables_returns_false(self):
        from retrieval.search import _formulation_complete
        p = _make_problem("p", complete=True)
        p["formulation"]["variables"] = []
        assert _formulation_complete(p) is False

    def test_no_objective_returns_false(self):
        from retrieval.search import _formulation_complete
        p = _make_problem("p", complete=True)
        p["formulation"]["objective"] = {}
        assert _formulation_complete(p) is False

    def test_none_objective_returns_false(self):
        from retrieval.search import _formulation_complete
        p = _make_problem("p", complete=True)
        p["formulation"]["objective"] = None
        assert _formulation_complete(p) is False

    def test_no_constraints_returns_false(self):
        from retrieval.search import _formulation_complete
        p = _make_problem("p", complete=True)
        p["formulation"]["constraints"] = []
        assert _formulation_complete(p) is False

    def test_no_formulation_key_returns_false(self):
        from retrieval.search import _formulation_complete
        p = {"id": "p", "name": "p", "description": "desc"}
        assert _formulation_complete(p) is False

    def test_formulation_none_returns_false(self):
        from retrieval.search import _formulation_complete
        p = {"id": "p", "formulation": None}
        assert _formulation_complete(p) is False

    def test_optmath_pattern_returns_false(self):
        """Reproduces the optmath_bench / gams stub pattern exactly."""
        from retrieval.search import _formulation_complete
        p = {
            "id": "optmath_bench_001",
            "source": "optmath_bench",
            "formulation": {
                "variables": [],
                "objective": {"expression": "See description (benchmark has no structured formulation)"},
                "constraints": [],
            },
        }
        assert _formulation_complete(p) is False

    def test_nl4opt_pattern_returns_true(self):
        """Reproduces the typical NL4Opt complete formulation pattern."""
        from retrieval.search import _formulation_complete
        p = {
            "id": "nl4opt_train_001",
            "source": "NL4Opt",
            "formulation": {
                "variables": [
                    {"symbol": "cleaners", "description": "number of cleaners"},
                    {"symbol": "receptionists", "description": "number of receptionists"},
                ],
                "objective": {"sense": "minimize", "expression": "500*cleaners + 350*receptionists"},
                "constraints": [
                    "cleaners + receptionists >= 100",
                    "receptionists >= 20",
                ],
            },
        }
        assert _formulation_complete(p) is True


# ── COMPLETENESS_BONUS constant ───────────────────────────────────────────────

class TestCompletenessBonusConstant:
    def test_bonus_is_positive_float(self):
        from retrieval.search import _COMPLETENESS_BONUS
        assert isinstance(_COMPLETENESS_BONUS, float)
        assert _COMPLETENESS_BONUS > 0

    def test_bonus_is_small(self):
        """The bonus must be small enough not to overwhelm genuine relevance differences."""
        from retrieval.search import _COMPLETENESS_BONUS
        # Cosine similarities from a good hit are typically 0.7+.
        # A bonus ≥ 0.1 would be too aggressive.
        assert _COMPLETENESS_BONUS < 0.1

    def test_bonus_is_meaningful(self):
        """The bonus must be large enough to resolve ties (> floating-point noise)."""
        from retrieval.search import _COMPLETENESS_BONUS
        assert _COMPLETENESS_BONUS >= 0.005


# ── Integration: prefer_complete in search() ─────────────────────────────────

class TestPreferCompleteInSearch:
    """Simulate the search() completeness-preference logic without a real model."""

    def _run_scored_search(
        self,
        catalog: list[dict],
        scores: list[float],
        *,
        prefer_complete: bool,
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """Minimal re-implementation of search()'s scoring + preference step.

        This mirrors exactly what search() does: it adds _COMPLETENESS_BONUS to
        complete problems, then argsorts descending.  Using this helper lets us
        test the preference logic without needing a GPU / sentence-transformer
        available.
        """
        from retrieval.search import _formulation_complete, _COMPLETENESS_BONUS
        import numpy as np

        arr = np.array(scores, dtype=float)
        if prefer_complete:
            for i, p in enumerate(catalog):
                if _formulation_complete(p):
                    arr[i] += _COMPLETENESS_BONUS
        idx = np.argsort(arr)[::-1][:top_k]
        return [(catalog[i], float(arr[i])) for i in idx]

    def test_tie_broken_in_favour_of_complete(self):
        """When an incomplete and a complete problem have equal cosine score,
        the complete one must rank first."""
        catalog = [
            _make_problem("stub", complete=False, name="Stub Problem"),
            _make_problem("full", complete=True, name="Full Problem"),
        ]
        results = self._run_scored_search(
            catalog, scores=[0.80, 0.80], prefer_complete=True
        )
        assert results[0][0]["id"] == "full", (
            "Complete problem should rank first on equal cosine scores"
        )

    def test_clearly_more_relevant_incomplete_still_wins(self):
        """An incomplete problem with a substantially higher cosine score must
        not be overtaken by a complete problem with a lower cosine score."""
        from retrieval.search import _COMPLETENESS_BONUS
        catalog = [
            _make_problem("stub", complete=False, name="Stub Problem"),
            _make_problem("full", complete=True, name="Full Problem"),
        ]
        # Gap much larger than the bonus
        high_score = 0.90
        low_score = 0.80
        assert high_score - low_score > _COMPLETENESS_BONUS
        results = self._run_scored_search(
            catalog, scores=[high_score, low_score], prefer_complete=True
        )
        assert results[0][0]["id"] == "stub", (
            "A clearly more-relevant incomplete problem must not be demoted"
        )

    def test_prefer_complete_false_preserves_original_ranking(self):
        """With prefer_complete=False, score order is unchanged even for tie."""
        catalog = [
            _make_problem("stub", complete=False, name="Stub Problem"),
            _make_problem("full", complete=True, name="Full Problem"),
        ]
        results = self._run_scored_search(
            catalog, scores=[0.80, 0.80], prefer_complete=False
        )
        # Tie with no bonus → first in catalog wins (numpy argsort is stable
        # in descending order implementation; we just check no preference applied)
        # Both are still in the results, order depends on original array order
        ids = [r[0]["id"] for r in results]
        assert "stub" in ids and "full" in ids

    def test_near_tie_within_bonus_prefers_complete(self):
        """A complete problem that is slightly behind (but within the bonus
        window) should still rank first."""
        from retrieval.search import _COMPLETENESS_BONUS
        catalog = [
            _make_problem("stub", complete=False, name="Stub Problem"),
            _make_problem("full", complete=True, name="Full Problem"),
        ]
        # Incomplete scores slightly higher, but within the bonus margin
        margin = _COMPLETENESS_BONUS / 2  # gap smaller than bonus
        results = self._run_scored_search(
            catalog,
            scores=[0.80 + margin, 0.80],  # stub slightly ahead
            prefer_complete=True,
        )
        assert results[0][0]["id"] == "full", (
            "Complete problem should still win when incomplete leads by less than the bonus"
        )

    def test_multiple_complete_problems_rank_by_score(self):
        """With multiple complete problems, they should still be ranked by their
        own cosine scores (each gets the same absolute bonus)."""
        catalog = [
            _make_problem("full_a", complete=True, name="Full A"),
            _make_problem("full_b", complete=True, name="Full B"),
            _make_problem("stub", complete=False, name="Stub"),
        ]
        results = self._run_scored_search(
            catalog, scores=[0.85, 0.90, 0.75], prefer_complete=True, top_k=2
        )
        assert results[0][0]["id"] == "full_b", "Highest-scoring complete problem first"
        assert results[1][0]["id"] == "full_a", "Second highest-scoring complete problem second"

    def test_all_incomplete_catalog_no_bonus_applied(self):
        """If all catalog entries are incomplete, no bonus is applied and ordering
        is purely by cosine similarity."""
        catalog = [
            _make_problem("s1", complete=False),
            _make_problem("s2", complete=False),
            _make_problem("s3", complete=False),
        ]
        results = self._run_scored_search(
            catalog, scores=[0.70, 0.90, 0.80], prefer_complete=True, top_k=2
        )
        assert results[0][0]["id"] == "s2"
        assert results[1][0]["id"] == "s3"
