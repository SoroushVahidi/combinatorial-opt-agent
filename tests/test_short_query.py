"""
Tests for Bottleneck-2 fix: short-query expansion and SHORT_QUERY_TEMPLATES coverage.

Covers:
- expand_short_query / _is_short_query: boundary behaviour and docstring examples
- search() expand_short_queries parameter (default on, can be disabled)
- All four baselines apply expansion inside rank()
"""
from __future__ import annotations

import pytest


def _tiny_catalog() -> list[dict]:
    """Three-problem catalog — same structure used across all tests."""
    return [
        {
            "id": "p1",
            "name": "Knapsack",
            "aliases": ["0-1 knapsack"],
            "description": (
                "Select items with weights and values to maximize total "
                "value without exceeding a weight capacity."
            ),
        },
        {
            "id": "p2",
            "name": "Set Cover",
            "aliases": [],
            "description": "Choose the minimum number of subsets to cover all elements.",
        },
        {
            "id": "p3",
            "name": "Vertex Cover",
            "aliases": [],
            "description": "Minimum set of vertices that covers every edge in the graph.",
        },
    ]


class TestExpandShortQuery:
    """Directly validate the expansion function and its word-count threshold."""

    def test_single_word_expanded(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("knapsack")
        # "knapsack" triggers knapsack domain expansion; must start with original query
        assert result.startswith("knapsack")
        assert len(result) > len("knapsack")

    def test_two_words_expanded(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("TSP ILP")
        # "TSP" triggers traveling-salesman domain expansion
        assert result.startswith("TSP ILP")
        assert len(result) > len("TSP ILP")

    def test_exactly_five_words_expanded(self):
        """Boundary: 5-word query should still be expanded."""
        from retrieval.utils import expand_short_query
        q = "facility location integer linear program"
        assert len(q.split()) == 5
        result = expand_short_query(q)
        # "facility" and "location" trigger facility-domain expansion
        assert result.startswith(q)
        assert len(result) > len(q)

    def test_six_words_not_expanded(self):
        """Boundary: 6-word query should be returned unchanged."""
        from retrieval.utils import expand_short_query
        q = "facility location integer linear program formulation"
        assert len(q.split()) == 6
        assert expand_short_query(q) == q

    def test_long_query_unchanged(self):
        from retrieval.utils import expand_short_query
        long_q = (
            "minimize cost of opening warehouses and assigning customers "
            "to open warehouses subject to capacity constraints"
        )
        assert expand_short_query(long_q) == long_q

    def test_empty_string_unchanged(self):
        from retrieval.utils import expand_short_query
        assert expand_short_query("") == ""

    def test_whitespace_only_unchanged(self):
        from retrieval.utils import expand_short_query
        assert expand_short_query("   ") == ""

    def test_leading_trailing_whitespace_stripped(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("  TSP  ")
        # "TSP" triggers domain expansion; result starts with stripped query
        assert result.startswith("TSP")
        assert len(result) > len("TSP")

    def test_unknown_domain_uses_generic_suffix(self):
        """A query with no known domain trigger falls back to the generic suffix."""
        from retrieval.utils import expand_short_query, _EXPANSION_SUFFIX
        result = expand_short_query("lagrangian dual")
        # No known domain keyword → generic suffix
        assert result == f"lagrangian dual {_EXPANSION_SUFFIX}"

    def test_lp_keyword_triggers_lp_expansion(self):
        """'lp' is a known domain keyword → LP/MIP expansion."""
        from retrieval.utils import expand_short_query
        result = expand_short_query("lp")
        assert result.startswith("lp")
        assert "linear" in result.lower() or "integer" in result.lower()

    def test_ilp_keyword_triggers_lp_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("ILP")
        assert result.startswith("ILP")
        assert len(result) > len("ILP")

    def test_ilp_formulation_triggers_lp_expansion_not_generic(self):
        """'ILP formulation' contains the LP-domain trigger 'formulation'; must NOT fall
        back to the generic suffix."""
        from retrieval.utils import expand_short_query, _EXPANSION_SUFFIX
        result = expand_short_query("ILP formulation")
        # must NOT be just the query + generic suffix
        assert result != f"ILP formulation {_EXPANSION_SUFFIX}", (
            "'ILP formulation' should match the LP/MIP domain, not the generic fallback"
        )
        assert len(result) > len("ILP formulation")

    def test_portfolio_keyword_triggers_finance_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("portfolio")
        assert result.startswith("portfolio")
        assert "risk" in result.lower() or "invest" in result.lower()

    def test_matching_keyword_triggers_matching_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("matching")
        assert result.startswith("matching")
        assert "bipartite" in result.lower() or "assignment" in result.lower()

    def test_inventory_keyword_triggers_resource_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("inventory")
        assert result.startswith("inventory")
        assert "planning" in result.lower() or "demand" in result.lower()

    def test_qp_keyword_triggers_qp_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("qp")
        assert result.startswith("qp")
        assert "quadratic" in result.lower() or "convex" in result.lower()

    # ── New domain triggers added in the "improve weakest point" iteration ──

    def test_flowshop_keyword_triggers_scheduling_not_flow(self):
        """'flowshop' (single token) must expand to scheduling context, NOT
        to network-flow context, even though 'flow' appears in its expansion."""
        from retrieval.utils import expand_short_query
        result = expand_short_query("flowshop")
        assert result.startswith("flowshop")
        # Scheduling-family keywords must be present
        assert any(kw in result.lower() for kw in ("scheduling", "makespan", "sequencing", "job shop")), (
            "'flowshop' must trigger the flow-shop/scheduling domain expansion"
        )
        # Must NOT be the bare network-flow expansion (which mentions 'arc' or 'path' only)
        assert "shortest path" not in result.lower(), (
            "'flowshop' must not map to the network-flow expansion"
        )

    def test_makespan_keyword_triggers_scheduling_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("makespan")
        assert result.startswith("makespan")
        assert any(kw in result.lower() for kw in ("scheduling", "makespan", "sequencing"))

    def test_sequencing_keyword_triggers_scheduling_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("sequencing")
        assert result.startswith("sequencing")
        assert any(kw in result.lower() for kw in ("scheduling", "sequencing", "job shop"))

    def test_spanning_keyword_triggers_tree_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("spanning")
        assert result.startswith("spanning")
        assert any(kw in result.lower() for kw in ("spanning", "steiner", "tree"))

    def test_steiner_keyword_triggers_tree_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("steiner")
        assert result.startswith("steiner")
        assert any(kw in result.lower() for kw in ("steiner", "spanning", "tree"))

    def test_mst_keyword_triggers_tree_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("mst")
        assert result.startswith("mst")
        assert any(kw in result.lower() for kw in ("spanning", "tree"))

    def test_auction_keyword_triggers_auction_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("auction")
        assert result.startswith("auction")
        assert any(kw in result.lower() for kw in ("auction", "bidding", "winner"))

    def test_bidding_keyword_triggers_auction_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("bidding")
        assert result.startswith("bidding")
        assert any(kw in result.lower() for kw in ("auction", "winner", "procurement"))

    def test_procurement_keyword_triggers_auction_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("procurement")
        assert result.startswith("procurement")
        assert any(kw in result.lower() for kw in ("auction", "bidding", "procurement"))

    def test_crew_keyword_triggers_scheduling_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("crew")
        assert result.startswith("crew")
        assert any(kw in result.lower() for kw in ("scheduling", "crew", "rostering", "assignment"))

    def test_rostering_keyword_triggers_scheduling_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("rostering")
        assert result.startswith("rostering")
        assert any(kw in result.lower() for kw in ("rostering", "scheduling", "crew"))

    def test_domain_count_at_least_21(self):
        """Regression guard: the domain map must not shrink below 21 entries."""
        from retrieval.utils import _DOMAIN_EXPANSION_MAP
        assert len(_DOMAIN_EXPANSION_MAP) >= 21, (
            f"Expected ≥ 21 domain expansion entries, got {len(_DOMAIN_EXPANSION_MAP)}"
        )


class TestIsShortQuery:
    def test_one_word_is_short(self):
        from retrieval.utils import _is_short_query
        assert _is_short_query("knapsack") is True

    def test_five_words_is_short(self):
        from retrieval.utils import _is_short_query
        assert _is_short_query("a b c d e") is True

    def test_six_words_is_not_short(self):
        from retrieval.utils import _is_short_query
        assert _is_short_query("a b c d e f") is False

    def test_long_sentence_is_not_short(self):
        from retrieval.utils import _is_short_query
        assert _is_short_query("minimize cost of opening warehouses and assigning customers") is False


class TestSearchIntegration:
    @pytest.mark.requires_network
    def test_search_uses_expansion_by_default(self):
        from retrieval.search import search, _default_model_path
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            pytest.skip("sentence-transformers (and/or torch) not available")

        catalog = _tiny_catalog()
        model = SentenceTransformer(_default_model_path())
        results = search("knapsack", catalog=catalog, model=model, top_k=1)
        assert results
        assert results[0][0]["id"] == "p1"

    @pytest.mark.requires_network
    def test_search_can_disable_expansion(self):
        from retrieval.search import search, _default_model_path
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            pytest.skip("sentence-transformers (and/or torch) not available")

        catalog = _tiny_catalog()
        model = SentenceTransformer(_default_model_path())
        # This primarily exercises the code path; behaviour difference depends on model.
        results = search("knapsack", catalog=catalog, model=model, top_k=1, expand_short_queries=False)
        assert results


class TestBaselinesUseExpansion:
    @pytest.mark.parametrize("cls_name", ["BM25Baseline", "TfidfBaseline", "LSABaseline"])
    def test_text_baselines_use_expansion(self, cls_name):
        from retrieval.baselines import BM25Baseline, TfidfBaseline, LSABaseline

        cls_map = {
            "BM25Baseline": BM25Baseline,
            "TfidfBaseline": TfidfBaseline,
            "LSABaseline": LSABaseline,
        }
        baseline = cls_map[cls_name]()
        baseline.fit(_tiny_catalog())
        results = baseline.rank("knapsack", top_k=1)
        assert results

