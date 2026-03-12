"""
Regression tests for retrieval reranking and short-query expansion improvements.

Covers:
- Deterministic lexical reranker (alias overlap, slot overlap, role-cue overlap, domain triggers)
- Domain-specific short-query expansion
- Ambiguity detection
- Multi-view text helper
- Integration: search() with rerank=True
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Tiny catalog for deterministic unit tests (no model needed)
# ---------------------------------------------------------------------------

def _make_catalog() -> list[dict]:
    """Small catalog with distinct domain signals."""
    return [
        {
            "id": "blend_01",
            "name": "Blending Problem",
            "aliases": ["diet problem", "feed mix", "mixture"],
            "description": (
                "Determine the optimal mix of ingredients to meet nutritional requirements "
                "at minimum cost. Percentage of each ingredient is a decision variable."
            ),
            "formulation": {
                "variables": [
                    {"symbol": "x_i", "description": "fraction of ingredient i in the blend"},
                ],
                "constraints": [
                    {"expression": "sum x_i = 1", "description": "proportions sum to one"},
                    {"expression": "x_i >= 0", "description": "non-negative ingredient fractions"},
                ],
                "objective": {"sense": "minimize", "expression": "sum c_i x_i"},
            },
        },
        {
            "id": "transport_01",
            "name": "Transportation Problem",
            "aliases": ["shipping problem", "distribution"],
            "description": (
                "Ship goods from multiple sources to multiple destinations minimizing "
                "total transportation cost subject to supply and demand constraints."
            ),
            "formulation": {
                "variables": [
                    {"symbol": "x_ij", "description": "amount shipped from source i to destination j"},
                ],
                "constraints": [
                    {"expression": "sum_j x_ij <= s_i", "description": "supply at source"},
                    {"expression": "sum_i x_ij >= d_j", "description": "demand at destination"},
                ],
                "objective": {"sense": "minimize", "expression": "sum c_ij x_ij"},
            },
        },
        {
            "id": "knapsack_01",
            "name": "0-1 Knapsack Problem",
            "aliases": ["binary knapsack", "0/1 knapsack"],
            "description": (
                "Select items with weights and values to maximize total value "
                "without exceeding a weight capacity."
            ),
            "formulation": {
                "variables": [
                    {"symbol": "x_i", "description": "1 if item i is selected"},
                ],
                "constraints": [
                    {"expression": "sum w_i x_i <= W", "description": "weight capacity constraint"},
                ],
                "objective": {"sense": "maximize", "expression": "sum v_i x_i"},
            },
        },
        {
            "id": "assign_01",
            "name": "Assignment Problem",
            "aliases": ["worker task assignment", "bipartite matching"],
            "description": (
                "Assign workers to tasks one-to-one to minimize total assignment cost."
            ),
            "formulation": {
                "variables": [
                    {"symbol": "x_ij", "description": "1 if worker i is assigned to task j"},
                ],
                "constraints": [
                    {"expression": "sum_j x_ij = 1", "description": "each worker assigned once"},
                    {"expression": "sum_i x_ij = 1", "description": "each task assigned once"},
                ],
                "objective": {"sense": "minimize", "expression": "sum c_ij x_ij"},
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tests for _tokenize, _schema_tokens, _extract_slot_vocabulary
# ---------------------------------------------------------------------------

class TestTokenization:
    def test_tokenize_basic(self):
        from retrieval.reranking import _tokenize
        tokens = _tokenize("Blending Problem cost")
        assert "blending" in tokens
        assert "problem" in tokens
        assert "cost" in tokens

    def test_tokenize_strips_punctuation(self):
        from retrieval.reranking import _tokenize
        tokens = _tokenize("x_ij: amount shipped.")
        assert "amount" in tokens
        assert "shipped" in tokens

    def test_extract_slot_vocabulary_has_variable_descriptions(self):
        from retrieval.reranking import _extract_slot_vocabulary
        cat = _make_catalog()
        tokens = _extract_slot_vocabulary(cat[0])  # blend_01
        assert "fraction" in tokens
        assert "ingredient" in tokens

    def test_extract_slot_vocabulary_has_constraint_descriptions(self):
        from retrieval.reranking import _extract_slot_vocabulary
        cat = _make_catalog()
        tokens = _extract_slot_vocabulary(cat[1])  # transport_01
        assert "supply" in tokens or "demand" in tokens

    def test_schema_tokens_includes_all_views(self):
        from retrieval.reranking import _schema_tokens
        cat = _make_catalog()
        tokens = _schema_tokens(cat[0])  # blend_01
        # Name
        assert "blending" in tokens
        # Alias
        assert "diet" in tokens or "mixture" in tokens
        # Description
        assert "nutritional" in tokens or "cost" in tokens
        # Slot vocab
        assert "fraction" in tokens or "ingredient" in tokens


# ---------------------------------------------------------------------------
# Tests for _rerank_score
# ---------------------------------------------------------------------------

class TestRerankScore:
    def test_alias_overlap_highest_for_matching_schema(self):
        """Query using exact alias term should get high alias overlap for that schema."""
        from retrieval.reranking import _rerank_score, _tokenize
        cat = _make_catalog()
        q_tokens = _tokenize("diet problem nutrient cost")
        blend_feat = _rerank_score(q_tokens, cat[0])  # blend_01 has alias "diet problem"
        transport_feat = _rerank_score(q_tokens, cat[1])  # transport_01
        assert blend_feat.alias_overlap > transport_feat.alias_overlap

    def test_slot_overlap_for_shipping_query(self):
        """A query with 'shipped amount route' should get higher slot overlap for transport."""
        from retrieval.reranking import _rerank_score, _tokenize
        cat = _make_catalog()
        q_tokens = _tokenize("shipped amount source destination")
        transport_feat = _rerank_score(q_tokens, cat[1])  # transport_01
        blend_feat = _rerank_score(q_tokens, cat[0])  # blend_01
        # transport has 'shipped' and 'source/destination' in slot vocab
        assert transport_feat.slot_overlap > blend_feat.slot_overlap

    def test_role_cue_overlap_for_capacity(self):
        """Query with 'capacity' should score high on role cues for knapsack."""
        from retrieval.reranking import _rerank_score, _tokenize
        cat = _make_catalog()
        q_tokens = _tokenize("knapsack capacity weight")
        knap_feat = _rerank_score(q_tokens, cat[2])  # knapsack
        blend_feat = _rerank_score(q_tokens, cat[0])  # blend
        # 'capacity' is in _CAPACITY_CUES; knapsack description also mentions capacity
        assert knap_feat.role_cue_overlap >= blend_feat.role_cue_overlap

    def test_domain_trigger_for_blending(self):
        """Query with 'mix ingredient' triggers blending domain, high for blend schema."""
        from retrieval.reranking import _rerank_score, _tokenize
        cat = _make_catalog()
        q_tokens = _tokenize("mix ingredient nutrient")
        blend_feat = _rerank_score(q_tokens, cat[0])
        assign_feat = _rerank_score(q_tokens, cat[3])
        assert blend_feat.domain_trigger_overlap > assign_feat.domain_trigger_overlap

    def test_empty_query_returns_zero(self):
        from retrieval.reranking import _rerank_score, _tokenize
        cat = _make_catalog()
        feat = _rerank_score(_tokenize(""), cat[0])
        assert feat.total == 0.0


# ---------------------------------------------------------------------------
# Tests for rerank()
# ---------------------------------------------------------------------------

class TestRerank:
    def _build_candidates(self) -> list[tuple[dict, float]]:
        """Simulate first-stage candidates with equal scores to test reranking signal."""
        cat = _make_catalog()
        # Give all equal retrieval scores so reranking determines the order
        return [(p, 0.5) for p in cat]

    def test_blending_query_ranks_blend_first(self):
        from retrieval.reranking import rerank
        candidates = self._build_candidates()
        results = rerank("diet problem ingredient mix percentage", candidates)
        assert results[0][0]["id"] == "blend_01", (
            f"Expected blend_01 first, got {results[0][0]['id']}"
        )

    def test_transport_query_ranks_transport_first(self):
        from retrieval.reranking import rerank
        candidates = self._build_candidates()
        results = rerank("shipping supply demand route source destination", candidates)
        assert results[0][0]["id"] == "transport_01", (
            f"Expected transport_01 first, got {results[0][0]['id']}"
        )

    def test_assignment_query_ranks_assign_first(self):
        from retrieval.reranking import rerank
        candidates = self._build_candidates()
        results = rerank("assign worker task bipartite", candidates)
        assert results[0][0]["id"] == "assign_01", (
            f"Expected assign_01 first, got {results[0][0]['id']}"
        )

    def test_rerank_preserves_length(self):
        from retrieval.reranking import rerank
        candidates = self._build_candidates()
        results = rerank("some query", candidates)
        assert len(results) == len(candidates)

    def test_rerank_returns_sorted_descending(self):
        from retrieval.reranking import rerank
        candidates = self._build_candidates()
        results = rerank("diet mix ingredient", candidates)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates_returns_empty(self):
        from retrieval.reranking import rerank
        assert rerank("any query", []) == []


# ---------------------------------------------------------------------------
# Tests for detect_ambiguity
# ---------------------------------------------------------------------------

class TestAmbiguityDetection:
    def test_ambiguous_when_scores_close(self):
        from retrieval.reranking import detect_ambiguity
        cat = _make_catalog()
        results = [(cat[0], 0.82), (cat[1], 0.81)]
        amb = detect_ambiguity(results, ambiguity_threshold=0.05)
        assert amb is not None
        assert amb.is_ambiguous is True
        assert amb.margin == pytest.approx(0.01, abs=1e-6)

    def test_not_ambiguous_when_margin_large(self):
        from retrieval.reranking import detect_ambiguity
        cat = _make_catalog()
        results = [(cat[0], 0.90), (cat[1], 0.70)]
        amb = detect_ambiguity(results, ambiguity_threshold=0.05)
        assert amb is not None
        assert amb.is_ambiguous is False

    def test_single_result_returns_none(self):
        from retrieval.reranking import detect_ambiguity
        cat = _make_catalog()
        assert detect_ambiguity([(cat[0], 0.9)]) is None

    def test_ids_set_correctly(self):
        from retrieval.reranking import detect_ambiguity
        cat = _make_catalog()
        results = [(cat[2], 0.85), (cat[3], 0.84)]
        amb = detect_ambiguity(results)
        assert amb.top_id == "knapsack_01"
        assert amb.second_id == "assign_01"


# ---------------------------------------------------------------------------
# Tests for domain-specific short-query expansion
# ---------------------------------------------------------------------------

class TestDomainExpansion:
    def test_diet_query_gets_blending_context(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("diet protein fat")
        # Should contain domain-specific blending vocabulary, not just generic suffix
        assert "blend" in result.lower() or "mix" in result.lower() or "ingredient" in result.lower()
        assert "diet protein fat" in result

    def test_transport_query_gets_transport_context(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("ship supply demand")
        assert "transport" in result.lower() or "route" in result.lower() or "network" in result.lower()

    def test_assignment_query_gets_assignment_context(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("assign worker task")
        assert "assign" in result.lower() or "scheduling" in result.lower()

    def test_facility_query_gets_facility_context(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("facility location")
        assert "facility" in result.lower() or "location" in result.lower() or "depot" in result.lower()

    def test_production_query_gets_production_context(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("produce profit")
        assert "production" in result.lower() or "manufacture" in result.lower()

    def test_unknown_domain_falls_back_to_generic(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("knapsack weight")
        # 'weight' and 'knapsack' both trigger knapsack domain expansion
        assert "knapsack" in result.lower() or "optimization" in result.lower()

    def test_long_query_unchanged_even_if_domain_match(self):
        """Long queries (> 5 words) should NOT be expanded even if domain keywords present."""
        from retrieval.utils import expand_short_query
        long_q = "ship goods from multiple sources to multiple destinations"
        result = expand_short_query(long_q)
        assert result == long_q

    def test_domain_expansion_preserves_original_query(self):
        from retrieval.utils import expand_short_query
        q = "diet cost"
        result = expand_short_query(q)
        assert result.startswith(q)

    # Backward-compatible: the original docstring examples still hold
    def test_generic_expansion_still_works_for_plain_knapsack(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("knapsack")
        # knapsack triggers the knapsack domain expansion; must still start with 'knapsack'
        assert result.startswith("knapsack")
        assert len(result) > len("knapsack")

    def test_tsp_ilp_expansion(self):
        from retrieval.utils import expand_short_query
        # "TSP ILP" — TSP triggers traveling salesman domain
        result = expand_short_query("TSP ILP")
        assert result.startswith("TSP ILP")
        assert len(result) > len("TSP ILP")

    def test_empty_query_unchanged(self):
        from retrieval.utils import expand_short_query
        assert expand_short_query("") == ""

    def test_whitespace_only_unchanged(self):
        from retrieval.utils import expand_short_query
        assert expand_short_query("   ") == ""


# ---------------------------------------------------------------------------
# Tests for multi-view text helper
# ---------------------------------------------------------------------------

class TestMultiViewText:
    def test_multi_view_adds_slot_content(self):
        from retrieval.search import _searchable_text
        cat = _make_catalog()
        p = cat[0]  # blend_01
        basic = _searchable_text(p, multi_view=False)
        multi = _searchable_text(p, multi_view=True)
        # Multi-view should be longer (includes slot vocab)
        assert len(multi) >= len(basic)

    def test_multi_view_contains_variable_description_tokens(self):
        from retrieval.search import _searchable_text
        cat = _make_catalog()
        p = cat[0]  # blend_01 has "fraction of ingredient i"
        multi = _searchable_text(p, multi_view=True)
        assert "fraction" in multi or "ingredient" in multi

    def test_default_is_not_multi_view(self):
        """Default multi_view=False; existing callers unaffected."""
        from retrieval.search import _searchable_text
        cat = _make_catalog()
        p = cat[0]
        default = _searchable_text(p)
        explicit_false = _searchable_text(p, multi_view=False)
        assert default == explicit_false


# ---------------------------------------------------------------------------
# Integration: BM25Baseline still works after utils change
# ---------------------------------------------------------------------------

class TestBaselineIntegration:
    def test_bm25_still_works_with_updated_utils(self):
        from retrieval.baselines import BM25Baseline
        cat = _make_catalog()
        bl = BM25Baseline()
        bl.fit(cat)
        results = bl.rank("diet blend ingredient percentage", top_k=2)
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert "blend_01" in ids

    def test_tfidf_transport_query(self):
        from retrieval.baselines import TfidfBaseline
        cat = _make_catalog()
        bl = TfidfBaseline()
        bl.fit(cat)
        results = bl.rank("ship source destination supply demand", top_k=2)
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert "transport_01" in ids
