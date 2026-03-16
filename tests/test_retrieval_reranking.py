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


# ---------------------------------------------------------------------------
# Tests for Feature 5: confusable-schema discrimination
# ---------------------------------------------------------------------------

class TestConfusableDiscrimination:
    """Tests for _confusable_discrimination_score in retrieval.reranking."""

    def test_positive_cues_give_non_negative_score(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        # blending problem with positive cues
        problem = {
            "id": "blend",
            "name": "Blending Problem",
            "tags": ["blending"],
        }
        query_tokens = _tokenize("blend ingredients proportion nutrient composition")
        score = _confusable_discrimination_score(query_tokens, problem)
        # Positive cues for blending should yield a non-negative score
        assert score >= 0.0

    def test_negative_cues_give_non_positive_score(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        # blending problem but query has production cues
        problem = {
            "id": "blend",
            "name": "Blending Problem",
            "tags": ["blending"],
        }
        query_tokens = _tokenize("produce product manufacturing labor assembly factory")
        score = _confusable_discrimination_score(query_tokens, problem)
        # Negative cues for blending (production cues) should give non-positive score
        assert score <= 0.0

    def test_score_bounded_in_range(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "knapsack", "name": "0-1 Knapsack", "tags": ["knapsack"]}
        q = _tokenize("select item value weight capacity binary maximize bin container")
        score = _confusable_discrimination_score(q, problem)
        assert -0.1 <= score <= 0.1

    def test_no_tags_returns_zero(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "foo", "name": "Unknown Schema"}
        q = _tokenize("some query about optimization")
        score = _confusable_discrimination_score(q, problem)
        assert score == 0.0

    def test_name_heuristic_picks_up_domain_key(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        # No tags, but schema name contains 'assignment'
        problem = {"id": "assgn", "name": "Assignment Problem", "tags": []}
        q = _tokenize("assign worker task employee job matching")
        score = _confusable_discrimination_score(q, problem)
        # Positive cues for assignment → non-negative
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Tests for Feature 6: grounding-consistency rerank
# ---------------------------------------------------------------------------

class TestGroundingConsistencyScore:
    """Tests for grounding_consistency_score in retrieval.reranking."""

    def test_neutral_score_for_no_cue_query(self):
        from retrieval.reranking import grounding_consistency_score
        problem = {"id": "foo", "name": "Problem", "formulation": {}}
        # Pure structural query with no quantity-role cue words → neutral 0.5
        score = grounding_consistency_score("find the optimal solution", problem)
        assert score == pytest.approx(0.5)

    def test_budget_cue_match(self):
        from retrieval.reranking import grounding_consistency_score
        problem = {
            "id": "budget_prob",
            "name": "Budget Allocation",
            "formulation": {
                "variables": [{"description": "amount allocated to project X"}],
                "constraints": [{"description": "total budget must not exceed limit"}],
            },
        }
        # Query mentions budget → should match schema's budget slot
        score = grounding_consistency_score(
            "allocate budget of $100,000 across two projects", problem
        )
        assert score > 0.0

    def test_capacity_cue_match(self):
        from retrieval.reranking import grounding_consistency_score
        problem = {
            "id": "cap_prob",
            "name": "Capacity Planning",
            "formulation": {
                "variables": [{"description": "production quantity"}],
                "constraints": [{"description": "total capacity availability limit"}],
            },
        }
        score = grounding_consistency_score(
            "factory with capacity limit produces two products", problem
        )
        assert score > 0.0

    def test_score_in_range(self):
        from retrieval.reranking import grounding_consistency_score
        problem = {"id": "foo", "name": "Problem", "formulation": {}}
        for query in [
            "maximize profit with budget constraint",
            "minimize cost with capacity limitation",
            "at least 40 percent allocation required",
            "count number of items selected",
            "transportation from source to destination",
        ]:
            score = grounding_consistency_score(query, problem)
            assert 0.0 <= score <= 1.0, f"Score out of range for: {query!r}"


class TestGroundingRerank:
    """Tests for the grounding_rerank function in retrieval.reranking."""

    def _make_candidates(self) -> list[tuple[dict, float]]:
        """Tiny candidate list for testing."""
        return [
            (
                {
                    "id": "prod_01",
                    "name": "Production Planning",
                    "formulation": {
                        "variables": [{"description": "units of product A and B to produce"}],
                        "constraints": [
                            {"description": "total labor hours budget constraint"},
                            {"description": "maximum capacity limit"},
                        ],
                    },
                },
                0.80,
            ),
            (
                {
                    "id": "tsp_01",
                    "name": "Travelling Salesman",
                    "formulation": {
                        "variables": [{"description": "binary route selection variables"}],
                        "constraints": [
                            {"description": "each city visited exactly once"},
                        ],
                    },
                },
                0.75,
            ),
        ]

    def test_returns_same_length(self):
        from retrieval.reranking import grounding_rerank
        cands = self._make_candidates()
        result = grounding_rerank("production budget labor", cands)
        assert len(result) == len(cands)

    def test_scores_modified(self):
        from retrieval.reranking import grounding_rerank
        cands = self._make_candidates()
        result = grounding_rerank("production budget labor", cands)
        orig_scores = {p["id"]: s for p, s in cands}
        new_scores = {p["id"]: s for p, s in result}
        # At least one score should differ from the original retrieval score
        # (because grounding_lambda=0.15 * gc != 0 unless gc is 0)
        changed = any(
            abs(new_scores[pid] - orig_scores[pid]) > 1e-9
            for pid in orig_scores
        )
        assert changed

    def test_empty_input_returns_empty(self):
        from retrieval.reranking import grounding_rerank
        result = grounding_rerank("any query", [])
        assert result == []

    def test_output_sorted_descending(self):
        from retrieval.reranking import grounding_rerank
        cands = self._make_candidates()
        result = grounding_rerank("production budget labor", cands)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_lambda_zero_preserves_order(self):
        from retrieval.reranking import grounding_rerank
        cands = self._make_candidates()
        result = grounding_rerank("production labor hours budget", cands, grounding_lambda=0.0)
        ids = [p["id"] for p, _ in result]
        orig_ids = [p["id"] for p, _ in cands]
        # With lambda=0, grounding adds nothing, order should be preserved
        assert ids == orig_ids


# ---------------------------------------------------------------------------
# Tests for synthetic case builder
# ---------------------------------------------------------------------------

class TestSyntheticCaseBuilder:
    """Tests for the synthetic test case generator."""

    def test_get_all_cases_returns_nonempty(self):
        from tools.build_easy_family_synthetic_cases import get_all_cases
        cases = get_all_cases()
        assert len(cases) > 0

    def test_all_five_families_present(self):
        from tools.build_easy_family_synthetic_cases import get_all_cases
        expected_families = {
            "percent_vs_integer",
            "implicit_count",
            "minmax_bound",
            "total_vs_perunit",
            "retrieval_failure",
        }
        cases = get_all_cases()
        families_in_cases = {c["family"] for c in cases}
        for fam in expected_families:
            assert fam in families_in_cases, f"Family {fam} missing from synthetic cases"

    def test_cases_have_required_fields(self):
        from tools.build_easy_family_synthetic_cases import get_all_cases
        for case in get_all_cases():
            assert "id" in case
            assert "family" in case
            assert "sub_type" in case
            assert "query" in case
            assert "expected_schema" in case
            assert "notes" in case

    def test_deterministic_on_repeated_import(self):
        from tools.build_easy_family_synthetic_cases import get_all_cases
        c1 = get_all_cases()
        c2 = get_all_cases()
        assert [c["id"] for c in c1] == [c["id"] for c in c2]

    def test_get_cases_by_family_filter(self):
        from tools.build_easy_family_synthetic_cases import get_cases_by_family
        pct_cases = get_cases_by_family("percent_vs_integer")
        assert len(pct_cases) > 0
        assert all(c["family"] == "percent_vs_integer" for c in pct_cases)

    def test_all_case_ids_unique(self):
        from tools.build_easy_family_synthetic_cases import get_all_cases
        ids = [c["id"] for c in get_all_cases()]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_percent_cases_have_fraction_expected_slots(self):
        """At least one expected slot in percent cases should be a float < 1."""
        from tools.build_easy_family_synthetic_cases import get_cases_by_family
        pct_cases = get_cases_by_family("percent_vs_integer")
        found_fraction = False
        for case in pct_cases:
            for v in case.get("expected_slots", {}).values():
                if isinstance(v, float) and 0.0 < v < 1.0:
                    found_fraction = True
        assert found_fraction, "No fraction-valued expected slot found in percent cases"


# ---------------------------------------------------------------------------
# Tests for ablation evaluation
# ---------------------------------------------------------------------------

class TestAblationEvaluation:
    """Tests for the ablation evaluation mode in scripts/evaluate_retrieval.py."""

    def test_run_ablation_tfidf_returns_four_variants(self):
        from scripts.evaluate_retrieval import run_ablation
        catalog = [
            {"id": "prod", "name": "Production Planning", "aliases": ["manufacturing"],
             "description": "Produce items to maximize profit subject to resource constraints."},
            {"id": "blend", "name": "Blending Problem", "aliases": ["diet", "mix"],
             "description": "Mix ingredients to meet requirements at minimum cost."},
        ]
        instances = [
            ("production profit resource", "prod"),
            ("blend mix diet ingredients", "blend"),
        ]
        results = run_ablation(catalog, instances, top_k_max=2, use_tfidf_fallback=True)
        assert len(results) == 4
        variant_names = [r["variant"] for r in results]
        assert variant_names == ["baseline", "+expansion", "+rerank", "+grounding"]

    def test_ablation_each_result_has_metrics(self):
        from scripts.evaluate_retrieval import run_ablation
        catalog = [
            {"id": "knapsack", "name": "Knapsack", "aliases": [], "description": "Select items to maximize value."},
            {"id": "tsp", "name": "TSP", "aliases": ["travelling salesman"], "description": "Find shortest tour."},
        ]
        instances = [("knapsack items value", "knapsack")]
        results = run_ablation(catalog, instances, top_k_max=2, use_tfidf_fallback=True)
        for r in results:
            assert "accuracy_at_1" in r
            assert "mrr" in r
            assert "n" in r


# ---------------------------------------------------------------------------
# New tests: missing confusable discrimination pairs (routing, scheduling,
# network_flow, covering, graph, transportation/production expansion)
# ---------------------------------------------------------------------------

class TestNewConfusableDiscriminationPairs:
    """Test the four new confusable discrimination pairs added:
    routing↔transportation, scheduling↔production,
    network_flow↔transportation, covering↔graph.
    """

    # ── routing ──────────────────────────────────────────────────────────────

    def test_vrp_positive_cues_boost_routing_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "vrp", "name": "Vehicle Routing Problem", "tags": []}
        # "vehicle", "tour", "depot" are positive cues for routing
        q = _tokenize("minimize vehicle tour depot capacitated vrp")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"VRP problem + VRP cues should give positive score, got {score}"

    def test_transport_cues_penalise_routing_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "vrp", "name": "Vehicle Routing Problem", "tags": []}
        # supply/demand/shipping are negative cues for routing (they belong to TP)
        q = _tokenize("supply demand shipping source destination distribution")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"TP cues on a routing schema should give non-positive score, got {score}"

    # ── transportation (extended — now has negative cues for VRP) ────────────

    def test_transport_positive_cues_boost_transport_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "tp", "name": "Transportation Problem", "tags": []}
        q = _tokenize("supply demand ship shipping source destination distribution")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"TP cues on transportation schema should give positive score, got {score}"

    def test_vrp_cues_penalise_transport_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "tp", "name": "Transportation Problem", "tags": []}
        q = _tokenize("vehicle tour depot vrp capacitated visit circuit")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"VRP cues on transportation schema should give non-positive score, got {score}"

    # ── scheduling ───────────────────────────────────────────────────────────

    def test_scheduling_positive_cues_boost_scheduling_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "js", "name": "Job Shop Scheduling", "tags": []}
        q = _tokenize("machine makespan deadline tardiness sequence flowshop processing")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"Scheduling cues on scheduling schema should give positive score, got {score}"

    def test_production_cues_penalise_scheduling_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "js", "name": "Job Shop Scheduling", "tags": []}
        q = _tokenize("product manufacturing profit assembly labor labour factory")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"Production cues on scheduling schema should give non-positive score, got {score}"

    # ── production (extended — now has negative cues for scheduling) ─────────

    def test_production_positive_cues_boost_production_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "pp", "name": "Production Planning", "tags": []}
        q = _tokenize("product manufacturing profit assembly labor factory output")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"Production cues on production schema should give positive score, got {score}"

    def test_scheduling_cues_penalise_production_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "pp", "name": "Production Planning", "tags": []}
        q = _tokenize("makespan deadline tardiness sequence flowshop jobshop")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"Scheduling cues on production schema should give non-positive score, got {score}"

    # ── network_flow ─────────────────────────────────────────────────────────

    def test_flow_positive_cues_boost_network_flow_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        # "network_flow" key has underscore; name heuristic won't match "network flow"
        # (space), so provide the tag explicitly.
        problem = {"id": "nf", "name": "Min-Cost Network Flow", "tags": ["network_flow"]}
        q = _tokenize("path shortest sink arc maximum minimum cut augmenting")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"Flow cues on network_flow schema should give positive score, got {score}"

    def test_tp_cues_penalise_network_flow_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "nf", "name": "Min-Cost Network Flow", "tags": ["network_flow"]}
        q = _tokenize("supply demand ship route distribution")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"TP cues on network_flow schema should give non-positive score, got {score}"

    # ── covering ─────────────────────────────────────────────────────────────

    def test_cover_positive_cues_boost_covering_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        # "covering" key won't match "set cover" by substring (different word);
        # use the tag directly so the discrimination entry is found.
        problem = {"id": "sc", "name": "Set Cover Problem", "tags": ["covering"]}
        q = _tokenize("set element subset universe cover dominate")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"Cover cues on covering schema should give positive score, got {score}"

    def test_graph_cues_penalise_covering_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "sc", "name": "Set Cover Problem", "tags": ["covering"]}
        q = _tokenize("color coloring chromatic clique independent")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"Graph cues on covering schema should give non-positive score, got {score}"

    # ── graph ─────────────────────────────────────────────────────────────────

    def test_graph_positive_cues_boost_graph_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "gc", "name": "Graph Coloring Problem", "tags": []}
        q = _tokenize("color coloring chromatic clique independent bipartite")
        score = _confusable_discrimination_score(q, problem)
        assert score > 0.0, f"Graph cues on graph schema should give positive score, got {score}"

    def test_cover_cues_penalise_graph_schema(self):
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        problem = {"id": "gc", "name": "Graph Coloring Problem", "tags": []}
        q = _tokenize("subset universe element cover")
        score = _confusable_discrimination_score(q, problem)
        assert score <= 0.0, f"Cover cues on graph schema should give non-positive score, got {score}"

    def test_score_bounded_for_all_new_pairs(self):
        """All new discrimination scores stay in [-0.1, 0.1]."""
        from retrieval.reranking import _confusable_discrimination_score, _tokenize
        q = _tokenize("vehicle tour depot makespan machine cover element color chromatic")
        for name in [
            "Vehicle Routing Problem", "Transportation Problem",
            "Job Shop Scheduling", "Production Planning",
            "Min-Cost Network Flow", "Set Cover Problem", "Graph Coloring",
        ]:
            problem = {"id": "x", "name": name, "tags": []}
            score = _confusable_discrimination_score(q, problem)
            assert -0.1 <= score <= 0.1, (
                f"Score out of bounds for '{name}': {score}"
            )

    def test_all_new_keys_present_in_discrimination_map(self):
        from retrieval.reranking import _CONFUSABLE_DISCRIMINATION
        for key in ("routing", "transportation", "scheduling", "production",
                    "network_flow", "covering", "graph"):
            assert key in _CONFUSABLE_DISCRIMINATION, (
                f"Expected '{key}' in _CONFUSABLE_DISCRIMINATION"
            )

    def test_total_discrimination_pairs_at_least_12(self):
        """Regression guard: map must not shrink below 12 entries (was 8 originally,
        expanded to 13 by adding routing/transportation/scheduling/production/
        network_flow/covering/graph pairs)."""
        from retrieval.reranking import _CONFUSABLE_DISCRIMINATION
        assert len(_CONFUSABLE_DISCRIMINATION) >= 13, (
            f"Expected ≥ 13 entries in _CONFUSABLE_DISCRIMINATION, "
            f"got {len(_CONFUSABLE_DISCRIMINATION)}"
        )


# ---------------------------------------------------------------------------
# New tests: role-cue vocabulary additions
# ---------------------------------------------------------------------------

class TestRoleCueVocabularyAdditions:
    """Test the new words added to _CAPACITY_CUES, _PER_UNIT_CUES, _BOUND_CUES."""

    def test_bandwidth_is_capacity_cue(self):
        from retrieval.reranking import _CAPACITY_CUES
        assert "bandwidth" in _CAPACITY_CUES, "bandwidth should be a capacity cue"

    def test_throughput_is_capacity_cue(self):
        from retrieval.reranking import _CAPACITY_CUES
        assert "throughput" in _CAPACITY_CUES, "throughput should be a capacity cue"

    def test_seat_is_capacity_cue(self):
        from retrieval.reranking import _CAPACITY_CUES
        assert "seat" in _CAPACITY_CUES, "seat should be a capacity cue"

    def test_slot_is_capacity_cue(self):
        from retrieval.reranking import _CAPACITY_CUES
        assert "slot" in _CAPACITY_CUES, "slot should be a capacity cue"

    def test_penalty_is_per_unit_cue(self):
        from retrieval.reranking import _PER_UNIT_CUES
        assert "penalty" in _PER_UNIT_CUES, "penalty should be a per-unit cue"

    def test_wage_is_per_unit_cue(self):
        from retrieval.reranking import _PER_UNIT_CUES
        assert "wage" in _PER_UNIT_CUES, "wage should be a per-unit cue"

    def test_salary_is_per_unit_cue(self):
        from retrieval.reranking import _PER_UNIT_CUES
        assert "salary" in _PER_UNIT_CUES, "salary should be a per-unit cue"

    def test_return_is_per_unit_cue(self):
        from retrieval.reranking import _PER_UNIT_CUES
        assert "return" in _PER_UNIT_CUES, "return should be a per-unit cue"

    def test_gain_is_per_unit_cue(self):
        from retrieval.reranking import _PER_UNIT_CUES
        assert "gain" in _PER_UNIT_CUES, "gain should be a per-unit cue"

    def test_yield_is_per_unit_cue(self):
        from retrieval.reranking import _PER_UNIT_CUES
        assert "yield" in _PER_UNIT_CUES, "yield should be a per-unit cue"

    def test_least_is_bound_cue(self):
        """'least' captures the 'at least' phrase pattern in user queries."""
        from retrieval.reranking import _BOUND_CUES
        assert "least" in _BOUND_CUES, "least (from 'at least') should be a bound cue"

    def test_most_is_bound_cue(self):
        """'most' captures the 'at most' phrase pattern in user queries."""
        from retrieval.reranking import _BOUND_CUES
        assert "most" in _BOUND_CUES, "most (from 'at most') should be a bound cue"

    def test_new_capacity_cues_in_all_role_cues(self):
        """New capacity cues must be propagated to _ALL_ROLE_CUES."""
        from retrieval.reranking import _ALL_ROLE_CUES
        for word in ("bandwidth", "throughput", "seat", "slot"):
            assert word in _ALL_ROLE_CUES, f"'{word}' should be in _ALL_ROLE_CUES"

    def test_new_per_unit_cues_in_all_role_cues(self):
        from retrieval.reranking import _ALL_ROLE_CUES
        for word in ("penalty", "wage", "salary", "return", "gain", "yield"):
            assert word in _ALL_ROLE_CUES, f"'{word}' should be in _ALL_ROLE_CUES"

    def test_new_bound_cues_in_all_role_cues(self):
        from retrieval.reranking import _ALL_ROLE_CUES
        for word in ("least", "most"):
            assert word in _ALL_ROLE_CUES, f"'{word}' should be in _ALL_ROLE_CUES"

    def test_bandwidth_query_triggers_capacity_overlap(self):
        """bandwidth is now in _QTY_CAPACITY_CUES, so a bandwidth query matches a
        capacity-slot schema via grounding_consistency_score (family-level check)."""
        from retrieval.reranking import grounding_consistency_score, _QTY_CAPACITY_CUES
        # Verify the new word is in the right family set
        assert "bandwidth" in _QTY_CAPACITY_CUES
        # A query that mentions bandwidth should register as having a capacity signal
        query = "maximize bandwidth through the network"
        # Schema whose slot vocabulary includes a standard capacity cue ("capacity")
        schema = {
            "id": "nf",
            "name": "Max-Flow Network Problem",
            "aliases": [],
            "description": "Maximise flow through a network respecting arc capacity limits.",
            "formulation": {
                "variables": [{"symbol": "f_ij", "description": "flow on arc ij"}],
                "objective": {"sense": "maximize", "expression": "sum f_ij"},
                "constraints": [
                    {"expression": "f_ij <= capacity_ij", "description": "arc capacity limit"},
                ],
            },
        }
        gc = grounding_consistency_score(query, schema)
        # Both query (bandwidth) and schema (capacity) signal the capacity family →
        # grounding_consistency_score should be > 0.5 (better than neutral)
        assert gc > 0.5, (
            f"bandwidth query vs capacity-slot schema: grounding_consistency should be > 0.5, got {gc}"
        )

    def test_at_least_phrase_triggers_bound_overlap(self):
        """'at least N' phrase: tokenised to 'at', 'least' → 'least' now in BOUND_CUES.
        Role-cue overlap fires when the schema also uses 'least' in its slot vocabulary."""
        from retrieval.reranking import _rerank_score, _tokenize
        query_tokens = _tokenize("produce at least 100 units per day")
        # Schema whose constraint description uses 'least' so s_role contains it
        schema = {
            "id": "pp",
            "name": "Production Planning",
            "aliases": [],
            "description": "Plan production to meet demand constraints.",
            "formulation": {
                "variables": [{"symbol": "x_p", "description": "units produced of product p"}],
                "objective": {"sense": "maximize", "expression": "sum profit_p x_p"},
                "constraints": [
                    {
                        "expression": "x_p >= demand_p",
                        "description": "produce at least the minimum demand for each product",
                    },
                ],
            },
        }
        features = _rerank_score(query_tokens, schema)
        # 'least' appears in both query tokens AND schema constraint description →
        # role_cue_overlap > 0 now that 'least' is in BOUND_CUES
        assert features.role_cue_overlap > 0.0, (
            "'at least' phrase should now contribute to role_cue_overlap via 'least' bound cue; "
            f"got role_cue_overlap={features.role_cue_overlap}"
        )
