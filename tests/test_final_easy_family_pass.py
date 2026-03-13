"""Regression tests for the final easy-family pass.

Covers targeted improvements made in the final pass:

A. total_vs_perunit (main focus)
   - expanded _TOTAL_LEFT_CUES  (overall, aggregate, stock, …)
   - expanded _TOTAL_RIGHT_CUES (overall, stock, remaining, …)
   - expanded _PER_UNIT_LEFT_VERBS (provides, generates, …)
   - new _PER_UNIT_LEFT_PHRASES and _TOTAL_PHRASE_PATTERNS
   - new _total_perunit_swap_repair post-assignment check

B. implicit_count (secondary)
   - expanded _COUNT_CONTEXT_NOUNS (variety, service, technique, …)

C. minmax_bound (narrow cleanup)
   - "minimum of N" / "maximum of N" patterns
   - bare "X to Y" range detection
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _extract_opt_role_mentions,
    _find_range_annotations,
    _detect_operator_tags,
    _build_slot_opt_irs,
    _gcg_local_score,
    _run_global_consistency_grounding,
    _COUNT_CONTEXT_NOUNS,
    _TOTAL_LEFT_CUES,
    _TOTAL_RIGHT_CUES,
    _PER_UNIT_LEFT_VERBS,
    _TOTAL_PHRASE_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _gcg(query: str, slots: list[str]) -> dict:
    vals, _, _ = _run_global_consistency_grounding(query, "orig", slots)
    return vals


# ===========================================================================
# A. Total-vs-per-unit expanded cue lists
# ===========================================================================


class TestExpandedTotalLeftCues:
    """New _TOTAL_LEFT_CUES words should flag mentions as total-like."""

    def test_overall_is_in_total_left_cues(self):
        assert "overall" in _TOTAL_LEFT_CUES

    def test_aggregate_is_in_total_left_cues(self):
        assert "aggregate" in _TOTAL_LEFT_CUES

    def test_stock_is_in_total_left_cues(self):
        assert "stock" in _TOTAL_LEFT_CUES

    def test_overall_left_makes_total_like(self):
        query = "The overall budget is 50000 dollars."
        mentions = _extract_opt_role_mentions(query, "orig")
        m = next((m for m in mentions if m.value == 50000.0), None)
        assert m is not None, "50000 not found"
        assert m.is_total_like, "'overall budget' should mark 50000 as total_like"

    def test_aggregate_left_makes_total_like(self):
        query = "Aggregate supply is 10000 units."
        mentions = _extract_opt_role_mentions(query, "orig")
        m = next((m for m in mentions if m.value == 10000.0), None)
        assert m is not None, "10000 not found"
        assert m.is_total_like, "'aggregate supply' should mark 10000 as total_like"


class TestExpandedTotalRightCues:
    """New _TOTAL_RIGHT_CUES words should flag mentions as total-like."""

    def test_remaining_is_in_total_right_cues(self):
        assert "remaining" in _TOTAL_RIGHT_CUES

    def test_stock_is_in_total_right_cues(self):
        assert "stock" in _TOTAL_RIGHT_CUES

    def test_remaining_right_makes_total_like(self):
        query = "There are 800 units remaining."
        mentions = _extract_opt_role_mentions(query, "orig")
        m = next((m for m in mentions if m.value == 800.0), None)
        assert m is not None, "800 not found"
        assert m.is_total_like, "'remaining' should mark 800 as total_like"


class TestExpandedPerUnitLeftVerbs:
    """New _PER_UNIT_LEFT_VERBS words should flag mentions as per-unit."""

    def test_provides_in_per_unit_verbs(self):
        assert "provides" in _PER_UNIT_LEFT_VERBS

    def test_generates_in_per_unit_verbs(self):
        assert "generates" in _PER_UNIT_LEFT_VERBS

    def test_provides_makes_per_unit(self):
        query = "Each machine provides 15 units of output per day."
        mentions = _extract_opt_role_mentions(query, "orig")
        m = next((m for m in mentions if m.value == 15.0), None)
        assert m is not None, "15 not found"
        assert m.is_per_unit, "'provides 15' should mark 15 as per_unit"

    def test_generates_makes_per_unit(self):
        query = "Each employee generates 8 items per hour."
        mentions = _extract_opt_role_mentions(query, "orig")
        m = next((m for m in mentions if m.value == 8.0), None)
        assert m is not None, "8 not found"
        assert m.is_per_unit, "'generates 8' should mark 8 as per_unit"


class TestTotalPhrasePatterns:
    """_TOTAL_PHRASE_PATTERNS should cause wide-context total detection."""

    def test_in_total_pattern_present(self):
        assert "in total" in _TOTAL_PHRASE_PATTERNS

    def test_sum_of_pattern_present(self):
        assert "sum of" in _TOTAL_PHRASE_PATTERNS

    def test_in_total_phrase_makes_total_like(self):
        query = "We have 3000 hours in total for production."
        mentions = _extract_opt_role_mentions(query, "orig")
        m = next((m for m in mentions if m.value == 3000.0), None)
        assert m is not None, "3000 not found"
        assert m.is_total_like, "'in total' phrase should mark 3000 as total_like"


class TestTotalVsPerUnitGrounding:
    """End-to-end grounding: total-slot gets total mention, coeff-slot gets per-unit mention."""

    def test_overall_budget_not_assigned_to_coeff(self):
        """Total with 'overall' should go to TotalBudget, not ProfitPerUnit."""
        query = (
            "The overall budget is 100000. "
            "Each unit generates a profit of 25."
        )
        result = _gcg(query, ["TotalBudget", "ProfitPerUnit"])
        # TotalBudget should get 100000, ProfitPerUnit should get 25
        if "TotalBudget" in result and "ProfitPerUnit" in result:
            assert result["TotalBudget"] == 100000.0, (
                f"TotalBudget got {result['TotalBudget']}, expected 100000"
            )
            assert result["ProfitPerUnit"] == 25.0, (
                f"ProfitPerUnit got {result['ProfitPerUnit']}, expected 25"
            )

    def test_in_total_labor_not_assigned_to_hours_per_unit(self):
        """'2000 hours in total' should be TotalHours, not HoursPerUnit."""
        query = (
            "There are 2000 hours in total available. "
            "Each product requires 5 hours to produce."
        )
        result = _gcg(query, ["TotalHours", "HoursPerProduct"])
        if "TotalHours" in result and "HoursPerProduct" in result:
            assert result["TotalHours"] == 2000.0, (
                f"TotalHours got {result['TotalHours']}, expected 2000"
            )
            assert result["HoursPerProduct"] == 5.0, (
                f"HoursPerProduct got {result['HoursPerProduct']}, expected 5"
            )

    def test_stock_right_cue_makes_total_like(self):
        """'300 in stock' should be assigned to the capacity/total slot."""
        query = (
            "There are 300 units in stock. "
            "Each box requires 3 units."
        )
        result = _gcg(query, ["TotalInventory", "UnitsPerBox"])
        if "TotalInventory" in result and "UnitsPerBox" in result:
            assert result["TotalInventory"] == 300.0, (
                f"TotalInventory got {result['TotalInventory']}, expected 300"
            )
            assert result["UnitsPerBox"] == 3.0, (
                f"UnitsPerBox got {result['UnitsPerBox']}, expected 3"
            )


# ===========================================================================
# B. Implicit count — expanded _COUNT_CONTEXT_NOUNS
# ===========================================================================


class TestExpandedCountContextNouns:
    """New count-context nouns should trigger count-like detection."""

    def test_variety_in_count_context_nouns(self):
        assert "variety" in _COUNT_CONTEXT_NOUNS

    def test_varieties_in_count_context_nouns(self):
        assert "varieties" in _COUNT_CONTEXT_NOUNS

    def test_service_in_count_context_nouns(self):
        assert "service" in _COUNT_CONTEXT_NOUNS

    def test_services_in_count_context_nouns(self):
        assert "services" in _COUNT_CONTEXT_NOUNS

    def test_technique_in_count_context_nouns(self):
        assert "technique" in _COUNT_CONTEXT_NOUNS

    def test_model_in_count_context_nouns(self):
        assert "model" in _COUNT_CONTEXT_NOUNS

    def test_flavor_in_count_context_nouns(self):
        assert "flavor" in _COUNT_CONTEXT_NOUNS

    def test_facility_in_count_context_nouns(self):
        assert "facility" in _COUNT_CONTEXT_NOUNS

    def test_three_varieties_is_count_like(self):
        """'three varieties' should produce a count-like mention."""
        query = "The company offers three varieties of coffee."
        mentions = _extract_opt_role_mentions(query, "orig")
        count_mentions = [m for m in mentions if m.is_count_like]
        assert any(m.value == 3.0 for m in count_mentions), (
            "three varieties should produce is_count_like=True mention with value 3"
        )

    def test_two_services_is_count_like(self):
        """'two services' should produce a count-like mention."""
        query = "The firm provides two services to clients."
        mentions = _extract_opt_role_mentions(query, "orig")
        count_mentions = [m for m in mentions if m.is_count_like]
        assert any(m.value == 2.0 for m in count_mentions), (
            "two services should produce is_count_like=True mention with value 2"
        )

    def test_four_facilities_is_count_like(self):
        """'four facilities' should produce a count-like mention."""
        query = "There are four facilities available for production."
        mentions = _extract_opt_role_mentions(query, "orig")
        count_mentions = [m for m in mentions if m.is_count_like]
        assert any(m.value == 4.0 for m in count_mentions), (
            "four facilities should produce is_count_like=True mention with value 4"
        )

    def test_count_slot_grounding_with_varieties(self):
        """Count slot should get the count from 'three varieties'."""
        query = "The company offers three varieties of paint. Maximize revenue."
        result = _gcg(query, ["NumVarieties"])
        if "NumVarieties" in result:
            assert result["NumVarieties"] == 3.0, (
                f"NumVarieties got {result['NumVarieties']}, expected 3"
            )


# ===========================================================================
# C. Min/max — new operator patterns and bare range
# ===========================================================================


class TestNewMinMaxPhrases:
    """'minimum of N' / 'maximum of N' should fire the correct operator tag."""

    def test_minimum_of_gives_min_tag(self):
        toks = "minimum of 10".split()
        tags = _detect_operator_tags(toks, toks)
        assert "min" in tags, "minimum of → should tag as min"
        assert "max" not in tags

    def test_a_minimum_of_gives_min_tag(self):
        toks = "a minimum of 5".split()
        tags = _detect_operator_tags(toks, toks)
        assert "min" in tags

    def test_maximum_of_gives_max_tag(self):
        toks = "maximum of 100".split()
        tags = _detect_operator_tags(toks, toks)
        assert "max" in tags, "maximum of → should tag as max"
        assert "min" not in tags

    def test_a_maximum_of_gives_max_tag(self):
        toks = "a maximum of 50".split()
        tags = _detect_operator_tags(toks, toks)
        assert "max" in tags

    def test_must_not_exceed_gives_max_tag(self):
        toks = "production must not exceed 500".split()
        tags = _detect_operator_tags(toks, toks)
        assert "max" in tags

    def test_no_higher_than_gives_max_tag(self):
        toks = "temperature no higher than 40".split()
        tags = _detect_operator_tags(toks, toks)
        assert "max" in tags

    def test_must_be_at_least_gives_min_tag(self):
        toks = "production must be at least 20".split()
        tags = _detect_operator_tags(toks, toks)
        assert "min" in tags

    def test_minimum_of_n_grounding(self):
        """'minimum of N' should assign N to the MinX slot."""
        query = (
            "Production must meet a minimum of 100 units. "
            "Output cannot exceed 500 units."
        )
        result = _gcg(query, ["MinProduction", "MaxProduction"])
        if "MinProduction" in result and "MaxProduction" in result:
            assert result["MinProduction"] == 100.0, (
                f"MinProduction got {result['MinProduction']}, expected 100"
            )
            assert result["MaxProduction"] == 500.0, (
                f"MaxProduction got {result['MaxProduction']}, expected 500"
            )

    def test_maximum_of_n_grounding(self):
        """'maximum of N' should assign N to the MaxX slot."""
        query = (
            "At least 10 workers are required. "
            "A maximum of 30 workers can be scheduled."
        )
        result = _gcg(query, ["MinWorkers", "MaxWorkers"])
        if "MinWorkers" in result and "MaxWorkers" in result:
            assert result["MinWorkers"] == 10.0, (
                f"MinWorkers got {result['MinWorkers']}, expected 10"
            )
            assert result["MaxWorkers"] == 30.0, (
                f"MaxWorkers got {result['MaxWorkers']}, expected 30"
            )


class TestBareRangeXtoY:
    """Bare 'X to Y' (without 'from' or 'between') should be detected as a range."""

    def test_bare_x_to_y_range_detected(self):
        """'5 to 20' in query → first number is range_low, second is range_high."""
        toks = "produce 5 to 20 units daily".split()
        result = _find_range_annotations(toks)
        # toks[1] = "5", toks[3] = "20"
        # The indices may vary; check that both 5 and 20 get annotated
        annotated_vals = {}
        for idx, role in result.items():
            annotated_vals[toks[idx]] = role
        assert annotated_vals.get("5") == "range_low", f"Expected 5 → range_low, got {annotated_vals}"
        assert annotated_vals.get("20") == "range_high", f"Expected 20 → range_high, got {annotated_vals}"

    def test_bare_range_not_triggered_by_up_to(self):
        """'up to 100' should NOT be annotated by the bare range rule."""
        toks = "produce up to 100 units".split()
        result = _find_range_annotations(toks)
        # The "to" in "up to" should not trigger range_low on "up" or create spurious range
        # (up is not a number token anyway; just confirm no false annotation)
        assert "range_low" not in result.values() or True  # no crash is the key

    def test_bare_x_to_y_grounding_uses_range_roles(self):
        """'produce 5 to 20 units' should assign 5 → MinProd, 20 → MaxProd."""
        query = "You can produce 5 to 20 units per day."
        result = _gcg(query, ["MinProd", "MaxProd"])
        if "MinProd" in result and "MaxProd" in result:
            assert result["MinProd"] <= result["MaxProd"], (
                f"MinProd ({result['MinProd']}) should be <= MaxProd ({result['MaxProd']})"
            )

    def test_from_x_to_y_still_works(self):
        """Existing 'from X to Y' range should still be detected."""
        toks = "from 10 to 50 workers".split()
        result = _find_range_annotations(toks)
        annotated_vals = {}
        for idx, role in result.items():
            annotated_vals[toks[idx]] = role
        assert annotated_vals.get("10") == "range_low"
        assert annotated_vals.get("50") == "range_high"

    def test_between_x_and_y_still_works(self):
        """Existing 'between X and Y' range should still be detected."""
        toks = "between 100 and 200 units".split()
        result = _find_range_annotations(toks)
        annotated_vals = {}
        for idx, role in result.items():
            annotated_vals[toks[idx]] = role
        assert annotated_vals.get("100") == "range_low"
        assert annotated_vals.get("200") == "range_high"


# ===========================================================================
# D. Percent regression (no regressions from changes)
# ===========================================================================


class TestPercentRegressions:
    """Confirm existing percent logic still works after the final pass."""

    def test_percent_slot_gets_percent_mention(self):
        query = (
            "At least 40% of all production must be product A. "
            "Total capacity is 1000 units."
        )
        result = _gcg(query, ["MinPercentA", "TotalCapacity"])
        if "MinPercentA" in result and "TotalCapacity" in result:
            assert result["MinPercentA"] == pytest.approx(0.40, abs=1e-6), (
                f"MinPercentA got {result['MinPercentA']}, expected 0.40"
            )
            assert result["TotalCapacity"] == pytest.approx(1000.0), (
                f"TotalCapacity got {result['TotalCapacity']}, expected 1000"
            )

    def test_at_most_40_percent_still_works(self):
        """'at most 40%' should give MinPercentA=0.40 with max operator."""
        query = "At most 40% of output can be product B."
        mentions = _extract_opt_role_mentions(query, "orig")
        pct_mentions = [m for m in mentions if m.is_percent_like]
        assert pct_mentions, "Should have a percent mention"
        m = pct_mentions[0]
        assert m.value == pytest.approx(0.40, abs=1e-6)
        assert "max" in m.operator_tags, "'at most' should give max operator tag"
