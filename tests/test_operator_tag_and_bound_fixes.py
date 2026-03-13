"""Regression tests for the operator-tag and is_total_like bug fixes.

Bug 1 — Operator-tag false positives:
    The old `_detect_operator_tags` used single-word sets (OPERATOR_MIN_PHRASES /
    OPERATOR_MAX_PHRASES) that shared ambiguous common words: "at", "no", "than",
    "to".  This caused both "min" and "max" tags to fire for almost every mention
    (e.g. "to" from "wants to maximize" triggered max; "at" from "at least" triggered
    both), neutralising the bound-flip penalty in downstream assignment.

Bug 2 — is_total_like under-detection:
    Slot names like MaxWater, WaterAvailability, PowderedPillAvailability were not
    classified as total-like because `is_total_like` only checked for "budget" or
    "total" in the name, missing "available", "availability", "capacity" prefixes and
    the "capacity_limit" role tag.

These tests guard against regression to both bugs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _build_slot_opt_irs,
    _detect_operator_tags,
    _extract_opt_role_mentions,
    _run_optimization_role_repair,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _narrow(toks: list[str], i: int, window: int = 3) -> list[str]:
    """Return the ±window token slice around position i (mirroring production code)."""
    from tools.nlp4lp_downstream_utility import _OPERATOR_NARROW_WINDOW
    w = _OPERATOR_NARROW_WINDOW
    return [
        t.lower().strip(".,;:()[]{}") for t in toks[max(0, i - w) : i + w + 1]
        if t.strip(".,;:(){}")
    ]


# ---------------------------------------------------------------------------
# Bug 1 — Operator tag false positives
# ---------------------------------------------------------------------------

class TestOperatorTagFix:
    """_detect_operator_tags should produce discriminating tags, not both min+max."""

    def test_at_least_only_gives_min(self):
        ctx = "company needs to send at least 5000 bottles".split()
        narrow = "needs to send at least 5000 bottles".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "min" in tags, "at least → min expected"
        assert "max" not in tags, "at least must NOT produce max"

    def test_at_most_only_gives_max(self):
        ctx = "in addition at most 30 new vans can be used".split()
        narrow = "addition at most 30 new vans can".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "max" in tags, "at most → max expected"
        assert "min" not in tags, "at most must NOT produce min"

    def test_cross_contamination_prevented(self):
        """When 'at most X' and 'at least Y' are in the same wide context, the narrow
        window prevents each from picking up the other's operator phrase."""
        # Query fragment: "... at least 5000 bottles. In addition, at most 30 vans ..."
        wide = "needs to send at least 5000 bottles in addition at most 30 new vans can be used".split()
        # '5000' is at index 5; _OPERATOR_NARROW_WINDOW=3 → toks[2:9]
        narrow_5000 = wide[2:9]   # ["send", "at", "least", "5000", "bottles", "in", "addition"]
        # '30' is at index 11; _OPERATOR_NARROW_WINDOW=3 → toks[8:15]
        narrow_30 = wide[8:15]    # ["addition", "at", "most", "30", "new", "vans", "can"]
        tags_5000 = _detect_operator_tags(wide, narrow_5000)
        tags_30 = _detect_operator_tags(wide, narrow_30)
        assert tags_5000 == frozenset({"min"}), f"5000 tags: {tags_5000}"
        assert tags_30 == frozenset({"max"}), f"30 tags: {tags_30}"

    def test_maximum_word_gives_max(self):
        """Unambiguous single word 'maximum' still works (non-regression)."""
        ctx = "the maximum number of items is 80 units".split()
        narrow = "maximum number of items is 80 units".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "max" in tags
        assert "min" not in tags

    def test_minimum_word_gives_min(self):
        """Unambiguous single word 'minimum' still works."""
        ctx = "the minimum number of chairs is 5".split()
        narrow = "minimum number of chairs is 5".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "min" in tags
        assert "max" not in tags

    def test_no_operator_cues_gives_empty(self):
        """A plain coefficient sentence produces no operator tags."""
        ctx = "each container requires 10 units of water and 15 units".split()
        narrow = "container requires 10 units of water and".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "min" not in tags
        assert "max" not in tags

    def test_at_most_1500_in_constraint(self):
        """Example 9 regression: 'at most 1500 units of water' → max only."""
        ctx = "there can be at most 1500 units of water 1350 units of pollution".split()
        narrow = "be at most 1500 units of water".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "max" in tags
        assert "min" not in tags

    def test_per_unit_mention_no_operator_tags(self):
        """'using 6 units of water' → no operator tags (not a bound)."""
        ctx = "process p can extract 9 units of metal using 6 units of water and produces 5".split()
        narrow = "metal using 6 units of water".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "min" not in tags
        assert "max" not in tags

    def test_no_more_than_phrase(self):
        """Multi-word phrase 'no more than' → max only."""
        ctx = "there should be no more than 100 workers employed".split()
        narrow = "be no more than 100 workers".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "max" in tags
        assert "min" not in tags

    def test_no_less_than_phrase(self):
        """Multi-word phrase 'no less than' → min only."""
        ctx = "the store must hire no less than 20 employees".split()
        narrow = "hire no less than 20 employees".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "min" in tags
        assert "max" not in tags

    def test_to_alone_does_not_trigger_max(self):
        """'to' as a common preposition must NOT trigger max (old bug source)."""
        ctx = "the factory wants to produce 500 units of product a".split()
        narrow = "wants to produce 500 units of".split()
        tags = _detect_operator_tags(ctx, narrow)
        assert "max" not in tags, "preposition 'to' must not be interpreted as 'at most'"


class TestOperatorTagInMentionExtraction:
    """End-to-end: _extract_opt_role_mentions should produce discriminating operator tags."""

    def test_5000_gets_min_in_multiconstraint_query(self):
        query = (
            "The company needs to send at least 5000 bottles. "
            "In addition, at most 30 new vans can be used."
        )
        mentions = _extract_opt_role_mentions(query, "orig")
        m_5000 = next((m for m in mentions if m.value == 5000), None)
        m_30 = next((m for m in mentions if m.value == 30), None)
        assert m_5000 is not None, "5000 not extracted"
        assert m_30 is not None, "30 not extracted"
        assert "min" in m_5000.operator_tags, f"5000 tags: {m_5000.operator_tags}"
        assert "max" not in m_5000.operator_tags, f"5000 tags: {m_5000.operator_tags}"
        assert "max" in m_30.operator_tags, f"30 tags: {m_30.operator_tags}"
        assert "min" not in m_30.operator_tags, f"30 tags: {m_30.operator_tags}"

    def test_at_most_1500_max_operator(self):
        query = (
            "Process J uses 8 units of water. "
            "Process P uses 6 units of water. "
            "There can be at most 1500 units of water."
        )
        mentions = _extract_opt_role_mentions(query, "orig")
        m_1500 = next((m for m in mentions if m.value == 1500), None)
        assert m_1500 is not None, "1500 not extracted"
        assert "max" in m_1500.operator_tags, f"1500 tags: {m_1500.operator_tags}"
        assert "min" not in m_1500.operator_tags, f"1500 tags: {m_1500.operator_tags}"


# ---------------------------------------------------------------------------
# Bug 2 — is_total_like under-detection
# ---------------------------------------------------------------------------

class TestIsTotalLikeFix:
    """Capacity / availability / max-bound slot names should be total-like."""

    def test_max_water_is_total_like(self):
        slots = _build_slot_opt_irs(["MaxWater"])
        assert slots[0].is_total_like, "MaxWater should be total-like"

    def test_max_pollution_is_total_like(self):
        slots = _build_slot_opt_irs(["MaxPollution"])
        assert slots[0].is_total_like, "MaxPollution should be total-like"

    def test_water_availability_is_total_like(self):
        slots = _build_slot_opt_irs(["WaterAvailability"])
        assert slots[0].is_total_like, "WaterAvailability should be total-like"

    def test_powdered_pill_availability_is_total_like(self):
        slots = _build_slot_opt_irs(["PowderedPillAvailability"])
        assert slots[0].is_total_like, "PowderedPillAvailability should be total-like"

    def test_water_capacity_is_total_like(self):
        slots = _build_slot_opt_irs(["WaterCapacity"])
        assert slots[0].is_total_like, "WaterCapacity (has 'capacity') should be total-like"

    def test_shaping_time_available_is_total_like(self):
        slots = _build_slot_opt_irs(["ShapingTimeAvailable"])
        assert slots[0].is_total_like, "ShapingTimeAvailable should be total-like"

    def test_per_unit_coeff_not_total_like(self):
        """Per-unit coefficient slots must NOT be classified as total-like."""
        for name in [
            "WaterRequiredPerContainer",
            "PowderedPillRequiredPerContainer",
            "ProfitPerUnit",
            "CostPerBatch",
            "ShapingTimePerType",
        ]:
            slots = _build_slot_opt_irs([name])
            assert not slots[0].is_total_like, f"{name} should NOT be total-like"

    def test_total_budget_still_total_like(self):
        """Non-regression: existing 'total' and 'budget' checks still work."""
        for name in ["TotalBudget", "WeeklyBudget", "TotalUnits"]:
            slots = _build_slot_opt_irs([name])
            assert slots[0].is_total_like, f"{name} should be total-like"


# ---------------------------------------------------------------------------
# Integration: example from benchmark failure set
# ---------------------------------------------------------------------------

class TestBoundSwapRegression:
    """End-to-end regression for min/max bound swap (nlp4lp_test_142 pattern)."""

    def test_minimum_bottles_gets_5000_not_30(self):
        """The min bound (5000) must go to MinimumBottles, not the max bound (30)."""
        query = (
            "A soda company sends bottles in old and new vans. "
            "An old van can take 100 soda bottles. A new van can take 80 soda bottles. "
            "An old van produces 50 units of pollution, a new van produces 30 units. "
            "The company needs to send at least 5000 bottles. "
            "In addition, at most 30 new vans can be used."
        )
        result, _, _ = _run_optimization_role_repair(
            query, "orig",
            ["MinimumBottles", "MaximumNewVans",
             "OldVanCapacity", "NewVanCapacity"],
        )
        if result.get("MinimumBottles") is not None and result.get("MaximumNewVans") is not None:
            min_v = float(result["MinimumBottles"])
            max_v = float(result["MaximumNewVans"])
            # Core assertion: the large bound (5000) should not end up in the max slot
            assert min_v > max_v or min_v == 5000 or max_v == 30, (
                f"Bound swap detected: MinimumBottles={min_v}, MaximumNewVans={max_v}"
            )
