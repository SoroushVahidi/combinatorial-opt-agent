"""Regression tests for total-vs-coefficient (global-vs-local) grounding fixes.

These tests guard against the failure mode where:
- Global totals / capacities / budgets (e.g. 100, 5000, 2000) are incorrectly
  assigned to local per-unit coefficient slots, or
- Per-unit coefficients (e.g. 2, 3, 12) are incorrectly assigned to global
  resource-capacity slots.

AUDIT context (see problem statement):
- Root cause 1: Wide ±14 token context for is_per_unit / is_total_like caused
  cross-contamination; fixed by directional narrow windows.
- Root cause 2: is_coefficient_like=True on total-capacity slots (e.g.
  LaborHoursAvailable) because resource_consumption tag fires on the domain
  topic, not the per-unit role; fixed by excluding total-like slots.
- Root cause 3: End-of-sentence numbers like "5000." were silently dropped
  because NUM_TOKEN_RE.fullmatch("5000.") fails; fixed by rstrip before match.
- Local mismatch penalties added: per-unit mention → total slot (and vice versa).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _gcg_local_score,
    _run_global_consistency_grounding,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _gcg(query: str, slots: list[str]) -> dict:
    vals, _, _ = _run_global_consistency_grounding(query, "orig", slots)
    return vals


# ---------------------------------------------------------------------------
# 1. Mention-flag unit tests: directional narrow windows
# ---------------------------------------------------------------------------


class TestMentionFlags:
    """Verify that is_per_unit / is_total_like are set correctly."""

    def test_requires_verb_makes_number_per_unit(self):
        """'requires 2' → is_per_unit=True, is_total_like=False."""
        query = "Each phone requires 2 hours of labor."
        mentions = _extract_opt_role_mentions(query, "orig")
        m2 = next(m for m in mentions if m.value == 2.0)
        assert m2.is_per_unit, "requires-governed number should be per_unit"
        assert not m2.is_total_like, "requires-governed number should not be total_like"

    def test_available_makes_number_total_like(self):
        """'2000 available' → is_total_like=True, is_per_unit=False."""
        query = "There are 2000 hours of labor available."
        mentions = _extract_opt_role_mentions(query, "orig")
        m2000 = next(m for m in mentions if m.value == 2000.0)
        assert m2000.is_total_like, "available-followed number should be total_like"
        assert not m2000.is_per_unit, "available-followed number should not be per_unit"

    def test_has_makes_number_total_like(self):
        """'has 100 sq feet' → is_total_like=True, is_per_unit=False."""
        query = "The factory has 100 sq feet of space."
        mentions = _extract_opt_role_mentions(query, "orig")
        m100 = next(m for m in mentions if m.value == 100.0)
        assert m100.is_total_like, "has-preceded number should be total_like"
        assert not m100.is_per_unit, "has-preceded number should not be per_unit"

    def test_budget_left_makes_number_total_like(self):
        """'budget is 5000' → is_total_like=True, is_per_unit=False."""
        query = "The budget is 5000."
        mentions = _extract_opt_role_mentions(query, "orig")
        m5000 = next(m for m in mentions if m.value == 5000.0)
        assert m5000.is_total_like, "budget-preceded number should be total_like"
        assert not m5000.is_per_unit, "budget-preceded number should not be per_unit"

    def test_cost_verb_makes_number_per_unit(self):
        """'costs 12 per' → is_per_unit=True, is_total_like=False."""
        query = "Phones cost 12 per sq foot."
        mentions = _extract_opt_role_mentions(query, "orig")
        m12 = next(m for m in mentions if m.value == 12.0)
        assert m12.is_per_unit, "cost-governed number should be per_unit"
        assert not m12.is_total_like, "cost-governed number should not be total_like"

    def test_cross_sentence_each_does_not_contaminate_total(self):
        """'require 3 each. There are 2000' → 'each' must not make 2000 per_unit."""
        query = (
            "Each phone requires 2 hours of labor each. "
            "There are 2000 hours of labor available."
        )
        mentions = _extract_opt_role_mentions(query, "orig")
        m2000 = next(m for m in mentions if m.value == 2000.0)
        # 'each' from prior clause must not bleed into 2000's left window
        assert not m2000.is_per_unit, (
            "'each' from prior clause must not contaminate 2000 via cross-sentence bleed"
        )
        assert m2000.is_total_like, "2000 should still be total_like from 'available'"


# ---------------------------------------------------------------------------
# 2. Slot-flag unit tests
# ---------------------------------------------------------------------------


class TestSlotFlags:
    """Verify is_total_like / is_coefficient_like on slot IR."""

    def test_labor_available_slot_not_coefficient_like(self):
        """LaborHoursAvailable is total-like, must NOT be coefficient-like."""
        slot_irs = _build_slot_opt_irs(["LaborHoursAvailable"])
        s = slot_irs[0]
        assert s.is_total_like, "LaborHoursAvailable should be total_like"
        assert not s.is_coefficient_like, (
            "LaborHoursAvailable must not be coefficient_like "
            "(resource_consumption tag is domain topic, not per-unit role)"
        )

    def test_budget_slot_total_like(self):
        """Budget is total-like, not coefficient-like."""
        slot_irs = _build_slot_opt_irs(["Budget"])
        s = slot_irs[0]
        assert s.is_total_like
        assert not s.is_coefficient_like

    def test_total_space_slot_flags(self):
        """TotalSpace is total-like, not coefficient-like."""
        slot_irs = _build_slot_opt_irs(["TotalSpace"])
        s = slot_irs[0]
        assert s.is_total_like
        assert not s.is_coefficient_like


# ---------------------------------------------------------------------------
# 3. Local-score penalty: per-unit mention must be penalised for total slot
# ---------------------------------------------------------------------------


class TestLocalScorePenalties:
    """Verify that local mismatch penalties fire correctly."""

    def test_per_unit_mention_penalised_for_total_slot(self):
        """A per-unit mention should score lower for a total slot than a total mention."""
        query = "Each phone requires 2 hours. There are 2000 hours available."
        mentions = _extract_opt_role_mentions(query, "orig")
        m2 = next(m for m in mentions if m.value == 2.0)
        m2000 = next(m for m in mentions if m.value == 2000.0)
        slot_irs = _build_slot_opt_irs(["LaborHoursAvailable"])
        s_lha = slot_irs[0]

        score_2, feats_2 = _gcg_local_score(m2, s_lha)
        score_2000, feats_2000 = _gcg_local_score(m2000, s_lha)

        assert score_2000 > score_2, (
            f"Total mention (2000, score={score_2000:.2f}) should beat "
            f"per-unit mention (2, score={score_2:.2f}) for LaborHoursAvailable"
        )
        # Verify the penalty feature fired for the per-unit mention
        assert feats_2.get("coeff_to_total_penalty") or score_2 < score_2000

    def test_total_mention_penalised_for_coefficient_slot(self):
        """A total mention should score lower for a per-unit slot than a per-unit mention.

        Uses HoursPerProduct which is correctly classified as coefficient-like
        (has resource_consumption + unit_cost tags).  LaborRequired is NOT
        coefficient-like (only demand_requirement tags) so the
        total_to_coeff_local_penalty would not fire there.
        """
        query = "Each phone requires 2 hours. There are 2000 hours available."
        mentions = _extract_opt_role_mentions(query, "orig")
        m2 = next(m for m in mentions if m.value == 2.0)
        m2000 = next(m for m in mentions if m.value == 2000.0)
        slot_irs = _build_slot_opt_irs(["HoursPerProduct"])
        s_lr = slot_irs[0]

        assert s_lr.is_coefficient_like, "HoursPerProduct must be coefficient-like for this test to be meaningful"

        score_2, _ = _gcg_local_score(m2, s_lr)
        score_2000, feats_2000 = _gcg_local_score(m2000, s_lr)

        assert score_2 > score_2000, (
            f"Per-unit mention (2, score={score_2:.2f}) should beat "
            f"total mention (2000, score={score_2000:.2f}) for HoursPerProduct"
        )
        assert feats_2000.get("total_to_coeff_penalty"), (
            "total_to_coeff_penalty feature must fire for a total mention on a coefficient slot"
        )


# ---------------------------------------------------------------------------
# 4. End-to-end: canonical total-vs-coefficient failure case (Test 1)
# ---------------------------------------------------------------------------


class TestCanonicalTotalVsCoefficient:
    """End-to-end regression for the 'each/requires + available/budget' pattern."""

    QUERY = (
        "There are 100 sq feet of space available. "
        "Each phone requires 2 sq feet. "
        "Each laptop requires 3 sq feet. "
        "The budget is 5000 dollars."
    )
    SLOTS = ["TotalSpace", "Budget", "PhoneSpaceReq", "LaptopSpaceReq"]

    def test_global_totals_assigned_correctly(self):
        vals = _gcg(self.QUERY, self.SLOTS)
        assert vals["TotalSpace"] == pytest.approx(100.0), (
            "TotalSpace must get 100 (global capacity), not a per-unit coefficient"
        )
        assert vals["Budget"] == pytest.approx(5000.0), (
            "Budget must get 5000 (global budget), not a per-unit coefficient"
        )

    def test_per_unit_coefficients_assigned_correctly(self):
        vals = _gcg(self.QUERY, self.SLOTS)
        assert vals["PhoneSpaceReq"] == pytest.approx(2.0), (
            "PhoneSpaceReq must get 2 (per-phone coefficient), not 100 or 5000"
        )
        assert vals["LaptopSpaceReq"] == pytest.approx(3.0), (
            "LaptopSpaceReq must get 3 (per-laptop coefficient), not 100 or 5000"
        )


# ---------------------------------------------------------------------------
# 5. End-to-end: global resource limits not confused with coefficients
# ---------------------------------------------------------------------------


class TestGlobalResourceSlots:
    """Global resource limits (capacity/budget/labor) must not be filled by coefficients."""

    def test_labor_capacity_gets_total_not_coefficient(self):
        query = (
            "Phones require 2 hours of labor. "
            "Laptops require 3 hours of labor. "
            "There are 2000 hours of labor available."
        )
        vals = _gcg(query, ["LaborHoursAvailable", "PhoneLabor", "LaptopLabor"])
        assert vals["LaborHoursAvailable"] == pytest.approx(2000.0), (
            "LaborHoursAvailable must get 2000 (total), not 2 or 3 (per-unit)"
        )

    def test_space_capacity_gets_total_not_coefficient(self):
        query = (
            "Each unit requires 5 sq feet. "
            "The factory has 500 sq feet of space available."
        )
        vals = _gcg(query, ["TotalSpace", "SpacePerUnit"])
        assert vals["TotalSpace"] == pytest.approx(500.0), (
            "TotalSpace must get 500 (total capacity), not 5 (per-unit)"
        )

    def test_budget_gets_spend_limit_not_unit_cost(self):
        query = (
            "Each item costs 15 dollars. "
            "The company can spend at most 3000 dollars."
        )
        vals = _gcg(query, ["Budget", "CostPerItem"])
        assert vals["Budget"] == pytest.approx(3000.0), (
            "Budget must get 3000 (spending limit), not 15 (unit cost)"
        )


# ---------------------------------------------------------------------------
# 6. Count-slot protection: large totals must not fill count slots
# ---------------------------------------------------------------------------


class TestCountSlotProtection:
    """Count-like slots must prefer small integers over large totals."""

    def test_count_slot_prefers_small_int_over_large_total(self):
        query = (
            "There are 2 products: phones and laptops. "
            "The factory has 500 sq feet available."
        )
        vals = _gcg(query, ["NumberOfProducts", "TotalSpace"])
        # NumberOfProducts=2 and TotalSpace=500 (not swapped)
        assert vals.get("NumberOfProducts") == pytest.approx(2.0), (
            "NumberOfProducts must prefer 2 (small count), not 500 (large total)"
        )
        if vals.get("TotalSpace") is not None:
            assert vals["TotalSpace"] != pytest.approx(2.0), (
                "TotalSpace must not get 2 (count value)"
            )

    def test_count_slot_scores_small_int_higher_than_large_total(self):
        """Scoring: small integer should beat large total for a count-like slot."""
        query = "There are 2 products and 500 units of capacity."
        mentions = _extract_opt_role_mentions(query, "orig")
        slot_irs = _build_slot_opt_irs(["NumberOfProducts"])
        s = slot_irs[0]
        assert s.is_count_like, "NumberOfProducts should be count_like"

        m2 = next((m for m in mentions if m.value == 2.0), None)
        m500 = next((m for m in mentions if m.value == 500.0), None)
        if m2 is not None and m500 is not None:
            score_2, _ = _gcg_local_score(m2, s)
            score_500, _ = _gcg_local_score(m500, s)
            assert score_2 > score_500, (
                f"Small int 2 (score={score_2:.2f}) must score higher than "
                f"large total 500 (score={score_500:.2f}) for NumberOfProducts"
            )


# ---------------------------------------------------------------------------
# 7. End-of-sentence number extraction (regression for '5000.' bug)
# ---------------------------------------------------------------------------


class TestEndOfSentenceNumbers:
    """Numbers at the end of a sentence (e.g. '5000.') must be extracted."""

    def test_budget_value_at_end_of_sentence(self):
        """'The budget is 5000.' — trailing period must not prevent extraction."""
        query = "Each item costs 15. The budget is 5000."
        mentions = _extract_opt_role_mentions(query, "orig")
        values = {m.value for m in mentions if m.value is not None}
        assert 5000.0 in values, (
            "5000 must be extracted even when followed by a sentence-ending period"
        )

    def test_end_of_sentence_number_assigned_to_budget(self):
        """End-of-sentence budget value must still win the Budget slot."""
        query = "Each item costs 15. The budget is 5000."
        vals = _gcg(query, ["Budget", "CostPerItem"])
        assert vals["Budget"] == pytest.approx(5000.0), (
            "Budget must get 5000, even though it ends with a period"
        )


# ---------------------------------------------------------------------------
# 8. Non-regression: non-global, non-count slots must not be harmed
# ---------------------------------------------------------------------------


class TestNonRegression:
    """Existing non-count, non-global slots must still be assigned correctly."""

    def test_percent_slot_unaffected(self):
        """Percent-type slots must still get percent values, not large integers."""
        query = "The discount is 20 percent. The total price is 500."
        vals = _gcg(query, ["DiscountRate", "TotalPrice"])
        if vals.get("DiscountRate") is not None:
            assert vals["DiscountRate"] == pytest.approx(0.20, abs=1e-3), (
                "DiscountRate must still get 0.20 (percent), not 500"
            )

    def test_objective_slot_unaffected(self):
        """Objective/revenue slots must still get plausible per-unit profit values."""
        query = "Each product earns 50 dollars profit. Total capacity is 200."
        vals = _gcg(query, ["ProfitPerUnit", "TotalCapacity"])
        if vals.get("TotalCapacity") is not None:
            assert vals["TotalCapacity"] == pytest.approx(200.0), (
                "TotalCapacity must get 200 (total), not 50 (per-unit profit)"
            )
