"""Tests for the four entity-anchoring and semantic-hint fixes applied to the
global-consistency grounding (GCG) pipeline.

Fix 1 – narrow_left entity-ID anchor in _gcg_local_score
    Single-character entity identifiers in the tight left-context window
    (e.g. 'b' from "Feed B contains") are used as discriminating signals
    so that per-entity slots are correctly distinguished even when the full
    context window contains both entity names.

Fix 2 – utilisation / utilization → percent in _expected_type
    The words "utilisation" and "utilization" are now classified as "percent"
    so that MinUtilisation / MaxUtilisation slots accept percent mentions.

Fix 3 – "rate" prefix not classified as percent in _expected_type
    "Rate" is only classified as percent when the slot name ENDS with "rate"
    (e.g. DiscountRate, TaxRate).  When "rate" appears as a prefix before an
    entity identifier (RateMachine1, Rate1) the slot is a plain float.

Fix 4 – tight-context cost/profit semantic hints in _gcg_local_score
    When a mention has an unambiguous tight-window cost context
    (is_cost_like=True, is_profit_like=False) it receives a bonus toward
    cost-containing slots and a penalty toward profit-containing slots (and
    vice-versa), resolving "profit is 8 and cost is 4" swaps.

Fix 5 – entity-coherence global penalty in _gcg_global_penalty
    Consecutive mentions where the second has no direct entity-letter anchor
    are rewarded when assigned to the same entity's slots as the preceding
    mention, fixing "Feed B 7 protein AND 15 fat" fat-value swaps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    GCG_GLOBAL_WEIGHTS,
    GCG_LOCAL_WEIGHTS,
    _build_slot_opt_irs,
    _expected_type,
    _extract_opt_role_mentions,
    _gcg_global_penalty,
    _gcg_local_score,
    _run_global_consistency_grounding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcg(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return _run_global_consistency_grounding(query, "orig", slots)


# ---------------------------------------------------------------------------
# Fix 1 – narrow_left single-char entity-ID anchor
# ---------------------------------------------------------------------------

class TestNarrowLeftEntityAnchor:
    """narrow_left_overlap must fire for single-char entity identifiers only,
    not for longer semantic-domain tokens that contaminate from preceding values."""

    def test_weight_exists_and_positive(self):
        assert "narrow_left_overlap" in GCG_LOCAL_WEIGHTS
        assert GCG_LOCAL_WEIGHTS["narrow_left_overlap"] > 0

    def test_single_letter_entity_anchor_fires(self):
        """val=7 in 'Feed B contains 7' has 'b' in narrow_left → FeedB slot gets bonus."""
        query = "Feed B contains 7 protein and 15 fat, while Feed A contains 10 protein and 8 fat."
        mentions = _extract_opt_role_mentions(query, "orig")
        slots = _build_slot_opt_irs(["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"])
        # val=7 should have higher score for FeedB slots than FeedA slots
        m7 = next(m for m in mentions if m.value == 7.0)
        s_feedb = next(s for s in slots if s.name == "ProteinFeedB")
        s_feeda = next(s for s in slots if s.name == "ProteinFeedA")
        sc_b, feats_b = _gcg_local_score(m7, s_feedb)
        sc_a, feats_a = _gcg_local_score(m7, s_feeda)
        assert "narrow_left_overlap" in feats_b, "FeedB slot should get narrow_left bonus"
        assert sc_b > sc_a, f"FeedB score {sc_b} should beat FeedA score {sc_a}"

    def test_single_digit_entity_anchor_fires(self):
        """val=20 in 'Machine 1 processes 20' has '1' in narrow_left → RateMachine1 bonus."""
        query = "Machine 1 processes 20 units per hour. Machine 2 processes 35 units per hour."
        mentions = _extract_opt_role_mentions(query, "orig")
        slots = _build_slot_opt_irs(["RateMachine1", "RateMachine2"])
        m20 = next(m for m in mentions if m.value == 20.0)
        s_m1 = next(s for s in slots if s.name == "RateMachine1")
        s_m2 = next(s for s in slots if s.name == "RateMachine2")
        sc_1, feats_1 = _gcg_local_score(m20, s_m1)
        sc_2, _ = _gcg_local_score(m20, s_m2)
        assert "narrow_left_overlap" in feats_1, "Machine1 slot should get narrow_left bonus"
        assert sc_1 > sc_2, f"RateMachine1 score {sc_1} should beat RateMachine2 {sc_2}"

    def test_long_token_does_not_fire(self):
        """'labor' in narrow_left of '5' ('3 labor hours AND 5 board feet') must NOT match
        LaborChair's slot_words, preventing false cross-attribute match."""
        query = "A chair requires 3 labor hours and 5 board feet of wood."
        mentions = _extract_opt_role_mentions(query, "orig")
        slots = _build_slot_opt_irs(["LaborChair", "WoodChair"])
        m5 = next(m for m in mentions if m.value == 5.0)
        s_labor = next(s for s in slots if s.name == "LaborChair")
        _, feats_5_labor = _gcg_local_score(m5, s_labor)
        assert "narrow_left_overlap" not in feats_5_labor, (
            "'labor' is a long token and must not trigger narrow_left_overlap for val=5"
        )

    def test_entity_b_query_assigns_correctly(self):
        """End-to-end: 'Feed B contains 7 protein ...' → ProteinFeedB=7, ProteinFeedA=10."""
        vals, _, _ = _gcg(
            "Feed B contains 7 protein and 15 fat, while Feed A contains 10 protein and 8 fat.",
            ["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        )
        assert vals.get("ProteinFeedB") == pytest.approx(7.0, rel=0.05)
        assert vals.get("ProteinFeedA") == pytest.approx(10.0, rel=0.05)

    def test_product_a_b_assign_correctly(self):
        """End-to-end: 'Product A requires 2 hours, Product B requires 5 hours' correct."""
        vals, _, _ = _gcg(
            "Product B requires 5 heating hours and 1 cooling hour. Product A requires 2 heating hours and 6 cooling hours.",
            ["HeatingA", "CoolingA", "HeatingB", "CoolingB"],
        )
        assert vals.get("HeatingB") == pytest.approx(5.0, rel=0.05)
        assert vals.get("HeatingA") == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Fix 2 – utilisation / utilization → percent
# ---------------------------------------------------------------------------

class TestUtilisationExpectedType:
    """_expected_type must classify utilisation/utilization names as 'percent'."""

    @pytest.mark.parametrize("name", [
        "MinUtilisation",
        "MaxUtilisation",
        "UtilisationA",
        "MinUtilization",
        "MaxUtilization",
        "UtilizationRate",
    ])
    def test_utilisation_is_percent(self, name: str):
        assert _expected_type(name) == "percent", (
            f"_expected_type('{name}') should be 'percent'"
        )

    def test_gcg_accepts_percent_for_utilisation_slots(self):
        """A percent mention should fill MinUtilisation/MaxUtilisation, not be rejected."""
        vals, _, _ = _gcg(
            "Machine utilisation must be at least 30% and at most 90%.",
            ["MinUtilisation", "MaxUtilisation"],
        )
        assert vals.get("MinUtilisation") is not None, "MinUtilisation should be filled"
        assert vals.get("MaxUtilisation") is not None, "MaxUtilisation should be filled"
        mn = float(vals["MinUtilisation"])
        mx = float(vals["MaxUtilisation"])
        assert mn < mx, f"MinUtilisation ({mn}) should be less than MaxUtilisation ({mx})"


# ---------------------------------------------------------------------------
# Fix 3 – "rate" prefix is float, not percent
# ---------------------------------------------------------------------------

class TestRatePrefixExpectedType:
    """'rate' is percent only when it is the FINAL segment of a name."""

    @pytest.mark.parametrize("name,expected", [
        ("DiscountRate", "percent"),
        ("TaxRate", "percent"),
        ("InterestRate", "percent"),
        ("GrowthRate", "percent"),
        ("MinRate", "percent"),
        ("MaxRate", "percent"),
        ("RateMachine1", "float"),   # 'rate' is a prefix, not a suffix
        ("RateMachine2", "float"),
        ("Rate1", "float"),
        ("RateA", "float"),
    ])
    def test_rate_classification(self, name: str, expected: str):
        result = _expected_type(name)
        assert result == expected, (
            f"_expected_type('{name}') = '{result}', expected '{expected}'"
        )

    def test_rate_machine_gcg_not_rejected(self):
        """RateMachine slots must not trigger type_incompatible when value is int/float."""
        vals, _, _ = _gcg(
            "Machine 1 processes 20 units per hour. Machine 2 processes 35 units per hour.",
            ["RateMachine1", "RateMachine2"],
        )
        assert vals.get("RateMachine1") == pytest.approx(20.0, rel=0.05)
        assert vals.get("RateMachine2") == pytest.approx(35.0, rel=0.05)


# ---------------------------------------------------------------------------
# Fix 4 – tight-context cost/profit semantic hints
# ---------------------------------------------------------------------------

class TestTightCostProfitHints:
    """When is_cost_like=True, is_profit_like=False the mention should prefer cost slots."""

    def test_weight_keys_exist(self):
        for key in (
            "tight_cost_match_bonus",
            "tight_cost_mismatch_penalty",
            "tight_profit_match_bonus",
            "tight_profit_mismatch_penalty",
        ):
            assert key in GCG_LOCAL_WEIGHTS, f"Missing key: {key}"

    def test_cost_mention_prefers_cost_slot(self):
        """'cost is 4' (is_cost_like=True, is_profit_like=False) should score higher
        for a CostX slot than a ProfitX slot."""
        query = "Product profit is 8 and cost is 4."
        mentions = _extract_opt_role_mentions(query, "orig")
        slots = _build_slot_opt_irs(["ProfitProductA", "CostProductA"])
        m4 = next(m for m in mentions if m.value == 4.0)
        s_cost = next(s for s in slots if s.name == "CostProductA")
        s_profit = next(s for s in slots if s.name == "ProfitProductA")
        assert m4.is_cost_like, "val=4 should be is_cost_like (tight context has 'cost')"
        assert not m4.is_profit_like, "val=4 should NOT be is_profit_like"
        sc_cost, feats_cost = _gcg_local_score(m4, s_cost)
        sc_profit, feats_profit = _gcg_local_score(m4, s_profit)
        assert "tight_cost_match" in feats_cost, "cost slot should get tight_cost_match bonus"
        assert "tight_cost_mismatch" in feats_profit, "profit slot should get tight_cost_mismatch penalty"
        assert sc_cost > sc_profit, (
            f"CostProductA score ({sc_cost}) should beat ProfitProductA ({sc_profit}) for cost mention"
        )

    def test_profit_cost_gcg_assignment(self):
        """End-to-end: 'Product profit is 8 and cost is 4' → ProfitProductA=8, CostProductA=4."""
        vals, _, _ = _gcg(
            "Product profit is 8 and cost is 4.",
            ["ProfitProductA", "CostProductA"],
        )
        assert vals.get("ProfitProductA") == pytest.approx(8.0, rel=0.05)
        assert vals.get("CostProductA") == pytest.approx(4.0, rel=0.05)


# ---------------------------------------------------------------------------
# Fix 5 – entity coherence global penalty
# ---------------------------------------------------------------------------

class TestEntityCoherenceGlobalPenalty:
    """Consecutive unanchored mentions should be rewarded for same-entity assignment."""

    def test_weight_keys_exist(self):
        assert "entity_coherence_reward" in GCG_GLOBAL_WEIGHTS
        assert "entity_coherence_penalty" in GCG_GLOBAL_WEIGHTS
        assert GCG_GLOBAL_WEIGHTS["entity_coherence_reward"] > 0
        assert GCG_GLOBAL_WEIGHTS["entity_coherence_penalty"] < 0

    def test_fat_values_coherent_with_protein_anchor(self):
        """'Feed B contains 7 protein AND 15 fat' → FatFeedB=15, FatFeedA=8 (not swapped)."""
        vals, _, _ = _gcg(
            "Feed B contains 7 protein and 15 fat, while Feed A contains 10 protein and 8 fat.",
            ["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        )
        assert vals.get("FatFeedB") == pytest.approx(15.0, rel=0.05), (
            f"FatFeedB should be 15 (same clause as ProteinFeedB=7), got {vals.get('FatFeedB')}"
        )
        assert vals.get("FatFeedA") == pytest.approx(8.0, rel=0.05), (
            f"FatFeedA should be 8 (same clause as ProteinFeedA=10), got {vals.get('FatFeedA')}"
        )

    def test_coherence_penalty_fires_for_cross_entity_unanchored(self):
        """_gcg_global_penalty should penalise assignments where an unanchored mention
        goes to a different entity's slot than its anchored predecessor."""
        query = "Feed B contains 7 protein and 15 fat, while Feed A contains 10 protein and 8 fat."
        mentions = _extract_opt_role_mentions(query, "orig")
        slots = _build_slot_opt_irs(["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"])
        slots_by_name = {s.name: s for s in slots}

        m7 = next(m for m in mentions if m.value == 7.0)
        m15 = next(m for m in mentions if m.value == 15.0)
        m10 = next(m for m in mentions if m.value == 10.0)
        m8 = next(m for m in mentions if m.value == 8.0)

        # Correct assignment: B-protein=7, B-fat=15, A-protein=10, A-fat=8
        correct = {
            "ProteinFeedB": m7, "FatFeedB": m15,
            "ProteinFeedA": m10, "FatFeedA": m8,
        }
        # Swapped fat assignment: B-fat=8, A-fat=15 (wrong)
        swapped = {
            "ProteinFeedB": m7, "FatFeedA": m15,
            "ProteinFeedA": m10, "FatFeedB": m8,
        }

        delta_correct, reasons_correct = _gcg_global_penalty(correct, slots_by_name, mentions)
        delta_swapped, reasons_swapped = _gcg_global_penalty(swapped, slots_by_name, mentions)

        # The correct assignment should score higher
        assert delta_correct > delta_swapped, (
            f"Correct assignment delta ({delta_correct}) should exceed swapped ({delta_swapped})"
        )
        # The swapped assignment should contain at least one entity_incoherent reason
        assert any("entity_incoherent" in r for r in reasons_swapped), (
            f"Swapped assignment should have entity_incoherent reason; reasons={reasons_swapped}"
        )
