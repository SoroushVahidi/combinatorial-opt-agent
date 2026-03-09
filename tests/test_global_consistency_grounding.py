"""
Unit tests for the Global Consistency Grounding (GCG) downstream method.

Covers:
- _detect_gcg_sibling_slots: min/max sibling detection
- _score_mention_slot_gcg: percent firewall, polarity mismatch, total/coeff cross-penalty,
  entity anchor, magnitude plausibility
- _gcg_conflict_repair: swaps min/max if min_value > max_value
- _run_global_consistency_grounding: end-to-end coverage, type assignment
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# _detect_gcg_sibling_slots
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectGcgSiblingSlots:
    def test_min_max_prefix(self):
        from tools.nlp4lp_downstream_utility import _detect_gcg_sibling_slots
        sibs = _detect_gcg_sibling_slots(["min_cost", "max_cost", "demand"])
        assert sibs["min_cost"]["max_sibling"] == "max_cost"
        assert sibs["max_cost"]["min_sibling"] == "min_cost"
        assert sibs["demand"]["min_sibling"] is None
        assert sibs["demand"]["max_sibling"] is None

    def test_lower_upper_prefix(self):
        from tools.nlp4lp_downstream_utility import _detect_gcg_sibling_slots
        sibs = _detect_gcg_sibling_slots(["lower_bound", "upper_bound"])
        assert sibs["lower_bound"]["max_sibling"] == "upper_bound"
        assert sibs["upper_bound"]["min_sibling"] == "lower_bound"

    def test_minimum_maximum_prefix(self):
        from tools.nlp4lp_downstream_utility import _detect_gcg_sibling_slots
        sibs = _detect_gcg_sibling_slots(["minimum_demand", "maximum_demand"])
        assert sibs["minimum_demand"]["max_sibling"] == "maximum_demand"
        assert sibs["maximum_demand"]["min_sibling"] == "minimum_demand"

    def test_no_sibling_when_unpaired(self):
        from tools.nlp4lp_downstream_utility import _detect_gcg_sibling_slots
        sibs = _detect_gcg_sibling_slots(["min_profit", "budget"])
        assert sibs["min_profit"]["max_sibling"] is None
        assert sibs["budget"]["min_sibling"] is None

    def test_empty_list(self):
        from tools.nlp4lp_downstream_utility import _detect_gcg_sibling_slots
        assert _detect_gcg_sibling_slots([]) == {}

    def test_multiple_pairs(self):
        from tools.nlp4lp_downstream_utility import _detect_gcg_sibling_slots
        sibs = _detect_gcg_sibling_slots(["min_cost", "max_cost", "min_demand", "max_demand"])
        assert sibs["min_cost"]["max_sibling"] == "max_cost"
        assert sibs["min_demand"]["max_sibling"] == "max_demand"


# ─────────────────────────────────────────────────────────────────────────────
# _score_mention_slot_gcg
# ─────────────────────────────────────────────────────────────────────────────

def _make_mention_opt(value, kind, operator_tags=frozenset(), role_tags=frozenset(),
                      fragment_type="", is_per_unit=False, is_total_like=False,
                      context_tokens=None, sentence_tokens=None):
    """Helper: build a minimal MentionOptIR for scoring tests."""
    from tools.nlp4lp_downstream_utility import MentionOptIR, NumTok
    raw = str(value) + ("%" if kind == "percent" else "")
    tok = NumTok(raw=raw, value=float(value), kind=kind)
    return MentionOptIR(
        mention_id=0,
        value=float(value),
        type_bucket=kind,
        raw_surface=raw,
        role_tags=frozenset(role_tags),
        operator_tags=frozenset(operator_tags),
        unit_tags=frozenset(),
        fragment_type=fragment_type,
        is_per_unit=is_per_unit,
        is_total_like=is_total_like,
        nearby_entity_tokens=frozenset(),
        nearby_resource_tokens=frozenset(),
        nearby_product_tokens=frozenset(),
        context_tokens=list(context_tokens or []),
        sentence_tokens=list(sentence_tokens or []),
        tok=tok,
    )


def _make_slot_opt(name, expected_type, operator_pref=frozenset(),
                   slot_role_tags=frozenset(), is_total_like=False,
                   is_coefficient_like=False, norm_tokens=None):
    """Helper: build a minimal SlotOptIR for scoring tests."""
    from tools.nlp4lp_downstream_utility import SlotOptIR, _normalize_tokens
    toks = norm_tokens if norm_tokens is not None else _normalize_tokens(name)
    return SlotOptIR(
        name=name,
        norm_tokens=toks,
        expected_type=expected_type,
        alias_tokens=set(),
        slot_role_tags=frozenset(slot_role_tags),
        operator_preference=frozenset(operator_pref),
        unit_preference=frozenset(),
        is_objective_like=False,
        is_bound_like=bool(operator_pref),
        is_total_like=is_total_like,
        is_coefficient_like=is_coefficient_like,
    )


class TestScoreMentionSlotGcg:
    def test_type_exact_bonus_percent(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg, GCG_WEIGHTS
        m = _make_mention_opt(50, "percent")
        s = _make_slot_opt("rate", "percent")
        score, feats = _score_mention_slot_gcg(m, s, has_percent_mention=True)
        assert feats.get("type_exact")
        assert score > 4.0

    def test_percent_firewall_applied_when_percent_exists(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg, GCG_WEIGHTS
        m = _make_mention_opt(200, "int")  # non-percent, non-currency → hits firewall check
        s = _make_slot_opt("rate", "percent")
        score_with_firewall, feats = _score_mention_slot_gcg(m, s, has_percent_mention=True)
        score_without_firewall, _ = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        # Firewall should apply a large negative penalty
        assert feats.get("percent_firewall")
        assert score_with_firewall < score_without_firewall - 5.0

    def test_percent_firewall_not_applied_when_no_percent_exists(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(200, "int")
        s = _make_slot_opt("rate", "percent")
        _, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        assert not feats.get("percent_firewall")

    def test_polarity_mismatch_penalty_min_slot_max_mention(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg, GCG_WEIGHTS
        m = _make_mention_opt(10, "int", operator_tags={"max"})
        s = _make_slot_opt("min_demand", "int", operator_pref={"min"})
        score_mismatch, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        # Also score a mention with matching polarity for comparison
        m_match = _make_mention_opt(10, "int", operator_tags={"min"})
        score_match, _ = _score_mention_slot_gcg(m_match, s, has_percent_mention=False)
        assert feats.get("polarity_mismatch_min")
        # Mismatch score must be significantly below the match score
        assert score_mismatch < score_match - 5.0

    def test_polarity_mismatch_penalty_max_slot_min_mention(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(10, "int", operator_tags={"min"})
        s = _make_slot_opt("max_capacity", "int", operator_pref={"max"})
        score_mismatch, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        m_match = _make_mention_opt(10, "int", operator_tags={"max"})
        score_match, _ = _score_mention_slot_gcg(m_match, s, has_percent_mention=False)
        assert feats.get("polarity_mismatch_max")
        assert score_mismatch < score_match - 5.0

    def test_polarity_match_bonus(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(10, "int", operator_tags={"min"})
        s = _make_slot_opt("min_demand", "int", operator_pref={"min"})
        score_match, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        m_mismatch = _make_mention_opt(10, "int", operator_tags={"max"})
        score_mismatch, _ = _score_mention_slot_gcg(m_mismatch, s, has_percent_mention=False)
        assert feats.get("polarity_match")
        assert score_match > score_mismatch + 5.0  # polarity match >> polarity mismatch

    def test_total_to_coeff_penalty(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(1000, "currency", is_total_like=True)
        s = _make_slot_opt("unit_profit", "currency", is_coefficient_like=True)
        score, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        assert feats.get("total_to_coeff_conflict")

    def test_coeff_to_total_penalty(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(5, "currency", is_per_unit=True)
        s = _make_slot_opt("total_budget", "currency", is_total_like=True)
        score, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        assert feats.get("coeff_to_total_conflict")

    def test_total_match_bonus(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m_total = _make_mention_opt(1000, "currency", is_total_like=True)
        m_perunit = _make_mention_opt(1000, "currency", is_per_unit=True)
        s = _make_slot_opt("total_budget", "currency", is_total_like=True)
        score_total, feats_total = _score_mention_slot_gcg(m_total, s, has_percent_mention=False)
        score_perunit, feats_perunit = _score_mention_slot_gcg(m_perunit, s, has_percent_mention=False)
        assert feats_total.get("total_match")
        assert feats_perunit.get("coeff_to_total_conflict")
        assert score_total > score_perunit + 3.0

    def test_entity_anchor_bonus(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg, _normalize_tokens
        # Slot name "profit" has norm_token "profit"; mention context contains "profit"
        m = _make_mention_opt(50, "currency", context_tokens=["profit", "per", "unit"])
        s = _make_slot_opt("profit", "currency", norm_tokens=["profit"])
        score_with, feats_with = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        m_no_ctx = _make_mention_opt(50, "currency", context_tokens=["cost", "total"])
        score_without, _ = _score_mention_slot_gcg(m_no_ctx, s, has_percent_mention=False)
        assert feats_with.get("entity_anchor")
        assert score_with > score_without + 1.5

    def test_percent_magnitude_penalty(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(150, "percent")  # 150% is implausible
        s = _make_slot_opt("rate", "percent")
        score, feats = _score_mention_slot_gcg(m, s, has_percent_mention=True)
        assert feats.get("percent_magnitude_bad")

    def test_count_decimal_penalty(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(3.5, "float")  # decimal value for an int slot
        s = _make_slot_opt("numItems", "int")
        score, feats = _score_mention_slot_gcg(m, s, has_percent_mention=False)
        assert feats.get("count_decimal_bad")

    def test_type_incompatible_hard_veto(self):
        from tools.nlp4lp_downstream_utility import _score_mention_slot_gcg
        m = _make_mention_opt(50, "percent")
        s = _make_slot_opt("budget", "currency")
        score, feats = _score_mention_slot_gcg(m, s, has_percent_mention=True)
        assert feats.get("type_incompatible")
        assert score < -1e7


# ─────────────────────────────────────────────────────────────────────────────
# _gcg_conflict_repair
# ─────────────────────────────────────────────────────────────────────────────

def _make_mention_with_value(mid, value, kind="int"):
    from tools.nlp4lp_downstream_utility import MentionOptIR, NumTok
    tok = NumTok(raw=str(value), value=float(value), kind=kind)
    return MentionOptIR(
        mention_id=mid, value=float(value), type_bucket=kind, raw_surface=str(value),
        role_tags=frozenset(), operator_tags=frozenset(), unit_tags=frozenset(),
        fragment_type="", is_per_unit=False, is_total_like=False,
        nearby_entity_tokens=frozenset(), nearby_resource_tokens=frozenset(),
        nearby_product_tokens=frozenset(), context_tokens=[], sentence_tokens=[], tok=tok,
    )


class TestGcgConflictRepair:
    def test_swap_when_min_greater_than_max(self):
        from tools.nlp4lp_downstream_utility import _gcg_conflict_repair, _detect_gcg_sibling_slots
        m_large = _make_mention_with_value(0, 20)
        m_small = _make_mention_with_value(1, 5)
        assignments = {"min_cost": m_large, "max_cost": m_small}
        siblings = _detect_gcg_sibling_slots(["min_cost", "max_cost"])
        repaired = _gcg_conflict_repair(assignments, siblings)
        # After repair, min_cost should have value 5 (smaller) and max_cost value 20 (larger)
        assert repaired["min_cost"].value == 5.0
        assert repaired["max_cost"].value == 20.0

    def test_no_swap_when_already_correct(self):
        from tools.nlp4lp_downstream_utility import _gcg_conflict_repair, _detect_gcg_sibling_slots
        m_small = _make_mention_with_value(0, 5)
        m_large = _make_mention_with_value(1, 20)
        assignments = {"min_cost": m_small, "max_cost": m_large}
        siblings = _detect_gcg_sibling_slots(["min_cost", "max_cost"])
        repaired = _gcg_conflict_repair(assignments, siblings)
        assert repaired["min_cost"].value == 5.0
        assert repaired["max_cost"].value == 20.0

    def test_no_swap_when_one_slot_missing(self):
        from tools.nlp4lp_downstream_utility import _gcg_conflict_repair, _detect_gcg_sibling_slots
        m = _make_mention_with_value(0, 20)
        assignments = {"min_cost": m}
        siblings = _detect_gcg_sibling_slots(["min_cost", "max_cost"])
        repaired = _gcg_conflict_repair(assignments, siblings)
        assert repaired["min_cost"].value == 20.0

    def test_lower_upper_swap(self):
        from tools.nlp4lp_downstream_utility import _gcg_conflict_repair, _detect_gcg_sibling_slots
        m_large = _make_mention_with_value(0, 100)
        m_small = _make_mention_with_value(1, 10)
        assignments = {"lower_bound": m_large, "upper_bound": m_small}
        siblings = _detect_gcg_sibling_slots(["lower_bound", "upper_bound"])
        repaired = _gcg_conflict_repair(assignments, siblings)
        assert repaired["lower_bound"].value == 10.0
        assert repaired["upper_bound"].value == 100.0

    def test_empty_assignments(self):
        from tools.nlp4lp_downstream_utility import _gcg_conflict_repair, _detect_gcg_sibling_slots
        siblings = _detect_gcg_sibling_slots(["min_x", "max_x"])
        repaired = _gcg_conflict_repair({}, siblings)
        assert repaired == {}


# ─────────────────────────────────────────────────────────────────────────────
# _run_global_consistency_grounding (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunGlobalConsistencyGrounding:
    def test_fills_slots_from_query(self):
        from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding
        fv, fm, fi = _run_global_consistency_grounding(
            "The budget is 500 dollars and the demand is 30 units",
            "orig",
            ["budget", "demand"],
        )
        assert "budget" in fv or "demand" in fv  # at least one filled

    def test_empty_scalar_list_returns_empty(self):
        from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding
        fv, fm, fi = _run_global_consistency_grounding(
            "The cost is 100", "orig", []
        )
        assert fv == {}
        assert fm == {}
        assert fi == {}

    def test_query_with_no_numbers_returns_empty(self):
        from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding
        fv, fm, fi = _run_global_consistency_grounding(
            "Maximize the total profit subject to demand constraints",
            "orig",
            ["profit", "demand"],
        )
        assert fv == {}

    def test_percent_slot_gets_percent_value(self):
        from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding
        fv, fm, fi = _run_global_consistency_grounding(
            "The interest rate is 5% and the total cost is 1000 dollars",
            "orig",
            ["interestRate", "totalCost"],
        )
        rate = fv.get("interestRate")
        # 5% should be recognised as percent kind and assigned to interestRate
        assert rate is not None
        # The value should be the percentage (5.0 or 0.05 depending on normalisation)
        assert rate in (5.0, 0.05)

    def test_min_max_conflict_repair_via_end_to_end(self):
        """After assignment, if min_value > max_value the two should be swapped."""
        from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding
        fv, fm, fi = _run_global_consistency_grounding(
            "The minimum cost is 5 and the maximum cost is 20",
            "orig",
            ["min_cost", "max_cost"],
        )
        min_v = fv.get("min_cost")
        max_v = fv.get("max_cost")
        if min_v is not None and max_v is not None:
            # After conflict repair, min should always be ≤ max
            assert min_v <= max_v

    def test_returns_three_tuple(self):
        from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding
        result = _run_global_consistency_grounding("cost is 10", "orig", ["cost"])
        assert isinstance(result, tuple) and len(result) == 3
        fv, fm, fi = result
        assert isinstance(fv, dict)
        assert isinstance(fm, dict)
        assert isinstance(fi, dict)
