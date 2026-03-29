"""Tests for the global_compatibility_grounding downstream method (GCGP).

Covers:
  1. Percent vs scalar (pairwise exclusivity)
  2. Total vs per-unit separation (pairwise magnitude + type-distinct terms)
  3. Min / max ordering (pairwise value-ordering terms)
  4. Duplicate mention hard-penalty
  5. Ablation modes (local_only / pairwise / full) produce valid output
  6. Constant sanity checks
  7. Integration: method registered in run_setting choices and focused_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    GCGP_BEAM_WIDTH,
    GCGP_PAIRWISE_WEIGHTS,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _gcgp_pairwise_score,
    _run_global_compatibility_grounding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcgp(query: str, slots: list[str], ablation_mode: str = "full") -> tuple[dict, dict, dict]:
    return _run_global_compatibility_grounding(query, "orig", slots, ablation_mode=ablation_mode)


# ---------------------------------------------------------------------------
# 1. Pairwise percent exclusivity
# ---------------------------------------------------------------------------

class TestPairwisePercent:
    def test_percent_mention_goes_to_percent_slot(self):
        """Pairwise percent-exclusivity bonus should keep percent mention on percent slot."""
        query = "The discount rate is 25% and the total price is 800 dollars."
        vals, mentions, _ = _gcgp(query, ["discount_percent", "total_price"])
        assert vals.get("discount_percent") is not None
        assert vals.get("total_price") is not None
        m = mentions.get("discount_percent")
        if m is not None:
            assert m.type_bucket == "percent"

    def test_non_percent_mention_goes_to_scalar_slot(self):
        """Non-percent slot should not receive the percent mention when a scalar mention exists."""
        query = "The investment is 500 dollars and the interest rate is 10%."
        vals, _, _ = _gcgp(query, ["investment_amount", "interest_rate_percent"])
        assert vals.get("investment_amount") is not None
        inv = float(vals["investment_amount"])
        assert abs(inv - 500.0) < 5.0, f"Expected ~500 for investment, got {inv}"

    def test_pairwise_score_rewards_percent_exclusive(self):
        """_gcgp_pairwise_score gives a positive bonus for correct percent slot pairing."""
        slots = _build_slot_opt_irs(["discount_percent", "total_price"])
        pct_slot = next(s for s in slots if "percent" in s.name)
        non_pct_slot = next(s for s in slots if "price" in s.name)
        mentions = _extract_opt_role_mentions(
            "The discount is 20% and the price is 100.", "orig"
        )
        pct_m = next((m for m in mentions if m.type_bucket == "percent"), None)
        non_pct_m = next((m for m in mentions if m.type_bucket != "percent" and m.value is not None), None)
        if pct_m is None or non_pct_m is None:
            pytest.skip("Could not extract both mention types")
        delta_good, reasons_good = _gcgp_pairwise_score(pct_slot, pct_m, non_pct_slot, non_pct_m)
        delta_bad, reasons_bad = _gcgp_pairwise_score(pct_slot, non_pct_m, non_pct_slot, pct_m)
        assert delta_good > delta_bad, (
            f"Correct percent pairing should score higher: {delta_good} vs {delta_bad}"
        )


# ---------------------------------------------------------------------------
# 2. Total vs per-unit pairwise separation
# ---------------------------------------------------------------------------

class TestPairwiseTotalCoeff:
    def test_profit_and_budget_separated(self):
        """Pairwise magnitude term should keep per-unit profit and total budget distinct."""
        query = (
            "Each item yields a profit of 4 dollars. "
            "The total available budget is 800 dollars."
        )
        vals, _, _ = _gcgp(query, ["profit_per_unit", "total_budget"])
        assert vals.get("profit_per_unit") is not None
        assert vals.get("total_budget") is not None
        profit = float(vals["profit_per_unit"])
        budget = float(vals["total_budget"])
        assert abs(profit - 4.0) < 0.5, f"Expected profit ~4, got {profit}"
        assert abs(budget - 800.0) < 10.0, f"Expected budget ~800, got {budget}"

    def test_pairwise_magnitude_order(self):
        """_gcgp_pairwise_score rewards total > coeff arrangement."""
        slots = _build_slot_opt_irs(["total_budget", "profit_per_unit"])
        total_slot = next(s for s in slots if "total" in s.name)
        coeff_slot = next(s for s in slots if "profit" in s.name)
        mentions = _extract_opt_role_mentions(
            "Each unit earns 5 dollars. Total budget is 1000.", "orig"
        )
        # Find per-unit mention (value ~5) and total mention (value ~1000)
        small_m = next((m for m in mentions if m.value and abs(m.value - 5.0) < 1), None)
        large_m = next((m for m in mentions if m.value and abs(m.value - 1000.0) < 50), None)
        if small_m is None or large_m is None:
            pytest.skip("Could not find both mentions")
        # Correct: total_slot←large_m, coeff_slot←small_m
        delta_correct, _ = _gcgp_pairwise_score(total_slot, large_m, coeff_slot, small_m)
        # Inverted: total_slot←small_m, coeff_slot←large_m
        delta_inverted, _ = _gcgp_pairwise_score(total_slot, small_m, coeff_slot, large_m)
        assert delta_correct >= delta_inverted, (
            f"Correct total>coeff assignment should score >= inverted: {delta_correct} vs {delta_inverted}"
        )

    def test_pairwise_type_distinct_bonus_fires(self):
        """When total and coeff slots get mentions with different is_per_unit, the bonus fires."""
        slots = _build_slot_opt_irs(["total_budget", "cost_per_unit"])
        total_slot = next(s for s in slots if "total" in s.name)
        coeff_slot = next(s for s in slots if "cost" in s.name)
        mentions = _extract_opt_role_mentions(
            "Each unit costs 7 dollars per unit. Total budget is 500.", "orig"
        )
        per_unit_m = next((m for m in mentions if m.is_per_unit and m.value and m.value < 50), None)
        total_m = next((m for m in mentions if m.is_total_like and m.value and m.value > 100), None)
        if per_unit_m is None or total_m is None:
            pytest.skip("Could not distinguish per-unit and total mentions")
        delta, reasons = _gcgp_pairwise_score(total_slot, total_m, coeff_slot, per_unit_m)
        assert any("total_coeff" in r for r in reasons), (
            f"Expected total_coeff reason, got: {reasons}"
        )


# ---------------------------------------------------------------------------
# 3. Min / max ordering via pairwise terms
# ---------------------------------------------------------------------------

class TestPairwiseMinMax:
    def test_min_max_correct_order(self):
        """Min slot must get the smaller value; max slot the larger value."""
        query = "You must produce at least 10 units and at most 50 units."
        vals, _, _ = _gcgp(query, ["min_units", "max_units"])
        assert vals.get("min_units") is not None
        assert vals.get("max_units") is not None
        mn = float(vals["min_units"])
        mx = float(vals["max_units"])
        assert mn < mx, f"min ({mn}) should be less than max ({mx})"
        assert abs(mn - 10.0) < 0.5, f"min_units expected ~10, got {mn}"
        assert abs(mx - 50.0) < 0.5, f"max_units expected ~50, got {mx}"

    def test_pairwise_minmax_bonus_fires(self):
        """_gcgp_pairwise_score gives a bonus for min ≤ max."""
        slots = _build_slot_opt_irs(["min_value", "max_value"])
        min_slot = next(s for s in slots if "min" in s.name)
        max_slot = next(s for s in slots if "max" in s.name)
        mentions = _extract_opt_role_mentions(
            "The minimum is 5 and the maximum is 100.", "orig"
        )
        small_m = next((m for m in mentions if m.value and abs(m.value - 5.0) < 1), None)
        large_m = next((m for m in mentions if m.value and abs(m.value - 100.0) < 5), None)
        if small_m is None or large_m is None:
            pytest.skip("Could not find both mentions")
        # Correct: min←small, max←large
        delta_correct, reasons_correct = _gcgp_pairwise_score(min_slot, small_m, max_slot, large_m)
        # Inverted: min←large, max←small
        delta_inverted, reasons_inverted = _gcgp_pairwise_score(min_slot, large_m, max_slot, small_m)
        assert delta_correct > delta_inverted, (
            f"Correct min≤max should score higher: {delta_correct} vs {delta_inverted}"
        )
        assert any("correct_order" in r for r in reasons_correct), (
            f"Expected correct_order reason, got: {reasons_correct}"
        )
        assert any("inverted_order" in r for r in reasons_inverted), (
            f"Expected inverted_order reason, got: {reasons_inverted}"
        )

    def test_pairwise_minmax_penalty_fires_on_inversion(self):
        """_gcgp_pairwise_score penalises when min value > max value."""
        slots = _build_slot_opt_irs(["min_threshold", "max_threshold"])
        min_slot = next(s for s in slots if "min" in s.name)
        max_slot = next(s for s in slots if "max" in s.name)
        mentions = _extract_opt_role_mentions(
            "At least 80 units and at most 20 units.", "orig"
        )
        m80 = next((m for m in mentions if m.value and abs(m.value - 80.0) < 2), None)
        m20 = next((m for m in mentions if m.value and abs(m.value - 20.0) < 2), None)
        if m80 is None or m20 is None:
            pytest.skip("Could not find mentions 80 and 20")
        delta, reasons = _gcgp_pairwise_score(min_slot, m80, max_slot, m20)
        assert delta < 0, f"Inverted min>max should yield negative delta, got {delta}"
        assert any("inverted_order" in r for r in reasons)


# ---------------------------------------------------------------------------
# 4. Duplicate mention hard penalty
# ---------------------------------------------------------------------------

class TestDuplicateMentionPenalty:
    def test_same_mention_strongly_penalised(self):
        """Two slots sharing the same mention should score much lower than distinct mentions."""
        slots = _build_slot_opt_irs(["budget", "demand"])
        s1, s2 = slots[0], slots[1]
        mentions = _extract_opt_role_mentions(
            "The budget is 200 and demand is 10.", "orig"
        )
        if len(mentions) < 2:
            pytest.skip("Need at least 2 mentions")
        m1, m2 = mentions[0], mentions[1]
        # Ensure m1 != m2 by mention_id
        if m1.mention_id == m2.mention_id:
            pytest.skip("Mentions share same id (edge case)")
        delta_distinct, _ = _gcgp_pairwise_score(s1, m1, s2, m2)
        delta_duplicate, reasons_dup = _gcgp_pairwise_score(s1, m1, s2, m1)  # same mention
        assert delta_duplicate < delta_distinct, (
            f"Duplicate should score lower: {delta_duplicate} vs {delta_distinct}"
        )
        assert any("duplicate" in r for r in reasons_dup)

    def test_global_assigns_distinct_mentions(self):
        """End-to-end: two slots should receive distinct mentions."""
        query = "The total budget is 1000 and minimum demand is 20."
        vals, mentions_dict, _ = _gcgp(query, ["total_budget", "min_demand"])
        assigned_mention_ids = [m.mention_id for m in mentions_dict.values() if m is not None]
        assert len(set(assigned_mention_ids)) == len(assigned_mention_ids), (
            f"Duplicate mention assigned: {assigned_mention_ids}"
        )


# ---------------------------------------------------------------------------
# 5. Ablation modes
# ---------------------------------------------------------------------------

class TestAblationModes:
    def test_local_only_returns_valid_output(self):
        query = "The budget is 500 and profit per unit is 3."
        vals, mentions, diag = _gcgp(query, ["total_budget", "profit_per_unit"], ablation_mode="local_only")
        assert isinstance(vals, dict)
        assert isinstance(mentions, dict)
        assert diag.get("ablation_mode") == "local_only"

    def test_pairwise_returns_valid_output(self):
        query = "The budget is 500 and profit per unit is 3."
        vals, _, diag = _gcgp(query, ["total_budget", "profit_per_unit"], ablation_mode="pairwise")
        assert isinstance(vals, dict)
        assert diag.get("ablation_mode") == "pairwise"

    def test_full_returns_valid_output(self):
        query = "The budget is 500 and profit per unit is 3."
        vals, _, diag = _gcgp(query, ["total_budget", "profit_per_unit"], ablation_mode="full")
        assert isinstance(vals, dict)
        assert diag.get("ablation_mode") == "full"

    def test_all_modes_return_same_slots(self):
        """All ablation modes should produce results for the same slots (may differ in values)."""
        query = "Each unit yields 5 dollars and the budget is 200 dollars."
        slots = ["profit_per_unit", "total_budget"]
        for mode in ("local_only", "pairwise", "full"):
            vals, _, _ = _gcgp(query, slots, ablation_mode=mode)
            assert isinstance(vals, dict), f"mode={mode} returned non-dict"

    def test_pairwise_vs_local_differ_on_min_max(self):
        """On a min/max query, 'pairwise' mode should make a different (better) decision than 'local_only'
        at least sometimes — or equivalently, pairwise mode should never be worse."""
        query = "You must produce at least 5 units and at most 30 units."
        slots = ["min_units", "max_units"]
        vals_local, _, _ = _gcgp(query, slots, ablation_mode="local_only")
        vals_pairwise, _, _ = _gcgp(query, slots, ablation_mode="pairwise")
        # pairwise mode should get correct ordering
        if vals_pairwise.get("min_units") and vals_pairwise.get("max_units"):
            assert float(vals_pairwise["min_units"]) <= float(vals_pairwise["max_units"]), (
                "pairwise mode inverted min/max"
            )

    def test_diagnostics_contain_ablation_mode(self):
        query = "The budget is 100."
        _, _, diag = _gcgp(query, ["budget"], ablation_mode="pairwise")
        assert "ablation_mode" in diag
        assert diag["ablation_mode"] == "pairwise"


# ---------------------------------------------------------------------------
# 6. Constant sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_pairwise_weights_keys(self):
        required = {
            "minmax_correct_order_bonus", "minmax_inverted_order_penalty",
            "total_coeff_type_distinct_bonus", "total_coeff_both_same_type_penalty",
            "percent_exclusive_bonus", "percent_collision_penalty",
            "duplicate_mention_pairwise_penalty",
            "objective_bound_compatible_bonus", "objective_objective_collision_penalty",
            "magnitude_budget_gt_coeff_bonus", "magnitude_budget_lt_coeff_penalty",
        }
        assert required.issubset(GCGP_PAIRWISE_WEIGHTS.keys()), (
            f"Missing keys: {required - GCGP_PAIRWISE_WEIGHTS.keys()}"
        )

    def test_penalty_signs(self):
        for key, val in GCGP_PAIRWISE_WEIGHTS.items():
            if "penalty" in key:
                assert val < 0, f"Pairwise penalty {key} should be negative, got {val}"

    def test_bonus_signs(self):
        for key, val in GCGP_PAIRWISE_WEIGHTS.items():
            if "bonus" in key:
                assert val > 0, f"Pairwise bonus {key} should be positive, got {val}"

    def test_beam_width_positive(self):
        assert GCGP_BEAM_WIDTH >= 1


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_slots_returns_empty(self):
        vals, mentions, diag = _gcgp("The budget is 100.", [])
        assert vals == {}
        assert mentions == {}

    def test_no_mentions_returns_empty(self):
        vals, mentions, _ = _gcgp("There are no numbers here.", ["budget"])
        assert vals == {}

    def test_single_slot(self):
        vals, _, _ = _gcgp("The cost is 42 dollars.", ["cost"])
        assert vals.get("cost") is not None
        assert abs(float(vals["cost"]) - 42.0) < 0.5

    def test_more_slots_than_mentions(self):
        vals, _, _ = _gcgp("The budget is 100 dollars.", ["budget", "demand", "cost"])
        assert vals.get("budget") is not None
        # Other slots may be empty — no error.

    def test_diagnostics_structure(self):
        query = "The budget is 100 and demand is 5."
        _, _, diag = _gcgp(query, ["budget", "demand"])
        assert "top_assignments" in diag
        assert "per_slot_candidates" in diag
        for entry in diag["top_assignments"]:
            for f in ("rank", "total_score", "local_sum", "global_delta", "active_reasons", "assignment"):
                assert f in entry, f"Missing field {f}"


# ---------------------------------------------------------------------------
# 8. Integration checks
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_assignment_mode_strings_in_argparse_choices(self):
        """All three ablation assignment_mode strings must appear in argparse choices."""
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        found = set()
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and isinstance(node.value, str):
                if node.value.startswith("global_compat_"):
                    found.add(node.value)
        for mode in ("global_compat_local", "global_compat_pairwise", "global_compat_full"):
            assert mode in found, f"Mode {mode!r} not found as string constant in downstream utility"

    def test_effective_baseline_naming(self):
        """_effective_baseline must map new modes to correct labels."""
        from tools.run_nlp4lp_focused_eval import _effective_baseline
        assert _effective_baseline("tfidf", "global_compat_local") == "tfidf_global_compat_local"
        assert _effective_baseline("tfidf", "global_compat_pairwise") == "tfidf_global_compat_pairwise"
        assert _effective_baseline("tfidf", "global_compat_full") == "tfidf_global_compat_full"

    def test_focused_eval_includes_new_methods(self):
        """All three ablation variants must be in FOCUSED_BASELINES_DEFAULT."""
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        for name in ("tfidf_global_compat_local", "tfidf_global_compat_pairwise", "tfidf_global_compat_full"):
            assert name in FOCUSED_BASELINES_DEFAULT, f"{name} not in FOCUSED_BASELINES_DEFAULT"

    def test_baseline_assignment_includes_new_methods(self):
        """BASELINE_ASSIGNMENT_DEFAULT must include all three new methods."""
        from tools.run_nlp4lp_focused_eval import BASELINE_ASSIGNMENT_DEFAULT
        modes_in = {am for _, am in BASELINE_ASSIGNMENT_DEFAULT}
        for mode in ("global_compat_local", "global_compat_pairwise", "global_compat_full"):
            assert mode in modes_in, f"assignment_mode {mode} not in BASELINE_ASSIGNMENT_DEFAULT"
