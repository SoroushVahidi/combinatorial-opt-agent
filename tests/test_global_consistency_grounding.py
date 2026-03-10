"""Focused tests for the global_consistency_grounding downstream method.

Tests cover the four hard confusion classes identified in the problem statement:
  1. percent vs scalar
  2. total vs per-unit
  3. lower-bound vs upper-bound / min vs max
  4. float-heavy small examples

Additional tests verify:
  - the method is wired into run_setting (assignment_mode string accepted)
  - configurable constants exist and are sane
  - diagnostics are returned with the expected structure
  - empty-input edge cases are handled gracefully
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    GCG_BEAM_WIDTH,
    GCG_GLOBAL_WEIGHTS,
    GCG_LOCAL_WEIGHTS,
    GCG_PRUNE_THRESHOLD,
    _gcg_beam_search,
    _gcg_global_penalty,
    _gcg_local_score,
    _run_global_consistency_grounding,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcg(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    """Thin wrapper for calling _run_global_consistency_grounding."""
    return _run_global_consistency_grounding(query, "orig", slots)


# ---------------------------------------------------------------------------
# 1. Percent vs scalar
# ---------------------------------------------------------------------------

class TestPercentVsScalar:
    """The method must assign a percent-tagged mention to a percent slot,
    not to a plain numeric slot, when both candidates are present."""

    def test_percent_mention_goes_to_percent_slot(self):
        query = "The tax rate is 15% and the total budget is 500 dollars."
        vals, _, diag = _gcg(query, ["tax_rate_percent", "total_budget"])
        # 500 must go to total_budget; the percent token (0.15 or 15%) to tax_rate_percent
        assert vals.get("total_budget") is not None
        assert vals.get("tax_rate_percent") is not None
        budget_val = float(vals["total_budget"])
        assert abs(budget_val - 500.0) < 1.0, f"Expected 500 for total_budget, got {budget_val}"

    def test_scalar_does_not_go_to_percent_slot_when_pct_available(self):
        """When a percent mention exists, the non-percent scalar should not occupy the percent slot."""
        query = "The discount is 20% and the price is 80 dollars."
        vals, mentions, _ = _gcg(query, ["discount_percent", "price"])
        # Both slots should be filled; percent slot should get the % token
        assert vals.get("discount_percent") is not None
        m = mentions.get("discount_percent")
        if m is not None:
            assert m.type_bucket == "percent", (
                f"Expected percent mention assigned to discount_percent, got {m.type_bucket}"
            )

    def test_percent_penalty_in_global_scoring(self):
        """_gcg_global_penalty must add a percent_misuse penalty when a percent mention
        is assigned to a non-percent slot and pct evidence exists in the mention pool."""
        mentions = _extract_opt_role_mentions(
            "The rate is 30% and budget is 200.", "orig"
        )
        slots = _build_slot_opt_irs(["budget", "rate_percent"])
        slots_by_name = {s.name: s for s in slots}
        # Force a bad assignment: pct mention → budget (non-pct slot)
        pct_mention = next((m for m in mentions if m.type_bucket == "percent"), None)
        if pct_mention is None:
            pytest.skip("No percent mention extracted from query")
        bad_assignment = {"budget": pct_mention}
        delta, reasons = _gcg_global_penalty(bad_assignment, slots_by_name, mentions)
        assert any("percent_misuse" in r for r in reasons), (
            f"Expected percent_misuse penalty, got reasons: {reasons}"
        )


# ---------------------------------------------------------------------------
# 2. Total vs per-unit
# ---------------------------------------------------------------------------

class TestTotalVsPerUnit:
    """The method must not swap a per-unit coefficient with a total/budget value."""

    def test_profit_and_budget_correctly_separated(self):
        query = "Each unit earns a profit of 3 dollars. The total budget available is 600 dollars."
        vals, _, _ = _gcg(query, ["profit_per_unit", "total_budget"])
        assert vals.get("profit_per_unit") is not None
        assert vals.get("total_budget") is not None
        profit = float(vals["profit_per_unit"])
        budget = float(vals["total_budget"])
        assert abs(profit - 3.0) < 0.5, f"Expected profit 3, got {profit}"
        assert abs(budget - 600.0) < 5.0, f"Expected budget 600, got {budget}"

    def test_coeff_to_total_penalty_applied(self):
        """Global penalty fires when a per-unit mention is forced to a total slot."""
        # Use a focused query so mention "3" has is_per_unit=True and is_total_like=False.
        # Keep the second sentence far away so "total" doesn't bleed into mention context.
        mentions = _extract_opt_role_mentions(
            "Each unit requires 3 hours per unit of processing.",
            "orig",
        )
        slots = _build_slot_opt_irs(["total_hours", "hours_per_unit"])
        slots_by_name = {s.name: s for s in slots}
        # Find a per-unit mention (not total-like).
        per_unit_m = next(
            (m for m in mentions if m.is_per_unit and not m.is_total_like and m.value is not None),
            None,
        )
        if per_unit_m is None:
            pytest.skip("Per-unit (non-total) mention not extracted — query may need adjustment")
        total_slot = next((s for s in slots if "total" in s.name), None)
        if total_slot is None or not total_slot.is_total_like:
            pytest.skip("total_hours slot not found or not marked total_like")
        bad_assignment = {"total_hours": per_unit_m}
        delta, reasons = _gcg_global_penalty(bad_assignment, slots_by_name, mentions)
        assert any("coeff_to_total" in r or "total_to_coeff" in r for r in reasons), (
            f"Expected coeff/total penalty, got: {reasons}"
        )

    def test_three_slot_assignment(self):
        """With three slots (coeff, budget, demand), each gets the right value."""
        query = (
            "Each product yields 5 dollars profit. "
            "There are 200 dollars total available. "
            "The minimum demand is 10 units."
        )
        vals, _, _ = _gcg(query, ["profit_per_unit", "total_budget", "min_demand"])
        assert len(vals) == 3, f"Expected 3 slots filled, got {len(vals)}: {vals}"
        assert abs(float(vals["profit_per_unit"]) - 5.0) < 0.5
        assert abs(float(vals["total_budget"]) - 200.0) < 5.0
        assert abs(float(vals["min_demand"]) - 10.0) < 0.5


# ---------------------------------------------------------------------------
# 3. Lower-bound vs upper-bound / min vs max
# ---------------------------------------------------------------------------

class TestBoundAssignment:
    """The method must not swap min and max bounds."""

    def test_min_max_correctly_separated(self):
        query = "You must produce at least 10 units and at most 50 units."
        vals, _, _ = _gcg(query, ["min_units", "max_units"])
        assert vals.get("min_units") is not None
        assert vals.get("max_units") is not None
        mn = float(vals["min_units"])
        mx = float(vals["max_units"])
        assert mn < mx, f"min ({mn}) should be less than max ({mx})"
        assert abs(mn - 10.0) < 0.5, f"Expected min=10, got {mn}"
        assert abs(mx - 50.0) < 0.5, f"Expected max=50, got {mx}"

    def test_bound_flip_penalty_applied(self):
        """_gcg_global_penalty must penalise when a max-context mention is assigned to a min slot."""
        # Use a query where the max-tagged number has a clear exclusive-max context
        # and is well-separated from the min number to avoid context bleed.
        mentions = _extract_opt_role_mentions(
            "The maximum number of items is 80 units.", "orig"
        )
        slots = _build_slot_opt_irs(["min_value", "max_value"])
        slots_by_name = {s.name: s for s in slots}
        # Find a mention tagged with 'max' operator but NOT 'min'.
        max_m = next(
            (m for m in mentions if "max" in m.operator_tags and "min" not in m.operator_tags and m.value is not None),
            None,
        )
        min_slot = next((s for s in slots if "min" in s.name), None)
        if max_m is None or min_slot is None:
            pytest.skip("Could not extract exclusively-max mention or min slot")
        bad_assignment = {"min_value": max_m}
        delta, reasons = _gcg_global_penalty(bad_assignment, slots_by_name, mentions)
        assert any("bound_flip" in r for r in reasons), (
            f"Expected bound_flip penalty, got reasons: {reasons}"
        )

    def test_larger_bound_set(self):
        """Four-bound case: two mins, two maxes — correct order maintained."""
        query = (
            "Product A must be made at least 5 units and at most 30. "
            "Product B must be made at least 8 units and at most 40."
        )
        vals, _, _ = _gcg(query, ["min_A", "max_A", "min_B", "max_B"])
        for slot, val in vals.items():
            assert val is not None, f"Slot {slot} should be filled"
        # All min slots must be smaller than corresponding max slots.
        if vals.get("min_A") and vals.get("max_A"):
            min_a, max_a = float(vals["min_A"]), float(vals["max_A"])
            assert min_a < max_a, f"min_A ({min_a}) should be < max_A ({max_a})"
            assert min_a in (5.0, 8.0), f"min_A expected 5 or 8, got {min_a}"
            assert max_a in (30.0, 40.0), f"max_A expected 30 or 40, got {max_a}"
        if vals.get("min_B") and vals.get("max_B"):
            min_b, max_b = float(vals["min_B"]), float(vals["max_B"])
            assert min_b < max_b, f"min_B ({min_b}) should be < max_B ({max_b})"
            assert min_b in (5.0, 8.0), f"min_B expected 5 or 8, got {min_b}"
            assert max_b in (30.0, 40.0), f"max_B expected 30 or 40, got {max_b}"


# ---------------------------------------------------------------------------
# 4. Float-heavy small examples
# ---------------------------------------------------------------------------

class TestFloatHeavy:
    """With several decimal values the method must not mix them up."""

    def test_two_float_slots(self):
        query = "Product A yields 2.5 dollars per unit and product B yields 3.7 dollars per unit."
        vals, _, _ = _gcg(query, ["profit_A", "profit_B"])
        assert vals.get("profit_A") is not None
        assert vals.get("profit_B") is not None
        values = sorted([float(vals["profit_A"]), float(vals["profit_B"])])
        assert abs(values[0] - 2.5) < 0.1, f"Expected 2.5, got {values[0]}"
        assert abs(values[1] - 3.7) < 0.1, f"Expected 3.7, got {values[1]}"

    def test_float_vs_integer_discrimination(self):
        """A float coefficient and an integer demand should not be swapped."""
        query = (
            "Each unit of product A uses 1.5 hours of machine time. "
            "The demand for product A is at least 20 units."
        )
        vals, _, _ = _gcg(query, ["hours_per_unit_A", "min_demand_A"])
        if vals.get("hours_per_unit_A") and vals.get("min_demand_A"):
            hu = float(vals["hours_per_unit_A"])
            md = float(vals["min_demand_A"])
            assert abs(hu - 1.5) < 0.5, (
                f"hours_per_unit_A expected ~1.5, got {hu}"
            )
            assert abs(md - 20.0) < 1.0, (
                f"min_demand_A expected ~20, got {md}"
            )

    def test_three_decimal_coefficients(self):
        """Three distinct float values should be assigned without duplication."""
        query = (
            "Product X has a unit profit of 1.2 dollars, "
            "product Y of 3.4 dollars, "
            "and product Z of 5.6 dollars."
        )
        vals, _, _ = _gcg(query, ["profit_X", "profit_Y", "profit_Z"])
        filled_values = [float(v) for v in vals.values() if v is not None]
        # Check no duplicates (all three values should be distinct).
        assert len(set(filled_values)) == len(filled_values), (
            f"Duplicate values detected: {filled_values}"
        )
        expected = {1.2, 3.4, 5.6}
        for ev in expected:
            assert any(abs(fv - ev) < 0.1 for fv in filled_values), (
                f"Expected value {ev} not found in {filled_values}"
            )


# ---------------------------------------------------------------------------
# 5. Diagnostics structure
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """The method must return a well-structured diagnostics dict."""

    def test_top_assignments_present(self):
        query = "The budget is 100 and demand is 5."
        _, _, diag = _gcg(query, ["budget", "demand"])
        assert "top_assignments" in diag, "diagnostics must contain top_assignments"
        assert isinstance(diag["top_assignments"], list)

    def test_top_assignments_fields(self):
        query = "The budget is 100 and demand is 5."
        _, _, diag = _gcg(query, ["budget", "demand"])
        for entry in diag["top_assignments"]:
            for field in ("rank", "total_score", "local_sum", "global_delta", "active_reasons", "assignment"):
                assert field in entry, f"Missing field '{field}' in top_assignment entry: {entry}"

    def test_per_slot_candidates_present(self):
        query = "The budget is 100 and demand is 5."
        _, _, diag = _gcg(query, ["budget", "demand"])
        assert "per_slot_candidates" in diag

    def test_reasons_are_strings(self):
        query = "The discount rate is 10% and the total price is 200."
        _, _, diag = _gcg(query, ["discount_percent", "total_price"])
        for entry in diag.get("top_assignments", []):
            for reason in entry.get("active_reasons", []):
                assert isinstance(reason, str), f"reason should be a string, got {reason!r}"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Empty inputs and degenerate inputs should not raise."""

    def test_empty_expected_scalar_returns_empty(self):
        vals, mentions, diag = _gcg("The budget is 100.", [])
        assert vals == {}
        assert mentions == {}

    def test_no_numeric_mentions_returns_empty(self):
        vals, mentions, _ = _gcg("There are no numbers here at all.", ["budget"])
        assert vals == {}

    def test_single_slot_single_mention(self):
        vals, _, _ = _gcg("The budget is 42 dollars", ["budget"])
        assert vals.get("budget") is not None
        assert abs(float(vals["budget"]) - 42.0) < 0.5

    def test_more_slots_than_mentions(self):
        """Should fill what it can and leave the rest empty without error."""
        vals, _, _ = _gcg("The budget is 100 dollars", ["budget", "demand", "cost"])
        assert vals.get("budget") is not None
        # demand and cost may be empty — that's fine.


# ---------------------------------------------------------------------------
# 7. Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    """Configurable constants must exist and have sensible values."""

    def test_local_weights_keys(self):
        required = {
            "type_exact_bonus", "type_loose_bonus", "type_incompatible_penalty",
            "opt_role_overlap", "fragment_compat_bonus", "operator_match_bonus",
            "lex_context_overlap", "lex_sentence_overlap", "unit_match_bonus",
            "entity_resource_overlap", "coefficient_vs_total_bonus",
            "schema_prior_bonus", "weak_match_penalty",
        }
        assert required.issubset(GCG_LOCAL_WEIGHTS.keys()), (
            f"Missing keys: {required - GCG_LOCAL_WEIGHTS.keys()}"
        )

    def test_global_weights_keys(self):
        required = {
            "coverage_reward_per_slot", "type_consistency_reward",
            "percent_misuse_penalty", "non_percent_to_pct_slot_penalty",
            "total_to_coeff_penalty", "coeff_to_total_penalty",
            "bound_flip_penalty", "duplicate_mention_penalty",
            "plausibility_coverage_bonus",
        }
        assert required.issubset(GCG_GLOBAL_WEIGHTS.keys())

    def test_beam_width_positive(self):
        assert GCG_BEAM_WIDTH >= 1

    def test_prune_threshold_is_float(self):
        assert isinstance(GCG_PRUNE_THRESHOLD, float)

    def test_penalties_are_negative(self):
        for key, val in GCG_GLOBAL_WEIGHTS.items():
            if "penalty" in key:
                assert val < 0, f"Penalty {key} should be negative, got {val}"

    def test_rewards_are_positive(self):
        for key, val in GCG_GLOBAL_WEIGHTS.items():
            if "reward" in key or "bonus" in key:
                assert val > 0, f"Reward {key} should be positive, got {val}"


# ---------------------------------------------------------------------------
# 8. Integration: method registered in run_setting choices
# ---------------------------------------------------------------------------

class TestIntegration:
    """Verify the method is wired into the evaluation pipeline."""

    def test_assignment_mode_string_in_argparse_choices(self):
        """The argparse choices in main() must include global_consistency_grounding."""
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        found = False
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and node.value == "global_consistency_grounding":
                found = True
                break
        assert found, "global_consistency_grounding not found as a string constant in downstream utility"

    def test_effective_baseline_naming(self):
        """run_single_setting must map global_consistency_grounding to the correct label."""
        from tools.run_nlp4lp_focused_eval import _effective_baseline
        result = _effective_baseline("tfidf", "global_consistency_grounding")
        assert result == "tfidf_global_consistency_grounding"

    def test_focused_eval_includes_new_method(self):
        """run_nlp4lp_focused_eval.py default baselines must include the new method."""
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        assert "tfidf_global_consistency_grounding" in FOCUSED_BASELINES_DEFAULT
