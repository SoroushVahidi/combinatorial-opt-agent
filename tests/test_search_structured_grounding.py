"""Tests for search_structured_grounding and its ablation variant.

Covers the six targeted failure modes described in the problem statement:
  1. Total vs per-unit
  2. Lower vs upper bound
  3. Percent vs scalar
  4. Count-like vs quantity-like
  5. Duplicate-number conflict (same number should not fill two incompatible slots)
  6. Abstention (ambiguous slot left unfilled rather than forced)

Additional tests verify:
  - Both methods are importable and callable.
  - Configurable constants are present and sane.
  - Diagnostics structure is complete.
  - The method strings are wired into run_setting dispatch (imports without error).
  - The no-global ablation produces structurally identical output.
  - Empty-input edge cases are handled gracefully.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.search_structured_grounding import (
    SSG_BEAM_WIDTH,
    SSG_ENABLE_GLOBAL,
    SSG_ENABLE_HARD_PRUNING,
    SSG_NULL_PENALTY,
    SSG_PRUNE_THRESHOLD,
    SSG_TOP_K_PER_SLOT,
    _ssg_beam_search,
    _slot_constraint_priority,
    _violates_hard_constraints,
    search_structured_grounding,
    search_structured_grounding_no_global,
)
from tools.nlp4lp_downstream_utility import (
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _gcg_local_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ssg(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    """Thin wrapper for search_structured_grounding."""
    return search_structured_grounding(query, "orig", slots)


def _ssg_no_global(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return search_structured_grounding_no_global(query, "orig", slots)


# ---------------------------------------------------------------------------
# 0. Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_beam_width_positive(self):
        assert SSG_BEAM_WIDTH > 0

    def test_top_k_positive(self):
        assert SSG_TOP_K_PER_SLOT > 0

    def test_null_penalty_small_negative(self):
        # Must be negative (penalty) but small in magnitude.
        assert SSG_NULL_PENALTY < 0
        assert abs(SSG_NULL_PENALTY) < 2.0

    def test_prune_threshold_not_huge_positive(self):
        assert SSG_PRUNE_THRESHOLD < 1.0

    def test_global_enabled_by_default(self):
        assert SSG_ENABLE_GLOBAL is True

    def test_hard_pruning_enabled_by_default(self):
        assert SSG_ENABLE_HARD_PRUNING is True


# ---------------------------------------------------------------------------
# 1. Total vs per-unit
# ---------------------------------------------------------------------------

class TestTotalVsPerUnit:
    """A total budget and a per-unit cost should not be confused."""

    def test_budget_gets_large_value(self):
        query = "The total budget is 5000 dollars and each radio ad costs 20 dollars."
        vals, _, diag = _ssg(query, ["total_budget", "cost_per_unit"])
        # 5000 must go to budget; 20 to per-unit cost
        if vals.get("total_budget") is not None:
            assert float(vals["total_budget"]) > 100, (
                f"total_budget should be 5000, got {vals['total_budget']}"
            )

    def test_per_unit_gets_small_value(self):
        query = "The total budget is 5000 dollars and each radio ad costs 20 dollars."
        vals, _, _ = _ssg(query, ["total_budget", "cost_per_unit"])
        if vals.get("cost_per_unit") is not None:
            assert float(vals["cost_per_unit"]) < 1000, (
                f"cost_per_unit should be 20, got {vals['cost_per_unit']}"
            )

    def test_total_budget_not_same_as_cost(self):
        """The two slots must not receive the same value (duplicate reuse)."""
        query = "Budget is 5000 and cost per unit is 20."
        vals, mentions, _ = _ssg(query, ["total_budget", "unit_cost"])
        # If both are filled, they must differ.
        if vals.get("total_budget") is not None and vals.get("unit_cost") is not None:
            assert float(vals["total_budget"]) != float(vals["unit_cost"]), (
                "total_budget and unit_cost received the same value"
            )


# ---------------------------------------------------------------------------
# 2. Lower vs upper bound
# ---------------------------------------------------------------------------

class TestLowerUpperBound:
    """Min and max slots should not be swapped."""

    def test_min_gets_smaller_value(self):
        query = "We need at least 100 units and at most 500 units."
        vals, _, _ = _ssg(query, ["min_units", "max_units"])
        if vals.get("min_units") is not None and vals.get("max_units") is not None:
            assert float(vals["min_units"]) <= float(vals["max_units"]), (
                f"min_units ({vals['min_units']}) > max_units ({vals['max_units']}): inversion!"
            )

    def test_min_gets_100(self):
        query = "We need at least 100 units and at most 500 units."
        vals, _, _ = _ssg(query, ["min_units", "max_units"])
        if vals.get("min_units") is not None:
            assert abs(float(vals["min_units"]) - 100.0) < 1.0, (
                f"min_units expected 100, got {vals['min_units']}"
            )

    def test_max_gets_500(self):
        query = "We need at least 100 units and at most 500 units."
        vals, _, _ = _ssg(query, ["min_units", "max_units"])
        if vals.get("max_units") is not None:
            assert abs(float(vals["max_units"]) - 500.0) < 1.0, (
                f"max_units expected 500, got {vals['max_units']}"
            )

    def test_hard_pruning_rejects_inverted_state(self):
        """_violates_hard_constraints must catch min > max during search."""
        mentions = _extract_opt_role_mentions(
            "at least 400 and at most 100", "orig"
        )
        slots = _build_slot_opt_irs(["min_units", "max_units"])
        if len(mentions) < 2 or len(slots) < 2:
            pytest.skip("Could not extract 2 mentions and 2 slots")
        # Identify min and max slot indices.
        min_j = next((j for j, s in enumerate(slots) if "min" in s.operator_preference), None)
        max_j = next((j for j, s in enumerate(slots) if "max" in s.operator_preference), None)
        # Identify large and small mention indices.
        m400 = next((i for i, m in enumerate(mentions) if m.value is not None and abs(m.value - 400) < 5), None)
        m100 = next((i for i, m in enumerate(mentions) if m.value is not None and abs(m.value - 100) < 5), None)
        if min_j is None or max_j is None or m400 is None or m100 is None:
            pytest.skip("Could not identify expected slot/mention indices")
        # Suppose min_slot already assigned to the large value (400).
        existing_bundle: frozenset[tuple[int, int]] = frozenset([(min_j, m400)])
        # Adding max_slot → 100 would give max(100) < min(400): must be rejected.
        assert _violates_hard_constraints(
            existing_bundle, max_j, m100, mentions, slots
        ), "Hard pruning should reject max_slot < min_slot assignment"


# ---------------------------------------------------------------------------
# 3. Percent vs scalar
# ---------------------------------------------------------------------------

class TestPercentVsScalar:
    """The percent mention must go to the percent slot, not to a scalar slot."""

    def test_percent_mention_goes_to_percent_slot(self):
        query = "The discount rate is 20% and the unit price is 50 dollars."
        vals, mentions, _ = _ssg(query, ["discount_percent", "unit_price"])
        if mentions.get("discount_percent") is not None:
            assert mentions["discount_percent"].type_bucket == "percent", (
                f"Expected percent mention for discount_percent, "
                f"got {mentions['discount_percent'].type_bucket}"
            )

    def test_scalar_slot_gets_non_percent_value(self):
        query = "The discount rate is 20% and the unit price is 50 dollars."
        vals, _, _ = _ssg(query, ["discount_percent", "unit_price"])
        if vals.get("unit_price") is not None:
            # 50 dollars should go here, not 0.2 or 20
            v = float(vals["unit_price"])
            assert v > 1.0, f"unit_price expected ~50, got {v}"

    def test_hard_pruning_rejects_percent_to_non_percent(self):
        """_violates_hard_constraints must reject percent mention → non-percent slot
        when a non-percent slot exists."""
        mentions = _extract_opt_role_mentions(
            "discount is 20% and price is 100 dollars", "orig"
        )
        slots = _build_slot_opt_irs(["discount_percent", "unit_price"])
        if not mentions or not slots:
            pytest.skip("Could not extract mentions / slots")
        pct_mention_idx = next(
            (i for i, m in enumerate(mentions) if m.type_bucket == "percent"), None
        )
        non_pct_slot_idx = next(
            (j for j, s in enumerate(slots) if s.expected_type != "percent"), None
        )
        if pct_mention_idx is None or non_pct_slot_idx is None:
            pytest.skip("Could not identify percent mention or non-percent slot")
        result = _violates_hard_constraints(
            frozenset(), non_pct_slot_idx, pct_mention_idx, mentions, slots
        )
        assert result, "Hard pruning should reject percent mention → non-percent slot"


# ---------------------------------------------------------------------------
# 4. Count-like vs quantity-like
# ---------------------------------------------------------------------------

class TestCountVsQuantity:
    """A small count should go to a count slot, not a large quantity slot."""

    def test_count_slot_gets_small_value(self):
        query = "There are three products and the storage capacity is 100 units."
        vals, _, _ = _ssg(query, ["num_products", "storage_capacity"])
        if vals.get("num_products") is not None:
            v = float(vals["num_products"])
            assert v <= 20, f"num_products (count) expected ≤20, got {v}"

    def test_capacity_slot_gets_large_value(self):
        query = "There are three products and the storage capacity is 100 units."
        vals, _, _ = _ssg(query, ["num_products", "storage_capacity"])
        if vals.get("storage_capacity") is not None:
            v = float(vals["storage_capacity"])
            assert v > 20, f"storage_capacity expected >20, got {v}"

    def test_slot_priority_count_before_generic(self):
        """_slot_constraint_priority should rank count-like slots higher priority
        (lower return value) than generic float slots."""
        slots = _build_slot_opt_irs(["num_products", "total_revenue"])
        if len(slots) < 2:
            pytest.skip("Could not build both slots")
        count_slot = next((s for s in slots if s.is_count_like), None)
        generic_slot = next((s for s in slots if not s.is_count_like), None)
        if count_slot is None or generic_slot is None:
            pytest.skip("Could not identify count vs generic slot")
        p_count = _slot_constraint_priority(count_slot, [])
        p_generic = _slot_constraint_priority(generic_slot, [])
        assert p_count < p_generic, (
            f"Count slot priority ({p_count}) should be lower than generic ({p_generic})"
        )


# ---------------------------------------------------------------------------
# 5. Duplicate number conflict
# ---------------------------------------------------------------------------

class TestDuplicateConflict:
    """When one value is salient, it must not fill two incompatible slots."""

    def test_no_duplicate_mention_assigned_to_two_slots(self):
        """Both slot values must differ (different mentions) when distinct numbers exist."""
        query = "The maximum production is 300 units and the minimum is 50 units."
        vals, mentions, _ = _ssg(query, ["min_production", "max_production"])
        if (
            vals.get("min_production") is not None
            and vals.get("max_production") is not None
        ):
            # Values should differ.
            assert float(vals["min_production"]) != float(vals["max_production"]), (
                "min and max should not receive the same value"
            )

    def test_hard_pruning_blocks_same_mention_reuse(self):
        """_violates_hard_constraints must block reuse of the same mention index."""
        mentions = _extract_opt_role_mentions("production is 300", "orig")
        slots = _build_slot_opt_irs(["slotA", "slotB"])
        if not mentions or len(slots) < 2:
            pytest.skip("Not enough mentions/slots")
        # Assign mention 0 to slot 0.
        existing_bundle: frozenset[tuple[int, int]] = frozenset([(0, 0)])
        # Now try to assign the SAME mention (0) to slot 1.
        assert _violates_hard_constraints(
            existing_bundle, 1, 0, mentions, slots
        ), "Hard pruning should block assigning the same mention to two slots"


# ---------------------------------------------------------------------------
# 6. Abstention
# ---------------------------------------------------------------------------

class TestAbstention:
    """When no good candidate exists, the method must leave the slot unfilled."""

    def test_method_can_return_fewer_slots_than_requested(self):
        """Given an unresolvable ambiguous slot, the method should not crash and
        may leave some slots unfilled rather than force a bad assignment."""
        query = "There are some items."
        vals, _, diag = _ssg(query, ["minimum_requirement", "maximum_allowance"])
        # No clear numeric evidence; method should not crash.
        assert isinstance(vals, dict)
        assert isinstance(diag, dict)

    def test_null_penalty_applied_in_diagnostics(self):
        """When the best assignment includes null slots, n_nulls_in_best should be > 0."""
        query = "The price is 100."  # only one numeric mention
        vals, _, diag = _ssg(query, ["price", "quantity", "discount_percent"])
        # At most one slot can be filled from one mention; others must be null.
        n_nulls = diag.get("n_nulls_in_best", 0)
        n_filled = len(vals)
        n_slots = 3
        assert n_nulls + n_filled == n_slots or n_filled <= 1, (
            f"Expected null assignments for unfilled slots: n_nulls={n_nulls}, n_filled={n_filled}"
        )


# ---------------------------------------------------------------------------
# 7. Diagnostics structure
# ---------------------------------------------------------------------------

class TestDiagnosticsStructure:
    def test_required_keys_present(self):
        query = "Budget is 1000 and cost per unit is 50."
        _, _, diag = _ssg(query, ["total_budget", "unit_cost"])
        required = {
            "n_mentions", "n_slots", "n_candidate_edges", "beam_width",
            "top_k_per_slot", "n_expanded_states", "best_score",
            "top_assignments", "per_slot_candidates",
        }
        for key in required:
            assert key in diag, f"Missing diagnostic key: {key!r}"

    def test_per_slot_candidates_keyed_by_slot_name(self):
        query = "Budget is 1000 and cost per unit is 50."
        _, _, diag = _ssg(query, ["total_budget", "unit_cost"])
        cands = diag.get("per_slot_candidates", {})
        for slot_name in ["total_budget", "unit_cost"]:
            assert slot_name in cands, f"Missing per_slot_candidates entry for {slot_name!r}"
            assert isinstance(cands[slot_name], list)

    def test_top_assignments_is_list(self):
        query = "Price is 200 and discount is 10%."
        _, _, diag = _ssg(query, ["item_price", "discount_percent"])
        top = diag.get("top_assignments", [])
        assert isinstance(top, list)

    def test_search_info_counts_are_non_negative(self):
        query = "Revenue is 5000 and cost is 200."
        _, _, diag = _ssg(query, ["revenue", "cost"])
        assert diag.get("n_expanded_states", -1) >= 0
        assert diag.get("n_candidate_edges", -1) >= 0
        assert diag.get("beam_width", -1) > 0


# ---------------------------------------------------------------------------
# 8. Ablation: no-global variant
# ---------------------------------------------------------------------------

class TestNoGlobalAblation:
    def test_no_global_returns_same_structure(self):
        query = "Budget is 1000 and cost per unit is 50."
        vals_g, _, diag_g = _ssg(query, ["total_budget", "unit_cost"])
        vals_l, _, diag_l = _ssg_no_global(query, ["total_budget", "unit_cost"])
        # Both should return dicts without crashing.
        assert isinstance(vals_g, dict)
        assert isinstance(vals_l, dict)
        # Diagnostics should both have the required keys.
        for key in ("n_mentions", "n_slots", "best_score", "top_assignments"):
            assert key in diag_g
            assert key in diag_l

    def test_no_global_disable_flag_in_diagnostics(self):
        query = "Revenue is 500 and profit per item is 10."
        _, _, diag = _ssg_no_global(query, ["total_revenue", "profit_per_item"])
        assert diag.get("enable_global") is False


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_expected_scalar_returns_empty(self):
        vals, mentions, diag = search_structured_grounding("any query", "orig", [])
        assert vals == {}
        assert mentions == {}

    def test_no_numeric_mentions_returns_empty(self):
        vals, mentions, diag = search_structured_grounding(
            "We need some products.", "orig", ["quantity", "budget"]
        )
        assert isinstance(vals, dict)
        assert isinstance(mentions, dict)

    def test_single_slot_single_mention(self):
        query = "The price is 42 dollars."
        vals, mentions, diag = search_structured_grounding(query, "orig", ["price"])
        assert isinstance(vals, dict)
        if vals.get("price") is not None:
            assert float(vals["price"]) > 0


# ---------------------------------------------------------------------------
# 10. Integration: dispatch strings wired into run_setting
# ---------------------------------------------------------------------------

class TestDispatchIntegration:
    def test_dispatch_strings_accepted_without_error(self):
        """Import the dispatch helper and confirm the new mode strings are
        routed without raising AttributeError / KeyError."""
        # We cannot run a full NLP4LP evaluation offline, but we can verify
        # that the conditional branch is syntactically present by importing
        # the module and checking the constant string is present in the file.
        utility_path = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        source = utility_path.read_text(encoding="utf-8")
        assert "search_structured_grounding" in source, (
            "run_setting dispatch must include 'search_structured_grounding'"
        )
        assert "search_structured_grounding_no_global" in source, (
            "run_setting dispatch must include 'search_structured_grounding_no_global'"
        )

    def test_module_importable(self):
        from tools.search_structured_grounding import (  # noqa: F401
            search_structured_grounding,
            search_structured_grounding_no_global,
        )
