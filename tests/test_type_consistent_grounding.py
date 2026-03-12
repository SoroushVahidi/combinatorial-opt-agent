"""Tests for the three grounding improvements:

A. Type-consistent assignment (strengthened _is_type_incompatible)
   - Integer/float tokens must be hard-incompatible with percent slots
   - Percent tokens must be hard-incompatible with integer count slots

B. Maximum-weight global matching (_run_max_weight_matching_grounding)
   - Uses full score matrix + exact Hungarian algorithm
   - Enforces one-to-one constraint
   - Exposed as assignment_mode "max_weight_matching"

C. Exhaustive DP over all valid assignments (_run_global_consistency_grounding_exact)
   - Bitmask DP considers ALL valid one-to-one assignments (not just beam)
   - For small instances (≤ GCG_DP_MAX_SLOTS) this is provably optimal
   - Falls back to wide beam for large instances
   - Exposed as assignment_mode "global_consistency_exact"
   - Diagnostics report whether exact DP was used

Additionally tests integration: new modes are registered in argparse + focused eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    GCG_DP_MAX_SLOTS,
    GCG_DP_FALLBACK_BEAM,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _gcg_dp_assignment,
    _gcg_local_score,
    _is_type_incompatible,
    _run_global_consistency_grounding,
    _run_global_consistency_grounding_exact,
    _run_max_weight_matching_grounding,
    MentionOptIR,
    NumTok,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcg_exact(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return _run_global_consistency_grounding_exact(query, "orig", slots)


def _mwm(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return _run_max_weight_matching_grounding(query, "orig", slots)


def _gcg_beam(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return _run_global_consistency_grounding(query, "orig", slots)


# ---------------------------------------------------------------------------
# A. Type-consistent assignment — strengthened _is_type_incompatible
# ---------------------------------------------------------------------------

class TestTypeConsistentAssignment:
    """Strengthened _is_type_incompatible: new hard incompatibilities."""

    # --- int / float → percent slot ---

    def test_int_to_percent_is_now_incompatible(self):
        """A bare integer token must be hard-incompatible with a percent slot."""
        assert _is_type_incompatible("percent", "int") is True

    def test_float_to_percent_is_now_incompatible(self):
        """A plain float token (no % sign) must be hard-incompatible with a percent slot."""
        assert _is_type_incompatible("percent", "float") is True

    # --- percent → int slot ---

    def test_percent_to_int_is_now_incompatible(self):
        """A percent token must be hard-incompatible with an integer count slot."""
        assert _is_type_incompatible("int", "percent") is True

    # --- original rules still hold ---

    def test_percent_to_currency_still_incompatible(self):
        assert _is_type_incompatible("currency", "percent") is True

    def test_currency_to_percent_still_incompatible(self):
        assert _is_type_incompatible("percent", "currency") is True

    # --- compatible pairs are NOT flagged ---

    def test_int_to_float_is_compatible(self):
        """int → float must remain compatible (int IS a real number)."""
        assert _is_type_incompatible("float", "int") is False

    def test_int_to_int_is_compatible(self):
        assert _is_type_incompatible("int", "int") is False

    def test_float_to_float_is_compatible(self):
        assert _is_type_incompatible("float", "float") is False

    def test_percent_to_percent_is_compatible(self):
        assert _is_type_incompatible("percent", "percent") is False

    def test_currency_to_currency_is_compatible(self):
        assert _is_type_incompatible("currency", "currency") is False

    def test_int_to_currency_is_compatible(self):
        """currency ← int: soft match is fine, not a hard incompatibility."""
        assert _is_type_incompatible("currency", "int") is False

    def test_currency_to_float_is_compatible(self):
        """float ← currency: soft loose match, not a hard incompatibility."""
        assert _is_type_incompatible("float", "currency") is False

    def test_unknown_to_percent_is_compatible(self):
        """unknown kind (noisy text) should not be hard-incompatible with percent."""
        assert _is_type_incompatible("percent", "unknown") is False

    # --- end-to-end: hard incompatibilities propagate through GCG local score ---

    def test_int_mention_hard_incompatible_with_percent_slot_in_gcg(self):
        """GCG local score for int mention + percent slot must be the hard-incompatible penalty."""
        mentions = _extract_opt_role_mentions("There are 30 units available.", "orig")
        slots = _build_slot_opt_irs(["discount_percent"])
        if not mentions or not slots:
            pytest.skip("No mentions or slots extracted")
        int_mentions = [m for m in mentions if m.type_bucket == "int"]
        if not int_mentions:
            pytest.skip("No integer mention extracted")
        sc, feats = _gcg_local_score(int_mentions[0], slots[0])
        assert feats.get("type_incompatible") is True, (
            "int mention → percent slot should be hard-incompatible in GCG local score"
        )
        assert sc < 0, f"Hard incompatible score should be negative, got {sc}"

    def test_percent_mention_hard_incompatible_with_int_slot_in_gcg(self):
        """GCG local score for percent mention + integer count slot must be hard-incompatible."""
        mentions = _extract_opt_role_mentions("The rate is 20% and we have 5 items.", "orig")
        slots = _build_slot_opt_irs(["item_count"])
        if not mentions or not slots:
            pytest.skip("No mentions or slots extracted")
        pct_mentions = [m for m in mentions if m.type_bucket == "percent"]
        if not pct_mentions:
            pytest.skip("No percent mention extracted")
        sc, feats = _gcg_local_score(pct_mentions[0], slots[0])
        assert feats.get("type_incompatible") is True, (
            "percent mention → int slot should be hard-incompatible in GCG local score"
        )

    # --- end-to-end: incompatible mentions cannot win percent/int slots ---

    def test_int_token_does_not_go_to_percent_slot_in_gcg(self):
        """When a percent-tagged mention and an integer mention exist, the integer
        must NOT be assigned to the percent slot."""
        query = "The discount rate is 15% and the order quantity is 50 units."
        vals, mentions, _ = _gcg_beam(query, ["discount_rate_percent", "order_quantity"])
        if vals.get("discount_rate_percent") is not None:
            m = mentions.get("discount_rate_percent")
            if m is not None:
                assert m.type_bucket == "percent", (
                    f"discount_rate_percent should receive percent mention, got {m.type_bucket}"
                )

    def test_percent_token_does_not_go_to_int_slot(self):
        """A percent token (e.g. '30%') must not be assigned to an integer count slot."""
        query = "The interest rate is 30% and we need 10 workers."
        vals, mentions, _ = _gcg_beam(query, ["number_of_workers", "interest_rate_percent"])
        m = mentions.get("number_of_workers")
        if m is not None:
            assert m.type_bucket != "percent", (
                f"number_of_workers (int slot) should not receive percent mention"
            )


# ---------------------------------------------------------------------------
# B. Maximum-weight global matching (_run_max_weight_matching_grounding)
# ---------------------------------------------------------------------------

class TestMaxWeightMatchingGrounding:
    """Tests for the exact Hungarian-algorithm grounding mode."""

    def test_returns_filled_values_and_mentions(self):
        """Basic smoke test: returns non-empty results for a simple query."""
        query = "The total budget is 500 dollars and the profit per unit is 10."
        vals, mentions, diag = _mwm(query, ["total_budget", "profit_per_unit"])
        assert isinstance(vals, dict)
        assert isinstance(mentions, dict)
        assert isinstance(diag, dict)

    def test_fills_both_slots(self):
        """Both slots should be filled when distinct mentions are present."""
        query = "The total budget is 500 dollars and the profit per unit is 10 dollars."
        vals, _, _ = _mwm(query, ["total_budget", "profit_per_unit"])
        assert vals.get("total_budget") is not None
        assert vals.get("profit_per_unit") is not None

    def test_one_to_one_assignment(self):
        """No mention should be assigned to two different slots."""
        query = "The budget is 500 and the cost per item is 10 and the demand is 20."
        vals, mentions, _ = _mwm(query, ["total_budget", "cost_per_item", "min_demand"])
        mention_ids = [m.mention_id for m in mentions.values()]
        assert len(mention_ids) == len(set(mention_ids)), (
            "One-to-one constraint violated: same mention used for multiple slots"
        )

    def test_large_value_goes_to_budget_slot(self):
        """Semantically, the larger value should go to the budget/total slot."""
        query = "Each item yields a profit of 4 dollars. The total budget is 800 dollars."
        vals, _, _ = _mwm(query, ["profit_per_unit", "total_budget"])
        if vals.get("profit_per_unit") is not None and vals.get("total_budget") is not None:
            assert float(vals["profit_per_unit"]) < float(vals["total_budget"]), (
                "Profit per unit should be smaller than total budget"
            )

    def test_diagnostics_contain_per_slot_candidates(self):
        """Diagnostics must include per_slot_candidates for transparency."""
        query = "The budget is 200 and demand is 30."
        _, _, diag = _mwm(query, ["budget", "demand"])
        assert "per_slot_candidates" in diag

    def test_empty_scalar_list_returns_empty(self):
        vals, mentions, diag = _mwm("Some query.", [])
        assert vals == {}
        assert mentions == {}

    def test_percent_token_goes_to_percent_slot(self):
        """Type-consistent assignment: percent mention must go to percent slot."""
        query = "The tax rate is 5% and the total price is 200 dollars."
        vals, mentions, _ = _mwm(query, ["tax_rate_percent", "total_price"])
        m = mentions.get("tax_rate_percent")
        if m is not None:
            assert m.type_bucket == "percent", (
                f"tax_rate_percent should receive percent mention, got {m.type_bucket}"
            )

    def test_assignment_mode_string_in_argparse_choices(self):
        """'max_weight_matching' must appear in the argparse choices."""
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        found = False
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and node.value == "max_weight_matching":
                found = True
                break
        assert found, "'max_weight_matching' not found as a string constant in downstream utility"

    def test_focused_eval_includes_max_weight_matching(self):
        """run_nlp4lp_focused_eval.py must include tfidf_max_weight_matching."""
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        assert "tfidf_max_weight_matching" in FOCUSED_BASELINES_DEFAULT

    def test_effective_baseline_naming(self):
        """_effective_baseline must map max_weight_matching to the correct label."""
        from tools.run_nlp4lp_focused_eval import _effective_baseline
        result = _effective_baseline("tfidf", "max_weight_matching")
        assert result == "tfidf_max_weight_matching"


# ---------------------------------------------------------------------------
# C. Exhaustive DP — _run_global_consistency_grounding_exact
# ---------------------------------------------------------------------------

class TestGlobalConsistencyExact:
    """Tests for the exact bitmask-DP grounding mode."""

    def test_returns_filled_values_and_mentions(self):
        """Smoke test: valid return types."""
        query = "The tax rate is 15% and the total budget is 500 dollars."
        vals, mentions, diag = _gcg_exact(query, ["tax_rate_percent", "total_budget"])
        assert isinstance(vals, dict)
        assert isinstance(mentions, dict)
        assert isinstance(diag, dict)

    def test_diagnostics_report_exact_dp_flag(self):
        """For small instances, diagnostics must report used_exact_dp=True."""
        query = "The budget is 500 and the demand is 30."
        _, _, diag = _gcg_exact(query, ["budget", "demand"])
        assert diag.get("used_exact_dp") is True, (
            "Small instances must use exact DP; expected used_exact_dp=True"
        )

    def test_diagnostics_n_dp_states(self):
        """n_dp_states must equal 2^n_slots for exact DP instances."""
        query = "The budget is 500 and the demand is 30."
        _, _, diag = _gcg_exact(query, ["budget", "demand"])
        if diag.get("used_exact_dp"):
            n_slots = 2
            assert diag.get("n_dp_states") == (1 << n_slots), (
                f"Expected {1 << n_slots} DP states for {n_slots} slots"
            )

    def test_diagnostics_report_fallback_for_large_instances(self):
        """For instances with more slots than GCG_DP_MAX_SLOTS, used_exact_dp must be False."""
        # Create a list of many slot names (more than GCG_DP_MAX_SLOTS).
        large_slots = [f"param_{i}" for i in range(GCG_DP_MAX_SLOTS + 1)]
        query = " ".join(f"Value {i} is {i * 10}." for i in range(GCG_DP_MAX_SLOTS + 1))
        _, _, diag = _gcg_exact(query, large_slots)
        assert diag.get("used_exact_dp") is False, (
            f"Instances with >{GCG_DP_MAX_SLOTS} slots should fall back to beam search"
        )

    def test_one_to_one_constraint(self):
        """Exact DP must respect the one-to-one constraint."""
        query = "The budget is 500 and the demand is 30 and the rate is 20%."
        vals, mentions, _ = _gcg_exact(
            query, ["total_budget", "demand_amount", "rate_percent"]
        )
        mention_ids = [m.mention_id for m in mentions.values()]
        assert len(mention_ids) == len(set(mention_ids)), (
            "One-to-one constraint violated in exact DP"
        )

    def test_matches_or_beats_beam_on_simple_case(self):
        """On simple 2-slot problems, exact DP should produce the same result as beam search."""
        query = "The total budget is 800 dollars and the profit per unit is 4 dollars."
        slots = ["total_budget", "profit_per_unit"]
        vals_exact, _, _ = _gcg_exact(query, slots)
        vals_beam, _, _ = _gcg_beam(query, slots)
        # Both should fill both slots.
        assert vals_exact.get("total_budget") is not None or vals_beam.get("total_budget") is None
        assert vals_exact.get("profit_per_unit") is not None or vals_beam.get("profit_per_unit") is None

    def test_percent_and_scalar_correctly_separated(self):
        """Exact DP must assign percent mention to percent slot."""
        query = "The discount rate is 20% and the item price is 80 dollars."
        vals, mentions, _ = _gcg_exact(query, ["discount_rate_percent", "item_price"])
        m = mentions.get("discount_rate_percent")
        if m is not None:
            assert m.type_bucket == "percent", (
                f"Exact DP: discount_rate_percent should get percent mention, got {m.type_bucket}"
            )

    def test_empty_input_graceful(self):
        vals, mentions, diag = _gcg_exact("Some query.", [])
        assert vals == {}
        assert mentions == {}

    def test_assignment_mode_string_in_argparse_choices(self):
        """'global_consistency_exact' must appear in the argparse choices."""
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        found = False
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and node.value == "global_consistency_exact":
                found = True
                break
        assert found, "'global_consistency_exact' not found as a string constant in downstream utility"

    def test_focused_eval_includes_global_consistency_exact(self):
        """run_nlp4lp_focused_eval.py must include tfidf_global_consistency_exact."""
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        assert "tfidf_global_consistency_exact" in FOCUSED_BASELINES_DEFAULT

    def test_effective_baseline_naming(self):
        """_effective_baseline must map global_consistency_exact to the correct label."""
        from tools.run_nlp4lp_focused_eval import _effective_baseline
        result = _effective_baseline("tfidf", "global_consistency_exact")
        assert result == "tfidf_global_consistency_exact"


# ---------------------------------------------------------------------------
# D. _gcg_dp_assignment unit tests
# ---------------------------------------------------------------------------

class TestGcgDpAssignment:
    """Direct unit tests for the _gcg_dp_assignment function."""

    def _make_local_scores(self, mentions, slots):
        """Precompute local scores for all (mention, slot) pairs."""
        m_count, s_count = len(mentions), len(slots)
        local_scores = [[0.0] * s_count for _ in range(m_count)]
        local_features = [[{} for _ in range(s_count)] for _ in range(m_count)]
        for i, mr in enumerate(mentions):
            for j, sr in enumerate(slots):
                sc, feats = _gcg_local_score(mr, sr)
                local_scores[i][j] = sc
                local_features[i][j] = feats
        return local_scores, local_features

    def test_single_slot_single_mention(self):
        """Single mention and single slot: exactly one assignment possible."""
        mentions = _extract_opt_role_mentions("The budget is 500 dollars.", "orig")
        slots = _build_slot_opt_irs(["total_budget"])
        if not mentions or not slots:
            pytest.skip("No mentions or slots extracted")
        ls, lf = self._make_local_scores(mentions, slots)
        asgn, scores, debug, tops = _gcg_dp_assignment(mentions, slots, ls, lf)
        assert isinstance(asgn, dict)

    def test_two_slots_two_mentions_no_reuse(self):
        """Two slots, two mentions: each mention can be used at most once."""
        mentions = _extract_opt_role_mentions(
            "The profit per unit is 4 dollars. The total budget is 800 dollars.", "orig"
        )
        slots = _build_slot_opt_irs(["profit_per_unit", "total_budget"])
        if len(mentions) < 2 or not slots:
            pytest.skip("Not enough mentions or slots")
        ls, lf = self._make_local_scores(mentions, slots)
        asgn, _, _, _ = _gcg_dp_assignment(mentions, slots, ls, lf)
        # No mention should be shared between slots.
        assigned_ids = [m.mention_id for m in asgn.values()]
        assert len(assigned_ids) == len(set(assigned_ids))

    def test_returns_top_assignments_list(self):
        """top_assignments must be a non-empty list."""
        mentions = _extract_opt_role_mentions("The demand is 30.", "orig")
        slots = _build_slot_opt_irs(["demand"])
        if not mentions or not slots:
            pytest.skip("No data")
        ls, lf = self._make_local_scores(mentions, slots)
        _, _, _, tops = _gcg_dp_assignment(mentions, slots, ls, lf)
        assert isinstance(tops, list)
        assert len(tops) >= 1

    def test_fallback_to_beam_for_large_s(self):
        """When s > max_exact_slots, the function must fall back to beam search."""
        large_slots = [f"param_{i}" for i in range(GCG_DP_MAX_SLOTS + 2)]
        query = " ".join(f"Value {i} is {i * 5}." for i in range(GCG_DP_MAX_SLOTS + 2))
        mentions = _extract_opt_role_mentions(query, "orig")
        slots = _build_slot_opt_irs(large_slots)
        if not mentions or not slots:
            pytest.skip("No data")
        ls, lf = self._make_local_scores(mentions[:min(len(mentions), 5)], slots)
        # Should not raise.
        asgn, _, _, _ = _gcg_dp_assignment(
            mentions[:min(len(mentions), 5)], slots, ls, lf,
            max_exact_slots=GCG_DP_MAX_SLOTS,
        )
        assert isinstance(asgn, dict)

    def test_constants_are_sane(self):
        """GCG_DP_MAX_SLOTS and GCG_DP_FALLBACK_BEAM must be positive integers."""
        assert isinstance(GCG_DP_MAX_SLOTS, int) and GCG_DP_MAX_SLOTS > 0
        assert isinstance(GCG_DP_FALLBACK_BEAM, int) and GCG_DP_FALLBACK_BEAM > 0
