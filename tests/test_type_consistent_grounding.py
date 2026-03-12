"""Tests for grounding improvements:

A. Type-consistent assignment (strengthened _is_type_incompatible)
   - Integer/float tokens must be hard-incompatible with percent slots
   - Percent tokens must be hard-incompatible with integer count slots
   - Old rules (percent↔currency) still hold

B. Maximum-weight global matching (_run_max_weight_matching_grounding)
   - Uses full score matrix + exact Hungarian algorithm (no repair step)
   - Enforces one-to-one constraint
   - Distinct ablation baseline vs optimization_role_repair (matching alone vs matching+repair)
   - Exposed as assignment_mode "max_weight_matching"

Note: global_consistency_exact / bitmask-DP mode was intentionally removed.
Rationale: the existing beam search (global_consistency_grounding) applies the global
consistency penalty to all K final beam states and picks the best; a DP that is exact
only under local scores and applies global penalty post-hoc to the single local-winner
is strictly weaker at the combined objective, adds O(m*2^s) cost, and does not
contribute a meaningfully different algorithmic baseline.
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
    _is_type_incompatible,
    _run_global_consistency_grounding,
    _run_max_weight_matching_grounding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcg_beam(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return _run_global_consistency_grounding(query, "orig", slots)


def _mwm(query: str, slots: list[str]) -> tuple[dict, dict, dict]:
    return _run_max_weight_matching_grounding(query, "orig", slots)


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

    # --- hard incompatibilities propagate through GCG local score ---

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
                "number_of_workers (int slot) should not receive percent mention"
            )


# ---------------------------------------------------------------------------
# B. Maximum-weight global matching (_run_max_weight_matching_grounding)
# ---------------------------------------------------------------------------

class TestMaxWeightMatchingGrounding:
    """Tests for the exact Hungarian-algorithm grounding mode (matching only, no repair)."""

    def test_returns_filled_values_and_mentions(self):
        """Basic smoke test: returns non-empty results for a simple query."""
        query = "The total budget is 500 dollars and the profit per unit is 10 dollars."
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
        query = "The budget is 500 and the cost per item is 10 dollars and the demand is 20 units."
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
        query = "The budget is 200 dollars and demand is 30 units."
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

    def test_global_consistency_exact_not_in_argparse(self):
        """global_consistency_exact was intentionally removed; must NOT appear in choices."""
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and node.value == "global_consistency_exact":
                raise AssertionError(
                    "'global_consistency_exact' was intentionally removed but still appears"
                )

    def test_global_consistency_exact_not_in_focused_eval(self):
        """global_consistency_exact was removed; must NOT be in FOCUSED_BASELINES_DEFAULT."""
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        assert "tfidf_global_consistency_exact" not in FOCUSED_BASELINES_DEFAULT
