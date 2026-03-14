"""Tests for literature-informed float type-match improvements.

Covers the four method ideas:
A. Numeracy / quantity normalization — _expected_type, _is_type_match
B. Slot semantic tagging — _slot_semantic_expansion, _slot_opt_role_expansion
C. Global consistency — GCG correctly discriminates float vs int when cues differ
D. Structured rules — per-unit and total-slot tagging

Regression cases target the known bottlenecks:
  - float/decimal values (int-as-float recognition)
  - percent vs scalar confusion
  - total vs per-unit confusion
  - lower-bound vs upper-bound / min vs max confusion
  - objective coefficient vs constraint bound confusion
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _expected_type,
    _is_type_match,
    _parse_num_token,
    _score_mention_slot,
    _slot_semantic_expansion,
    _slot_opt_role_expansion,
    _run_global_consistency_grounding,
    MentionRecord,
    NumTok,
    SlotRecord,
    _normalize_tokens,
    _slot_aliases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mention(tok_str: str, ctx: list[str] | None = None) -> MentionRecord:
    ctx = ctx or []
    tok = _parse_num_token(tok_str, set(ctx))
    return MentionRecord(
        index=0,
        tok=tok,
        context_tokens=ctx,
        sentence_tokens=ctx,
        cue_words=set(),
    )


def _make_slot(name: str) -> SlotRecord:
    et = _expected_type(name)
    aliases = _slot_aliases(name)
    alias_tokens: set[str] = set()
    for a in aliases:
        alias_tokens.update(_normalize_tokens(a))
    return SlotRecord(
        name=name,
        norm_tokens=_normalize_tokens(name),
        expected_type=et,
        aliases=aliases,
        alias_tokens=alias_tokens,
    )


def _gcg(query: str, slots: list[str]):
    return _run_global_consistency_grounding(query, "orig", slots)


# ---------------------------------------------------------------------------
# A. Numeracy — _expected_type extended integer patterns
# ---------------------------------------------------------------------------

class TestExpectedTypeExtendedInt:
    """_expected_type must classify discrete quantities as 'int', not 'float'."""

    def test_total_workers_is_int(self):
        assert _expected_type("TotalWorkers") == "int"

    def test_number_of_shifts_is_int(self):
        assert _expected_type("NumberOfShifts") == "int"

    def test_number_of_days_is_int(self):
        assert _expected_type("NumberOfDays") == "int"

    def test_number_of_machines_is_int(self):
        assert _expected_type("NumberOfMachines") == "int"

    def test_number_of_vehicles_is_int(self):
        assert _expected_type("NumberOfVehicles") == "int"

    def test_total_batches_is_int(self):
        assert _expected_type("TotalBatches") == "int"

    def test_number_of_persons_is_int(self):
        assert _expected_type("NumberOfPersons") == "int"

    def test_number_of_trips_is_int(self):
        assert _expected_type("NumberOfTrips") == "int"

    def test_currency_takes_precedence_over_extended_int(self):
        """'TotalBudget' must remain 'currency', not be reclassified as 'int'."""
        assert _expected_type("TotalBudget") == "currency"

    def test_total_cost_remains_currency(self):
        assert _expected_type("TotalCost") == "currency"

    def test_continuous_float_slot_stays_float(self):
        """Slots without discrete-count cues must remain 'float'."""
        assert _expected_type("RequiredEggsPerSandwich") == "float"
        assert _expected_type("BakingTimePerType") == "float"
        assert _expected_type("AmountPerPill") == "float"


# ---------------------------------------------------------------------------
# A. Numeracy — _is_type_match helper
# ---------------------------------------------------------------------------

class TestIsTypeMatch:
    """_is_type_match must treat int as a full match for float slots."""

    def test_float_slot_int_token_is_match(self):
        """Core fix: integer token IS a valid float value."""
        assert _is_type_match("float", "int") is True

    def test_float_slot_float_token_is_match(self):
        assert _is_type_match("float", "float") is True

    def test_int_slot_int_token_is_match(self):
        assert _is_type_match("int", "int") is True

    def test_percent_slot_percent_token_is_match(self):
        assert _is_type_match("percent", "percent") is True

    def test_currency_slot_currency_token_is_match(self):
        assert _is_type_match("currency", "currency") is True

    def test_float_slot_currency_token_is_not_match(self):
        """Currency token should NOT be a full float match."""
        assert _is_type_match("float", "currency") is False

    def test_int_slot_float_token_is_not_match(self):
        """A decimal float should NOT be a full int match."""
        assert _is_type_match("int", "float") is False

    def test_percent_slot_int_token_is_not_match(self):
        assert _is_type_match("percent", "int") is False

    def test_currency_slot_percent_token_is_not_match(self):
        assert _is_type_match("currency", "percent") is False


# ---------------------------------------------------------------------------
# A. Numeracy — _score_mention_slot type bonuses
# ---------------------------------------------------------------------------

class TestScoreMentionSlotTypeBonus:
    """_score_mention_slot must give full type_match for int token on float slot."""

    def test_int_token_float_slot_gives_type_match(self):
        """'RequiredEggsPerSandwich' (float) + '2' (int) → type_match=True."""
        m = _make_mention("2")
        s = _make_slot("RequiredEggsPerSandwich")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True
        assert not feats.get("type_loose_match")

    def test_float_token_float_slot_gives_type_match(self):
        """A decimal float token must still give type_match for float slot."""
        m = _make_mention("1.5")
        s = _make_slot("RequiredEggsPerSandwich")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True

    def test_percent_token_percent_slot_gives_type_match(self):
        m = _make_mention("30%")
        s = _make_slot("MaxFractionWassaAds")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True

    def test_currency_token_currency_slot_gives_type_match(self):
        m = _make_mention("$500000")
        s = _make_slot("Budget")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True

    def test_int_token_float_slot_scores_higher_than_before(self):
        """Full type_match_bonus (3.0) must replace old 0.5× loose bonus (1.5)."""
        m = _make_mention("5")
        s = _make_slot("AmountPerPill")
        score, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True
        # Full bonus is 3.0; old code gave at most 1.5 for type_loose_match
        assert score >= 3.0

    def test_currency_token_on_float_slot_is_loose_not_exact(self):
        """Currency token on a plain float slot should be loose, not exact."""
        m = _make_mention("$5000")
        s = _make_slot("AmountPerPill")
        _, feats = _score_mention_slot(m, s)
        assert not feats.get("type_match")
        assert feats.get("type_loose_match") is True


# ---------------------------------------------------------------------------
# B. Slot semantic tagging
# ---------------------------------------------------------------------------

class TestSlotSemanticTags:
    """_slot_semantic_expansion must add per-unit and total tags."""

    def test_per_unit_slot_has_per_unit_tag(self):
        tags = _slot_semantic_expansion("RequiredEggsPerSandwich")
        # "per" is in the name
        assert "per_unit" in tags

    def test_total_slot_has_total_tag(self):
        tags = _slot_semantic_expansion("TotalAvailableEggs")
        assert "total" in tags or "aggregate" in tags

    def test_percent_slot_has_percentage_tag(self):
        tags = _slot_semantic_expansion("MaxFractionWassaAds")
        assert "percentage" in tags or "ratio" in tags or "proportion" in tags

    def test_budget_slot_has_budget_tag(self):
        tags = _slot_semantic_expansion("AdvertisingBudget")
        assert "budget" in tags

    def test_min_slot_has_lower_bound_tag(self):
        tags = _slot_semantic_expansion("MinimumDemand")
        assert "lower_bound" in tags or "minimum" in tags

    def test_max_slot_has_upper_bound_tag(self):
        tags = _slot_semantic_expansion("MaximumCapacity")
        assert "upper_bound" in tags or "maximum" in tags


class TestSlotOptRoleTags:
    """_slot_opt_role_expansion must add per-unit and total tags."""

    def test_per_unit_slot_gets_objective_coeff_tag(self):
        tags = _slot_opt_role_expansion("ProfitPerUnit")
        assert "objective_coeff" in tags

    def test_available_slot_gets_total_available_tag(self):
        tags = _slot_opt_role_expansion("TotalAvailableEggs")
        assert "total_available" in tags or "capacity_limit" in tags

    def test_each_slot_gets_resource_consumption_tag(self):
        tags = _slot_opt_role_expansion("TimeRequiredEachProduct")
        assert "resource_consumption" in tags or "objective_coeff" in tags


# ---------------------------------------------------------------------------
# C. Global consistency — float vs integer discrimination
# ---------------------------------------------------------------------------

class TestGCGFloatVsInteger:
    """GCG must correctly assign float coefficients and integer counts."""

    def test_decimal_coefficient_not_swapped_with_integer_demand(self):
        """A decimal coefficient (1.5) should not be assigned to a count slot."""
        query = (
            "Each unit of product A requires 1.5 hours of machine time. "
            "The minimum demand for product A is 20 units."
        )
        vals, _, _ = _gcg(query, ["hours_per_unit_A", "min_demand_A"])
        if vals.get("hours_per_unit_A") is not None and vals.get("min_demand_A") is not None:
            assert abs(float(vals["hours_per_unit_A"]) - 1.5) < 0.1, (
                f"Expected hours_per_unit_A ≈ 1.5, got {vals['hours_per_unit_A']}"
            )

    def test_integer_token_valid_for_float_slot(self):
        """An integer token should be assignable to a float-typed slot."""
        query = "Each sandwich requires 2 eggs and 3 strips of bacon."
        vals, _, _ = _gcg(query, ["RequiredEggsPerSandwich", "RequiredBaconPerSandwich"])
        # Both should be filled (no hard incompatibility)
        assert vals.get("RequiredEggsPerSandwich") is not None
        assert vals.get("RequiredBaconPerSandwich") is not None

    def test_integer_token_value_correct_for_float_slot(self):
        query = "Each sandwich requires 2 eggs."
        vals, _, _ = _gcg(query, ["RequiredEggsPerSandwich"])
        assert vals.get("RequiredEggsPerSandwich") is not None
        assert abs(float(vals["RequiredEggsPerSandwich"]) - 2.0) < 0.1


# ---------------------------------------------------------------------------
# C. Global consistency — percent vs scalar
# ---------------------------------------------------------------------------

class TestGCGPercentVsScalar:
    """GCG must not confuse percentage tokens with plain scalar slots."""

    def test_percent_goes_to_percent_slot(self):
        query = "The interest rate is 5% and the loan amount is 10000 dollars."
        vals, _, _ = _gcg(query, ["interest_rate_percent", "loan_amount"])
        if vals.get("interest_rate_percent") is not None:
            # percent value should be ≤ 1 (stored as fraction) or recognise as pct
            v = float(vals["interest_rate_percent"])
            assert v < 1.0 or abs(v - 5.0) < 0.1

    def test_scalar_does_not_go_to_percent_slot(self):
        query = "The discount rate is 20% and the unit price is 50."
        vals, _, _ = _gcg(query, ["discount_rate_percent", "unit_price"])
        if vals.get("unit_price") is not None:
            v = float(vals["unit_price"])
            # unit_price should NOT be the percent value
            assert v > 1.0, f"unit_price should be 50, not a percent fraction; got {v}"


# ---------------------------------------------------------------------------
# C. Global consistency — lower-bound vs upper-bound
# ---------------------------------------------------------------------------

class TestGCGBoundDirection:
    """GCG must respect min/max cue words for bound assignment."""

    def test_at_least_goes_to_min_slot(self):
        query = "Production must be at least 100 units and at most 500 units."
        vals, _, _ = _gcg(query, ["min_production", "max_production"])
        if vals.get("min_production") is not None and vals.get("max_production") is not None:
            assert float(vals["min_production"]) < float(vals["max_production"]), (
                f"min={vals['min_production']} should be < max={vals['max_production']}"
            )

    def test_lower_bound_value_smaller_than_upper_bound(self):
        query = "You need at least 10 workers but no more than 30 workers."
        vals, _, _ = _gcg(query, ["min_workers", "max_workers"])
        if vals.get("min_workers") is not None and vals.get("max_workers") is not None:
            assert float(vals["min_workers"]) <= float(vals["max_workers"]), (
                f"min_workers={vals['min_workers']} should be ≤ max_workers={vals['max_workers']}"
            )


# ---------------------------------------------------------------------------
# C. Global consistency — total vs per-unit
# ---------------------------------------------------------------------------

class TestGCGTotalVsPerUnit:
    """GCG must distinguish total-quantity mentions from per-unit coefficients."""

    def test_total_budget_not_confused_with_unit_cost(self):
        query = (
            "The total advertising budget is $5000. "
            "Each radio ad costs $20."
        )
        vals, _, _ = _gcg(query, ["AdvertisingBudget", "CostPerRadioAd"])
        if vals.get("AdvertisingBudget") is not None and vals.get("CostPerRadioAd") is not None:
            budget = float(vals["AdvertisingBudget"])
            cost = float(vals["CostPerRadioAd"])
            assert budget > cost, (
                f"Budget ({budget}) should be greater than unit cost ({cost})"
            )


# ---------------------------------------------------------------------------
# D. Structured rules — int-as-float does not pollute int slots
# ---------------------------------------------------------------------------

class TestIntSlotIntegrity:
    """Promoting int→float must NOT cause float tokens to match int slots."""

    def test_float_token_not_full_match_for_int_slot(self):
        """A decimal '1.5' should NOT be a full type_match for 'NumSandwichTypes'."""
        m = _make_mention("1.5")
        s = _make_slot("NumSandwichTypes")
        _, feats = _score_mention_slot(m, s)
        # Should be loose, not exact type_match
        assert not feats.get("type_match"), (
            "Decimal 1.5 should not be a full type_match for an integer slot"
        )

    def test_int_token_full_match_for_int_slot(self):
        m = _make_mention("3")
        s = _make_slot("NumSandwichTypes")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True

    def test_percent_slot_int_token_is_not_type_match(self):
        """Integer '30' without % sign should not be a full type_match for percent slot."""
        m = _make_mention("30")
        s = _make_slot("MaxFractionWassaAds")
        _, feats = _score_mention_slot(m, s)
        assert not feats.get("type_match"), (
            "Plain integer '30' must not be a full type_match for a percent slot"
        )


# ---------------------------------------------------------------------------
# E. Quantity-constraint slots are no longer misclassified as 'currency'
# ---------------------------------------------------------------------------

class TestQuantityConstraintSlotTypes:
    """demand / capacity / minimum / maximum / limit are quantity constraints,
    not monetary values.  They should be typed 'float' so that plain integer
    tokens (the most common representation in NL) receive a full type_match."""

    def test_demand_slot_is_float(self):
        assert _expected_type("Demand") == "float"

    def test_minimum_demand_slot_is_float(self):
        assert _expected_type("MinimumDemand") == "float"

    def test_max_demand_slot_is_float(self):
        assert _expected_type("MaxDemand") == "float"

    def test_capacity_slot_is_float(self):
        assert _expected_type("Capacity") == "float"

    def test_max_capacity_slot_is_float(self):
        assert _expected_type("MaxCapacity") == "float"

    def test_min_capacity_slot_is_float(self):
        assert _expected_type("MinCapacity") == "float"

    def test_minimum_slot_is_float(self):
        assert _expected_type("Minimum") == "float"

    def test_maximum_slot_is_float(self):
        assert _expected_type("Maximum") == "float"

    def test_limit_slot_is_float(self):
        assert _expected_type("Limit") == "float"

    def test_time_limit_slot_is_float(self):
        assert _expected_type("TimeLimit") == "float"

    def test_maximum_capacity_slot_is_float(self):
        assert _expected_type("MaximumCapacity") == "float"

    def test_minimum_production_slot_is_float(self):
        assert _expected_type("MinimumProduction") == "float"

    # BudgetLimit / CostCapacity: monetary keyword is checked first → still 'currency'
    def test_budget_limit_stays_currency(self):
        """'BudgetLimit' contains 'budget' → must remain 'currency'."""
        assert _expected_type("BudgetLimit") == "currency"

    def test_profit_limit_stays_currency(self):
        """'ProfitLimit' contains 'profit' → must remain 'currency'."""
        assert _expected_type("ProfitLimit") == "currency"

    # Scoring consequence: int token on a formerly-currency, now-float slot must
    # now get a FULL type_match (not the old weak-match penalty of -1.0).
    def test_int_token_on_minimum_demand_slot_type_match(self):
        """`MinimumDemand` + integer '100' must yield type_match=True, score ≥ 3.0."""
        m = _make_mention("100")
        s = _make_slot("MinimumDemand")
        score, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True, (
            "Integer token on MinimumDemand (float) should be a full type_match"
        )
        assert score >= 3.0, (
            f"Expected score ≥ 3.0 (type_match_bonus), got {score}"
        )

    def test_int_token_on_max_capacity_slot_type_match(self):
        m = _make_mention("500")
        s = _make_slot("MaxCapacity")
        score, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True
        assert score >= 3.0


# ---------------------------------------------------------------------------
# F. Currency slots accept plain integer/float tokens (no '$' sign)
# ---------------------------------------------------------------------------

class TestCurrencySlotPlainNumericTokens:
    """Monetary slot values often appear without an explicit '$' in NL text.
    A plain integer or float token IS a valid monetary assignment and must be
    counted as a full type_match."""

    def test_is_type_match_currency_int(self):
        """_is_type_match('currency', 'int') must be True."""
        assert _is_type_match("currency", "int") is True

    def test_is_type_match_currency_float(self):
        """_is_type_match('currency', 'float') must be True."""
        assert _is_type_match("currency", "float") is True

    def test_int_token_on_unit_cost_slot_type_match(self):
        """'UnitCost' (currency) + small int '50' → type_match=True, score ≥ 3.0."""
        m = _make_mention("50")          # kind=int (below 1000, no $ prefix)
        s = _make_slot("UnitCost")
        score, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True, (
            "Plain integer on a currency slot should be a full type_match"
        )
        assert score >= 3.0

    def test_float_token_on_price_slot_type_match(self):
        """'Price' (currency) + decimal '4.99' → type_match=True."""
        m = _make_mention("4.99")        # kind=float
        s = _make_slot("Price")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True, (
            "Decimal token on a currency slot should be a full type_match"
        )

    def test_currency_token_on_currency_slot_still_type_match(self):
        """Explicit '$' token on currency slot must still give type_match=True."""
        m = _make_mention("$5000")
        s = _make_slot("TotalBudget")
        _, feats = _score_mention_slot(m, s)
        assert feats.get("type_match") is True
