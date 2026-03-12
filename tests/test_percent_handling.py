"""Regression tests for percent vs integer/float type incompatibility.

Covers:
  1. Explicit percent expressions (40%, 25 percent, forty percent)
  2. Fraction words (half, one-third, quarter, three-quarters)
  3. Rate-context decimals (0.4 with "rate" context is percent-like)
  4. Non-percent numerals stay non-percent (40 workers, 0.4 tons)
  5. Slot typing with pct / ratio / proportion / share in slot name
  6. Hard incompatibility rules (float→percent, int→percent, percent→int)
  7. End-to-end grounding: percent mention routes only to percent slot
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _expected_type,
    _extract_num_tokens,
    _extract_opt_role_mentions,
    _is_type_incompatible,
    _run_global_consistency_grounding,
)


# ===========================================================================
# 1. Explicit percent expressions
# ===========================================================================

class TestExplicitPercentExpressions:
    """40%, 25 percent, forty percent → type_bucket='percent', is_percent_like=True."""

    def test_digit_percent_sign(self):
        ms = _extract_opt_role_mentions(
            "At least 40% of the budget must go to marketing.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.4) < 1e-9), None)
        assert m is not None, "40% should produce value≈0.4"
        assert m.type_bucket == "percent"
        assert m.is_percent_like is True

    def test_digit_space_percent_word(self):
        ms = _extract_opt_role_mentions(
            "No more than 25 percent of workers can be assigned to night shift.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.25) < 1e-9), None)
        assert m is not None, "25 percent should produce value≈0.25"
        assert m.type_bucket == "percent"
        assert m.is_percent_like is True

    def test_word_number_percent(self):
        ms = _extract_opt_role_mentions(
            "At least forty percent of workers must rest.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.4) < 1e-9), None)
        assert m is not None, "forty percent should produce value≈0.4"
        assert m.type_bucket == "percent"
        assert m.is_percent_like is True

    def test_twenty_percent(self):
        ms = _extract_opt_role_mentions(
            "Allocate twenty percent to advertising.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.2) < 1e-9), None)
        assert m is not None, "twenty percent should produce value≈0.2"
        assert m.type_bucket == "percent"


# ===========================================================================
# 2. Fraction words
# ===========================================================================

class TestFractionWords:
    """half, one-third, quarter, three-quarters → kind='percent'."""

    def test_half(self):
        ms = _extract_opt_role_mentions(
            "At least half of the total production must be exported.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.5) < 1e-9), None)
        assert m is not None, "half should produce value=0.5"
        assert m.type_bucket == "percent"
        assert m.is_percent_like is True

    def test_one_third(self):
        ms = _extract_opt_role_mentions(
            "Exactly one-third of production must be exported.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 1.0 / 3.0) < 1e-6), None)
        assert m is not None, "one-third should produce value≈0.333"
        assert m.type_bucket == "percent"

    def test_quarter(self):
        ms = _extract_opt_role_mentions(
            "No more than a quarter of the budget can be spent here.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.25) < 1e-9), None)
        assert m is not None, "quarter should produce value=0.25"
        assert m.type_bucket == "percent"

    def test_three_quarters(self):
        ms = _extract_opt_role_mentions(
            "At least three-quarters of production must be domestic.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.75) < 1e-9), None)
        assert m is not None, "three-quarters should produce value=0.75"
        assert m.type_bucket == "percent"

    def test_old_extraction_path_half(self):
        """_extract_num_tokens (older constrained assignment path) also handles 'half'."""
        toks = _extract_num_tokens("At least half of production must be exported.", "orig")
        m = next((t for t in toks if t.value is not None and abs(t.value - 0.5) < 1e-9), None)
        assert m is not None
        assert m.kind == "percent"

    def test_fraction_word_not_enumeration_item(self):
        """'half and quarter' should NOT produce an enumeration-derived count of 2."""
        ms = _extract_opt_role_mentions("half and quarter of the budget.", "orig")
        # No derived_count mention should appear (fraction words are excluded from enum items)
        derived = [m for m in ms if "derived_count" in m.role_tags]
        assert len(derived) == 0, "Fraction words must not be treated as enumeration items"


# ===========================================================================
# 3. Rate-context decimal (0.4 with "rate" in context)
# ===========================================================================

class TestRateContextDecimal:
    """0.4 in a 'rate' context should be treated as percent-like."""

    def test_defect_rate_decimal(self):
        ms = _extract_opt_role_mentions(
            "The defect rate is 0.4.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.4) < 1e-9), None)
        assert m is not None
        assert m.type_bucket == "percent", (
            "0.4 in a rate context should be typed percent"
        )
        assert m.is_percent_like is True

    def test_fraction_context_decimal(self):
        """0.3 with 'fraction' context should also be percent-like."""
        ms = _extract_opt_role_mentions(
            "The fraction exported is 0.3.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.3) < 1e-9), None)
        assert m is not None
        assert m.type_bucket == "percent"


# ===========================================================================
# 4. Non-percent numerals stay non-percent
# ===========================================================================

class TestNonPercentNumerals:
    """Ordinary numbers without percent context must NOT be classified as percent."""

    def test_integer_workers_machines(self):
        ms = _extract_opt_role_mentions(
            "There are 40 workers and 25 machines.", "orig"
        )
        for m in ms:
            assert m.type_bucket != "percent", (
                f"Plain integer {m.value} should not be percent-typed"
            )
            assert m.is_percent_like is False

    def test_decimal_tons_not_percent(self):
        ms = _extract_opt_role_mentions(
            "The company has 0.4 tons of material.", "orig"
        )
        m = next((x for x in ms if x.value is not None and abs(x.value - 0.4) < 1e-9), None)
        assert m is not None
        assert m.type_bucket != "percent", (
            "0.4 tons should NOT be treated as percent-like"
        )
        assert m.is_percent_like is False

    def test_large_budget_not_percent(self):
        ms = _extract_opt_role_mentions(
            "The total budget is 50000 dollars.", "orig"
        )
        for m in ms:
            assert m.type_bucket != "percent"


# ===========================================================================
# 5. Slot typing with pct / ratio / proportion / share
# ===========================================================================

class TestSlotTyping:
    """_expected_type should return 'percent' for slot names with pct/ratio/proportion/share."""

    def test_pct_suffix(self):
        assert _expected_type("DiscountPct") == "percent"
        assert _expected_type("MarketingPct") == "percent"

    def test_ratio_substring(self):
        assert _expected_type("WaterRatio") == "percent"
        assert _expected_type("LandRatio") == "percent"

    def test_proportion_substring(self):
        assert _expected_type("ExportProportion") == "percent"

    def test_share_substring(self):
        assert _expected_type("MarketingShare") == "percent"
        assert _expected_type("LaborShare") == "percent"

    def test_rate_substring(self):
        assert _expected_type("DefectRate") == "percent"
        assert _expected_type("InterestRate") == "percent"

    def test_fraction_substring(self):
        assert _expected_type("FractionAllocated") == "percent"

    def test_percent_substring(self):
        assert _expected_type("DiscountPercent") == "percent"

    def test_non_percent_slots_not_affected(self):
        """Ensure non-percent slots are not accidentally typed as percent."""
        assert _expected_type("NumWorkers") == "int"
        assert _expected_type("TotalBudget") == "currency"
        assert _expected_type("Revenue") == "currency"
        assert _expected_type("NumProducts") == "int"


# ===========================================================================
# 6. Hard incompatibility rules
# ===========================================================================

class TestHardIncompatibility:
    """percent ↔ int/float must be hard incompatible in both directions."""

    def test_percent_slot_rejects_int(self):
        assert _is_type_incompatible("percent", "int") is True

    def test_percent_slot_rejects_float(self):
        assert _is_type_incompatible("percent", "float") is True

    def test_percent_slot_rejects_currency(self):
        assert _is_type_incompatible("percent", "currency") is True

    def test_int_slot_rejects_percent(self):
        assert _is_type_incompatible("int", "percent") is True

    def test_float_slot_rejects_percent(self):
        assert _is_type_incompatible("float", "percent") is True

    def test_currency_slot_rejects_percent(self):
        assert _is_type_incompatible("currency", "percent") is True

    def test_compatible_pairs_unchanged(self):
        """Sanity: normal int/float compatibility still holds."""
        assert _is_type_incompatible("int", "int") is False
        assert _is_type_incompatible("float", "int") is False
        assert _is_type_incompatible("float", "float") is False
        assert _is_type_incompatible("percent", "percent") is False
        assert _is_type_incompatible("currency", "currency") is False


# ===========================================================================
# 7. End-to-end grounding: percent mention routes only to percent slot
# ===========================================================================

class TestEndToEndPercentGrounding:
    """GCG should route percent mentions exclusively to percent-typed slots."""

    def test_40pct_to_marketing_pct_slot(self):
        result, _, _ = _run_global_consistency_grounding(
            "At least 40% of the budget must go to marketing.",
            "orig",
            ["MarketingPct"],
        )
        assert "MarketingPct" in result
        assert abs(result["MarketingPct"] - 0.4) < 1e-6

    def test_percent_mention_not_assigned_to_count_slot(self):
        """40% must not fill NumWorkers (int slot); only 5 should."""
        result, _, _ = _run_global_consistency_grounding(
            "We need 40% marketing allocation and 5 workers.",
            "orig",
            ["MarketingPct", "NumWorkers"],
        )
        assert result.get("MarketingPct", None) is not None
        assert abs(result["MarketingPct"] - 0.4) < 1e-6
        assert result.get("NumWorkers", None) is not None
        assert abs(result["NumWorkers"] - 5.0) < 1e-6

    def test_plain_int_not_assigned_to_percent_slot(self):
        """40 workers and 25% marketing: counts go to count slots, percent to percent."""
        result, _, _ = _run_global_consistency_grounding(
            "40 workers and 25% must be in marketing.",
            "orig",
            ["NumWorkers", "MarketingShare"],
        )
        assert result.get("NumWorkers", None) is not None
        assert abs(result["NumWorkers"] - 40.0) < 1e-6
        assert result.get("MarketingShare", None) is not None
        assert abs(result["MarketingShare"] - 0.25) < 1e-6

    def test_half_assigned_to_fraction_slot(self):
        """'half' should be extracted as 0.5 (percent) and fill a fraction/rate slot."""
        result, _, _ = _run_global_consistency_grounding(
            "At least half of the total production must be exported.",
            "orig",
            ["ExportFraction"],
        )
        assert "ExportFraction" in result
        assert abs(result["ExportFraction"] - 0.5) < 1e-6

    def test_one_third_assigned_to_share_slot(self):
        result, _, _ = _run_global_consistency_grounding(
            "Exactly one-third of production must be exported.",
            "orig",
            ["ExportShare"],
        )
        assert "ExportShare" in result
        assert abs(result["ExportShare"] - 1.0 / 3.0) < 1e-4
