"""Group 1 hard-family improvements — measure/attribute-aware linking tests.

These tests validate the camel-case slot-name splitting and numeric-token
boundary stop introduced in Group 1.  They complement the Stage 8 distractor-
suppression tests in ``test_harder_grounding_distractor_suppression.py``.

New capabilities verified here:
  * ``_split_camel_case`` correctly splits camelCase slot names
  * ``_slot_measure_tokens`` returns both the full token and split tokens
  * ``narrow_measure_overlap`` now fires because slot norm_tokens include
    individual component words (e.g. "heating", "cooling", "labor", "wood")
  * Numeric-token boundary stop keeps each mention's right context local,
    preventing cross-measure contamination in "3 heating hours and 5 cooling"

Scenarios covered:
  A. Reversed-order protein vs fat            (was fragile before Group 1)
  B. Reversed-order heating vs cooling        (was fragile before Group 1)
  C. Total labor hours vs per-product hours   (was broken before Group 1)
  D. Labor vs material / wood (Step 10 E)    (new in Group 1)
  E. Cost vs profit (both orderings)
  F. Easy-family regressions (percent, count, bounds, total-vs-per-unit)
  G. Helper function unit tests
"""
from __future__ import annotations

import pytest

from tools.nlp4lp_downstream_utility import _split_camel_case, _slot_measure_tokens
from tools.relation_aware_linking import run_relation_aware_grounding


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _ground(query: str, slots: list[str]) -> dict[str, float]:
    """Run full-ablation grounding and return slot → value mapping."""
    _, mentions, _ = run_relation_aware_grounding(query, "orig", slots)
    return {sn: m.value for sn, m in mentions.items()}


# ---------------------------------------------------------------------------
# G. Helper function unit tests
# ---------------------------------------------------------------------------

class TestSplitCamelCase:
    """_split_camel_case must decompose slot names into component words."""

    def test_two_words(self):
        assert _split_camel_case("HeatingHours") == ["heating", "hours"]

    def test_two_words_cooling(self):
        assert _split_camel_case("CoolingHours") == ["cooling", "hours"]

    def test_three_words(self):
        assert _split_camel_case("TotalLaborHours") == ["total", "labor", "hours"]

    def test_four_words_with_per(self):
        assert _split_camel_case("LaborHoursPerProduct") == [
            "labor", "hours", "per", "product",
        ]

    def test_protein_feed(self):
        assert _split_camel_case("ProteinFeedA") == ["protein", "feed", "a"]

    def test_fat_feed(self):
        assert _split_camel_case("FatFeedA") == ["fat", "feed", "a"]

    def test_already_lowercase(self):
        assert _split_camel_case("profit") == ["profit"]

    def test_underscore_separated(self):
        result = _split_camel_case("labor_hours")
        assert result == ["labor", "hours"]

    def test_single_word_capitalized(self):
        assert _split_camel_case("Budget") == ["budget"]

    def test_all_caps_abbreviation(self):
        # e.g. "ABCCost" → ['abc', 'cost']
        result = _split_camel_case("ABCCost")
        assert "cost" in result


class TestSlotMeasureTokens:
    """_slot_measure_tokens must return the full lowercase identifier AND splits."""

    def test_full_token_present(self):
        tokens = _slot_measure_tokens("HeatingHours")
        assert "heatinghours" in tokens

    def test_split_tokens_present(self):
        tokens = _slot_measure_tokens("HeatingHours")
        assert "heating" in tokens
        assert "hours" in tokens

    def test_no_duplicates(self):
        tokens = _slot_measure_tokens("HeatingHours")
        assert len(tokens) == len(set(tokens))

    def test_total_labor_hours_all_components(self):
        tokens = _slot_measure_tokens("TotalLaborHours")
        for word in ("total", "labor", "hours"):
            assert word in tokens, f"'{word}' should be in {tokens}"

    def test_labor_per_product_all_components(self):
        tokens = _slot_measure_tokens("LaborHoursPerProduct")
        for word in ("labor", "hours", "per", "product"):
            assert word in tokens, f"'{word}' should be in {tokens}"


# ---------------------------------------------------------------------------
# A. Reversed-order protein vs fat (was fragile — now correct by logic)
# ---------------------------------------------------------------------------

class TestReversedProteinFat:
    """When fat appears before protein in text, assignments must not swap."""

    def test_fat_before_protein_protein_slot(self):
        q = "Feed A contains 8 fat and 10 protein."
        result = _ground(q, ["ProteinFeedA", "FatFeedA"])
        assert result["ProteinFeedA"] == pytest.approx(10.0), (
            "10 (protein context) should fill ProteinFeedA even when fat appears first"
        )

    def test_fat_before_protein_fat_slot(self):
        q = "Feed A contains 8 fat and 10 protein."
        result = _ground(q, ["ProteinFeedA", "FatFeedA"])
        assert result["FatFeedA"] == pytest.approx(8.0), (
            "8 (fat context) should fill FatFeedA even when fat appears first"
        )

    def test_no_swap_fat_before_protein(self):
        q = "Feed A contains 8 fat and 10 protein."
        result = _ground(q, ["ProteinFeedA", "FatFeedA"])
        assert result["ProteinFeedA"] != pytest.approx(8.0), "ProteinFeedA must not get the fat value"
        assert result["FatFeedA"] != pytest.approx(10.0), "FatFeedA must not get the protein value"


# ---------------------------------------------------------------------------
# B. Reversed-order heating vs cooling (was fragile — now correct by logic)
# ---------------------------------------------------------------------------

class TestReversedHeatingCooling:
    """When cooling appears before heating in text, assignments must not swap."""

    def test_cooling_before_heating_heating_slot(self):
        q = "Regular glass requires 5 cooling hours and 3 heating hours."
        result = _ground(q, ["HeatingHours", "CoolingHours"])
        assert result["HeatingHours"] == pytest.approx(3.0), (
            "3 (heating context) should fill HeatingHours even when cooling appears first"
        )

    def test_cooling_before_heating_cooling_slot(self):
        q = "Regular glass requires 5 cooling hours and 3 heating hours."
        result = _ground(q, ["HeatingHours", "CoolingHours"])
        assert result["CoolingHours"] == pytest.approx(5.0), (
            "5 (cooling context) should fill CoolingHours even when cooling appears first"
        )

    def test_no_swap_cooling_before_heating(self):
        q = "Regular glass requires 5 cooling hours and 3 heating hours."
        result = _ground(q, ["HeatingHours", "CoolingHours"])
        assert result["HeatingHours"] != pytest.approx(5.0), "Heating slot must not get cooling value"
        assert result["CoolingHours"] != pytest.approx(3.0), "Cooling slot must not get heating value"


# ---------------------------------------------------------------------------
# C. Total labor hours vs per-product hours (was broken — now fixed)
# ---------------------------------------------------------------------------

class TestTotalVsPerProductLaborHours:
    """total-available hours must not be assigned to a per-product coefficient slot."""

    def test_total_hours_slot(self):
        q = "There are 2000 labor hours available, and each product requires 3 labor hours."
        result = _ground(q, ["TotalLaborHours", "LaborHoursPerProduct"])
        assert result["TotalLaborHours"] == pytest.approx(2000.0), (
            "2000 (total available) should fill TotalLaborHours"
        )

    def test_per_product_hours_slot(self):
        q = "There are 2000 labor hours available, and each product requires 3 labor hours."
        result = _ground(q, ["TotalLaborHours", "LaborHoursPerProduct"])
        assert result["LaborHoursPerProduct"] == pytest.approx(3.0), (
            "3 (per-product coefficient) should fill LaborHoursPerProduct"
        )

    def test_no_swap_total_per_product(self):
        q = "There are 2000 labor hours available, and each product requires 3 labor hours."
        result = _ground(q, ["TotalLaborHours", "LaborHoursPerProduct"])
        assert result["TotalLaborHours"] != pytest.approx(3.0), (
            "TotalLaborHours must not receive the per-unit coefficient"
        )
        assert result["LaborHoursPerProduct"] != pytest.approx(2000.0), (
            "LaborHoursPerProduct must not receive the total available hours"
        )


# ---------------------------------------------------------------------------
# D. Labor vs material / wood  (Step 10 E — new in Group 1)
# ---------------------------------------------------------------------------

class TestLaborVsMaterial:
    """Per-product labor hours and wood units must reach their correct slots."""

    def test_labor_hours_slot(self):
        q = "Each table requires 4 labor hours and 6 units of wood."
        result = _ground(q, ["LaborHoursPerTable", "WoodUnitsPerTable"])
        assert result["LaborHoursPerTable"] == pytest.approx(4.0), (
            "4 (labor context) should fill LaborHoursPerTable"
        )

    def test_wood_units_slot(self):
        q = "Each table requires 4 labor hours and 6 units of wood."
        result = _ground(q, ["LaborHoursPerTable", "WoodUnitsPerTable"])
        assert result["WoodUnitsPerTable"] == pytest.approx(6.0), (
            "6 (wood context) should fill WoodUnitsPerTable"
        )

    def test_no_labor_material_swap(self):
        q = "Each table requires 4 labor hours and 6 units of wood."
        result = _ground(q, ["LaborHoursPerTable", "WoodUnitsPerTable"])
        assert result["LaborHoursPerTable"] != pytest.approx(6.0), (
            "LaborHoursPerTable must not receive the wood value"
        )
        assert result["WoodUnitsPerTable"] != pytest.approx(4.0), (
            "WoodUnitsPerTable must not receive the labor value"
        )

    def test_wood_before_labor(self):
        """Reversed order: wood value before labor value."""
        q = "Each table uses 6 wood units and 4 labor hours."
        result = _ground(q, ["LaborHoursPerTable", "WoodUnitsPerTable"])
        assert result["LaborHoursPerTable"] == pytest.approx(4.0), (
            "4 (labor context) should fill LaborHoursPerTable even when wood appears first"
        )
        assert result["WoodUnitsPerTable"] == pytest.approx(6.0), (
            "6 (wood context) should fill WoodUnitsPerTable even when wood appears first"
        )


# ---------------------------------------------------------------------------
# E. Cost vs profit  (both orderings)
# ---------------------------------------------------------------------------

class TestCostVsProfitBothOrders:
    """profit and cost values must reach their correct slots regardless of order."""

    def test_profit_before_cost_profit_slot(self):
        q = "Product X yields 12 dollars profit and costs 5 dollars to produce."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["ProfitPerUnit"] == pytest.approx(12.0), (
            "12 (profit context) should fill ProfitPerUnit"
        )

    def test_profit_before_cost_cost_slot(self):
        q = "Product X yields 12 dollars profit and costs 5 dollars to produce."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["CostPerUnit"] == pytest.approx(5.0), (
            "5 (cost context) should fill CostPerUnit"
        )

    def test_cost_before_profit_profit_slot(self):
        q = "It costs 5 dollars to produce product X and earns 12 dollars profit."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["ProfitPerUnit"] == pytest.approx(12.0), (
            "12 (profit context) should fill ProfitPerUnit even when cost appears first"
        )

    def test_cost_before_profit_cost_slot(self):
        q = "It costs 5 dollars to produce product X and earns 12 dollars profit."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["CostPerUnit"] == pytest.approx(5.0), (
            "5 (cost context) should fill CostPerUnit even when cost appears first"
        )


# ---------------------------------------------------------------------------
# F. Easy-family regression guard
# ---------------------------------------------------------------------------

class TestEasyFamilyRegressions:
    """Group 1 changes must not break any previously-working easy-family behavior."""

    def test_percent_upper_bound_preserved(self):
        q = "At most 40% of workers can be assigned to night shift."
        result = _ground(q, ["MaxNightFraction"])
        assert result["MaxNightFraction"] == pytest.approx(0.4), (
            "Percent extraction must still yield a normalized fraction"
        )

    def test_derived_count_two_types_preserved(self):
        q = "There are two types of jars."
        result = _ground(q, ["NumJarTypes"])
        assert result["NumJarTypes"] == pytest.approx(2.0), (
            "Word-number count must still be extracted"
        )

    def test_bound_directions_preserved(self):
        q = "At least 10 and at most 20 units."
        result = _ground(q, ["MinUnits", "MaxUnits"])
        assert result["MinUnits"] == pytest.approx(10.0), "min bound must still go to MinUnits"
        assert result["MaxUnits"] == pytest.approx(20.0), "max bound must still go to MaxUnits"

    def test_total_vs_per_unit_preserved(self):
        q = "There are 2000 labor hours available, and each product requires 2 labor hours."
        result = _ground(q, ["TotalLaborHours", "LaborHoursPerProduct"])
        assert result["TotalLaborHours"] == pytest.approx(2000.0)
        assert result["LaborHoursPerProduct"] == pytest.approx(2.0)
