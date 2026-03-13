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
  * ``is_cost_like``, ``is_profit_like``, ``is_demand_like``,
    ``is_resource_like``, ``is_time_like`` flags on MentionOptIR (Stage 2)
  * ``narrow_measure_conflict`` post-processing (Stage 5)
  * ``role_family_mismatch`` distractor suppression (Stage 6)

Scenarios covered:
  A. Reversed-order protein vs fat            (was fragile before Group 1)
  B. Reversed-order heating vs cooling        (was fragile before Group 1)
  C. Total labor hours vs per-product hours   (was broken before Group 1)
  D. Labor vs material / wood (Step 10 E)    (new in Group 1)
  E. Cost vs profit (both orderings)
  F. Easy-family regressions (percent, count, bounds, total-vs-per-unit)
  G. Helper function unit tests
  H. Stage 2 structured mention flags         (new in Group 1 completion)
  I. Stage 5 competing-measure conflict       (new in Group 1 completion)
  J. Stage 6 role-family mismatch             (new in Group 1 completion)
"""
from __future__ import annotations

import pytest

from tools.nlp4lp_downstream_utility import (
    _split_camel_case,
    _slot_measure_tokens,
    _extract_opt_role_mentions,
)
from tools.relation_aware_linking import (
    build_mention_slot_links,
    relation_aware_local_score,
    run_relation_aware_grounding,
)


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


# ---------------------------------------------------------------------------
# H. Stage 2 — structured mention role-family flags
# ---------------------------------------------------------------------------

class TestMentionRoleFamilyFlags:
    """MentionOptIR must carry correct role-family flags from tight ±2-token window."""

    def test_cost_like_flag_set_for_costs_context(self):
        q = "It costs 5 dollars to produce product X."
        mentions = _extract_opt_role_mentions(q, "orig")
        m = next((m for m in mentions if m.value == 5.0), None)
        assert m is not None
        assert m.is_cost_like is True, "5 directly after 'costs' must be flagged is_cost_like"
        assert m.is_profit_like is False, "5 has no profit cue in tight window"

    def test_profit_like_flag_set_for_profit_context(self):
        q = "Product X yields 12 dollars profit."
        mentions = _extract_opt_role_mentions(q, "orig")
        m = next((m for m in mentions if m.value == 12.0), None)
        assert m is not None
        assert m.is_profit_like is True, "12 before 'profit' must be flagged is_profit_like"
        assert m.is_cost_like is False, "12 has no cost cue in tight window"

    def test_resource_like_flag_set_for_labor_context(self):
        q = "Each product requires 4 labor hours."
        mentions = _extract_opt_role_mentions(q, "orig")
        m = next((m for m in mentions if m.value == 4.0), None)
        assert m is not None
        assert m.is_resource_like is True, "4 next to 'labor' must be flagged is_resource_like"

    def test_time_like_flag_set_for_hours_context(self):
        q = "Each product requires 4 labor hours."
        mentions = _extract_opt_role_mentions(q, "orig")
        m = next((m for m in mentions if m.value == 4.0), None)
        assert m is not None
        assert m.is_time_like is True, "4 next to 'hours' must be flagged is_time_like"

    def test_demand_like_flag_set_for_demand_context(self):
        # "demand of 100" — "demand" is within the ±2 tight window of 100
        q = "There is a demand of 100 units."
        mentions = _extract_opt_role_mentions(q, "orig")
        m = next((m for m in mentions if m.value == 100.0), None)
        assert m is not None
        assert m.is_demand_like is True, "100 immediately after 'demand of' must be flagged is_demand_like"

    def test_flags_false_for_unrelated_context(self):
        """A plain integer with no role cues in the tight window must have all flags False."""
        q = "There are 7 products available."
        mentions = _extract_opt_role_mentions(q, "orig")
        m = next((m for m in mentions if m.value == 7.0), None)
        assert m is not None
        assert m.is_cost_like is False
        assert m.is_profit_like is False
        assert m.is_demand_like is False

    def test_tight_window_prevents_bleed_profit_before_cost(self):
        """In 'yields 12 profit and costs 5', 5 must be is_cost_like only (not profit_like)."""
        q = "Product X yields 12 dollars profit and costs 5 dollars to produce."
        mentions = _extract_opt_role_mentions(q, "orig")
        m5 = next((m for m in mentions if m.value == 5.0), None)
        assert m5 is not None
        assert m5.is_cost_like is True, "5 after 'costs' must be cost_like"
        assert m5.is_profit_like is False, (
            "5 must NOT be profit_like: 'profit' belongs to 12's territory, "
            "not the ±2-token tight window of 5"
        )

    def test_tight_window_prevents_bleed_cost_before_profit(self):
        """In 'costs 5 ... and yields 12 profit', 12 must be is_profit_like only."""
        q = "Product X costs 5 dollars to produce and yields 12 dollars profit."
        mentions = _extract_opt_role_mentions(q, "orig")
        m12 = next((m for m in mentions if m.value == 12.0), None)
        assert m12 is not None
        assert m12.is_profit_like is True, "12 near 'yields profit' must be profit_like"
        assert m12.is_cost_like is False, (
            "'costs' belongs to 5's territory; must not bleed into 12's tight window"
        )


# ---------------------------------------------------------------------------
# I. Stage 5 — competing-measure conflict (narrow_measure_conflict)
# ---------------------------------------------------------------------------

class TestNarrowMeasureConflict:
    """narrow_measure_conflict must be positive when another slot matches the context better."""

    def _links_for(self, query: str, slots: list[str]):
        links, _, _, _, _ = build_mention_slot_links(query, "orig", slots)
        return links

    def test_conflict_zero_for_clear_match(self):
        """3 in 'requires 3 heating hours' should have conflict=0 for HeatingHoursPerUnit."""
        links = self._links_for(
            "Regular glass requires 3 heating hours and 5 cooling hours.",
            ["HeatingHoursPerUnit", "CoolingHoursPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 3.0 and lnk.slot_name == "HeatingHoursPerUnit"
        )
        assert lnk.narrow_measure_conflict == 0, (
            "3 next to 'heating' should have NO conflict for HeatingHoursPerUnit"
        )

    def test_conflict_positive_for_wrong_slot(self):
        """3 in 'requires 3 heating hours' should have conflict>0 for CoolingHoursPerUnit."""
        links = self._links_for(
            "Regular glass requires 3 heating hours and 5 cooling hours.",
            ["HeatingHoursPerUnit", "CoolingHoursPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 3.0 and lnk.slot_name == "CoolingHoursPerUnit"
        )
        assert lnk.narrow_measure_conflict > 0, (
            "3 next to 'heating' (not cooling) should have positive conflict for CoolingHoursPerUnit"
        )

    def test_conflict_reduces_wrong_slot_score(self):
        """The scoring penalty for conflict must lower score of the wrong slot."""
        links = self._links_for(
            "Regular glass requires 3 heating hours and 5 cooling hours.",
            ["HeatingHoursPerUnit", "CoolingHoursPerUnit"],
        )
        heat_lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 3.0 and lnk.slot_name == "HeatingHoursPerUnit"
        )
        cool_lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 3.0 and lnk.slot_name == "CoolingHoursPerUnit"
        )
        heat_score, _ = relation_aware_local_score(heat_lnk, "semantic")
        cool_score, _ = relation_aware_local_score(cool_lnk, "semantic")
        assert heat_score > cool_score, (
            "3 (heating context) must score higher for HeatingHoursPerUnit than CoolingHoursPerUnit"
        )

    def test_total_vs_per_unit_conflict(self):
        """4 (labor cue) should have high conflict for Budget slot when total budget is present."""
        links = self._links_for(
            "There are 100 workers, each product needs 4 labor hours, and total budget is 5000.",
            ["LaborHoursPerProduct", "Budget"],
        )
        budget_lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 4.0 and lnk.slot_name == "Budget"
        )
        assert budget_lnk.narrow_measure_conflict > 0, (
            "4 (labor cue) should have conflict when assigned to Budget slot"
        )


# ---------------------------------------------------------------------------
# J. Stage 6 — role-family mismatch distractor suppression
# ---------------------------------------------------------------------------

class TestRoleFamilyMismatch:
    """role_family_mismatch must fire for clear role-family conflicts."""

    def _links_for(self, query: str, slots: list[str]):
        links, _, _, _, _ = build_mention_slot_links(query, "orig", slots)
        return links

    def test_cost_mention_to_profit_slot_mismatch(self):
        """5 (cost context) assigned to ProfitPerUnit must be flagged as mismatch."""
        links = self._links_for(
            "Product X costs 5 dollars to produce and yields 12 dollars profit.",
            ["ProfitPerUnit", "CostPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 5.0 and lnk.slot_name == "ProfitPerUnit"
        )
        assert lnk.role_family_mismatch is True, (
            "5 (cost cue only) → ProfitPerUnit should be flagged as role_family_mismatch"
        )

    def test_profit_mention_to_cost_slot_mismatch(self):
        """12 (profit context) assigned to CostPerUnit must be flagged as mismatch."""
        links = self._links_for(
            "Product X costs 5 dollars to produce and yields 12 dollars profit.",
            ["ProfitPerUnit", "CostPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 12.0 and lnk.slot_name == "CostPerUnit"
        )
        assert lnk.role_family_mismatch is True, (
            "12 (profit cue only) → CostPerUnit should be flagged as role_family_mismatch"
        )

    def test_cost_mention_to_cost_slot_no_mismatch(self):
        """5 (cost context) assigned to CostPerUnit must NOT be flagged as mismatch."""
        links = self._links_for(
            "Product X costs 5 dollars to produce and yields 12 dollars profit.",
            ["ProfitPerUnit", "CostPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 5.0 and lnk.slot_name == "CostPerUnit"
        )
        assert lnk.role_family_mismatch is False, (
            "5 (cost cue) → CostPerUnit should NOT be flagged as mismatch"
        )

    def test_profit_mention_to_profit_slot_no_mismatch(self):
        """12 (profit context) assigned to ProfitPerUnit must NOT be flagged as mismatch."""
        links = self._links_for(
            "Product X costs 5 dollars to produce and yields 12 dollars profit.",
            ["ProfitPerUnit", "CostPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 12.0 and lnk.slot_name == "ProfitPerUnit"
        )
        assert lnk.role_family_mismatch is False, (
            "12 (profit cue) → ProfitPerUnit should NOT be flagged as mismatch"
        )

    def test_demand_mention_to_profit_slot_mismatch(self):
        """100 (demand context) → ProfitPerUnit must be flagged as role-family mismatch."""
        links = self._links_for(
            "Each unit has a demand of 100, and the profit per unit is 25.",
            ["Demand", "ProfitPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 100.0 and lnk.slot_name == "ProfitPerUnit"
        )
        assert lnk.role_family_mismatch is True, (
            "100 (demand cue) → ProfitPerUnit should be flagged as role_family_mismatch"
        )

    def test_resource_time_mention_to_profit_slot_mismatch(self):
        """4 (labor/hours context) → ProfitPerUnit must be flagged as role-family mismatch."""
        links = self._links_for(
            "Each product requires 4 labor hours, and the profit per unit is 25.",
            ["LaborHoursPerProduct", "ProfitPerUnit"],
        )
        lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 4.0 and lnk.slot_name == "ProfitPerUnit"
        )
        assert lnk.role_family_mismatch is True, (
            "4 (labor/hours cue) → ProfitPerUnit should be flagged as role_family_mismatch"
        )

    def test_mismatch_penalty_lowers_score(self):
        """role_family_mismatch must lower the semantic-mode score of the mismatched pair."""
        links = self._links_for(
            "Product X costs 5 dollars to produce and yields 12 dollars profit.",
            ["ProfitPerUnit", "CostPerUnit"],
        )
        # 12 → ProfitPerUnit (no mismatch) vs 12 → CostPerUnit (mismatch)
        profit_lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 12.0 and lnk.slot_name == "ProfitPerUnit"
        )
        cost_lnk = next(
            lnk for lnk in links
            if lnk.mention_feats.value == 12.0 and lnk.slot_name == "CostPerUnit"
        )
        profit_score, _ = relation_aware_local_score(profit_lnk, "semantic")
        cost_score, _ = relation_aware_local_score(cost_lnk, "semantic")
        assert profit_score > cost_score, (
            "12 (profit cue) must score higher for ProfitPerUnit than CostPerUnit"
        )

    def test_ambiguous_context_no_mismatch(self):
        """When both cost and profit cues appear (ambiguous tight window), no mismatch fires."""
        # This tests the conservative rule: only fire when exactly one flag is set.
        # In this short sentence, 'cost' and 'profit' are both in the tight window of each number.
        # We verify that no mismatch fires to avoid false positives.
        links = self._links_for(
            "Cost 5 profit 12.",
            ["ProfitPerUnit", "CostPerUnit"],
        )
        # For mentions that have BOTH is_cost_like and is_profit_like, mismatch must not fire
        for lnk in links:
            mf = lnk.mention_feats
            if mf.is_cost_like and mf.is_profit_like:
                assert lnk.role_family_mismatch is False, (
                    f"Ambiguous context (both cost_like and profit_like) for value={mf.value} "
                    f"→ {lnk.slot_name} should NOT trigger role_family_mismatch"
                )
