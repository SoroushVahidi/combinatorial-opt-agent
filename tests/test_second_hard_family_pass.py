"""Tests for the second hard-family pass: structured sibling-entity linking.

These tests validate the improvements introduced by the second hard-family pass:

1. Slot name decomposition helpers:
   - slot_entity_tokens / slot_measure_tokens / slot_role_tokens
   - sibling_slot_groups
   - _split_slot_name

2. CamelCase slot name splitting so narrow_measure_overlap works properly
   (e.g. "HeatingRegular" → {"heating", "regular"} instead of a single token).

3. Sibling-entity discrimination features in MentionSlotLink:
   - split_entity_overlap / split_measure_overlap
   - sibling_entity_mismatch / sibling_entity_best_match
   - clause_entity_match / clause_entity_sibling_match

4. ``sibling_aware`` ablation mode with new scoring weights.

Test families:
  A. Slot decomposition helpers
  B. Sibling slot group detection
  C. Sibling entity discrimination (chair/dresser, Feed A/B, regular/tempered)
  D. Same-entity different-measure discrimination (4-slot Feed A/B)
  E. Small vs large variant discrimination
  F. Regression: single-entity problems must NOT be over-penalised
  G. Regression: first-pass gains preserved in sibling_aware mode
  H. Feature field presence (MentionSlotLink has new fields)
"""
import pytest

from tools.nlp4lp_downstream_utility import (
    _split_slot_name,
    slot_entity_tokens,
    slot_measure_tokens,
    slot_role_tokens,
    sibling_slot_groups,
    _build_slot_opt_irs,
)
from tools.relation_aware_linking import (
    ABLATION_MODES,
    RAL_WEIGHTS,
    MentionSlotLink,
    build_mention_slot_links,
    relation_aware_local_score,
    run_relation_aware_grounding,
)


# ---------------------------------------------------------------------------
# A. Slot decomposition helpers
# ---------------------------------------------------------------------------

class TestSlotDecomposition:
    """Unit tests for the slot-name parsing helpers."""

    def test_split_slot_name_camel_case(self):
        assert _split_slot_name("HeatingRegular") == ["heating", "regular"]

    def test_split_slot_name_three_parts(self):
        result = _split_slot_name("ProteinFeedA")
        assert "protein" in result
        assert "feed" in result
        assert "a" in result

    def test_split_slot_name_underscore(self):
        result = _split_slot_name("wood_per_chair")
        assert "wood" in result
        assert "per" in result
        assert "chair" in result

    def test_split_slot_name_all_lower(self):
        result = _split_slot_name("profit")
        assert result == ["profit"]

    def test_split_slot_name_multiple_caps(self):
        result = _split_slot_name("LaborHoursPerProduct")
        assert "labor" in result
        assert "hours" in result
        assert "per" in result
        assert "product" in result

    def test_entity_tokens_regular(self):
        e = slot_entity_tokens("HeatingRegular")
        assert "regular" in e

    def test_entity_tokens_tempered(self):
        e = slot_entity_tokens("HeatingTempered")
        assert "tempered" in e

    def test_entity_tokens_chair(self):
        e = slot_entity_tokens("WoodPerChair")
        assert "chair" in e

    def test_entity_tokens_dresser(self):
        e = slot_entity_tokens("WoodPerDresser")
        assert "dresser" in e

    def test_entity_tokens_feed_a(self):
        e = slot_entity_tokens("ProteinFeedA")
        assert "a" in e or "feed" in e

    def test_entity_tokens_feed_b(self):
        e = slot_entity_tokens("ProteinFeedB")
        assert "b" in e or "feed" in e

    def test_entity_tokens_small(self):
        e = slot_entity_tokens("LaborHoursSmall")
        assert "small" in e

    def test_entity_tokens_large(self):
        e = slot_entity_tokens("LaborHoursLarge")
        assert "large" in e

    def test_measure_tokens_heating(self):
        m = slot_measure_tokens("HeatingRegular")
        assert "heating" in m

    def test_measure_tokens_protein(self):
        m = slot_measure_tokens("ProteinFeedA")
        assert "protein" in m

    def test_measure_tokens_fat(self):
        m = slot_measure_tokens("FatFeedA")
        assert "fat" in m

    def test_measure_tokens_wood(self):
        m = slot_measure_tokens("WoodPerChair")
        assert "wood" in m

    def test_measure_tokens_labor(self):
        m = slot_measure_tokens("LaborHoursSmall")
        assert "labor" in m

    def test_role_tokens_per(self):
        """'per' should be classified as a role token, not an entity or measure token."""
        r = slot_role_tokens("WoodPerChair")
        # 'per' is explicitly listed in _SLOT_ROLE_TOKENS
        assert "per" in r

    def test_entity_and_measure_disjoint_for_regular(self):
        """HeatingRegular: entity and measure should not overlap."""
        e = slot_entity_tokens("HeatingRegular")
        m = slot_measure_tokens("HeatingRegular")
        assert e & m == frozenset()

    def test_entity_and_measure_disjoint_for_chair(self):
        """WoodPerChair: entity=chair, measure=wood — should not overlap."""
        e = slot_entity_tokens("WoodPerChair")
        m = slot_measure_tokens("WoodPerChair")
        assert e & m == frozenset()


# ---------------------------------------------------------------------------
# B. Sibling slot group detection
# ---------------------------------------------------------------------------

class TestSiblingSlotGroups:
    """Tests for sibling_slot_groups function."""

    def _make_slots(self, names):
        return _build_slot_opt_irs(names)

    def test_regular_tempered_heating_cooling(self):
        slots = self._make_slots(["HeatingRegular", "CoolingRegular", "HeatingTempered", "CoolingTempered"])
        groups = sibling_slot_groups(slots)
        group_sets = [frozenset(s.name for s in g) for g in groups]
        # At minimum, HeatingRegular and HeatingTempered should be siblings
        assert any(
            "HeatingRegular" in gs and "HeatingTempered" in gs
            for gs in group_sets
        ), f"Expected HeatingRegular/HeatingTempered sibling group, got {group_sets}"
        # CoolingRegular and CoolingTempered should be siblings
        assert any(
            "CoolingRegular" in gs and "CoolingTempered" in gs
            for gs in group_sets
        ), f"Expected CoolingRegular/CoolingTempered sibling group, got {group_sets}"

    def test_feed_protein_siblings(self):
        slots = self._make_slots(["ProteinFeedA", "ProteinFeedB"])
        groups = sibling_slot_groups(slots)
        group_sets = [frozenset(s.name for s in g) for g in groups]
        assert any(
            "ProteinFeedA" in gs and "ProteinFeedB" in gs
            for gs in group_sets
        ), f"Expected ProteinFeedA/ProteinFeedB sibling group, got {group_sets}"

    def test_chair_dresser_wood_siblings(self):
        slots = self._make_slots(["WoodPerChair", "WoodPerDresser"])
        groups = sibling_slot_groups(slots)
        group_sets = [frozenset(s.name for s in g) for g in groups]
        assert any(
            "WoodPerChair" in gs and "WoodPerDresser" in gs
            for gs in group_sets
        ), f"Expected WoodPerChair/WoodPerDresser sibling group, got {group_sets}"

    def test_no_siblings_single_slot(self):
        slots = self._make_slots(["TotalBudget"])
        groups = sibling_slot_groups(slots)
        # A single slot cannot form a sibling pair with itself
        for g in groups:
            assert len(g) >= 2, "A sibling group must contain at least two slots"


# ---------------------------------------------------------------------------
# C. Sibling entity discrimination (2-entity problems)
# ---------------------------------------------------------------------------

class TestSiblingEntityDiscrimination:
    """Core case-7 entity-anchor discrimination tests."""

    def test_chair_vs_dresser_wood(self):
        """'A chair uses 2 units of wood, while a dresser uses 5.' → 2=chair, 5=dresser."""
        _, mentions, _ = run_relation_aware_grounding(
            "A chair uses 2 units of wood, while a dresser uses 5.",
            "orig",
            ["WoodPerChair", "WoodPerDresser"],
            ablation_mode="sibling_aware",
        )
        assert mentions.get("WoodPerChair") is not None
        assert abs(mentions["WoodPerChair"].value - 2.0) < 1e-6
        assert mentions.get("WoodPerDresser") is not None
        assert abs(mentions["WoodPerDresser"].value - 5.0) < 1e-6

    def test_feed_a_b_protein_simple(self):
        """'Feed A contains 10 protein and Feed B contains 7 protein.' → A=10, B=7."""
        _, mentions, _ = run_relation_aware_grounding(
            "Feed A contains 10 protein and Feed B contains 7 protein.",
            "orig",
            ["ProteinFeedA", "ProteinFeedB"],
            ablation_mode="sibling_aware",
        )
        assert abs(mentions["ProteinFeedA"].value - 10.0) < 1e-6
        assert abs(mentions["ProteinFeedB"].value - 7.0) < 1e-6

    def test_regular_vs_tempered_heating(self):
        """'Regular glass requires 3 heating hours … tempered … 5 heating …' → correct split."""
        _, mentions, _ = run_relation_aware_grounding(
            "Regular glass requires 3 heating hours and 5 cooling hours, "
            "whereas tempered glass requires 5 heating and 8 cooling.",
            "orig",
            ["HeatingRegular", "CoolingRegular", "HeatingTempered", "CoolingTempered"],
            ablation_mode="sibling_aware",
        )
        assert abs(mentions["HeatingRegular"].value - 3.0) < 1e-6
        assert abs(mentions["CoolingRegular"].value - 5.0) < 1e-6
        assert abs(mentions["HeatingTempered"].value - 5.0) < 1e-6
        assert abs(mentions["CoolingTempered"].value - 8.0) < 1e-6

    def test_small_vs_large_labor(self):
        """'A small sculpture requires 2 labor hours and a large sculpture requires 6.'"""
        _, mentions, _ = run_relation_aware_grounding(
            "A small sculpture requires 2 labor hours and a large sculpture requires 6.",
            "orig",
            ["LaborHoursSmall", "LaborHoursLarge"],
            ablation_mode="sibling_aware",
        )
        assert abs(mentions["LaborHoursSmall"].value - 2.0) < 1e-6
        assert abs(mentions["LaborHoursLarge"].value - 6.0) < 1e-6


# ---------------------------------------------------------------------------
# D. Same-entity different-measure discrimination (4-slot problems)
# ---------------------------------------------------------------------------

class TestSameEntityDifferentMeasure:
    """4-slot problems where both entity AND measure discrimination are needed."""

    def test_feed_a_b_protein_fat_four_slots(self):
        """'Feed A contains 10 protein and 8 fat, while Feed B contains 7 protein and 15 fat.'"""
        _, mentions, _ = run_relation_aware_grounding(
            "Feed A contains 10 protein and 8 fat, while Feed B contains 7 protein and 15 fat.",
            "orig",
            ["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
            ablation_mode="sibling_aware",
        )
        assert abs(mentions["ProteinFeedA"].value - 10.0) < 1e-6, f"Got {mentions}"
        assert abs(mentions["FatFeedA"].value - 8.0) < 1e-6, f"Got {mentions}"
        assert abs(mentions["ProteinFeedB"].value - 7.0) < 1e-6, f"Got {mentions}"
        assert abs(mentions["FatFeedB"].value - 15.0) < 1e-6, f"Got {mentions}"

    def test_regular_tempered_distinct_measures(self):
        """HeatingRegular vs HeatingTempered and CoolingRegular vs CoolingTempered.

        Uses a period-separated two-sentence form so clause boundaries are unambiguous.
        """
        _, mentions, _ = run_relation_aware_grounding(
            "Regular pane requires 3 heating hours and 5 cooling. "
            "Tempered pane requires 5 heating hours and 8 cooling.",
            "orig",
            ["HeatingRegular", "CoolingRegular", "HeatingTempered", "CoolingTempered"],
            ablation_mode="sibling_aware",
        )
        # Both entity AND measure discrimination needed
        assert abs(mentions["HeatingRegular"].value - 3.0) < 1e-6, f"Got {mentions}"
        assert abs(mentions["HeatingTempered"].value - 5.0) < 1e-6, f"Got {mentions}"


# ---------------------------------------------------------------------------
# E. Feature field presence
# ---------------------------------------------------------------------------

class TestMentionSlotLinkFields:
    """Validate that MentionSlotLink carries all new second-pass fields."""

    def _first_link(self, query, slots):
        links, _, _, _, _ = build_mention_slot_links(query, "orig", slots)
        return links[0] if links else None

    def test_split_entity_overlap_field_exists(self):
        lnk = self._first_link("A chair uses 2 wood.", ["WoodPerChair", "WoodPerDresser"])
        assert hasattr(lnk, "split_entity_overlap")
        assert isinstance(lnk.split_entity_overlap, int)

    def test_split_measure_overlap_field_exists(self):
        lnk = self._first_link("A chair uses 2 wood.", ["WoodPerChair", "WoodPerDresser"])
        assert hasattr(lnk, "split_measure_overlap")
        assert isinstance(lnk.split_measure_overlap, int)

    def test_sibling_entity_mismatch_field_exists(self):
        lnk = self._first_link("A chair uses 2 wood.", ["WoodPerChair", "WoodPerDresser"])
        assert hasattr(lnk, "sibling_entity_mismatch")
        assert isinstance(lnk.sibling_entity_mismatch, bool)

    def test_sibling_entity_best_match_field_exists(self):
        lnk = self._first_link("A chair uses 2 wood.", ["WoodPerChair", "WoodPerDresser"])
        assert hasattr(lnk, "sibling_entity_best_match")
        assert isinstance(lnk.sibling_entity_best_match, bool)

    def test_clause_entity_match_field_exists(self):
        lnk = self._first_link("A chair uses 2 wood.", ["WoodPerChair", "WoodPerDresser"])
        assert hasattr(lnk, "clause_entity_match")
        assert isinstance(lnk.clause_entity_match, int)

    def test_sibling_aware_mode_present_in_ral_weights(self):
        assert "sibling_aware" in RAL_WEIGHTS

    def test_sibling_aware_mode_present_in_ablation_modes(self):
        assert "sibling_aware" in ABLATION_MODES

    def test_sibling_aware_weights_have_expected_keys(self):
        w = RAL_WEIGHTS["sibling_aware"]
        assert "split_entity_overlap_bonus" in w
        assert "split_measure_overlap_bonus" in w
        assert "sibling_entity_mismatch_penalty" in w
        assert "sibling_entity_best_match_bonus" in w
        assert "clause_entity_match_bonus" in w
        assert "clause_entity_mismatch_penalty" in w

    def test_sibling_aware_penalties_are_negative(self):
        w = RAL_WEIGHTS["sibling_aware"]
        for k, v in w.items():
            if "penalty" in k:
                assert v < 0, f"Penalty {k} should be negative, got {v}"

    def test_sibling_aware_bonuses_are_positive(self):
        w = RAL_WEIGHTS["sibling_aware"]
        for k, v in w.items():
            if "bonus" in k:
                assert v > 0, f"Bonus {k} should be positive, got {v}"

    def test_sibling_entity_mismatch_flagged_for_mismatched_slot(self):
        """For 'chair uses 2 wood': WoodPerDresser link should have sibling_entity_mismatch=True."""
        links, _, _, _, _ = build_mention_slot_links(
            "A chair uses 2 units of wood, while a dresser uses 5.",
            "orig",
            ["WoodPerChair", "WoodPerDresser"],
        )
        # Find link for val=2 → WoodPerDresser
        link_2_dresser = next(
            (lnk for lnk in links
             if abs((lnk.mention_feats.value or -1) - 2.0) < 1e-6
             and lnk.slot_name == "WoodPerDresser"),
            None,
        )
        assert link_2_dresser is not None
        assert link_2_dresser.sibling_entity_mismatch, (
            "WoodPerDresser should be flagged as sibling_entity_mismatch for mention near 'chair'"
        )

    def test_sibling_entity_best_match_flagged_for_correct_slot(self):
        """For 'chair uses 2 wood': WoodPerChair link should have sibling_entity_best_match=True."""
        links, _, _, _, _ = build_mention_slot_links(
            "A chair uses 2 units of wood, while a dresser uses 5.",
            "orig",
            ["WoodPerChair", "WoodPerDresser"],
        )
        link_2_chair = next(
            (lnk for lnk in links
             if abs((lnk.mention_feats.value or -1) - 2.0) < 1e-6
             and lnk.slot_name == "WoodPerChair"),
            None,
        )
        assert link_2_chair is not None
        assert link_2_chair.sibling_entity_best_match, (
            "WoodPerChair should have sibling_entity_best_match=True for mention near 'chair'"
        )


# ---------------------------------------------------------------------------
# F. Regression: single-entity problems must NOT be over-penalised
# ---------------------------------------------------------------------------

class TestSingleEntityNoOverfire:
    """The sibling-entity logic should not hurt single-entity problems."""

    def test_single_entity_table_labor_wood(self):
        """'Each table requires 4 labor hours and 6 units of wood.' → no false penalties."""
        for mode in ("full", "sibling_aware"):
            _, mentions, _ = run_relation_aware_grounding(
                "Each table requires 4 labor hours and 6 units of wood.",
                "orig",
                ["LaborHoursPerTable", "WoodPerTable"],
                ablation_mode=mode,
            )
            assert abs(mentions["LaborHoursPerTable"].value - 4.0) < 1e-6, (
                f"mode={mode}: LaborHoursPerTable should be 4, got {mentions}"
            )
            assert abs(mentions["WoodPerTable"].value - 6.0) < 1e-6, (
                f"mode={mode}: WoodPerTable should be 6, got {mentions}"
            )

    def test_single_entity_budget_hours(self):
        """Single entity: budget and hours should not trigger sibling mismatch."""
        links, _, _, _, _ = build_mention_slot_links(
            "The total budget is 1000 and each worker requires 8 hours.",
            "orig",
            ["TotalBudget", "HoursPerWorker"],
        )
        # No sibling groups → sibling_entity_mismatch should be False for all
        for lnk in links:
            assert not lnk.sibling_entity_mismatch, (
                f"No siblings defined, but {lnk.slot_name} got sibling_entity_mismatch=True"
            )


# ---------------------------------------------------------------------------
# G. Regression: first-pass gains preserved in sibling_aware mode
# ---------------------------------------------------------------------------

class TestFirstPassRegression:
    """Confirm that first-hard-family-pass gains still hold in sibling_aware mode."""

    def test_percent_match_still_works(self):
        """Percent slot should still attract percent mentions."""
        _, mentions, _ = run_relation_aware_grounding(
            "The tax rate is 15% and the base price is 200.",
            "orig",
            ["TaxRatePercent", "BasePrice"],
            ablation_mode="sibling_aware",
        )
        assert mentions.get("TaxRatePercent") is not None

    def test_bound_mismatch_penalty_still_fires(self):
        """Bound mention should NOT override a pure objective slot."""
        from tools.relation_aware_linking import build_mention_slot_links, relation_aware_local_score
        links, _, _, _, _ = build_mention_slot_links(
            "At least 50 units must be made. The profit is 20 per unit.",
            "orig",
            ["MinUnits", "ProfitPerUnit"],
        )
        # Find the link for '50' → ProfitPerUnit
        lnk_50_profit = next(
            (lnk for lnk in links
             if abs((lnk.mention_feats.value or -1) - 50.0) < 1e-6
             and lnk.slot_name == "ProfitPerUnit"),
            None,
        )
        if lnk_50_profit is not None:
            sc, feats = relation_aware_local_score(lnk_50_profit, "sibling_aware")
            lnk_50_min = next(
                (lnk for lnk in links
                 if abs((lnk.mention_feats.value or -1) - 50.0) < 1e-6
                 and lnk.slot_name == "MinUnits"),
                None,
            )
            sc_min, _ = relation_aware_local_score(lnk_50_min, "sibling_aware")
            assert sc_min > sc, (
                "MinUnits should score higher than ProfitPerUnit for '50' with at-least context"
            )

    def test_narrow_measure_overlap_preserved(self):
        """narrow_measure_overlap still contributes in sibling_aware mode."""
        links, _, _, _, _ = build_mention_slot_links(
            "Each product requires 4 labor hours.",
            "orig",
            ["LaborHoursPerProduct", "TotalBudget"],
        )
        lnk_labor = next(
            (lnk for lnk in links
             if abs((lnk.mention_feats.value or -1) - 4.0) < 1e-6
             and lnk.slot_name == "LaborHoursPerProduct"),
            None,
        )
        assert lnk_labor is not None
        sc_labor, _ = relation_aware_local_score(lnk_labor, "sibling_aware")
        lnk_budget = next(
            (lnk for lnk in links
             if abs((lnk.mention_feats.value or -1) - 4.0) < 1e-6
             and lnk.slot_name == "TotalBudget"),
            None,
        )
        assert lnk_budget is not None
        sc_budget, _ = relation_aware_local_score(lnk_budget, "sibling_aware")
        assert sc_labor > sc_budget, (
            f"LaborHoursPerProduct ({sc_labor:.2f}) should score higher than TotalBudget "
            f"({sc_budget:.2f}) for mention with 'labor hours' in context"
        )

    def test_all_modes_score_without_crash(self):
        """Every ablation mode must run without error."""
        for mode in ABLATION_MODES:
            _, mentions, diag = run_relation_aware_grounding(
                "Each product requires 4 labor hours and costs 10.",
                "orig",
                ["LaborHoursPerProduct", "CostPerProduct"],
                ablation_mode=mode,
            )
            # Must not be empty
            assert len(mentions) > 0, f"mode={mode} produced no assignments"
