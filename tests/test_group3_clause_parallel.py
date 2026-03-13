"""Group 3 clause-parallel structure + structured reranking tests.

This test module validates the Group 3 improvements to relation_aware_linking:

1. Left-directional entity anchor overlap (left_entity_anchor_overlap):
   Discriminates sibling entities in parallel-clause patterns by using only
   the left-side narrow context of each mention.

2. Alpha-digit split in _split_camel_case:
   "LaborProduct1" → ["labor", "product", "1"] so numeric suffixes become
   proper entity-discriminating tokens.

3. Swap repair (detect_and_repair_parallel_swaps):
   Post-assignment check that corrects reversed sibling-slot assignments when
   the left-anchor evidence clearly favours the swap.

4. Clause splitting (split_into_clauses / ClauseSpan):
   Lightweight deterministic clause segmentation.

5. Clause summaries (build_clause_summaries):
   Entity cue and measure cue extraction per clause.

6. Parallel clause detection (detect_parallel_clauses):
   Identifies parallel-pattern clause pairs.

Group 3 design constraints (preserved by all tests here):
- deterministic, no learned models
- no regression on Group 1 / Group 2 / easy-family tests
- conservative: neutral when no clause structure is present
"""
from __future__ import annotations

import pytest
from tools.relation_aware_linking import (
    build_mention_slot_links,
    relation_aware_local_score,
    run_relation_aware_grounding,
    _LEFT_ANCHOR_MEASURE_EXCLUDE,
)
from tools.nlp4lp_downstream_utility import _split_camel_case, _slot_measure_tokens
from tools.clause_aware_linking import (
    ClauseSpan,
    ClauseSummary,
    split_into_clauses,
    build_clause_summaries,
    detect_parallel_clauses,
    detect_and_repair_parallel_swaps,
)


# ---------------------------------------------------------------------------
# A. _split_camel_case digit-split (Group 3 prerequisite)
# ---------------------------------------------------------------------------


class TestSplitCamelCaseDigit:
    """Ensure _split_camel_case handles digit suffixes for entity disambiguation."""

    def test_product1_splits_to_product_1(self):
        assert "product" in _split_camel_case("LaborProduct1")
        assert "1" in _split_camel_case("LaborProduct1")

    def test_product2_splits_to_product_2(self):
        assert "product" in _split_camel_case("LaborProduct2")
        assert "2" in _split_camel_case("LaborProduct2")

    def test_type1_type2(self):
        parts1 = _split_camel_case("CostType1")
        parts2 = _split_camel_case("CostType2")
        assert "1" in parts1
        assert "2" in parts2
        assert "type" in parts1
        assert "type" in parts2

    def test_no_regression_feed_a(self):
        # Single-letter uppercase suffix should still work as before.
        parts = _split_camel_case("ProteinFeedA")
        assert "protein" in parts
        assert "feed" in parts
        assert "a" in parts

    def test_no_regression_labor_hours(self):
        parts = _split_camel_case("LaborHoursPerProduct")
        assert "labor" in parts
        assert "hours" in parts
        assert "per" in parts
        assert "product" in parts

    def test_slot_measure_tokens_product1(self):
        toks = _slot_measure_tokens("LaborProduct1")
        # Should contain the digit as a separate token for entity discrimination.
        assert "product" in toks
        assert "1" in toks


# ---------------------------------------------------------------------------
# B. Left entity anchor overlap in scoring
# ---------------------------------------------------------------------------


class TestLeftEntityAnchorOverlap:
    """Verify that left_entity_anchor_overlap is computed correctly."""

    def _get_link(self, query, slots, mention_val, slot_name):
        links, _, _, _, _ = build_mention_slot_links(query, "orig", slots)
        for lnk in links:
            if (
                abs((lnk.mention_feats.value or -999) - mention_val) < 1e-6
                and lnk.slot_name == slot_name
            ):
                return lnk
        return None

    def test_chair_gets_chair_anchor(self):
        q = "Chair requires 2 wood units and dresser requires 5 wood units."
        lnk = self._get_link(q, ["ChairWood", "DresserWood"], 2.0, "ChairWood")
        assert lnk is not None
        assert lnk.left_entity_anchor_overlap >= 1, (
            "mention 2 should have 'chair' in its left anchor overlapping ChairWood"
        )

    def test_dresser_gets_dresser_anchor(self):
        q = "Chair requires 2 wood units and dresser requires 5 wood units."
        lnk = self._get_link(q, ["ChairWood", "DresserWood"], 5.0, "DresserWood")
        assert lnk is not None
        assert lnk.left_entity_anchor_overlap >= 1

    def test_chair_anchor_is_zero_for_dresser_slot(self):
        # 2 (in chair clause) should have 0 overlap with DresserWood entity tokens.
        q = "Chair requires 2 wood units and dresser requires 5 wood units."
        lnk = self._get_link(q, ["ChairWood", "DresserWood"], 2.0, "DresserWood")
        assert lnk is not None
        assert lnk.left_entity_anchor_overlap == 0

    def test_product_b_7_gets_b_anchor(self):
        q = "Product B requires 7 labor hours and product A requires 3 labor hours."
        lnk = self._get_link(q, ["LaborHoursA", "LaborHoursB"], 7.0, "LaborHoursB")
        assert lnk is not None
        assert lnk.left_entity_anchor_overlap >= 1

    def test_product_a_3_gets_a_anchor(self):
        q = "Product B requires 7 labor hours and product A requires 3 labor hours."
        lnk = self._get_link(q, ["LaborHoursA", "LaborHoursB"], 3.0, "LaborHoursA")
        assert lnk is not None
        assert lnk.left_entity_anchor_overlap >= 1

    def test_no_heating_contamination_for_5_cooling(self):
        # In "3 heating hours and 5 cooling hours", "heating" appears in 5's left
        # context.  It must NOT be counted for HeatingHours entity overlap because
        # "heating" is in _LEFT_ANCHOR_MEASURE_EXCLUDE.
        q = "Regular glass requires 3 heating hours and 5 cooling hours."
        lnk = self._get_link(q, ["HeatingHours", "CoolingHours"], 5.0, "HeatingHours")
        assert lnk is not None
        # The slot entity words for HeatingHours after measure-exclusion should
        # contain nothing (or at most 0 matching tokens from the left anchor).
        assert lnk.left_entity_anchor_overlap == 0, (
            "HeatingHours entity anchor words should be empty after filtering 'heating'/'hours'"
        )


# ---------------------------------------------------------------------------
# C. Full grounding — previously failing cases now fixed
# ---------------------------------------------------------------------------


class TestParallelEntityAssignment:
    """Group 3: end-to-end grounding of parallel-entity patterns."""

    def test_chair_dresser_wood_correct(self):
        """Case: two entities, same measure. 2→Chair, 5→Dresser."""
        result, _, _ = run_relation_aware_grounding(
            "Chair requires 2 wood units and dresser requires 5 wood units.",
            "orig",
            ["ChairWood", "DresserWood"],
        )
        assert result.get("ChairWood") == pytest.approx(2.0), "Chair wood should be 2"
        assert result.get("DresserWood") == pytest.approx(5.0), "Dresser wood should be 5"

    def test_labor_hours_reversed_product_b_first(self):
        """Case: reversed order, Product B mentioned first."""
        result, _, _ = run_relation_aware_grounding(
            "Product B requires 7 labor hours and product A requires 3 labor hours.",
            "orig",
            ["LaborHoursA", "LaborHoursB"],
        )
        assert result.get("LaborHoursA") == pytest.approx(3.0), "LaborHoursA should be 3"
        assert result.get("LaborHoursB") == pytest.approx(7.0), "LaborHoursB should be 7"

    def test_product1_product2_numeric_suffix(self):
        """Case: digit-suffixed slot names require alpha-digit split."""
        result, _, _ = run_relation_aware_grounding(
            "Product 1 uses 3 hours of labor. Product 2 uses 7 hours of labor.",
            "orig",
            ["LaborProduct1", "LaborProduct2"],
        )
        assert result.get("LaborProduct1") == pytest.approx(3.0), (
            "LaborProduct1 should get 3 (from Product 1 clause)"
        )
        assert result.get("LaborProduct2") == pytest.approx(7.0), (
            "LaborProduct2 should get 7 (from Product 2 clause)"
        )

    def test_feed_a_feed_b_two_measures(self):
        """Case: two entities, two measures — all four numbers correct."""
        result, _, _ = run_relation_aware_grounding(
            "Feed A contains 10 protein and 8 fat, while Feed B contains 7 protein and 15 fat.",
            "orig",
            ["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        )
        assert result.get("ProteinFeedA") == pytest.approx(10.0)
        assert result.get("FatFeedA") == pytest.approx(8.0)
        assert result.get("ProteinFeedB") == pytest.approx(7.0)
        assert result.get("FatFeedB") == pytest.approx(15.0)

    def test_two_clause_same_measure_protein_only(self):
        """Case: two clauses, one shared measure only."""
        result, _, _ = run_relation_aware_grounding(
            "Feed A contains 10 protein and Feed B contains 7 protein.",
            "orig",
            ["ProteinFeedA", "ProteinFeedB"],
        )
        assert result.get("ProteinFeedA") == pytest.approx(10.0), (
            "10 is in Feed A clause and should go to ProteinFeedA"
        )
        assert result.get("ProteinFeedB") == pytest.approx(7.0), (
            "7 is in Feed B clause and should go to ProteinFeedB"
        )

    def test_regular_tempered_heating_cooling(self):
        """Case: four-slot parallel clause (problem-8 style)."""
        result, _, _ = run_relation_aware_grounding(
            "Regular glass requires 3 heating hours and 5 cooling hours, "
            "whereas tempered glass requires 5 heating and 8 cooling.",
            "orig",
            ["HeatingRegular", "CoolingRegular", "HeatingTempered", "CoolingTempered"],
        )
        assert result.get("HeatingRegular") == pytest.approx(3.0), "HeatingRegular should be 3"
        assert result.get("CoolingRegular") == pytest.approx(5.0), "CoolingRegular should be 5"
        assert result.get("HeatingTempered") == pytest.approx(5.0), "HeatingTempered should be 5"
        assert result.get("CoolingTempered") == pytest.approx(8.0), "CoolingTempered should be 8"


# ---------------------------------------------------------------------------
# D. No overfiring on simple single-entity cases
# ---------------------------------------------------------------------------


class TestNoOverfiringSimpleCases:
    """Group 3 must not break single-entity or single-measure problems."""

    def test_heating_cooling_single_entity(self):
        """Must not swap heating/cooling when there is only one glass type."""
        result, _, _ = run_relation_aware_grounding(
            "Regular glass requires 3 heating hours and 5 cooling hours.",
            "orig",
            ["HeatingHours", "CoolingHours"],
        )
        assert result.get("HeatingHours") == pytest.approx(3.0), (
            "HeatingHours must not get cooling value"
        )
        assert result.get("CoolingHours") == pytest.approx(5.0), (
            "CoolingHours must not get heating value"
        )

    def test_protein_fat_single_feed(self):
        """Must preserve measure discrimination for single-entity feed case."""
        result, _, _ = run_relation_aware_grounding(
            "Feed A contains 10 protein and 8 fat.",
            "orig",
            ["ProteinFeedA", "FatFeedA"],
        )
        assert result.get("ProteinFeedA") == pytest.approx(10.0)
        assert result.get("FatFeedA") == pytest.approx(8.0)

    def test_labor_wood_single_product(self):
        """Single entity, two measures: no clause/entity confusion."""
        result, _, _ = run_relation_aware_grounding(
            "Each table requires 4 labor hours and 6 units of wood.",
            "orig",
            ["LaborHoursPerTable", "WoodUnitsPerTable"],
        )
        assert result.get("LaborHoursPerTable") == pytest.approx(4.0)
        assert result.get("WoodUnitsPerTable") == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# E. Cross-clause contamination prevention
# ---------------------------------------------------------------------------


class TestCrossClauseContamination:
    """Group 3: numbers must not drift to the wrong entity's slots."""

    def test_protein_stays_in_feed_a_clause(self):
        """Feed A has 10 protein. Feed B has 15 fat. No cross-drift."""
        result, _, _ = run_relation_aware_grounding(
            "Feed A has 10 protein. Feed B has 15 fat.",
            "orig",
            ["ProteinFeedA", "FatFeedB"],
        )
        assert result.get("ProteinFeedA") == pytest.approx(10.0), (
            "protein number should not drift to FatFeedB"
        )
        assert result.get("FatFeedB") == pytest.approx(15.0)

    def test_small_large_labor(self):
        """Small 2, large 6 — entity discrimination via left anchor."""
        result, _, _ = run_relation_aware_grounding(
            "Small sculptures require 2 labor hours and large sculptures require 6.",
            "orig",
            ["LaborSmall", "LaborLarge"],
        )
        assert result.get("LaborSmall") == pytest.approx(2.0), (
            "LaborSmall should get 2 from 'small' clause"
        )
        assert result.get("LaborLarge") == pytest.approx(6.0), (
            "LaborLarge should get 6 from 'large' clause"
        )


# ---------------------------------------------------------------------------
# F. Clause splitting helpers
# ---------------------------------------------------------------------------


class TestSplitIntoClauses:
    """Verify split_into_clauses produces reasonable boundaries."""

    def test_single_sentence_gives_one_clause(self):
        text = "Each table requires 4 labor hours and 6 units of wood."
        clauses = split_into_clauses(text)
        assert len(clauses) == 1

    def test_while_creates_two_clauses(self):
        text = "Feed A contains 10 protein, while Feed B contains 7 protein."
        clauses = split_into_clauses(text)
        assert len(clauses) == 2

    def test_whereas_creates_two_clauses(self):
        text = "Regular glass requires 3 heating hours, whereas tempered glass requires 5."
        clauses = split_into_clauses(text)
        assert len(clauses) == 2

    def test_two_sentences_gives_two_clauses(self):
        text = "Product 1 uses 3 hours. Product 2 uses 7 hours."
        clauses = split_into_clauses(text)
        assert len(clauses) == 2

    def test_clause_span_type(self):
        text = "Chair requires 2 wood, while dresser requires 5."
        clauses = split_into_clauses(text)
        assert all(isinstance(c, ClauseSpan) for c in clauses)
        assert all(len(c.text) > 0 for c in clauses)

    def test_content_tokens_are_lowercase(self):
        text = "Feed A contains 10 protein."
        clauses = split_into_clauses(text)
        for clause in clauses:
            for tok in clause.content_tokens:
                assert tok == tok.lower(), f"Token {tok!r} should be lowercase"

    def test_empty_string_gives_empty_list(self):
        assert split_into_clauses("") == []

    def test_no_boundary_on_and_alone(self):
        # "and" should NOT be a standalone clause boundary.
        text = "3 heating hours and 5 cooling hours."
        clauses = split_into_clauses(text)
        # Should remain as a single clause.
        assert len(clauses) == 1

    def test_clause_indices_sequential(self):
        text = "Feed A has 10. Feed B has 7. Feed C has 3."
        clauses = split_into_clauses(text)
        for i, c in enumerate(clauses):
            assert c.clause_idx == i


# ---------------------------------------------------------------------------
# G. Clause summaries
# ---------------------------------------------------------------------------


class TestBuildClauseSummaries:
    """Verify build_clause_summaries extracts entity and measure cues."""

    def _summaries_for(self, text, slots):
        from tools.relation_aware_linking import build_mention_slot_links
        links, mentions, _, _, _ = build_mention_slot_links(text, "orig", slots)
        clauses = split_into_clauses(text)
        return build_clause_summaries(clauses, mentions), clauses

    def test_two_clause_summaries_count(self):
        text = "Feed A contains 10 protein, while Feed B contains 7 protein."
        summaries, _ = self._summaries_for(text, ["ProteinFeedA", "ProteinFeedB"])
        assert len(summaries) == 2

    def test_feed_a_clause_has_entity_cue(self):
        text = "Feed A contains 10 protein, while Feed B contains 7 protein."
        summaries, _ = self._summaries_for(text, ["ProteinFeedA", "ProteinFeedB"])
        # First clause should have "Feed" or "A" as entity cue.
        first_cues = summaries[0].entity_cue_tokens
        assert first_cues, "First (Feed A) clause should have entity cues"

    def test_protein_appears_as_measure_cue(self):
        text = "Feed A contains 10 protein, while Feed B contains 7 protein."
        summaries, _ = self._summaries_for(text, ["ProteinFeedA", "ProteinFeedB"])
        all_measure_cues = set().union(*(s.measure_cue_tokens for s in summaries))
        assert "protein" in all_measure_cues, "protein should appear as a measure cue"

    def test_summary_type(self):
        text = "Regular glass requires 3 heating hours."
        summaries, _ = self._summaries_for(text, ["HeatingHours"])
        assert len(summaries) >= 1
        assert isinstance(summaries[0], ClauseSummary)


# ---------------------------------------------------------------------------
# H. Parallel clause detection
# ---------------------------------------------------------------------------


class TestDetectParallelClauses:
    """Verify detect_parallel_clauses identifies genuine parallel patterns."""

    def _make_summaries(self, text, slots):
        from tools.relation_aware_linking import build_mention_slot_links
        links, mentions, _, _, _ = build_mention_slot_links(text, "orig", slots)
        clauses = split_into_clauses(text)
        return build_clause_summaries(clauses, mentions)

    def test_feed_a_feed_b_parallel(self):
        text = "Feed A contains 10 protein, while Feed B contains 7 protein."
        summaries = self._make_summaries(text, ["ProteinFeedA", "ProteinFeedB"])
        parallel = detect_parallel_clauses(summaries)
        assert len(parallel) >= 1, "Feed A / Feed B with protein should be parallel"

    def test_single_clause_no_parallel(self):
        text = "Each table requires 4 labor hours and 6 units of wood."
        summaries = self._make_summaries(text, ["LaborHoursPerTable", "WoodUnitsPerTable"])
        parallel = detect_parallel_clauses(summaries)
        # Single clause cannot form a parallel pair.
        assert len(parallel) == 0

    def test_parallel_pairs_are_ordered_tuples(self):
        text = "Feed A has 10 protein, while Feed B has 7 protein."
        summaries = self._make_summaries(text, ["ProteinFeedA", "ProteinFeedB"])
        parallel = detect_parallel_clauses(summaries)
        for pair in parallel:
            assert len(pair) == 2
            assert pair[0] < pair[1]


# ---------------------------------------------------------------------------
# I. Swap repair
# ---------------------------------------------------------------------------


class TestDetectAndRepairParallelSwaps:
    """Verify swap repair corrects reversed sibling assignments."""

    def test_swap_feed_a_feed_b_reversed(self):
        """Simulate a reversed assignment and verify swap repair fixes it."""
        from tools.nlp4lp_downstream_utility import MentionOptIR
        from tools.relation_aware_linking import build_mention_slot_links

        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        slots = ["ProteinFeedA", "ProteinFeedB"]
        links, mentions, _, _, _ = build_mention_slot_links(q, "orig", slots)
        for lnk in links:
            relation_aware_local_score(lnk, "full")

        # Find the 10 and 7 mentions.
        m10 = next(m for m in mentions if abs(m.value - 10.0) < 1e-6)
        m7 = next(m for m in mentions if abs(m.value - 7.0) < 1e-6)

        # Artificially reverse the assignment.
        reversed_values = {"ProteinFeedA": 7.0, "ProteinFeedB": 10.0}
        reversed_mentions = {"ProteinFeedA": m7, "ProteinFeedB": m10}

        new_vals, new_ments, swap_log = detect_and_repair_parallel_swaps(
            reversed_values, reversed_mentions, links
        )
        assert new_vals.get("ProteinFeedA") == pytest.approx(10.0), (
            "Swap repair should restore 10 → ProteinFeedA"
        )
        assert new_vals.get("ProteinFeedB") == pytest.approx(7.0), (
            "Swap repair should restore 7 → ProteinFeedB"
        )
        assert len(swap_log) >= 1, "A swap should have been logged"

    def test_no_spurious_swap_heating_cooling(self):
        """Swap repair must NOT fire on correct single-entity heating/cooling."""
        from tools.relation_aware_linking import build_mention_slot_links, best_assignment_greedy

        q = "Regular glass requires 3 heating hours and 5 cooling hours."
        slots = ["HeatingHours", "CoolingHours"]
        links, mentions, slot_irs, _, _ = build_mention_slot_links(q, "orig", slots)
        for lnk in links:
            relation_aware_local_score(lnk, "full")

        # Correct assignment from greedy (should already be correct).
        filled_v, filled_m, _ = best_assignment_greedy(links, slot_irs, mentions, "full")
        new_vals, new_ments, swap_log = detect_and_repair_parallel_swaps(
            filled_v, filled_m, links
        )
        # Should NOT have swapped.
        assert new_vals.get("HeatingHours") == pytest.approx(3.0), (
            "HeatingHours must stay at 3 — no spurious swap"
        )
        assert new_vals.get("CoolingHours") == pytest.approx(5.0), (
            "CoolingHours must stay at 5 — no spurious swap"
        )
        assert len(swap_log) == 0, "No swap should be logged for already-correct assignment"

    def test_swap_log_format(self):
        """Swap log entries should name the slots and show the before→after values."""
        from tools.nlp4lp_downstream_utility import MentionOptIR
        from tools.relation_aware_linking import build_mention_slot_links

        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        slots = ["ProteinFeedA", "ProteinFeedB"]
        links, mentions, _, _, _ = build_mention_slot_links(q, "orig", slots)
        for lnk in links:
            relation_aware_local_score(lnk, "full")

        m10 = next(m for m in mentions if abs(m.value - 10.0) < 1e-6)
        m7 = next(m for m in mentions if abs(m.value - 7.0) < 1e-6)
        reversed_values = {"ProteinFeedA": 7.0, "ProteinFeedB": 10.0}
        reversed_mentions = {"ProteinFeedA": m7, "ProteinFeedB": m10}

        _, _, swap_log = detect_and_repair_parallel_swaps(
            reversed_values, reversed_mentions, links
        )
        assert len(swap_log) >= 1, "Expected at least one swap log entry"
        entry = swap_log[0]
        # Log must reference both slot names.
        assert "ProteinFeedA" in entry, f"Expected 'ProteinFeedA' in log entry: {entry!r}"
        assert "ProteinFeedB" in entry, f"Expected 'ProteinFeedB' in log entry: {entry!r}"
        # Log must record the original values before the swap.
        assert "10" in entry and "7" in entry, (
            f"Expected original values 10 and 7 in log entry: {entry!r}"
        )
        # Log must include the anchor improvement marker.
        assert "anchor" in entry.lower(), f"Expected 'anchor' in log entry: {entry!r}"


# ---------------------------------------------------------------------------
# J. Ablation mode: swap repair only fires in "full" mode
# ---------------------------------------------------------------------------


class TestAblationMode:
    """Group 3 left_entity_anchor_bonus is in 'full' mode only."""

    def test_left_anchor_bonus_zero_in_semantic_mode(self):
        """left_entity_anchor_overlap should not contribute in 'semantic' mode."""
        q = "Chair requires 2 wood units and dresser requires 5 wood units."
        links, _, _, _, _ = build_mention_slot_links(q, "orig", ["ChairWood", "DresserWood"])
        for lnk in links:
            if abs((lnk.mention_feats.value or -999) - 2.0) < 1e-6 and lnk.slot_name == "ChairWood":
                sc_full, feats_full = relation_aware_local_score(lnk, "full")
                sc_sem, feats_sem = relation_aware_local_score(lnk, "semantic")
                # In semantic mode, left_entity_anchor_overlap should NOT be in features.
                assert "left_entity_anchor_overlap" not in feats_sem, (
                    "left_entity_anchor_bonus should not appear in semantic mode"
                )
                return
        pytest.fail("Could not find the 2.0→ChairWood link")


# ---------------------------------------------------------------------------
# K. Clause-local alignment feature computation
# ---------------------------------------------------------------------------


class TestClauseLocalAlignment:
    """Group 3: clause_entity_overlap and cross_clause_entity_penalty features."""

    def _get_link(self, query, slots, mention_val, slot_name):
        links, _, _, _, _ = build_mention_slot_links(query, "orig", slots)
        for lnk in links:
            if (
                abs((lnk.mention_feats.value or -999) - mention_val) < 1e-6
                and lnk.slot_name == slot_name
            ):
                return lnk
        return None

    def test_clause_entity_overlap_fires_for_correct_clause(self):
        """Feed A mention (10) should have clause_entity_overlap > 0 for ProteinFeedA."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedA")
        assert lnk is not None
        assert lnk.clause_entity_overlap >= 1, (
            "mention 10 in Feed-A clause should overlap with ProteinFeedA entity words"
        )

    def test_clause_entity_overlap_lower_for_wrong_clause(self):
        """Feed A mention (10) should have lower/zero clause_entity_overlap for ProteinFeedB."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk_correct = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedA")
        lnk_wrong = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedB")
        assert lnk_correct is not None
        assert lnk_wrong is not None
        # Correct slot should have >= entity overlap of wrong slot.
        assert lnk_correct.clause_entity_overlap >= lnk_wrong.clause_entity_overlap

    def test_cross_clause_penalty_fires_for_wrong_clause(self):
        """Feed-B mention (7) should have cross_clause_entity_penalty > 0 for ProteinFeedA."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 7.0, "ProteinFeedA")
        assert lnk is not None
        assert lnk.cross_clause_entity_penalty >= 1, (
            "mention 7 in Feed-B clause should be penalised for ProteinFeedA "
            "(best entity match is in Feed-A clause)"
        )

    def test_cross_clause_penalty_zero_for_correct_clause(self):
        """Feed-A mention (10) should have cross_clause_penalty = 0 for ProteinFeedA."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedA")
        assert lnk is not None
        assert lnk.cross_clause_entity_penalty == 0, (
            "correct clause assignment must not receive a cross-clause penalty"
        )

    def test_clause_features_neutral_for_single_clause(self):
        """Single-clause query: all clause alignment fields must stay at 0."""
        q = "Each table requires 4 labor hours and 6 units of wood."
        links, _, _, _, _ = build_mention_slot_links(
            q, "orig", ["LaborHoursPerTable", "WoodUnitsPerTable"]
        )
        for lnk in links:
            assert lnk.clause_entity_overlap == 0, (
                f"clause_entity_overlap must be 0 for single-clause query, "
                f"got {lnk.clause_entity_overlap} for {lnk.slot_name}"
            )
            assert lnk.cross_clause_entity_penalty == 0, (
                f"cross_clause_entity_penalty must be 0 for single-clause query"
            )

    def test_clause_entity_alignment_bonus_appears_in_semantic_mode(self):
        """clause_entity_overlap should contribute to score in 'semantic' mode."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedA")
        assert lnk is not None
        if lnk.clause_entity_overlap > 0:
            _, feats = relation_aware_local_score(lnk, "semantic")
            assert "clause_entity_overlap" in feats, (
                "clause_entity_overlap bonus should appear in semantic mode features"
            )

    def test_clause_entity_alignment_bonus_appears_in_full_mode(self):
        """clause_entity_overlap should contribute to score in 'full' mode."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedA")
        assert lnk is not None
        if lnk.clause_entity_overlap > 0:
            _, feats = relation_aware_local_score(lnk, "full")
            assert "clause_entity_overlap" in feats, (
                "clause_entity_overlap bonus should appear in full mode features"
            )

    def test_clause_features_absent_in_basic_and_ops_modes(self):
        """Clause features must NOT appear in basic/ops mode features (ablation boundary)."""
        q = "Feed A contains 10 protein, while Feed B contains 7 protein."
        lnk = self._get_link(q, ["ProteinFeedA", "ProteinFeedB"], 10.0, "ProteinFeedA")
        assert lnk is not None
        for mode in ("basic", "ops"):
            _, feats = relation_aware_local_score(lnk, mode)
            assert "clause_entity_overlap" not in feats, (
                f"clause_entity_overlap must not appear in {mode} mode"
            )
            assert "cross_clause_entity_penalty" not in feats, (
                f"cross_clause_entity_penalty must not appear in {mode} mode"
            )

    def test_grounding_feed_a_b_two_measures_semantic_mode(self):
        """Four-slot Feed A/B case must also be correct in semantic mode with clause features."""
        result, _, _ = run_relation_aware_grounding(
            "Feed A contains 10 protein and 8 fat, while Feed B contains 7 protein and 15 fat.",
            "orig",
            ["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
            ablation_mode="semantic",
        )
        assert result.get("ProteinFeedA") == pytest.approx(10.0)
        assert result.get("FatFeedA") == pytest.approx(8.0)
        assert result.get("ProteinFeedB") == pytest.approx(7.0)
        assert result.get("FatFeedB") == pytest.approx(15.0)

    def test_no_cross_clause_penalty_for_single_entity_heating_cooling(self):
        """Single-entity heating/cooling must not fire any cross-clause penalty."""
        q = "Regular glass requires 3 heating hours and 5 cooling hours."
        links, _, _, _, _ = build_mention_slot_links(q, "orig", ["HeatingHours", "CoolingHours"])
        for lnk in links:
            assert lnk.cross_clause_entity_penalty == 0, (
                "No cross-clause penalty should fire for single-clause queries"
            )
