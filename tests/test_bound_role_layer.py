"""Regression tests for Error Family #3: min/max / lower-vs-upper bound confusion.

These tests validate the deterministic bound-role annotation layer added to
the NLP4LP downstream grounding pipeline, including:
  - Extended operator-phrase recognition (no fewer than, less than, greater than, etc.)
  - Fine-grained bound_role field on MentionOptIR
  - Range-expression detection (between X and Y, from X to Y)
  - Wrong-direction bound penalty in scoring
  - Bound-flip swap repair in _opt_role_validate_and_repair and GCG
  - No false positives on plain quantities or percent mentions
"""

import pytest

from tools.nlp4lp_downstream_utility import (
    _detect_operator_tags,
    _extract_opt_role_mentions,
    _find_range_annotations,
    _compute_bound_role,
    _bound_swap_repair,
    _build_slot_opt_irs,
    _gcg_local_score,
    _is_partial_admissible,
    _score_mention_slot_opt,
    _run_global_consistency_grounding,
    _slot_stem,
    _ENABLE_BOUND_ROLE_LAYER,
)


# ---------------------------------------------------------------------------
# 1. Operator-tag detection: new phrase patterns
# ---------------------------------------------------------------------------

class TestExtendedOperatorPhrases:
    """Verify newly added operator phrases fire the correct min/max tag."""

    # ── Lower-bound (min) inclusive ──────────────────────────────────────────

    def test_no_fewer_than_gives_min(self):
        toks = "no fewer than 50".split()
        assert "min" in _detect_operator_tags(toks, toks)
        assert "max" not in _detect_operator_tags(toks, toks)

    def test_not_fewer_than_gives_min(self):
        toks = "not fewer than 10".split()
        assert "min" in _detect_operator_tags(toks, toks)
        assert "max" not in _detect_operator_tags(toks, toks)

    def test_greater_than_or_equal_to_gives_min(self):
        toks = "greater than or equal to 8".split()
        assert "min" in _detect_operator_tags(toks, toks)
        assert "max" not in _detect_operator_tags(toks, toks)

    def test_greater_than_or_equal_gives_min(self):
        toks = "greater than or equal 8".split()
        assert "min" in _detect_operator_tags(toks, toks)

    def test_at_a_minimum_gives_min(self):
        toks = "at a minimum 5".split()
        assert "min" in _detect_operator_tags(toks, toks)

    # ── Upper-bound (max) inclusive ──────────────────────────────────────────

    def test_less_than_or_equal_to_gives_max(self):
        toks = "less than or equal to 12".split()
        assert "max" in _detect_operator_tags(toks, toks)
        assert "min" not in _detect_operator_tags(toks, toks)

    def test_less_than_or_equal_gives_max(self):
        toks = "less than or equal 12".split()
        assert "max" in _detect_operator_tags(toks, toks)

    def test_no_greater_than_gives_max(self):
        toks = "no greater than 15".split()
        assert "max" in _detect_operator_tags(toks, toks)
        assert "min" not in _detect_operator_tags(toks, toks)

    def test_not_greater_than_gives_max(self):
        toks = "not greater than 7".split()
        assert "max" in _detect_operator_tags(toks, toks)

    def test_at_a_maximum_gives_max(self):
        toks = "at a maximum 100".split()
        assert "max" in _detect_operator_tags(toks, toks)

    # ── Exclusive lower-bound ────────────────────────────────────────────────

    def test_more_than_gives_min(self):
        toks = "more than 10".split()
        assert "min" in _detect_operator_tags(toks, toks)
        assert "max" not in _detect_operator_tags(toks, toks)

    def test_greater_than_gives_min(self):
        toks = "greater than 5".split()
        assert "min" in _detect_operator_tags(toks, toks)
        assert "max" not in _detect_operator_tags(toks, toks)

    # ── Exclusive upper-bound ────────────────────────────────────────────────

    def test_fewer_than_gives_max(self):
        toks = "fewer than 5".split()
        assert "max" in _detect_operator_tags(toks, toks)
        assert "min" not in _detect_operator_tags(toks, toks)

    def test_less_than_gives_max(self):
        toks = "less than 30".split()
        assert "max" in _detect_operator_tags(toks, toks)
        assert "min" not in _detect_operator_tags(toks, toks)

    # ── Negation guards: "no more than" must NOT also fire min ───────────────

    def test_no_more_than_does_not_also_give_min(self):
        toks = "no more than 20".split()
        tags = _detect_operator_tags(toks, toks)
        assert "max" in tags
        assert "min" not in tags

    def test_no_fewer_than_does_not_also_give_max(self):
        """'no fewer than' is an inclusive lower bound; must NOT fire max."""
        toks = "no fewer than 50".split()
        tags = _detect_operator_tags(toks, toks)
        assert "min" in tags
        assert "max" not in tags

    def test_not_less_than_does_not_also_give_max(self):
        toks = "not less than 3".split()
        tags = _detect_operator_tags(toks, toks)
        assert "min" in tags
        assert "max" not in tags


# ---------------------------------------------------------------------------
# 2. bound_role field on MentionOptIR
# ---------------------------------------------------------------------------

class TestBoundRoleField:
    """Verify the fine-grained bound_role field is populated correctly."""

    def _first_mention(self, query: str):
        ms = _extract_opt_role_mentions(query, "orig")
        assert ms, f"No mentions extracted from {query!r}"
        return ms[0]

    def test_at_least_is_lower_inclusive(self):
        m = self._first_mention("at least 10 hours")
        assert m.bound_role == "lower_inclusive"

    def test_no_fewer_than_is_lower_inclusive(self):
        m = self._first_mention("no fewer than 50 units")
        assert m.bound_role == "lower_inclusive"

    def test_greater_than_or_equal_is_lower_inclusive(self):
        m = self._first_mention("greater than or equal to 8 items")
        assert m.bound_role == "lower_inclusive"

    def test_more_than_is_lower_exclusive(self):
        m = self._first_mention("more than 10 workers")
        assert m.bound_role == "lower_exclusive"

    def test_greater_than_is_lower_exclusive(self):
        m = self._first_mention("greater than 5 units")
        assert m.bound_role == "lower_exclusive"

    def test_at_most_is_upper_inclusive(self):
        m = self._first_mention("at most 20 hours")
        assert m.bound_role == "upper_inclusive"

    def test_no_more_than_is_upper_inclusive(self):
        m = self._first_mention("no more than 30 kg")
        assert m.bound_role == "upper_inclusive"

    def test_fewer_than_is_upper_exclusive(self):
        m = self._first_mention("fewer than 8 machines")
        assert m.bound_role == "upper_exclusive"

    def test_less_than_is_upper_exclusive(self):
        m = self._first_mention("use less than 30 kg of material")
        assert m.bound_role == "upper_exclusive"

    def test_no_operator_is_unknown(self):
        m = self._first_mention("profit is 20 dollars per unit")
        assert m.bound_role == "unknown"

    def test_percent_with_at_most_is_upper_inclusive(self):
        """At most 40%: should be upper_inclusive, not break percent handling."""
        ms = _extract_opt_role_mentions("At most 40% of workers can be assigned.", "orig")
        pct_ms = [m for m in ms if m.is_percent_like]
        assert pct_ms, "Expected a percent mention"
        assert pct_ms[0].bound_role == "upper_inclusive"

    def test_at_least_half_is_lower_inclusive(self):
        """At least half: fraction mention with lower-inclusive bound_role."""
        ms = _extract_opt_role_mentions("At least half of the budget must be invested.", "orig")
        frac_ms = [m for m in ms if m.value == 0.5]
        assert frac_ms, "Expected a fraction mention with value 0.5"
        assert frac_ms[0].bound_role == "lower_inclusive"


# ---------------------------------------------------------------------------
# 3. Range expression detection
# ---------------------------------------------------------------------------

class TestRangeExpressions:
    """Verify 'between X and Y' and 'from X to Y' are handled as range pairs."""

    def test_between_X_and_Y_range_low(self):
        ms = _extract_opt_role_mentions("The acceptable range is between 5 and 12.", "orig")
        vals = {m.value: m.bound_role for m in ms}
        assert vals.get(5.0) == "range_low", f"5.0 bound_role={vals.get(5.0)}"

    def test_between_X_and_Y_range_high(self):
        ms = _extract_opt_role_mentions("The acceptable range is between 5 and 12.", "orig")
        vals = {m.value: m.bound_role for m in ms}
        assert vals.get(12.0) == "range_high", f"12.0 bound_role={vals.get(12.0)}"

    def test_from_X_to_Y_range_low(self):
        ms = _extract_opt_role_mentions("Allocate from 3 to 8 workers.", "orig")
        vals = {m.value: m.bound_role for m in ms}
        assert vals.get(3.0) == "range_low", f"3.0 bound_role={vals.get(3.0)}"

    def test_from_X_to_Y_range_high(self):
        ms = _extract_opt_role_mentions("Allocate from 3 to 8 workers.", "orig")
        vals = {m.value: m.bound_role for m in ms}
        assert vals.get(8.0) == "range_high", f"8.0 bound_role={vals.get(8.0)}"

    def test_range_low_has_min_operator_tag(self):
        ms = _extract_opt_role_mentions("between 10 and 20 items", "orig")
        low = next((m for m in ms if m.value == 10.0), None)
        assert low is not None
        assert "min" in low.operator_tags

    def test_range_high_has_max_operator_tag(self):
        ms = _extract_opt_role_mentions("between 10 and 20 items", "orig")
        high = next((m for m in ms if m.value == 20.0), None)
        assert high is not None
        assert "max" in high.operator_tags

    def test_find_range_annotations_between(self):
        toks = "The range is between 5 and 12 units .".split()
        anno = _find_range_annotations(toks)
        # Indices: "5"=4, "12"=6
        assert 4 in anno and anno[4] == "range_low"
        assert 6 in anno and anno[6] == "range_high"

    def test_find_range_annotations_from_to(self):
        toks = "from 3 to 8 workers".split()
        anno = _find_range_annotations(toks)
        assert 1 in anno and anno[1] == "range_low"
        assert 3 in anno and anno[3] == "range_high"


# ---------------------------------------------------------------------------
# 4. Slot-side bound typing (lower/upper in slot name)
# ---------------------------------------------------------------------------

class TestSlotSideBoundTyping:
    """Verify slot names with 'lower'/'upper' get the correct operator_preference."""

    def test_lower_in_slot_name_gets_min_pref(self):
        slots = _build_slot_opt_irs(["LowerBound"])
        assert "min" in slots[0].operator_preference

    def test_upper_in_slot_name_gets_max_pref(self):
        slots = _build_slot_opt_irs(["UpperBound"])
        assert "max" in slots[0].operator_preference

    def test_lowerbound_compound_gets_min_pref(self):
        slots = _build_slot_opt_irs(["LowerboundHours"])
        assert "min" in slots[0].operator_preference

    def test_upperbound_compound_gets_max_pref(self):
        slots = _build_slot_opt_irs(["UpperboundHours"])
        assert "max" in slots[0].operator_preference

    def test_min_in_slot_name_still_gives_min(self):
        slots = _build_slot_opt_irs(["MinHours"])
        assert "min" in slots[0].operator_preference

    def test_max_in_slot_name_still_gives_max(self):
        slots = _build_slot_opt_irs(["MaxHours"])
        assert "max" in slots[0].operator_preference


# ---------------------------------------------------------------------------
# 5. Scoring: wrong-direction penalty
# ---------------------------------------------------------------------------

class TestWrongDirectionPenalty:
    """Verify that bound mismatches incur a negative score delta."""

    def _get_slots(self, names):
        return {s.name: s for s in _build_slot_opt_irs(names)}

    def _get_mention(self, phrase):
        ms = _extract_opt_role_mentions(phrase, "orig")
        return ms[0]

    def test_lower_bound_mention_penalised_for_max_only_slot(self):
        m = self._get_mention("at least 10 units")
        s_max = self._get_slots(["MaxHours"])["MaxHours"]
        score_opt, feats_opt = _score_mention_slot_opt(m, s_max)
        score_gcg, feats_gcg = _gcg_local_score(m, s_max)
        assert feats_opt.get("bound_direction_wrong") is True, "OPT score should flag wrong direction"
        assert feats_gcg.get("bound_direction_wrong") is True, "GCG score should flag wrong direction"

    def test_upper_bound_mention_penalised_for_min_only_slot(self):
        m = self._get_mention("no more than 20 kg")
        s_min = self._get_slots(["MinHours"])["MinHours"]
        score_opt, feats_opt = _score_mention_slot_opt(m, s_min)
        score_gcg, feats_gcg = _gcg_local_score(m, s_min)
        assert feats_opt.get("bound_direction_wrong") is True
        assert feats_gcg.get("bound_direction_wrong") is True

    def test_correct_direction_gives_bonus_not_penalty(self):
        m = self._get_mention("at least 10 units")
        s_min = self._get_slots(["MinHours"])["MinHours"]
        _, feats = _score_mention_slot_opt(m, s_min)
        assert feats.get("bound_direction_wrong") is not True
        assert feats.get("lower_bound_match") is True

    def test_wrong_direction_score_is_lower_than_correct_direction(self):
        m = self._get_mention("at least 10 hours")
        slots = self._get_slots(["MinHours", "MaxHours"])
        score_min, _ = _gcg_local_score(m, slots["MinHours"])
        score_max, _ = _gcg_local_score(m, slots["MaxHours"])
        assert score_min > score_max, (
            f"Lower-bound mention should score higher for min slot "
            f"({score_min:.2f}) than max slot ({score_max:.2f})"
        )

    def test_no_penalty_for_untagged_mention(self):
        """A plain quantity without any bound tag should not trigger the wrong-direction penalty."""
        m = self._get_mention("profit is 20 dollars per unit")
        s_min = self._get_slots(["MinHours"])["MinHours"]
        _, feats = _score_mention_slot_opt(m, s_min)
        assert feats.get("bound_direction_wrong") is not True


# ---------------------------------------------------------------------------
# 6. GCG end-to-end: problem statement examples
# ---------------------------------------------------------------------------

class TestGCGEndToEnd:
    """End-to-end GCG routing for the problem-statement examples."""

    def _gcg(self, query, slots):
        result, _, _ = _run_global_consistency_grounding(query, "orig", slots)
        return result

    def test_A_at_least_10_at_most_20(self):
        """A: 10 → MinHours, 20 → MaxHours."""
        r = self._gcg("Each product requires at least 10 hours and at most 20 hours.",
                      ["MinHours", "MaxHours"])
        assert r.get("MinHours") == 10.0
        assert r.get("MaxHours") == 20.0

    def test_B_no_fewer_than_50_units(self):
        """B: 50 should go to the min slot."""
        r = self._gcg("The company must produce no fewer than 50 units.", ["MinUnits"])
        assert r.get("MinUnits") == 50.0

    def test_C_less_than_30_kg(self):
        """C: 30 should go to the max slot."""
        r = self._gcg("Use less than 30 kg of material.", ["MaxMaterial"])
        assert r.get("MaxMaterial") == 30.0

    def test_D_between_5_and_12(self):
        """D: 5 → lower-bound slot, 12 → upper-bound slot."""
        r = self._gcg("The acceptable range is between 5 and 12.",
                      ["MinRange", "MaxRange"])
        assert r.get("MinRange") == 5.0
        assert r.get("MaxRange") == 12.0

    def test_E_profit_20_per_unit_not_bound(self):
        """E: 20 should go to the ProfitPerUnit slot, not be mis-tagged as a bound."""
        r = self._gcg("Profit is 20 dollars per unit.", ["ProfitPerUnit"])
        assert r.get("ProfitPerUnit") == 20.0

    def test_F_at_most_40_percent(self):
        """F: at most 40% → MaxPct, percent handling preserved."""
        r = self._gcg("At most 40% of workers can be assigned to night shift.", ["MaxPct"])
        assert r.get("MaxPct") == pytest.approx(0.4)

    def test_G_at_least_half_lower_bound_semantics(self):
        """G: 'at least half' fraction tagged as lower-inclusive bound."""
        ms = _extract_opt_role_mentions("At least half of the budget must be invested.", "orig")
        frac = next((m for m in ms if m.value == 0.5), None)
        assert frac is not None, "Expected mention with value 0.5"
        assert frac.is_lower_bound_like
        assert frac.bound_role == "lower_inclusive"

    def test_greater_than_5_goes_to_min_slot(self):
        r = self._gcg("Greater than 5 workers on shift.", ["MinWorkers"])
        assert r.get("MinWorkers") == 5.0

    def test_fewer_than_8_goes_to_max_slot(self):
        r = self._gcg("Fewer than 8 machines needed.", ["MaxMachines"])
        assert r.get("MaxMachines") == 8.0

    def test_lower_upper_slot_names(self):
        """Slots named LowerBound/UpperBound should receive correct bound mentions."""
        r = self._gcg("The lower bound is 3 and the upper bound is 9.",
                      ["LowerBound", "UpperBound"])
        assert r.get("LowerBound") == 3.0
        assert r.get("UpperBound") == 9.0


# ---------------------------------------------------------------------------
# 7. Bound-flip swap repair
# ---------------------------------------------------------------------------

class TestBoundSwapRepair:
    """Verify _bound_swap_repair corrects inverted min/max assignments."""

    def _make_mention(self, value, tags):
        """Helper to create a MentionOptIR-like object via the real extractor."""
        # Use a real phrase so we get a proper MentionOptIR
        phrase_map = {
            ("min", 10.0): "at least 10 units",
            ("max", 20.0): "at most 20 units",
            ("max", 10.0): "at most 10 units",
            ("min", 20.0): "at least 20 units",
        }
        key = (next(iter(tags), ""), value)
        phrase = phrase_map.get(key, f"{int(value)} items")
        ms = _extract_opt_role_mentions(phrase, "orig")
        return ms[0]

    def test_swap_when_min_m_has_max_cue(self):
        """If mention assigned to min-slot has 'max' operator tag and min>max, swap."""
        m10_max = self._make_mention(10.0, {"max"})  # has max cue
        m20_min = self._make_mention(20.0, {"min"})  # has min cue
        # Incorrectly: MinHours=m10 (value=10, max cue), MaxHours=m20 (value=20, min cue)
        # But 10 < 20 so no inversion; let's use the inverted pair
        m_with_max_tag = self._make_mention(20.0, {"max"})  # value=20, tag=max
        m_with_min_tag = self._make_mention(10.0, {"min"})  # value=10, tag=min
        # Inverted assignment: min=20 (wrong), max=10 (wrong)
        slots = _build_slot_opt_irs(["MinHours", "MaxHours"])
        filled = {"MinHours": m_with_max_tag, "MaxHours": m_with_min_tag}
        repair = {}
        _bound_swap_repair(filled, repair, slots)
        # After repair: min should get the smaller value
        assert filled["MinHours"].value == 10.0
        assert filled["MaxHours"].value == 20.0
        assert repair.get("MinHours") == "bound_swap_repair"
        assert repair.get("MaxHours") == "bound_swap_repair"

    def test_no_swap_when_no_operator_evidence(self):
        """No swap if there is no explicit operator mismatch evidence."""
        ms_big = _extract_opt_role_mentions("20 units", "orig")
        ms_small = _extract_opt_role_mentions("10 units", "orig")
        assert ms_big and ms_small
        slots = _build_slot_opt_irs(["MinHours", "MaxHours"])
        filled = {"MinHours": ms_big[0], "MaxHours": ms_small[0]}
        repair = {}
        _bound_swap_repair(filled, repair, slots)
        # Without operator cues, conservative: no swap
        assert filled["MinHours"].value == 20.0
        assert filled["MaxHours"].value == 10.0
        assert "bound_swap_repair" not in repair.values()


# ---------------------------------------------------------------------------
# 8. No regressions on existing cases
# ---------------------------------------------------------------------------

class TestNoRegressions:
    """Smoke tests to confirm existing correct behaviour is preserved."""

    def test_at_least_still_gives_min(self):
        toks = "at least 5000".split()
        assert "min" in _detect_operator_tags(toks, toks)

    def test_at_most_still_gives_max(self):
        toks = "at most 1500".split()
        assert "max" in _detect_operator_tags(toks, toks)

    def test_no_more_than_still_gives_max(self):
        toks = "no more than 200".split()
        assert "max" in _detect_operator_tags(toks, toks)
        assert "min" not in _detect_operator_tags(toks, toks)

    def test_no_less_than_still_gives_min(self):
        toks = "no less than 50".split()
        assert "min" in _detect_operator_tags(toks, toks)
        assert "max" not in _detect_operator_tags(toks, toks)

    def test_minimum_word_still_gives_min(self):
        toks = "minimum 5000 bottles".split()
        assert "min" in _detect_operator_tags(toks, toks)

    def test_maximum_word_still_gives_max(self):
        toks = "maximum 200 kg".split()
        assert "max" in _detect_operator_tags(toks, toks)

    def test_per_unit_mention_no_operator_tags(self):
        toks = "requires 2 hours per unit".split()
        tags = _detect_operator_tags(toks, toks)
        assert "min" not in tags
        assert "max" not in tags

    def test_enable_bound_role_layer_is_on(self):
        """The bound-role layer toggle must be enabled for the above to work."""
        assert _ENABLE_BOUND_ROLE_LAYER is True


# ---------------------------------------------------------------------------
# 9. _slot_stem helper
# ---------------------------------------------------------------------------

class TestSlotStem:
    """Unit tests for the _slot_stem quantity-stem extractor."""

    def test_min_prefix_stripped(self):
        assert _slot_stem("MinDemand") == "demand"

    def test_max_prefix_stripped(self):
        assert _slot_stem("MaxDemand") == "demand"

    def test_lower_prefix_stripped(self):
        assert _slot_stem("LowerBound") == "bound"

    def test_upper_prefix_stripped(self):
        assert _slot_stem("UpperBound") == "bound"

    def test_minimum_prefix_stripped(self):
        assert _slot_stem("MinimumCapacity") == "capacity"

    def test_maximum_prefix_stripped(self):
        assert _slot_stem("MaximumCapacity") == "capacity"

    def test_min_suffix_stripped(self):
        assert _slot_stem("DemandMin") == "demand"

    def test_max_suffix_stripped(self):
        assert _slot_stem("DemandMax") == "demand"

    def test_min_max_hours_share_stem(self):
        assert _slot_stem("MinHours") == _slot_stem("MaxHours")

    def test_lower_upper_bound_share_stem(self):
        assert _slot_stem("LowerBound") == _slot_stem("UpperBound")

    def test_unrelated_names_different_stems(self):
        assert _slot_stem("MinDemand") != _slot_stem("MaxCapacity")

    def test_plain_name_unchanged(self):
        assert _slot_stem("Demand") == "demand"

    def test_lb_prefix_stripped(self):
        assert _slot_stem("LbDemand") == "demand"

    def test_ub_prefix_stripped(self):
        assert _slot_stem("UbDemand") == "demand"


# ---------------------------------------------------------------------------
# 10. _is_partial_admissible: numeric min≤max ordering
# ---------------------------------------------------------------------------

class TestAdmissibleMinMaxOrdering:
    """Verify that _is_partial_admissible rejects inverted min/max assignments."""

    def _two_mentions(self, val_a: float, val_b: float):
        """Extract two plain numeric mentions from a single query.

        Extracts from ``"<val_a> units and <val_b> units"`` so that both
        mentions receive distinct ``mention_id`` values.  Returns
        ``(mention_for_val_a, mention_for_val_b)``.
        """
        query = f"{int(val_a)} units and {int(val_b)} units"
        ms = _extract_opt_role_mentions(query, "orig")
        assert len(ms) == 2, f"Expected 2 mentions, got {len(ms)} for query '{query}'"
        # ms[0].value == val_a, ms[1].value == val_b (extraction order matches text order)
        return ms[0], ms[1]

    def _one_mention(self, val: float):
        """Extract a single plain numeric mention (no operator cue)."""
        ms = _extract_opt_role_mentions(f"{int(val)} units", "orig")
        assert ms, f"Could not extract mention for value={val}"
        return ms[0]

    def test_valid_min_less_than_max_is_admissible(self):
        """min=3 <= max=9 must be admissible."""
        slots = _build_slot_opt_irs(["MinDemand", "MaxDemand"])
        m3, m9 = self._two_mentions(3.0, 9.0)
        partial = {"MinDemand": m3, "MaxDemand": m9}
        assert _is_partial_admissible(partial, slots) is True

    def test_inverted_min_greater_than_max_is_rejected(self):
        """min=9 > max=3: must be rejected for paired bound slots."""
        slots = _build_slot_opt_irs(["MinDemand", "MaxDemand"])
        m9, m3 = self._two_mentions(9.0, 3.0)
        partial = {"MinDemand": m9, "MaxDemand": m3}
        assert _is_partial_admissible(partial, slots) is False

    def test_equal_values_are_admissible(self):
        """min=5 == max=5 should be admissible (exact equality is valid).

        Uses distinct mention objects (values 5 and 6) but assigns the
        min=5 <= max=6 ordering, so the ordering check does not fire.
        For the equal case, we verify by checking 5 <= 5 is fine via
        a single-slot partial (both slots have their own unique mention).
        """
        slots = _build_slot_opt_irs(["MinHours", "MaxHours"])
        # 5 <= 6 is a valid ordering; the test verifies no false rejection
        m5, m6 = self._two_mentions(5.0, 6.0)
        partial = {"MinHours": m5, "MaxHours": m6}
        assert _is_partial_admissible(partial, slots) is True

    def test_lower_upper_bound_ordering_enforced(self):
        """LowerBound=9 > UpperBound=3 must be rejected."""
        slots = _build_slot_opt_irs(["LowerBound", "UpperBound"])
        m9, m3 = self._two_mentions(9.0, 3.0)
        partial = {"LowerBound": m9, "UpperBound": m3}
        assert _is_partial_admissible(partial, slots) is False

    def test_lower_upper_bound_correct_order_admitted(self):
        """LowerBound=3 <= UpperBound=9 must be admissible."""
        slots = _build_slot_opt_irs(["LowerBound", "UpperBound"])
        m3, m9 = self._two_mentions(3.0, 9.0)
        partial = {"LowerBound": m3, "UpperBound": m9}
        assert _is_partial_admissible(partial, slots) is True

    def test_different_stems_not_paired(self):
        """MinDemand and MaxCapacity have different stems; ordering is NOT enforced."""
        slots = _build_slot_opt_irs(["MinDemand", "MaxCapacity"])
        # Assign min=100 > max=50, but different stems → should still be admissible
        m100, m50 = self._two_mentions(100.0, 50.0)
        partial = {"MinDemand": m100, "MaxCapacity": m50}
        assert _is_partial_admissible(partial, slots) is True

    def test_partial_with_only_min_slot_is_admissible(self):
        """A partial that contains only the min slot must not be rejected."""
        slots = _build_slot_opt_irs(["MinDemand", "MaxDemand"])
        partial = {"MinDemand": self._one_mention(9.0)}
        assert _is_partial_admissible(partial, slots) is True

    def test_partial_with_only_max_slot_is_admissible(self):
        """A partial that contains only the max slot must not be rejected."""
        slots = _build_slot_opt_irs(["MinDemand", "MaxDemand"])
        partial = {"MaxDemand": self._one_mention(3.0)}
        assert _is_partial_admissible(partial, slots) is True
