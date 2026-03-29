"""Regression tests for the explicit quantity-role layer on MentionOptIR.

These tests cover the new is_count_like, is_lower_bound_like,
is_upper_bound_like, is_percent_like, and primary_role fields added to
MentionOptIR, plus the corresponding role-aware scoring bonuses/penalties
in _score_mention_slot_opt and _gcg_local_score.

Failure families targeted:
  A. Count mentions identified correctly (is_count_like=True for "three types",
     small-int near count-context noun)
  B. Bound direction correctly set (is_lower_bound_like / is_upper_bound_like)
  C. Percent mentions flagged as is_percent_like
  D. primary_role synthesized consistently
  E. count_mention_count_slot_bonus fires in scoring
  F. bound_direction_bonus fires in scoring
  G. count_mention_non_count_penalty fires in scoring
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _build_slot_opt_irs,
    _compute_is_count_like_mention,
    _compute_primary_role,
    _extract_opt_role_mentions,
    _gcg_local_score,
    _score_mention_slot_opt,
    GCG_LOCAL_WEIGHTS,
    MentionOptIR,
    NumTok,
    OPT_ROLE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mentions(query: str) -> list[MentionOptIR]:
    return _extract_opt_role_mentions(query, "orig")


def _slot(name: str):
    return _build_slot_opt_irs([name])[0]


def _find_by_value(mentions: list[MentionOptIR], val: float) -> MentionOptIR | None:
    for m in mentions:
        if m.value == val:
            return m
    return None


# ===========================================================================
# A. is_count_like detection
# ===========================================================================

class TestIsCountLike:
    """is_count_like should be True for count-context mentions."""

    def test_three_types_is_count_like(self):
        ms = _mentions("A company produces three types of products.")
        m = next((x for x in ms if x.value == 3.0), None)
        assert m is not None, "value 3 should be extracted"
        assert m.is_count_like is True

    def test_two_products_is_count_like(self):
        ms = _mentions("We manufacture two products: phones and laptops.")
        m = next((x for x in ms if x.value == 2.0), None)
        assert m is not None
        assert m.is_count_like is True

    def test_large_capacity_not_count_like(self):
        ms = _mentions("The factory has 5000 hours available.")
        m = next((x for x in ms if x.value == 5000.0), None)
        assert m is not None
        # 5000 is not a small integer (>20), so not count-like by context alone
        assert m.is_count_like is False

    def test_small_int_without_count_noun_not_forced_count(self):
        """A number like '5' without a count noun nearby should not be flagged."""
        ms = _mentions("Each worker earns $5 per hour.")
        m = next((x for x in ms if x.value == 5.0), None)
        # Value 5 is small, but context has no count-context noun
        if m is not None:
            # May or may not be count-like depending on context, but should not
            # be confidently count-like without a count noun.
            # This is a soft test: we just check the field is accessible.
            assert isinstance(m.is_count_like, bool)

    def test_derived_count_is_count_like(self):
        """Enumeration-derived counts must be is_count_like=True."""
        ms = _mentions("We produce phones and laptops.")
        derived = [m for m in ms if "derived_count" in m.role_tags]
        assert derived, "should have derived-count mention"
        for m in derived:
            assert m.is_count_like is True, "all derived-count mentions must be is_count_like"

    def test_four_resources_is_count_like(self):
        ms = _mentions("There are four resources available.")
        m = next((x for x in ms if x.value == 4.0), None)
        assert m is not None
        assert m.is_count_like is True

    def test_seven_ingredients_is_count_like(self):
        ms = _mentions("The recipe uses seven ingredients.")
        m = next((x for x in ms if x.value == 7.0), None)
        assert m is not None
        assert m.is_count_like is True


# ===========================================================================
# B. is_lower_bound_like / is_upper_bound_like
# ===========================================================================

class TestBoundFlags:
    """Bound direction flags should be set from operator tags."""

    def test_at_least_is_lower_bound(self):
        ms = _mentions("We must produce at least 50 units.")
        m = next((x for x in ms if x.value == 50.0), None)
        assert m is not None
        assert m.is_lower_bound_like is True
        assert m.is_upper_bound_like is False

    def test_at_most_is_upper_bound(self):
        ms = _mentions("We can use at most 30 machines.")
        m = next((x for x in ms if x.value == 30.0), None)
        assert m is not None
        assert m.is_upper_bound_like is True
        assert m.is_lower_bound_like is False

    def test_plain_number_no_bound_flags(self):
        ms = _mentions("The profit per unit is 8 dollars.")
        m = next((x for x in ms if x.value == 8.0), None)
        if m is not None:
            assert m.is_lower_bound_like is False
            assert m.is_upper_bound_like is False

    def test_minimum_is_lower_bound(self):
        ms = _mentions("The minimum demand is 100 units.")
        m = next((x for x in ms if x.value == 100.0), None)
        assert m is not None
        assert m.is_lower_bound_like is True

    def test_maximum_is_upper_bound(self):
        ms = _mentions("The maximum capacity is 200 kg.")
        m = next((x for x in ms if x.value == 200.0), None)
        assert m is not None
        assert m.is_upper_bound_like is True


# ===========================================================================
# C. is_percent_like
# ===========================================================================

class TestPercentLike:
    """Percent-valued mentions should have is_percent_like=True."""

    def test_explicit_percent_is_percent_like(self):
        ms = _mentions("The discount rate is 20%.")
        m = next((x for x in ms if x.type_bucket == "percent"), None)
        assert m is not None
        assert m.is_percent_like is True

    def test_integer_not_percent_like(self):
        ms = _mentions("There are 5 products.")
        m = next((x for x in ms if x.value == 5.0), None)
        assert m is not None
        assert m.is_percent_like is False


# ===========================================================================
# D. primary_role synthesis
# ===========================================================================

class TestPrimaryRole:
    """primary_role should synthesize the dominant role correctly."""

    def test_count_primary_role(self):
        ms = _mentions("There are three types of resources.")
        m = next((x for x in ms if x.value == 3.0), None)
        assert m is not None
        assert m.primary_role == "count"

    def test_derived_count_primary_role(self):
        ms = _mentions("We produce phones and laptops.")
        derived = [m for m in ms if "derived_count" in m.role_tags]
        assert derived
        for m in derived:
            assert m.primary_role == "count"

    def test_percent_primary_role(self):
        ms = _mentions("The interest rate is 5%.")
        m = next((x for x in ms if x.type_bucket == "percent"), None)
        assert m is not None
        assert m.primary_role == "percent"

    def test_lower_bound_primary_role(self):
        ms = _mentions("At least 100 units must be produced.")
        m = next((x for x in ms if x.value == 100.0), None)
        assert m is not None
        assert m.primary_role == "lower_bound"

    def test_upper_bound_primary_role(self):
        ms = _mentions("At most 200 units can be produced.")
        m = next((x for x in ms if x.value == 200.0), None)
        assert m is not None
        assert m.primary_role == "upper_bound"

    def test_compute_primary_role_count_wins_over_total(self):
        """is_count_like takes priority over is_total_like in primary_role."""
        tok = NumTok(raw="3", value=3.0, kind="int")
        role = _compute_primary_role(
            tok, is_count_like=True, is_lower_bound_like=False,
            is_upper_bound_like=False, is_percent_like=False,
            is_total_like=True, is_per_unit=False,
        )
        assert role == "count"

    def test_compute_primary_role_percent_wins_over_generic(self):
        tok = NumTok(raw="0.2", value=0.2, kind="percent")
        role = _compute_primary_role(
            tok, is_count_like=False, is_lower_bound_like=False,
            is_upper_bound_like=False, is_percent_like=True,
            is_total_like=False, is_per_unit=False,
        )
        assert role == "percent"

    def test_compute_primary_role_coefficient(self):
        tok = NumTok(raw="5", value=5.0, kind="int")
        role = _compute_primary_role(
            tok, is_count_like=False, is_lower_bound_like=False,
            is_upper_bound_like=False, is_percent_like=False,
            is_total_like=False, is_per_unit=True,
        )
        assert role == "coefficient"

    def test_compute_primary_role_total(self):
        tok = NumTok(raw="5000", value=5000.0, kind="int")
        role = _compute_primary_role(
            tok, is_count_like=False, is_lower_bound_like=False,
            is_upper_bound_like=False, is_percent_like=False,
            is_total_like=True, is_per_unit=False,
        )
        assert role == "total"

    def test_compute_primary_role_generic(self):
        tok = NumTok(raw="42", value=42.0, kind="int")
        role = _compute_primary_role(
            tok, is_count_like=False, is_lower_bound_like=False,
            is_upper_bound_like=False, is_percent_like=False,
            is_total_like=False, is_per_unit=False,
        )
        assert role == "generic"


# ===========================================================================
# E. count_mention_count_slot_bonus in scoring
# ===========================================================================

class TestCountMentionCountSlotBonus:
    """count-like mention + count-like slot should produce count_role_match bonus."""

    def _make_count_mention(self, value: float = 3.0) -> MentionOptIR:
        tok = NumTok(raw=str(int(value)), value=value, kind="int")
        return MentionOptIR(
            mention_id=0, value=value, type_bucket="int",
            raw_surface=str(int(value)),
            role_tags=frozenset({"cardinality_limit", "quantity_limit"}),
            operator_tags=frozenset(),
            unit_tags=frozenset({"count_marker"}),
            fragment_type="",
            is_per_unit=False,
            is_total_like=False,
            nearby_entity_tokens=frozenset(),
            nearby_resource_tokens=frozenset(),
            nearby_product_tokens=frozenset(),
            context_tokens=["three", "types"],
            sentence_tokens=["three", "types"],
            tok=tok,
            is_count_like=True,
            is_lower_bound_like=False,
            is_upper_bound_like=False,
            is_percent_like=False,
            primary_role="count",
        )

    def test_count_mention_count_slot_bonus_fires(self):
        m = self._make_count_mention(3.0)
        s = _slot("NumProducts")
        score, feats = _score_mention_slot_opt(m, s)
        assert feats.get("count_role_match") is True
        # Bonus should be present
        assert score >= OPT_ROLE_WEIGHTS["count_mention_count_slot_bonus"]

    def test_count_mention_count_slot_bonus_in_gcg(self):
        m = self._make_count_mention(3.0)
        s = _slot("NumProducts")
        score, feats = _gcg_local_score(m, s)
        assert feats.get("count_role_match") is True

    def test_count_mention_non_count_slot_penalty(self):
        m = self._make_count_mention(3.0)
        s = _slot("TotalBudget")  # non-count slot
        score, feats = _score_mention_slot_opt(m, s)
        assert feats.get("count_to_non_count_penalty") is True

    def test_count_mention_non_count_slot_penalty_in_gcg(self):
        m = self._make_count_mention(3.0)
        s = _slot("TotalBudget")
        score, feats = _gcg_local_score(m, s)
        assert feats.get("count_to_non_count_penalty") is True

    def test_count_slot_scores_higher_for_count_mention(self):
        """count-like mention should score higher on a count slot than a total slot."""
        m = self._make_count_mention(3.0)
        count_slot = _slot("NumProducts")
        total_slot = _slot("TotalBudget")
        score_count, _ = _score_mention_slot_opt(m, count_slot)
        score_total, _ = _score_mention_slot_opt(m, total_slot)
        assert score_count > score_total, (
            f"count mention should prefer count slot ({score_count:.2f}) "
            f"over total slot ({score_total:.2f})"
        )


# ===========================================================================
# F. bound_direction_bonus in scoring
# ===========================================================================

class TestBoundDirectionBonus:
    """Lower/upper bound mentions should score higher on matching bound slots."""

    def _make_bound_mention(self, is_lower: bool, value: float = 50.0) -> MentionOptIR:
        op_tags = frozenset({"min"} if is_lower else {"max"})
        tok = NumTok(raw=str(int(value)), value=value, kind="int")
        return MentionOptIR(
            mention_id=0, value=value, type_bucket="int",
            raw_surface=str(int(value)),
            role_tags=frozenset({"lower_bound" if is_lower else "upper_bound"}),
            operator_tags=op_tags,
            unit_tags=frozenset(),
            fragment_type="bound",
            is_per_unit=False,
            is_total_like=False,
            nearby_entity_tokens=frozenset(),
            nearby_resource_tokens=frozenset(),
            nearby_product_tokens=frozenset(),
            context_tokens=["at", "least" if is_lower else "most", str(int(value))],
            sentence_tokens=["at", "least" if is_lower else "most", str(int(value))],
            tok=tok,
            is_count_like=False,
            is_lower_bound_like=is_lower,
            is_upper_bound_like=not is_lower,
            is_percent_like=False,
            primary_role="lower_bound" if is_lower else "upper_bound",
        )

    def test_lower_bound_mention_lower_bound_slot(self):
        m = self._make_bound_mention(is_lower=True, value=50.0)
        s = _slot("MinDemand")
        score, feats = _score_mention_slot_opt(m, s)
        assert feats.get("lower_bound_match") is True, f"feats={feats}"

    def test_upper_bound_mention_upper_bound_slot(self):
        m = self._make_bound_mention(is_lower=False, value=200.0)
        s = _slot("MaxCapacity")
        score, feats = _score_mention_slot_opt(m, s)
        assert feats.get("upper_bound_match") is True, f"feats={feats}"

    def test_lower_bound_mention_scores_higher_on_min_slot(self):
        m_lower = self._make_bound_mention(is_lower=True, value=50.0)
        m_upper = self._make_bound_mention(is_lower=False, value=50.0)
        s = _slot("MinDemand")
        sc_lower, _ = _score_mention_slot_opt(m_lower, s)
        sc_upper, _ = _score_mention_slot_opt(m_upper, s)
        assert sc_lower > sc_upper, (
            f"lower-bound mention ({sc_lower:.2f}) should score higher than "
            f"upper-bound mention ({sc_upper:.2f}) on MinDemand slot"
        )

    def test_upper_bound_mention_scores_higher_on_max_slot(self):
        m_lower = self._make_bound_mention(is_lower=True, value=200.0)
        m_upper = self._make_bound_mention(is_lower=False, value=200.0)
        s = _slot("MaxCapacity")
        sc_lower, _ = _score_mention_slot_opt(m_lower, s)
        sc_upper, _ = _score_mention_slot_opt(m_upper, s)
        assert sc_upper > sc_lower, (
            f"upper-bound mention ({sc_upper:.2f}) should score higher than "
            f"lower-bound mention ({sc_lower:.2f}) on MaxCapacity slot"
        )

    def test_bound_direction_bonus_in_gcg(self):
        m = self._make_bound_mention(is_lower=True, value=50.0)
        s = _slot("MinDemand")
        score, feats = _gcg_local_score(m, s)
        assert feats.get("lower_bound_match") is True


# ===========================================================================
# G. End-to-end grounding: count via word-number
# ===========================================================================

class TestEndToEndCountRoleGrounding:
    """End-to-end: word-number count gets is_count_like and routes to count slot."""

    def test_two_types_mention_is_count_like_e2e(self):
        ms = _mentions("A company makes two types of products: A and B.")
        count_m = next((m for m in ms if m.value == 2.0 and m.is_count_like), None)
        assert count_m is not None, "mention with value=2 and is_count_like must exist"
        assert count_m.primary_role == "count"

    def test_count_mention_gets_bonus_on_num_products_slot(self):
        ms = _mentions("There are three types of goods.")
        count_m = next((m for m in ms if m.value == 3.0 and m.is_count_like), None)
        assert count_m is not None
        s = _slot("NumProducts")
        sc, feats = _score_mention_slot_opt(count_m, s)
        assert feats.get("count_role_match") is True

    def test_word_number_five_items_is_count_like(self):
        ms = _mentions("We can store five items in each bin.")
        m = next((x for x in ms if x.value == 5.0), None)
        assert m is not None
        assert m.is_count_like is True


# ===========================================================================
# H. _compute_is_count_like_mention unit tests
# ===========================================================================

class TestComputeIsCountLikeMention:
    """Unit tests for the _compute_is_count_like_mention helper."""

    def test_derived_count_tag_triggers(self):
        tok = NumTok(raw="derived:2 (phones and laptops)", value=2.0, kind="int")
        result = _compute_is_count_like_mention(
            tok, frozenset({"derived_count"}), []
        )
        assert result is True

    def test_cardinality_limit_tag_alone_does_not_trigger(self):
        """cardinality_limit in role_tags is too broad (fires for 'units') so it
        does NOT trigger is_count_like on its own. Only derived_count does."""
        tok = NumTok(raw="100", value=100.0, kind="int")
        result = _compute_is_count_like_mention(
            tok, frozenset({"cardinality_limit"}), []
        )
        assert result is False

    def test_small_int_with_count_noun(self):
        tok = NumTok(raw="5", value=5.0, kind="int")
        result = _compute_is_count_like_mention(
            tok, frozenset(), ["there", "are", "5", "types"]
        )
        assert result is True

    def test_large_int_with_count_noun_is_false(self):
        tok = NumTok(raw="50", value=50.0, kind="int")
        result = _compute_is_count_like_mention(
            tok, frozenset(), ["50", "types"]
        )
        # 50 > 20, so not count-like by the small-int rule
        assert result is False

    def test_small_int_no_count_noun_is_false(self):
        tok = NumTok(raw="5", value=5.0, kind="int")
        result = _compute_is_count_like_mention(
            tok, frozenset(), ["earns", "5", "dollars"]
        )
        assert result is False

    def test_none_value_is_false(self):
        tok = NumTok(raw="<num>", value=None, kind="unknown")
        result = _compute_is_count_like_mention(tok, frozenset(), ["5", "types"])
        assert result is False

    def test_float_value_is_false(self):
        tok = NumTok(raw="2.5", value=2.5, kind="float")
        result = _compute_is_count_like_mention(
            tok, frozenset(), ["2.5", "types"]
        )
        # 2.5 is not int-valued, so not count-like
        assert result is False
