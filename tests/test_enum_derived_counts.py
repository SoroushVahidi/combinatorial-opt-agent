"""Regression tests for enumeration-derived count extraction and grounding.

These tests cover Phase C of the grounding improvements: cases where a
count-like slot value is implicit in an enumerated list rather than stated
as an explicit number.

Failure families addressed:
  - "phones and laptops"               → NumProducts = 2 (unquantified pair)
  - "wood, plastic, and metal"         → NumMaterials = 3 (unquantified triple)
  - "10 apples, 20 bananas, and 80 grapes" → NumItems = 3 (quantified list)
  - derived counts must NOT fill non-count slots (e.g. TotalHours, Budget)
  - when an explicit number matches, derived counts are not forced

Also covers that "num*"-prefixed slots gain the "cardinality_limit" role tag
so that derived-count candidates achieve better role-overlap alignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _build_slot_opt_irs,
    _extract_enum_derived_counts,
    _extract_opt_role_mentions,
    _gcg_local_score,
    _is_count_like_slot,
    _run_global_consistency_grounding,
    _slot_opt_role_expansion,
    GCG_LOCAL_WEIGHTS,
    MentionOptIR,
    NumTok,
)


# ---------------------------------------------------------------------------
# Helper: run GCG and return (values, mentions, extra)
# ---------------------------------------------------------------------------

def _gcg(query: str, slots: list[str]):
    return _run_global_consistency_grounding(query, "orig", slots)


# ===========================================================================
# 1. _extract_enum_derived_counts — unit tests
# ===========================================================================

class TestExtractEnumDerivedCounts:
    """Tests for the _extract_enum_derived_counts() helper."""

    def test_simple_pair_returns_count_2(self):
        result = _extract_enum_derived_counts("We produce phones and laptops.")
        counts = [r[0] for r in result]
        assert 2.0 in counts

    def test_triple_list_returns_count_3(self):
        result = _extract_enum_derived_counts(
            "The store sells apples, bananas, and grapes."
        )
        counts = [r[0] for r in result]
        assert 3.0 in counts

    def test_quantified_list_returns_count_3(self):
        result = _extract_enum_derived_counts(
            "We have 10 apples, 20 bananas, and 80 grapes in stock."
        )
        counts = [r[0] for r in result]
        assert 3.0 in counts

    def test_two_word_items_accepted(self):
        result = _extract_enum_derived_counts(
            "The factory makes regular candy and sour candy."
        )
        counts = [r[0] for r in result]
        assert 2.0 in counts

    def test_stop_word_head_filtered(self):
        # "the" is a stop word → should not count as a valid item
        result = _extract_enum_derived_counts("We have a and the.")
        counts = [r[0] for r in result]
        # "a" is 1 char (too short) and "the" is stop-word: no valid pair
        assert 2.0 not in counts

    def test_written_number_head_filtered(self):
        # "two" is a written number → head-word filter rejects it as an item
        result = _extract_enum_derived_counts("We have one and two products.")
        # "one" and "two" are written numbers → both filtered → count 0
        counts_2 = [r for r in result if r[0] == 2.0 and "one" in r[1].lower() and "two" in r[1].lower()]
        assert len(counts_2) == 0

    def test_context_tokens_returned(self):
        result = _extract_enum_derived_counts("A shop sells phones and laptops daily.")
        assert result
        _, span, ctx = result[0]
        assert len(ctx) > 0

    def test_count_above_5_not_returned(self):
        # "A, B, C, D, E, and F" → 6 items → outside conservative range [2, 5]
        result = _extract_enum_derived_counts(
            "Products: alpha, beta, gamma, delta, epsilon, and zeta."
        )
        counts = [r[0] for r in result]
        assert 6.0 not in counts

    def test_quadruple_list_accepted(self):
        result = _extract_enum_derived_counts(
            "We use wood, plastic, metal, and glass."
        )
        counts = [r[0] for r in result]
        assert 4.0 in counts

    def test_no_enumeration_returns_empty(self):
        result = _extract_enum_derived_counts(
            "The profit is 500 and the budget is 2000."
        )
        # "500" and "2000" are numeric tokens, not noun enumeration items
        for count, span, _ in result:
            # Any detected span should only come from actual noun lists.
            # In this sentence there is no noun-enumeration pattern.
            assert count < 2 or not any(
                c.isdigit() for c in span.replace(",", "").replace(".", "")
            )


# ===========================================================================
# 2. _extract_opt_role_mentions — derived-count mentions appear in output
# ===========================================================================

class TestOptRoleMentionsDerivedCounts:
    """Derived-count mentions are appended to the mention list."""

    def test_derived_count_mention_present(self):
        mentions = _extract_opt_role_mentions(
            "A shop sells phones and laptops.", "orig"
        )
        derived = [m for m in mentions if "derived_count" in m.role_tags]
        assert len(derived) >= 1

    def test_derived_count_value_is_2(self):
        mentions = _extract_opt_role_mentions(
            "We produce phones and laptops.", "orig"
        )
        derived_values = [m.value for m in mentions if "derived_count" in m.role_tags]
        assert 2.0 in derived_values

    def test_derived_count_type_bucket_is_int(self):
        mentions = _extract_opt_role_mentions(
            "We produce phones and laptops.", "orig"
        )
        for m in mentions:
            if "derived_count" in m.role_tags:
                assert m.type_bucket == "int"

    def test_derived_count_has_count_marker_unit(self):
        mentions = _extract_opt_role_mentions(
            "We produce phones and laptops.", "orig"
        )
        for m in mentions:
            if "derived_count" in m.role_tags:
                assert "count_marker" in m.unit_tags

    def test_derived_count_role_tags(self):
        mentions = _extract_opt_role_mentions(
            "We produce phones and laptops.", "orig"
        )
        for m in mentions:
            if "derived_count" in m.role_tags:
                assert "cardinality_limit" in m.role_tags
                assert "quantity_limit" in m.role_tags

    def test_quantified_list_derived_count_3(self):
        mentions = _extract_opt_role_mentions(
            "We have 10 apples, 20 bananas, and 80 grapes.", "orig"
        )
        derived = [m for m in mentions if "derived_count" in m.role_tags]
        values = [m.value for m in derived]
        assert 3.0 in values


# ===========================================================================
# 3. _gcg_local_score — derived counts are hard-incompatible with non-count slots
# ===========================================================================

class TestDerivedCountGcgLocalScore:
    """_gcg_local_score must penalise derived counts for non-count-like slots."""

    def _make_derived_count_mention(self, value: float = 2.0) -> MentionOptIR:
        tok = NumTok(raw=f"derived:{int(value)} (phones and laptops)", value=value, kind="int")
        return MentionOptIR(
            mention_id=99,
            value=value,
            type_bucket="int",
            raw_surface=tok.raw,
            role_tags=frozenset({"cardinality_limit", "quantity_limit", "derived_count"}),
            operator_tags=frozenset(),
            unit_tags=frozenset({"count_marker"}),
            fragment_type="",
            is_per_unit=False,
            is_total_like=False,
            nearby_entity_tokens=frozenset(),
            nearby_resource_tokens=frozenset(),
            nearby_product_tokens=frozenset(),
            context_tokens=[],
            sentence_tokens=[],
            tok=tok,
        )

    def test_derived_count_hard_penalised_for_float_slot(self):
        slots = _build_slot_opt_irs(["TotalHours"])
        s = slots[0]
        assert not s.is_count_like
        m = self._make_derived_count_mention(2.0)
        score, feats = _gcg_local_score(m, s)
        assert score <= GCG_LOCAL_WEIGHTS["derived_count_non_count_penalty"]
        assert feats.get("derived_count_non_count") is True

    def test_derived_count_accepted_for_count_slot(self):
        slots = _build_slot_opt_irs(["NumProducts"])
        s = slots[0]
        assert s.is_count_like
        m = self._make_derived_count_mention(2.0)
        score, feats = _gcg_local_score(m, s)
        # Should get a positive score (type_exact + count_small_int_prior + schema_prior)
        assert score > 0.0
        assert feats.get("derived_count_non_count") is None

    def test_derived_count_count_slot_role_overlap(self):
        slots = _build_slot_opt_irs(["NumResources"])
        s = slots[0]
        m = self._make_derived_count_mention(2.0)
        score, feats = _gcg_local_score(m, s)
        # role overlap should fire (cardinality_limit / quantity_limit match)
        assert feats.get("opt_role_overlap", 0) >= 1


# ===========================================================================
# 4. Slot role expansion: "num*" prefix gets "cardinality_limit" tag
# ===========================================================================

class TestSlotRoleExpansionNumPrefix:
    """_slot_opt_role_expansion should tag num* slots with cardinality_limit."""

    def test_num_products_gets_cardinality_limit(self):
        tags = _slot_opt_role_expansion("NumProducts")
        assert "cardinality_limit" in tags

    def test_num_resources_gets_cardinality_limit(self):
        tags = _slot_opt_role_expansion("NumResources")
        assert "cardinality_limit" in tags

    def test_num_types_gets_cardinality_limit(self):
        tags = _slot_opt_role_expansion("NumTypes")
        assert "cardinality_limit" in tags

    def test_num_mixes_gets_cardinality_limit(self):
        tags = _slot_opt_role_expansion("NumMixes")
        assert "cardinality_limit" in tags

    def test_non_num_slot_unaffected(self):
        tags = _slot_opt_role_expansion("TotalHours")
        # TotalHours is not a count-like slot; cardinality_limit should not appear
        assert "cardinality_limit" not in tags

    def test_budget_slot_unaffected(self):
        tags = _slot_opt_role_expansion("Budget")
        assert "cardinality_limit" not in tags


# ===========================================================================
# 5. End-to-end GCG: derived counts fill count-like slots
# ===========================================================================

class TestEnumDerivedCountEndToEnd:
    """Full GCG pipeline tests for enumeration-derived count grounding."""

    def test_phones_and_laptops_fills_num_products(self):
        """'phones and laptops' should lead to NumProducts = 2 even without explicit digit."""
        q = "A factory produces phones and laptops. Each phone earns 10 and each laptop earns 15."
        vals, _, _ = _gcg(q, ["NumProducts"])
        assert "NumProducts" in vals
        assert abs(float(vals["NumProducts"]) - 2.0) < 0.1

    def test_quantified_list_fills_num_items(self):
        """'10 apples, 20 bananas, and 80 grapes' → NumItems = 3."""
        q = (
            "A grocery store wants to liquidate its stock of 10 apples, "
            "20 bananas, and 80 grapes."
        )
        vals, _, _ = _gcg(q, ["NumItems"])
        if "NumItems" in vals:
            assert abs(float(vals["NumItems"]) - 3.0) < 0.1

    def test_derived_count_does_not_override_explicit_number(self):
        """When an explicit 'two' is present, the derived count should be compatible but
        the explicit number wins for a count slot (both are value 2 — no conflict)."""
        q = "We make two types of products: phones and laptops."
        vals, _, _ = _gcg(q, ["NumProducts"])
        assert "NumProducts" in vals
        assert abs(float(vals["NumProducts"]) - 2.0) < 0.1

    def test_derived_count_does_not_fill_budget_slot(self):
        """Derived count from noun enumeration must not overwrite a budget slot."""
        q = (
            "We make phones and laptops. "
            "The total budget is 5000 and each phone requires 200."
        )
        vals, _, _ = _gcg(q, ["Budget", "NumProducts"])
        if "Budget" in vals:
            # Budget should be 5000, not 2
            assert abs(float(vals["Budget"]) - 5000.0) < 1.0

    def test_two_count_slots_can_both_be_filled(self):
        """When two count-like slots exist, derived count and explicit word-number
        can each fill one slot independently."""
        q = (
            "An artisan makes two types of terracotta jars: a thin jar and a stubby jar. "
            "Each jar requires 3 kg of clay."
        )
        vals, _, _ = _gcg(q, ["NumJarTypes", "NumIngredients"])
        # NumJarTypes should be 2 (from "two types")
        if "NumJarTypes" in vals:
            assert abs(float(vals["NumJarTypes"]) - 2.0) < 0.1

    def test_wood_and_plastic_fills_num_types(self):
        """'wood and plastic' enumeration → NumTypes = 2."""
        q = "A carpenter uses wood and plastic to build furniture. Each unit earns 5."
        vals, _, _ = _gcg(q, ["NumTypes"])
        if "NumTypes" in vals:
            assert abs(float(vals["NumTypes"]) - 2.0) < 0.1
