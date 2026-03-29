"""Regression tests for count-like slot grounding protections.

These tests cover the failure family where the pipeline was assigning
coefficient values, fractions, or large resource amounts to count-like
slots such as NumProducts, NumResources, NumMixes, NumCandyTypes.

Failure examples from benchmarks:
  - NumProducts = 50 instead of 2
  - NumResources = 20 instead of 2
  - NumMixes = 60 instead of 2
  - NumCandyTypes = 0.2 instead of 2
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _build_slot_opt_irs,
    _build_slot_records,
    _expected_type,
    _gcg_local_score,
    _is_count_like_slot,
    _run_global_consistency_grounding,
    _score_mention_slot,
    MentionRecord,
    NumTok,
    SlotRecord,
    _normalize_tokens,
    _slot_aliases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mention_record(
    value: float, kind: str, ctx: list[str] | None = None, index: int = 0
) -> MentionRecord:
    ctx = ctx or []
    tok = NumTok(raw=str(value), value=value, kind=kind)
    return MentionRecord(
        index=index,
        tok=tok,
        context_tokens=ctx,
        sentence_tokens=ctx,
        cue_words=set(),
    )


def _make_slot_record(name: str) -> SlotRecord:
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
        is_count_like=_is_count_like_slot(name),
    )


def _gcg(query: str, slots: list[str]):
    return _run_global_consistency_grounding(query, "orig", slots)


# ---------------------------------------------------------------------------
# Phase 1: _is_count_like_slot detection
# ---------------------------------------------------------------------------


class TestIsCountLikeSlot:
    """_is_count_like_slot must correctly classify slot names."""

    def test_num_products_is_count_like(self):
        assert _is_count_like_slot("NumProducts") is True

    def test_num_resources_is_count_like(self):
        assert _is_count_like_slot("NumResources") is True

    def test_num_mixes_is_count_like(self):
        assert _is_count_like_slot("NumMixes") is True

    def test_num_candy_types_is_count_like(self):
        assert _is_count_like_slot("NumCandyTypes") is True

    def test_num_types_is_count_like(self):
        assert _is_count_like_slot("NumTypes") is True

    def test_num_items_is_count_like(self):
        assert _is_count_like_slot("NumItems") is True

    def test_number_of_machines_is_count_like(self):
        assert _is_count_like_slot("NumberOfMachines") is True

    def test_num_workers_is_count_like(self):
        assert _is_count_like_slot("NumWorkers") is True

    def test_num_tasks_is_count_like(self):
        assert _is_count_like_slot("NumTasks") is True

    def test_num_batches_is_count_like(self):
        assert _is_count_like_slot("NumBatches") is True

    def test_budget_is_not_count_like(self):
        assert _is_count_like_slot("Budget") is False

    def test_cost_is_not_count_like(self):
        assert _is_count_like_slot("CostPerUnit") is False

    def test_fraction_slot_is_not_count_like(self):
        assert _is_count_like_slot("MaxFractionWassaAds") is False

    def test_total_hours_is_not_count_like(self):
        assert _is_count_like_slot("TotalHours") is False

    def test_slot_ir_has_count_like_flag(self):
        irs = _build_slot_opt_irs(["NumProducts", "Budget"])
        flags = {s.name: s.is_count_like for s in irs}
        assert flags["NumProducts"] is True
        assert flags["Budget"] is False

    def test_slot_record_has_count_like_flag(self):
        recs = _build_slot_records(["NumProducts", "Budget"])
        flags = {s.name: s.is_count_like for s in recs}
        assert flags["NumProducts"] is True
        assert flags["Budget"] is False


# ---------------------------------------------------------------------------
# Phase 2: Hard incompatibility — non-integer float → count-like slot
# ---------------------------------------------------------------------------


class TestCountSlotFloatIncompatible:
    """Non-integer decimal (float) values must be hard-incompatible with count slots."""

    def test_decimal_float_blocked_for_count_slot_gcg(self):
        """A decimal like 0.2 must not be assigned to NumCandyTypes."""
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        q = "The defect rate is 0.2 and there are candy types to consider."
        mentions = _extract_opt_role_mentions(q, "orig")
        slots = _build_slot_opt_irs(["NumCandyTypes"])
        s = slots[0]
        for m in mentions:
            if m.value is not None and abs(m.value - 0.2) < 1e-9:
                score, feats = _gcg_local_score(m, s)
                assert score < 0, (
                    f"0.2 (kind={m.type_bucket}) should be incompatible with NumCandyTypes, got score={score}"
                )

    def test_decimal_float_blocked_via_score_mention_slot(self):
        """_score_mention_slot must return a hard incompatibility score for float → count slot."""
        m = _make_mention_record(0.2, "float")
        s = _make_slot_record("NumCandyTypes")
        score, feats = _score_mention_slot(m, s)
        assert score < 0, f"Expected negative score, got {score}"
        assert feats.get("count_slot_float_incompatible") is True

    def test_decimal_1_5_blocked_for_count_slot(self):
        m = _make_mention_record(1.5, "float")
        s = _make_slot_record("NumProducts")
        score, feats = _score_mention_slot(m, s)
        assert score < 0
        assert feats.get("count_slot_float_incompatible") is True

    def test_integer_float_not_blocked(self):
        """An integer like 2.0 stored as 'int' kind must NOT be blocked."""
        m = _make_mention_record(2.0, "int")
        s = _make_slot_record("NumProducts")
        score, feats = _score_mention_slot(m, s)
        assert score > 0
        assert not feats.get("count_slot_float_incompatible")

    def test_float_incompatible_does_not_affect_non_count_slot(self):
        """A decimal float should still be allowed on a non-count (float) slot."""
        m = _make_mention_record(0.2, "float")
        s = _make_slot_record("RequiredEggsPerSandwich")
        score, feats = _score_mention_slot(m, s)
        assert score > 0, f"0.2 should be valid for a float slot, got {score}"
        assert not feats.get("count_slot_float_incompatible")


# ---------------------------------------------------------------------------
# Phase 3: Small-integer cardinality prior
# ---------------------------------------------------------------------------


class TestCountSlotSmallIntPrior:
    """Small integers get a cardinality bonus on count-like slots."""

    def test_small_int_gets_prior_bonus(self):
        m = _make_mention_record(2.0, "int")
        s = _make_slot_record("NumProducts")
        score, feats = _score_mention_slot(m, s)
        assert feats.get("count_small_int_prior") is True

    def test_small_int_gets_gcg_prior_bonus(self):
        slots = _build_slot_opt_irs(["NumProducts"])
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        q = "two products"
        mentions = _extract_opt_role_mentions(q, "orig")
        if mentions:
            score, feats = _gcg_local_score(mentions[0], slots[0])
            assert feats.get("count_small_int_prior") is True

    def test_small_int_beats_large_int_for_count_slot(self):
        """On count slots, 2 must score higher than 20 (above plausible_max)."""
        m_small = _make_mention_record(2.0, "int")
        m_large = _make_mention_record(20.0, "int")
        s = _make_slot_record("NumProducts")
        score_small, _ = _score_mention_slot(m_small, s)
        score_large, _ = _score_mention_slot(m_large, s)
        assert score_small > score_large, (
            f"Small int (2) should score higher than 20 on count slot; "
            f"got {score_small} vs {score_large}"
        )

    def test_large_int_penalty_applied(self):
        """Very large integers (> 50) should get the large-int penalty."""
        m = _make_mention_record(60.0, "int")
        s = _make_slot_record("NumMixes")
        score, feats = _score_mention_slot(m, s)
        assert feats.get("count_large_int_penalty") is True

    def test_count_prior_not_applied_to_non_count_slot(self):
        """Small-int prior must NOT be applied to non-count float slots."""
        m = _make_mention_record(2.0, "int")
        s = _make_slot_record("RequiredEggsPerSandwich")
        _, feats = _score_mention_slot(m, s)
        assert not feats.get("count_small_int_prior")
        assert not feats.get("count_large_int_penalty")

    def test_count_prior_not_applied_to_currency_slot(self):
        m = _make_mention_record(2.0, "int")
        s = _make_slot_record("Budget")
        _, feats = _score_mention_slot(m, s)
        assert not feats.get("count_small_int_prior")


# ---------------------------------------------------------------------------
# Phase 4: End-to-end grounding regression tests (benchmark failures)
# ---------------------------------------------------------------------------


class TestCountSlotGroundingRegressions:
    """Grounding must prefer small cardinality candidates for count-like slots."""

    def test_num_products_prefers_2_over_20(self):
        """'20 resources and two product types' — NumProducts should be 2."""
        q = "There are 20 resources and two product types."
        vals, _, _ = _gcg(q, ["NumProducts"])
        assert vals.get("NumProducts") is not None
        assert abs(float(vals["NumProducts"]) - 2.0) < 0.1, (
            f"NumProducts should be 2 (from 'two product types'), got {vals['NumProducts']}"
        )

    def test_num_mixes_prefers_2_over_60(self):
        """'Two candy mixes use 60 grams each' — NumMixes should be 2."""
        q = "Two candy mixes use 60 grams each."
        vals, _, _ = _gcg(q, ["NumMixes"])
        assert vals.get("NumMixes") is not None
        assert abs(float(vals["NumMixes"]) - 2.0) < 0.1, (
            f"NumMixes should be 2, got {vals['NumMixes']}"
        )

    def test_num_jar_types_prefers_2_over_50(self):
        """Artisan example: NumJarTypes should be 2, not 50."""
        q = "An artisan makes two types of terracotta jars and can produce 50 per day."
        vals, _, _ = _gcg(q, ["NumJarTypes"])
        assert vals.get("NumJarTypes") is not None
        assert abs(float(vals["NumJarTypes"]) - 2.0) < 0.1, (
            f"NumJarTypes should be 2, got {vals['NumJarTypes']}"
        )

    def test_num_resources_prefers_2_over_20(self):
        """'two resources: wood and plastic' — NumResources should be 2."""
        q = "There are two resources: wood and plastic. Total is 20 units."
        vals, _, _ = _gcg(q, ["NumResources"])
        assert vals.get("NumResources") is not None
        assert abs(float(vals["NumResources"]) - 2.0) < 0.1

    def test_num_candy_types_rejects_fraction(self):
        """A fraction 0.2 (mix ratio) must not be assigned to NumCandyTypes."""
        q = "fraction of mix A is 0.2, mix B is 0.8"
        vals, _, _ = _gcg(q, ["NumCandyTypes"])
        # 0.2 should be blocked as percent-context float; slot left empty is acceptable.
        if vals.get("NumCandyTypes") is not None:
            v = float(vals["NumCandyTypes"])
            assert v > 1.0, (
                f"NumCandyTypes must not be assigned a fractional value 0.2; got {v}"
            )

    def test_num_candy_types_with_correct_count(self):
        """When two is explicitly mentioned, NumCandyTypes should be 2 even if 0.2 exists."""
        q = "The defect rate is 0.2. There are two candy types."
        vals, _, _ = _gcg(q, ["NumCandyTypes"])
        assert vals.get("NumCandyTypes") is not None
        assert abs(float(vals["NumCandyTypes"]) - 2.0) < 0.1

    def test_num_products_three_with_large_hours(self):
        """'three products and 400 total hours' — NumProducts should be 3."""
        q = "There are three products and 400 total hours available."
        vals, _, _ = _gcg(q, ["NumProducts", "TotalHours"])
        assert vals.get("NumProducts") is not None
        assert abs(float(vals["NumProducts"]) - 3.0) < 0.1, (
            f"NumProducts should be 3, got {vals['NumProducts']}"
        )
        assert vals.get("TotalHours") is not None
        assert abs(float(vals["TotalHours"]) - 400.0) < 0.1

    def test_non_count_slot_unaffected_by_count_rules(self):
        """Non-count float slots must not be harmed by count-slot rules."""
        q = "Each sandwich requires 2 eggs and 1.5 strips of bacon."
        vals, _, _ = _gcg(q, ["RequiredEggsPerSandwich", "RequiredBaconPerSandwich"])
        # Both should be fillable; the specific values are secondary here.
        filled = sum(1 for v in vals.values() if v is not None)
        assert filled >= 1, "Non-count float slots should still be fillable"

    def test_budget_slot_unaffected_by_count_rules(self):
        """Currency/budget slots must not be harmed by count-slot rules."""
        q = "The total budget is $5000 and there are 2 products."
        vals, _, _ = _gcg(q, ["Budget", "NumProducts"])
        if vals.get("Budget") is not None:
            assert float(vals["Budget"]) >= 100, (
                f"Budget should be large (5000), not 2; got {vals['Budget']}"
            )
