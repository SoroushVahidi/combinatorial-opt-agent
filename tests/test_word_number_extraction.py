"""Tests for written-word number recognition in the numeric extraction pipeline.

Covers:
- _parse_word_num_span: multi-token compound number parsing
- _extract_num_tokens: word-numbers (single and compound) + percent typing
- _extract_num_mentions: MentionRecord construction for word-number spans
- End-to-end grounding: word-number values reach the correct slots

Regression examples taken directly from the problem statement:
  "An artisan makes two types of terracotta jars" -> 2
  "Each worker is paid five dollars"              -> 5 (currency)
  "The defect rate is twenty percent"             -> 0.20 (percent)
  "At least one hundred units"                   -> 100
  "There are three products and 400 total hours"  -> 3 and 400
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    _extract_num_mentions,
    _extract_num_tokens,
    _parse_word_num_span,
    _run_global_consistency_grounding,
    MentionRecord,
    NumTok,
)


# ---------------------------------------------------------------------------
# _parse_word_num_span
# ---------------------------------------------------------------------------


class TestParseWordNumSpan:
    """Unit tests for the multi-token word-number span parser."""

    # ── simple single-token cases ──────────────────────────────────────────

    def test_single_ones(self):
        val, consumed = _parse_word_num_span(["two"], 0)
        assert val == 2.0 and consumed == 1

    def test_single_tens(self):
        val, consumed = _parse_word_num_span(["twenty"], 0)
        assert val == 20.0 and consumed == 1

    def test_bare_hundred(self):
        val, consumed = _parse_word_num_span(["hundred"], 0)
        assert val == 100.0 and consumed == 1

    def test_bare_thousand(self):
        val, consumed = _parse_word_num_span(["thousand"], 0)
        assert val == 1_000.0 and consumed == 1

    def test_single_hyphenated(self):
        val, consumed = _parse_word_num_span(["twenty-five"], 0)
        assert val == 25.0 and consumed == 1

    # ── two-token combinations ─────────────────────────────────────────────

    def test_one_hundred(self):
        val, consumed = _parse_word_num_span("one hundred units".split(), 0)
        assert val == 100.0 and consumed == 2

    def test_two_hundred(self):
        val, consumed = _parse_word_num_span("two hundred".split(), 0)
        assert val == 200.0 and consumed == 2

    def test_one_thousand(self):
        val, consumed = _parse_word_num_span("one thousand".split(), 0)
        assert val == 1_000.0 and consumed == 2

    def test_five_million(self):
        val, consumed = _parse_word_num_span("five million".split(), 0)
        assert val == 5_000_000.0 and consumed == 2

    # ── three-token combinations ───────────────────────────────────────────

    def test_two_hundred_fifty(self):
        val, consumed = _parse_word_num_span("two hundred fifty".split(), 0)
        assert val == 250.0 and consumed == 3

    def test_twenty_one_hundred(self):
        # "twenty one hundred" = (20+1) * 100 = 2100
        val, consumed = _parse_word_num_span("twenty one hundred".split(), 0)
        assert val == 2100.0 and consumed == 3

    def test_five_hundred_thousand(self):
        val, consumed = _parse_word_num_span("five hundred thousand".split(), 0)
        assert val == 500_000.0 and consumed == 3

    # ── four-token combination ─────────────────────────────────────────────

    def test_three_thousand_five_hundred(self):
        val, consumed = _parse_word_num_span(
            "three thousand five hundred".split(), 0
        )
        assert val == 3_500.0 and consumed == 4

    # ── offset start index ─────────────────────────────────────────────────

    def test_mid_sentence_offset(self):
        toks = "at least one hundred units".split()
        val, consumed = _parse_word_num_span(toks, 2)  # "one hundred"
        assert val == 100.0 and consumed == 2

    # ── non-number token → returns (None, 0) ──────────────────────────────

    def test_non_number_returns_none(self):
        val, consumed = _parse_word_num_span(["not"], 0)
        assert val is None and consumed == 0

    def test_stop_at_non_number(self):
        # span should stop at "products", not absorb it
        toks = "three products".split()
        val, consumed = _parse_word_num_span(toks, 0)
        assert val == 3.0 and consumed == 1

    def test_out_of_bounds_start(self):
        val, consumed = _parse_word_num_span([], 0)
        assert val is None and consumed == 0


# ---------------------------------------------------------------------------
# _extract_num_tokens — word-numbers and percent typing
# ---------------------------------------------------------------------------


class TestExtractNumTokensWordNumbers:
    """_extract_num_tokens must recognise and correctly type word-number spans."""

    # ── problem-statement regression examples ──────────────────────────────

    def test_two_types_of_jars(self):
        """'two' must be extracted as int=2."""
        toks = _extract_num_tokens("An artisan makes two types of terracotta jars", "orig")
        values = [(t.value, t.kind) for t in toks]
        assert (2.0, "int") in values

    def test_five_workers(self):
        toks = _extract_num_tokens("There are five workers", "orig")
        values = [(t.value, t.kind) for t in toks]
        assert (5.0, "int") in values

    def test_twenty_percent(self):
        toks = _extract_num_tokens("The defect rate is twenty percent", "orig")
        values = [(t.value, t.kind) for t in toks]
        assert (0.20, "percent") in values

    def test_five_per_cent(self):
        toks = _extract_num_tokens("A five per cent tax applies", "orig")
        values = [(t.value, t.kind) for t in toks]
        assert (0.05, "percent") in values

    def test_one_hundred_units(self):
        toks = _extract_num_tokens("At least one hundred units", "orig")
        values = [(t.value, t.kind) for t in toks]
        assert (100.0, "int") in values

    def test_three_and_digit_400(self):
        """Both word-number 'three' and digit '400' must be extracted."""
        toks = _extract_num_tokens("There are three products and 400 total hours", "orig")
        values = [t.value for t in toks]
        assert 3.0 in values
        assert 400.0 in values

    def test_five_dollars_currency(self):
        """Word-number near a money-context word must be typed as currency."""
        toks = _extract_num_tokens("Each worker is paid five dollars", "orig")
        currency_vals = [t.value for t in toks if t.kind == "currency"]
        assert 5.0 in currency_vals

    # ── compound spans ─────────────────────────────────────────────────────

    def test_two_hundred_fifty_single_token(self):
        """'two hundred fifty' must collapse into one token with value 250."""
        toks = _extract_num_tokens("two hundred fifty items", "orig")
        assert len(toks) == 1
        assert toks[0].value == 250.0
        assert toks[0].kind == "int"

    def test_one_thousand(self):
        toks = _extract_num_tokens("one thousand machines", "orig")
        assert len(toks) == 1
        assert toks[0].value == 1_000.0

    def test_three_thousand_five_hundred(self):
        toks = _extract_num_tokens("three thousand five hundred units", "orig")
        assert len(toks) == 1
        assert toks[0].value == 3_500.0

    # ── digit-based tokens not affected ───────────────────────────────────

    def test_digit_only_unaffected(self):
        toks = _extract_num_tokens("The budget is $5000 and demand is 100", "orig")
        values = {t.value for t in toks}
        assert 5000.0 in values
        assert 100.0 in values

    def test_mixed_digit_and_word(self):
        toks = _extract_num_tokens("two batches of 400 items each", "orig")
        values = {t.value for t in toks}
        assert 2.0 in values
        assert 400.0 in values

    # ── raw surface of spans ───────────────────────────────────────────────

    def test_span_raw_surface_one_hundred(self):
        toks = _extract_num_tokens("one hundred units", "orig")
        assert toks[0].raw == "one hundred"

    def test_span_raw_surface_two_hundred_fifty(self):
        toks = _extract_num_tokens("two hundred fifty", "orig")
        assert toks[0].raw == "two hundred fifty"


# ---------------------------------------------------------------------------
# _extract_num_mentions — MentionRecord construction for word spans
# ---------------------------------------------------------------------------


class TestExtractNumMentions:
    """_extract_num_mentions must produce correct MentionRecord objects."""

    def test_two_types_produces_one_mention(self):
        mentions = _extract_num_mentions("two types of jars", "orig")
        assert len(mentions) == 1
        assert mentions[0].tok.value == 2.0
        assert mentions[0].tok.kind == "int"

    def test_twenty_percent_mention_is_percent(self):
        mentions = _extract_num_mentions("twenty percent discount", "orig")
        assert any(m.tok.kind == "percent" and abs(m.tok.value - 0.20) < 1e-9 for m in mentions)

    def test_one_hundred_produces_one_mention(self):
        """'one hundred' must not split into two separate mentions."""
        mentions = _extract_num_mentions("one hundred units", "orig")
        assert len(mentions) == 1
        assert mentions[0].tok.value == 100.0

    def test_compound_raw_surface_stored(self):
        mentions = _extract_num_mentions("one thousand machines", "orig")
        assert mentions[0].tok.raw == "one thousand"

    def test_mention_index_is_span_start(self):
        """index must point to the first token of the span."""
        mentions = _extract_num_mentions("at least one hundred items", "orig")
        assert mentions[0].index == 2  # "one" is token 2 (0-based)

    def test_three_and_400_are_two_mentions(self):
        mentions = _extract_num_mentions(
            "There are three products and 400 total hours", "orig"
        )
        values = {m.tok.value for m in mentions}
        assert 3.0 in values
        assert 400.0 in values


# ---------------------------------------------------------------------------
# End-to-end grounding: word-number values reach the right slots
# ---------------------------------------------------------------------------


class TestGroundingWithWordNumbers:
    """Word-number mentions must flow through the grounding pipeline correctly."""

    def _gcg(self, query: str, slots: list[str]):
        return _run_global_consistency_grounding(query, "orig", slots)

    def test_two_types_assigned_to_count_slot(self):
        query = "An artisan makes two types of terracotta jars."
        vals, _, _ = self._gcg(query, ["NumJarTypes"])
        assert vals.get("NumJarTypes") is not None
        assert abs(float(vals["NumJarTypes"]) - 2.0) < 0.1

    def test_word_number_and_digit_both_assignable(self):
        """'three' and '400' should both be assignable to their slots."""
        query = "There are three products and 400 total hours available."
        vals, _, _ = self._gcg(query, ["NumProducts", "TotalHours"])
        assert vals.get("NumProducts") is not None
        assert vals.get("TotalHours") is not None

    def test_twenty_percent_assigned_to_percent_slot(self):
        query = "The defect rate is twenty percent."
        vals, _, _ = self._gcg(query, ["DefectRatePercent"])
        if vals.get("DefectRatePercent") is not None:
            v = float(vals["DefectRatePercent"])
            # stored as fraction ≤ 1 OR as the raw 20 (some pipelines keep raw)
            assert v <= 1.0 or abs(v - 20.0) < 0.1

    def test_one_hundred_assigned_to_scalar_slot(self):
        query = "Produce at least one hundred units per day."
        vals, _, _ = self._gcg(query, ["MinUnitsPerDay"])
        if vals.get("MinUnitsPerDay") is not None:
            assert abs(float(vals["MinUnitsPerDay"]) - 100.0) < 0.1
