from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (  # noqa: E402
    _extract_multiplicative_ratio_tokens,
    _extract_num_mentions,
    _extract_num_tokens,
)


def _ratio_tokens(text: str) -> list[tuple[str, float | None, str]]:
    return [
        (tok.raw, tok.value, tok.kind)
        for tok in _extract_num_tokens(text, "orig")
        if tok.raw.startswith("RATIO_WORD:")
    ]


def test_extracts_twice_double_and_two_times_as_multiplier_two():
    assert _ratio_tokens("At least twice the capacity is required.") == [
        ("RATIO_WORD:twice", 2.0, "percent")
    ]
    assert _ratio_tokens("DOUBLE the profit target.") == [
        ("RATIO_WORD:twice", 2.0, "percent")
    ]
    assert _ratio_tokens("Use two times the regular amount.") == [
        ("RATIO_WORD:twice", 2.0, "percent")
    ]


def test_extracts_triple_and_three_times_as_multiplier_three():
    assert _ratio_tokens("Triple the minimum production.") == [
        ("RATIO_WORD:triple", 3.0, "percent")
    ]
    assert _ratio_tokens("Set demand to three times last year's value.") == [
        ("RATIO_WORD:triple", 3.0, "percent")
    ]


def test_punctuation_casing_and_order_are_deterministic():
    toks = _ratio_tokens("TWICE, then triple; two times again and double.")
    assert toks == [
        ("RATIO_WORD:twice", 2.0, "percent"),
        ("RATIO_WORD:triple", 3.0, "percent"),
    ]


def test_does_not_extract_ordinary_cardinal_language_or_hyphenated_double_check():
    assert _ratio_tokens("There are two products and three machines.") == []
    assert _ratio_tokens("Double-check the two products before solving.") == []


def test_existing_word_numbers_are_not_deduplicated_as_equivalent_ratio_evidence():
    toks = _extract_num_tokens("Two times the number of items.", "orig")
    assert toks[0].raw.lower() == "two"
    assert (toks[0].value, toks[0].kind) == (2.0, "int")
    assert (toks[1].raw, toks[1].value, toks[1].kind) == ("RATIO_WORD:twice", 2.0, "percent")


def test_ratio_mentions_include_source_index_and_context():
    mentions = [
        m for m in _extract_num_mentions("Products need three times the capacity.", "orig")
        if m.tok.raw.startswith("RATIO_WORD:")
    ]
    assert len(mentions) == 1
    assert mentions[0].index == 2
    assert mentions[0].tok.raw == "RATIO_WORD:triple"
    assert mentions[0].tok.value == 3.0
    assert mentions[0].tok.kind == "percent"
    assert "capacity" in mentions[0].context_tokens


def test_helper_preserves_stage_a_raw_labels():
    assert [(i, tok.raw, tok.value, tok.kind) for i, tok in _extract_multiplicative_ratio_tokens("twice triple")] == [
        (0, "RATIO_WORD:twice", 2.0, "percent"),
        (6, "RATIO_WORD:triple", 3.0, "percent"),
    ]
