from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.role_quantity_stage_a_diagnostic import (
    compatibility_score,
    extract_mention_diags,
    slot_metadata,
)


def _mention_by_value(query: str, value: float):
    mentions = extract_mention_diags(query, "orig")
    for mention in mentions:
        if mention.value == value:
            return mention
    raise AssertionError(f"value {value} not extracted from {query!r}")


def test_per_unit_and_total_cues_are_distinguished():
    per_unit = _mention_by_value("Each product requires 3 hours. Total hours available are 100.", 3.0)
    total = _mention_by_value("Each product requires 3 hours. Total hours available are 100.", 100.0)

    assert "per_unit" in per_unit.quantity_forms
    assert "constraint_coefficient" in per_unit.roles
    assert "total" in total.quantity_forms
    assert "rhs_capacity" in total.roles


def test_bound_polarity_features():
    lower = _mention_by_value("At least 20 units and at most 50 units may be produced.", 20.0)
    upper = _mention_by_value("At least 20 units and at most 50 units may be produced.", 50.0)

    assert "lower_bound" in lower.roles
    assert "upper_bound" in upper.roles


def test_percent_currency_and_rate_features():
    percent = _mention_by_value("At least 20% must be invested.", 0.2)
    currency = _mention_by_value("The budget is $5000.", 5000.0)
    rate = _mention_by_value("The machine processes 30 units per hour.", 30.0)

    assert "percent" in percent.quantity_forms
    assert "currency" in currency.quantity_forms
    assert "rate" in rate.quantity_forms


def test_slot_metadata_and_compatibility_prefer_matching_quantity_form():
    slot = slot_metadata("TotalHoursAvailable")
    total = _mention_by_value("Each product requires 3 hours. Total hours available are 100.", 100.0)
    per_unit = _mention_by_value("Each product requires 3 hours. Total hours available are 100.", 3.0)

    total_score, total_reasons = compatibility_score(total, slot)
    per_score, _ = compatibility_score(per_unit, slot)

    assert "total" in slot.expected_quantity_forms
    assert total_score > per_score
    assert "form:total" in total_reasons


def test_feature_extraction_is_deterministic():
    query = "Each product requires 3 hours and earns $5 profit."
    first = extract_mention_diags(query, "orig")
    second = extract_mention_diags(query, "orig")

    assert first == second
