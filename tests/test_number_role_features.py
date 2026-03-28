from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.features.number_role_features import (
    extract_number_mentions, annotate_relevance, detect_quantity_families, NumberMention
)


class TestAdditionCueDetection:
    def test_addition_cue_found(self):
        text = "You gained 5 more apples."
        mentions = extract_number_mentions(text)
        assert mentions
        m = mentions[0]
        assert any(c in m.nearby_operator_cues for c in ["gained", "more"])

class TestSubtractionCueDetection:
    def test_subtraction_cue_found(self):
        text = "She spent 20 dollars at the store."
        mentions = extract_number_mentions(text)
        assert mentions
        m = mentions[0]
        assert any(c in m.nearby_operator_cues for c in ["spent"])

class TestRateCueDetection:
    def test_rate_cue_found(self):
        text = "The machine produces 10 units per hour."
        mentions = extract_number_mentions(text)
        assert mentions
        # find mention with value 10
        m10 = next((m for m in mentions if m.value == 10), None)
        assert m10 is not None
        assert any(c in m10.nearby_operator_cues for c in ["per", "hourly", "each", "every"])

class TestLowerBoundDetection:
    def test_at_least_gives_lower_bound(self):
        text = "You must produce at least 5 units."
        mentions = extract_number_mentions(text)
        m5 = next((m for m in mentions if m.value == 5), None)
        assert m5 is not None
        assert m5.bound_role == "lower"

    def test_no_fewer_than_gives_lower_bound(self):
        text = "Order no fewer than 10 items."
        mentions = extract_number_mentions(text)
        m10 = next((m for m in mentions if m.value == 10), None)
        assert m10 is not None
        assert m10.bound_role == "lower"

class TestUpperBoundDetection:
    def test_at_most_gives_upper_bound(self):
        text = "You can use at most 8 machines."
        mentions = extract_number_mentions(text)
        m8 = next((m for m in mentions if m.value == 8), None)
        assert m8 is not None
        assert m8.bound_role == "upper"

    def test_no_more_than_gives_upper_bound(self):
        text = "Spend no more than 100 dollars."
        mentions = extract_number_mentions(text)
        m100 = next((m for m in mentions if m.value == 100), None)
        assert m100 is not None
        assert m100.bound_role == "upper"

class TestRangePartnerDetection:
    def test_between_and_range(self):
        text = "Between 3 and 7 workers are needed."
        mentions = extract_number_mentions(text)
        assert len([m for m in mentions if m.range_partner_detected]) >= 1

class TestRelevanceLabelRequired:
    def test_target_cue_gives_role_required(self):
        text = "Maximize profit with 50 units."
        mentions = extract_number_mentions(text)
        annotated = annotate_relevance(mentions, text)
        m50 = next((m for m in annotated if m.value == 50), None)
        assert m50 is not None
        assert m50.relevance_label == "role_required"

class TestRelevanceLabelOptional:
    def test_addition_only_gives_role_optional(self):
        text = "She bought 3 more books."
        mentions = extract_number_mentions(text)
        annotated = annotate_relevance(mentions, text)
        m3 = next((m for m in annotated if m.value == 3), None)
        assert m3 is not None
        assert m3.relevance_label in ("role_optional", "role_unknown")

class TestRelevanceLabelIrrelevant:
    def test_year_like_number_role_irrelevant(self):
        text = "The regulation was established in 2023."
        mentions = extract_number_mentions(text)
        annotated = annotate_relevance(mentions, text)
        m2023 = next((m for m in annotated if m.value == 2023), None)
        assert m2023 is not None
        assert m2023.relevance_label == "role_irrelevant"

class TestQuantityFamilyGrouping:
    def test_min_max_pair_same_family(self):
        text = "The factory needs at least 10 workers and no more than 20 workers."
        mentions = extract_number_mentions(text)
        with_families = detect_quantity_families(mentions, text)
        m10 = next((m for m in with_families if m.value == 10), None)
        m20 = next((m for m in with_families if m.value == 20), None)
        assert m10 is not None and m20 is not None
        assert m10.quantity_family_id is not None
        assert m10.quantity_family_id == m20.quantity_family_id
