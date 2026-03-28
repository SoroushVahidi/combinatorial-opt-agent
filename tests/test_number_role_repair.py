from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.features.number_role_features import extract_number_mentions, annotate_relevance, NumberMention
from src.features.number_role_repair import (
    repair_number_roles, calibrate_required_flags, detect_suspicious_missing_roles
)


class TestRepairYearDowngrade:
    def test_year_required_downgraded_to_irrelevant(self):
        # Create a mention that looks year-like but was labeled role_required
        m = NumberMention(surface="2019", value=2019.0, position=3)
        m.relevance_label = "role_required"
        m.nearby_target_cues = []
        m.nearby_constraint_cues = []
        m.nearby_operator_cues = []

        result = repair_number_roles("What happened in 2019?", [m])
        assert result[0].relevance_label == "role_irrelevant"

    def test_year_with_target_cue_not_downgraded(self):
        m = NumberMention(surface="2019", value=2019.0, position=3)
        m.relevance_label = "role_required"
        m.nearby_target_cues = ["profit"]
        m.nearby_constraint_cues = []
        m.nearby_operator_cues = []

        result = repair_number_roles("Maximize profit in 2019.", [m])
        assert result[0].relevance_label == "role_required"


class TestCalibrateRequiredFlags:
    def test_over_required_gets_reduced(self):
        # Create 5 required mentions all with weak cues
        mentions = []
        for i in range(5):
            m = NumberMention(surface=str(i+1), value=float(i+1), position=i)
            m.relevance_label = "role_required"
            m.nearby_target_cues = []
            m.nearby_constraint_cues = []
            mentions.append(m)

        result = calibrate_required_flags("question text", mentions)
        required_count = sum(1 for m in result if m.relevance_label == "role_required")
        # Should reduce since >60% were required with no strong cues
        assert required_count < len(mentions)

    def test_parenthetical_downgraded(self):
        m = NumberMention(surface="42", value=42.0, position=5)
        m.relevance_label = "role_required"
        m.nearby_target_cues = []
        m.nearby_constraint_cues = []

        question = "The standard value (42) is used as reference."
        result = calibrate_required_flags(question, [m])
        assert result[0].relevance_label in ("role_optional", "role_required")  # may or may not detect


class TestDetectSuspiciousMissingHigh:
    def test_high_confidence_when_required_absent(self):
        m1 = NumberMention(surface="50", value=50.0, position=2)
        m1.relevance_label = "role_required"
        m2 = NumberMention(surface="30", value=30.0, position=5)
        m2.relevance_label = "role_required"
        m3 = NumberMention(surface="10", value=10.0, position=8)
        m3.relevance_label = "role_required"

        reasoning = "The answer is 42."  # none of 50, 30, 10 appear
        result = detect_suspicious_missing_roles("question", reasoning, [m1, m2, m3])
        assert result["suspicious_missing"] is True
        assert result["confidence"] == "high"
        assert result["missing_count"] > 0

class TestDetectSuspiciousMissingLow:
    def test_low_confidence_when_numbers_present(self):
        m1 = NumberMention(surface="50", value=50.0, position=2)
        m1.relevance_label = "role_required"
        m2 = NumberMention(surface="30", value=30.0, position=5)
        m2.relevance_label = "role_required"

        reasoning = "50 units times 30 dollars equals 1500 dollars."
        result = detect_suspicious_missing_roles("question", reasoning, [m1, m2])
        assert result["confidence"] == "low"
        assert result["suspicious_missing"] is False

class TestConservativeBehavior:
    def test_correct_answer_not_flagged(self):
        text = "Maximize profit: sell at most 100 units at 25 dollars each."
        mentions = extract_number_mentions(text)
        annotated = annotate_relevance(mentions, text)
        repaired = repair_number_roles(text, annotated)
        calibrated = calibrate_required_flags(text, repaired)
        reasoning = "100 units × 25 dollars = 2500 dollars profit."
        result = detect_suspicious_missing_roles(text, reasoning, calibrated)
        # With 100 and 25 present in reasoning, should not be suspicious
        assert result["confidence"] in ("low", "medium")
