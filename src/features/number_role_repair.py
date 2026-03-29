"""Post-hoc repair and calibration of number role labels."""
from __future__ import annotations
import re
from src.features.number_role_features import NumberMention

def repair_number_roles(question_text: str, mentions: list[NumberMention]) -> list[NumberMention]:
    """
    Rules:
    1. If role_required but no target/constraint/rate cues AND value looks like year (1900-2099) → role_irrelevant
    2. If multiple mentions have same value but different sentences, keep nearest to target/constraint cue, downgrade others to role_optional
    3. If mention's nearby_quantity_words don't overlap with question nouns → downgrade role_required to role_optional
    Returns new list (no mutation).
    """
    import copy
    result = [copy.copy(m) for m in mentions]

    # Rule 1
    for m in result:
        if m.relevance_label == "role_required":
            has_strong_cue = bool(m.nearby_target_cues or m.nearby_constraint_cues or m.nearby_operator_cues)
            if not has_strong_cue and 1900 <= m.value <= 2099:
                m.relevance_label = "role_irrelevant"

    # Rule 2: group by value, keep best
    from collections import defaultdict
    by_value = defaultdict(list)
    for i, m in enumerate(result):
        by_value[m.value].append(i)

    for val, indices in by_value.items():
        if len(indices) > 1:
            def score(idx):
                m = result[idx]
                return len(m.nearby_target_cues) + len(m.nearby_constraint_cues)
            best = max(indices, key=score)
            for idx in indices:
                if idx != best and result[idx].relevance_label == "role_required":
                    result[idx].relevance_label = "role_optional"

    # Rule 3: question nouns heuristic
    stopwords = {"what", "which", "when", "where", "does", "will", "should", "many", "much", "from", "with", "that", "this", "have", "been", "each", "they", "them", "their"}
    question_words = set(w.lower().strip(".,?!") for w in question_text.split() if len(w) > 3 and w.lower() not in stopwords)

    for m in result:
        if m.relevance_label == "role_required" and m.nearby_quantity_words:
            qty_words = set(w.lower() for w in m.nearby_quantity_words)
            if question_words and qty_words and not qty_words.intersection(question_words):
                m.relevance_label = "role_optional"

    return result


def calibrate_required_flags(question_text: str, mentions: list[NumberMention]) -> list[NumberMention]:
    """
    Conservative calibration:
    - If >60% labeled role_required, downgrade weak-cue ones to role_optional
    - Numbers in parenthetical context → role_optional
    - Numbers after "e.g.", "i.e." → role_irrelevant
    """
    import copy
    result = [copy.copy(m) for m in mentions]

    if not result:
        return result

    required = [m for m in result if m.relevance_label == "role_required"]
    ratio = len(required) / len(result)

    if ratio > 0.60:
        for m in result:
            if m.relevance_label == "role_required":
                strong = bool(m.nearby_target_cues or m.nearby_constraint_cues)
                if not strong:
                    m.relevance_label = "role_optional"

    # Parenthetical context
    paren_pattern = re.compile(r'\(([^)]*)\)')
    paren_contents = []
    for match in paren_pattern.finditer(question_text):
        paren_contents.append((match.start(), match.end(), match.group(1)))

    for m in result:
        num_str = m.surface
        try:
            idx = question_text.index(num_str)
            for ps, pe, _ in paren_contents:
                if ps < idx < pe:
                    m.relevance_label = "role_optional"
                    break
        except ValueError:
            pass

    # e.g. / i.e. pattern
    eg_pattern = re.compile(r'(?:e\.g\.|i\.e\.|e\.g,|i\.e,)\s*(\S+)', re.IGNORECASE)
    for match in eg_pattern.finditer(question_text):
        eg_num = match.group(1).strip(".,;")
        for m in result:
            if m.surface == eg_num:
                m.relevance_label = "role_irrelevant"

    return result


def detect_suspicious_missing_roles(
    question_text: str,
    reasoning_text: str,
    mentions: list[NumberMention],
) -> dict:
    """
    Detect when a missing role is suspicious.
    Returns dict with suspicious_missing, confidence, evidence, required_count, used_count, missing_count.
    """
    required = [m for m in mentions if m.relevance_label == "role_required"]
    required_count = len(required)

    used_count = 0
    evidence = []

    for m in required:
        val_str = str(int(m.value)) if m.value == int(m.value) else str(m.value)
        if val_str in reasoning_text or m.surface in reasoning_text:
            used_count += 1
        else:
            evidence.append(f"Required number {m.surface} (value={m.value}) not found in reasoning")

    missing_count = required_count - used_count

    if required_count == 0:
        return {
            "suspicious_missing": False,
            "confidence": "low",
            "evidence": [],
            "required_count": 0,
            "used_count": 0,
            "missing_count": 0,
        }

    ratio = missing_count / required_count

    if ratio > 0.5:
        suspicious = True
        confidence = "high"
    elif ratio > 0.25:
        suspicious = True
        confidence = "medium"
    else:
        suspicious = False
        confidence = "low"

    return {
        "suspicious_missing": suspicious,
        "confidence": confidence,
        "evidence": evidence,
        "required_count": required_count,
        "used_count": used_count,
        "missing_count": missing_count,
    }
