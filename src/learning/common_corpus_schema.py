"""Canonical schema for the common learning corpus (JSONL) and validation helpers."""

from __future__ import annotations

from typing import Any

# Required top-level keys for every record.
REQUIRED_KEYS = frozenset({"dataset", "split", "instance_id", "source_path", "problem_text"})

# Optional but standard keys.
OPTIONAL_KEYS = frozenset({
    "schema_name", "schema_description", "slots", "numeric_mentions",
    "gold_slot_assignments", "role_labels", "entity_labels", "bound_labels", "metadata",
})

ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

# Slot object keys.
SLOT_KEYS = frozenset({"slot_id", "slot_name", "slot_text", "slot_role", "expected_type", "variable_entity"})

# Numeric mention keys.
MENTION_KEYS = frozenset({
    "mention_id", "surface", "normalized_value", "type_bucket", "sentence_id",
    "char_start", "char_end", "local_context", "unit", "operator_cues",
})

DATASET_NAMES = frozenset({"nlp4lp", "nl4opt", "tatqa", "finqa"})
SPLIT_NAMES = frozenset({"train", "dev", "test"})


def validate_record(rec: dict[str, Any]) -> list[str]:
    """Validate one corpus record. Returns list of error messages (empty if valid)."""
    errs: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not a dict"]
    missing = REQUIRED_KEYS - set(rec.keys())
    if missing:
        errs.append(f"missing required keys: {sorted(missing)}")
    if rec.get("dataset") not in DATASET_NAMES:
        errs.append(f"invalid dataset: {rec.get('dataset')}")
    if rec.get("split") not in SPLIT_NAMES:
        errs.append(f"invalid split: {rec.get('split')}")
    for key in rec:
        if key not in ALL_KEYS:
            errs.append(f"unknown key: {key}")
    slots = rec.get("slots")
    if slots is not None:
        if not isinstance(slots, list):
            errs.append("slots must be a list")
        else:
            for i, s in enumerate(slots):
                if not isinstance(s, dict):
                    errs.append(f"slots[{i}] is not a dict")
                else:
                    for k in s:
                        if k not in SLOT_KEYS:
                            errs.append(f"slots[{i}] unknown key: {k}")
    mentions = rec.get("numeric_mentions")
    if mentions is not None:
        if not isinstance(mentions, list):
            errs.append("numeric_mentions must be a list")
        else:
            for i, m in enumerate(mentions):
                if not isinstance(m, dict):
                    errs.append(f"numeric_mentions[{i}] is not a dict")
                else:
                    for k in m:
                        if k not in MENTION_KEYS:
                            errs.append(f"numeric_mentions[{i}] unknown key: {k}")
    return errs


def slot_to_dict(
    slot_id: str,
    slot_name: str,
    slot_text: str | None = None,
    slot_role: str | None = None,
    expected_type: str | None = None,
    variable_entity: str | None = None,
) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "slot_name": slot_name,
        "slot_text": slot_text,
        "slot_role": slot_role,
        "expected_type": expected_type,
        "variable_entity": variable_entity,
    }


def mention_to_dict(
    mention_id: str,
    surface: str,
    normalized_value: float | int | None,
    type_bucket: str,
    sentence_id: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    local_context: str | None = None,
    unit: str | None = None,
    operator_cues: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "mention_id": mention_id,
        "surface": surface,
        "normalized_value": normalized_value,
        "type_bucket": type_bucket,
        "sentence_id": sentence_id,
        "char_start": char_start,
        "char_end": char_end,
        "local_context": local_context,
        "unit": unit,
        "operator_cues": operator_cues or [],
    }
