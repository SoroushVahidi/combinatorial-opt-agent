# Number Role Repair

Post-hoc repair and calibration layer that corrects mislabeled number role mentions.

## Public API

```python
from src.features.number_role_repair import (
    repair_number_roles, calibrate_required_flags, detect_suspicious_missing_roles
)

repaired   = repair_number_roles(question_text, mentions)
calibrated = calibrate_required_flags(question_text, repaired)
report     = detect_suspicious_missing_roles(question_text, reasoning_text, calibrated)
```

## `repair_number_roles`

Applies three rules:

1. **Year downgrade**: `role_required` with no strong cue and year-like value (1900–2099) → `role_irrelevant`.
2. **Duplicate value deduplication**: when the same value appears multiple times, only the instance with the strongest cues stays `role_required`; others → `role_optional`.
3. **Noun overlap filter**: if a `role_required` mention's quantity words have no overlap with question content words → `role_optional`.

## `calibrate_required_flags`

- If >60% of mentions are `role_required` with weak cues, downgrade weak ones to `role_optional`.
- Numbers inside parentheses → `role_optional`.
- Numbers after "e.g." or "i.e." → `role_irrelevant`.

## `detect_suspicious_missing_roles`

Returns a dict:

```python
{
  "suspicious_missing": bool,
  "confidence": "low" | "medium" | "high",
  "evidence": [str, ...],
  "required_count": int,
  "used_count": int,
  "missing_count": int,
}
```

Confidence is `high` if >50% of required numbers are absent from reasoning, `medium` if >25%, else `low`.
