# Number Role Features

Extracts numeric mentions from problem text and annotates each with operator cues, bound roles, range detection, relevance labels, and quantity family IDs.

## Key Components

- **`NumberMention`** dataclass: holds `surface`, `value`, `position`, cue lists, `bound_role`, `relevance_label`, `quantity_family_id`.
- **Cue lists**: `ADDITION_CUES`, `SUBTRACTION_CUES`, `RATE_CUES`, `TARGET_CUES`, `CONSTRAINT_CUES`, `LOWER_BOUND_PHRASES`, `UPPER_BOUND_PHRASES`.

## Public API

```python
from src.features.number_role_features import (
    extract_number_mentions, annotate_relevance, detect_quantity_families
)

mentions = extract_number_mentions(text)        # detect + annotate cues
mentions = annotate_relevance(mentions, text)   # refine relevance labels
mentions = detect_quantity_families(mentions, text)  # group bound pairs
```

## Relevance Labels

| Label | Condition |
|---|---|
| `role_required` | Has target, constraint, or rate cue nearby |
| `role_irrelevant` | Year-like (1900–2099) with no strong cues, or ≥1B |
| `role_optional` | Has add/subtract cue but no target/constraint |
| `role_unknown` | Fallback |

## Bound Roles

- `lower`: phrases like "at least", "no fewer than", "≥"
- `upper`: phrases like "at most", "no more than", "≤"

## Quantity Families

Two mentions share a family if they are a lower/upper pair sharing nearby nouns, or share a unit word within 10 tokens.
