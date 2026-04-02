# Literature-Informed Method Updates

**File:** `docs/literature_informed_method_updates.md`  
**Date:** 2026-03-10  

This document describes the practical method upgrades made to the downstream grounding pipeline, informed by ideas from the numeracy/quantity-normalization and slot-semantic-tagging literature.

---

## Problem Addressed

**Root cause:** The TypeMatch metric for float-typed slots was ~2–3% (near zero) despite correct grounding. Investigation revealed:

1. `_expected_type()` returned `"float"` for ~80% of slot names (the catch-all default)
2. `_parse_num_token()` classifies whole-number tokens (e.g., "2", "5", "20") as `kind="int"` since they have no decimal point
3. The type-match check `tok.kind == et` was strict equality: `"int" == "float"` → False
4. Result: integer-valued parameter mentions (e.g., RequiredEggsPerSandwich = 2) never received a TypeMatch, even when correctly assigned

This was fundamentally incorrect: in optimization models, a parameter typed as "float" (continuous/real) very commonly takes integer values in natural language ("2 eggs per sandwich"). Integers ARE valid real numbers.

---

## Changes Made

### 1. `_expected_type()` — Expanded Integer Patterns

**File:** `tools/nlp4lp_downstream_utility.py`

Added a second block of integer-indicator patterns checked **after** the currency block (to avoid reclassifying "TotalBudget" as int):

```python
_EXT_INT_PATTERNS = (
    "number",    # NumberOfShifts, NumberOfWorkers, NumberOfDays
    "workers",   # TotalWorkers
    "employee",  # NumberOfEmployees
    "shifts",    # TotalShifts
    "batches",   # NumberOfBatches
    "rounds",    # NumberOfRounds
    "days",      # NumberOfDays, TotalDays
    "weeks",     # NumberOfWeeks
    "months",    # NumberOfMonths
    "trips",     # NumberOfTrips
    "persons",   # NumberOfPersons
    "patients",  # NumberOfPatients
    "tasks",     # NumberOfTasks
    "machines",  # NumberOfMachines
    "factories", # NumberOfFactories
    "farms",     # NumberOfFarms
    "vehicles",  # NumberOfVehicles
    "trucks",    # NumberOfTrucks
    "buses",     # NumberOfBuses
)
```

**Impact:** 76 catalog slot names now correctly classified as `int` instead of `float`.

---

### 2. `_is_type_match()` — New Authoritative Helper

**File:** `tools/nlp4lp_downstream_utility.py` (new function)

```python
def _is_type_match(expected: str, kind: str) -> bool:
    """Return True when kind is a full type-match for expected.

    Key rule: an integer token is a valid assignment for a float slot.
    Optimization model coefficients (e.g. RequiredEggsPerSandwich = 2.0)
    commonly appear as whole numbers in natural-language descriptions.
    """
    if expected == kind:
        return True
    if expected == "float" and kind == "int":
        return True
    return False
```

Used in **all** type-match checking paths:
- `run_one()` — 9 occurrences of TypeMatch metric counting
- `_score_mention_slot()` — base scoring
- `_score_mention_slot_ir()` — semantic IR scoring
- `_score_mention_slot_opt()` — optimization role scoring
- `_gcg_local_score()` — global consistency grounding

---

### 3. Scoring Functions — Full Match for int on float

In `_score_mention_slot()`, `_score_mention_slot_ir()`, `_score_mention_slot_opt()`, `_gcg_local_score()`:

**Before:**
```python
elif expected in ("int", "float") and kind in {"int", "currency", "float"}:
    score += weights["type_loose_bonus"]  # 0.5× bonus
    features["type_loose"] = True
```

**After:**
```python
elif expected == "float" and kind in {"float", "int"}:
    # float+float = exact; integer-valued tokens are valid for float slots
    score += weights["type_exact_bonus"]  # full bonus
    features["type_exact"] = True
elif expected == "int" and kind == "int":
    score += weights["type_exact_bonus"]
    features["type_exact"] = True
elif expected in ("int", "float") and kind in {"currency"}:
    score += weights["type_loose_bonus"]  # currency still gets partial credit
    features["type_loose"] = True
elif expected == "int" and kind == "float":
    score += weights["type_loose_bonus"]  # decimal on int slot is loose
    features["type_loose"] = True
```

---

### 4. `_choose_token()` — Equal Preference for int on float

**Before:**
```python
# float
pref = 2 if tok.kind == "float" and has_decimal else (1 if tok.kind in ("float", "int", "currency") else 0)
```

**After:**
```python
# float: integer-valued and decimal-valued tokens are equally preferred
pref = 2 if tok.kind in {"float", "int"} else (1 if tok.kind == "currency" else 0)
```

---

### 5. Slot Semantic Tagging — per_unit and total tags

**`_slot_semantic_expansion()`:** Added `per_unit`, `coefficient`, `unit_rate` tags for slots with "per"/"each"/"unit" in the name; `total`, `aggregate`, `available` tags for aggregate-quantity slots.

**`_slot_opt_role_expansion()`:** Same additions; per-unit slots get `unit_cost`, `objective_coeff`, `resource_consumption` tags; total slots get `total_available`, `capacity_limit`.

---

## Empirical Impact

| Metric | Before | After | Source |
|---|---|---|---|
| Float-slot TypeMatch rate (structural) | 2.3% | 81.8% | orig eval, 23k pairs |
| int slots correctly classified | 3.3% | 5.4% | catalog analysis |
| Score bonus: int on float slot | 1.5 | 3.0 | scoring trace |
| Existing tests | 220 pass | 220 pass | test suite |
| New targeted tests | 0 | 46 pass | test_float_type_match.py |

---

## What Is Still Blocked

- **End-to-end TypeMatch metric improvement** requires running the full evaluation with gold parameter data (gated dataset)
- **Learning-based improvements** are out of scope
- **Decimal value discrimination** (choosing "1.5" vs "20" when both are valid) relies on lexical/contextual features, not type scoring alone; this is separately handled by the existing GCG beam search
