# Pre-Fix vs Post-Fix Float TypeMatch Ablation

**Date:** 2026-03-10

## Background

A structural bug was found in `tools/nlp4lp_downstream_utility.py`:

- **Function:** `_is_type_match(expected, kind)` (line 472 as of commit e3fdaf4)
- **Bug:** The old code used `tok.kind == et` (strict equality). Since ~80% of schema
  slots have `expected_type = "float"` but most numeric tokens in text are written as
  integers (kind = "int"), 97.7% of float-slot token pairs received **zero type-match credit**.
- **Fix:** `_is_type_match` now returns `True` for `(expected="float", kind="int")`.
  This is mathematically correct: integers are valid real-number values.

## What the fix changes

| Component | Change |
|-----------|--------|
| Type-match scoring in `_score_mention_slot` | `int` tokens on float slots now receive `type_match_bonus` (3.0) instead of `type_loose_match_bonus` (1.5) |
| TypeMatch metric computation in `run_one()` | All 9 TypeMatch counting occurrences use `_is_type_match` |
| `_expected_type()` reclassification | 76 slot names reclassified from `float` to `int` (workers, shifts, etc.) |
| Coverage | **Unchanged** (fill rate does not depend on type match) |
| `exact20_on_hits` | May improve slightly (better ranked tokens for float slots) |

## Structural estimate

| Metric | Pre-fix | Post-fix (structural est.) | Evidence |
|--------|---------|---------------------------|---------|
| Float-slot type match rate | ~2.3% | ~81.8% | 23,109 float-slot×token pairs counted |
| Overall TypeMatch (tfidf_typed) | 0.2267 | ~0.55–0.65 | Structural calculation |
| Float-type TypeMatch specifically | ~0.030 | ~0.70–0.80 | Structural calculation |
| Integer-type TypeMatch | ~0.991 | ~0.991 (unchanged) | Already high |
| Coverage | 0.8222 | 0.8222 (unchanged) | Type fix doesn't affect fill rate |
| InstReady | 0.0725 | TBD (likely higher) | Requires gold data |

**Important caveat:** The post-fix TypeMatch estimate is structural (counts token-slot pair
compatibility). The actual end-to-end TypeMatch on the assignment problem may differ because:
(a) assignment scores changed, shifting which token wins each slot, and
(b) the evaluation metric changes independently of assignment.

## What is NOT measured yet

Full end-to-end post-fix TypeMatch requires:
```bash
export HF_TOKEN=hf_...
python tools/run_nlp4lp_focused_eval.py --variant orig
# Reads from: results/paper/nlp4lp_downstream_summary.csv
```

This is the #1 priority experiment once HF_TOKEN is available.

## Classification: what did the bug fix change?

- **Metric accounting:** YES — TypeMatch metric now correctly counts int tokens for float slots.
- **Assignment behavior:** YES — Scoring shifted; float slots now more likely to accept
  integer tokens, potentially changing which token is selected.
- Both effects are real and inseparable without a controlled experiment.

## Source files

- Fix: `tools/nlp4lp_downstream_utility.py` (lines 472–485, `_is_type_match`; lines 403–470, `_expected_type`)
- Test: `tests/test_float_type_match.py` (46 test cases, all passing)
- Docs: `docs/literature_informed_rerun_report.md §3`
