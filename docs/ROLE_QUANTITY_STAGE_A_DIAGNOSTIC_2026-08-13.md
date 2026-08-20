# Role-Quantity Stage-A Diagnostic - 2026-08-13

**Status:** `STAGE_A_NO_GO` for implementing role-quantity factorized
grounding as the next main method improvement.

This is a diagnostic-only pass. It does not alter typed-greedy behavior, does
not edit manuscript files, does not change canonical result tables, and does
not use external APIs.

## 1. Hypothesis

The precommitted Stage-A question was:

> Does explicit role/quantity information contain enough new signal to change
> typed-greedy's wrong mention choice in at least 10 targeted schema-hit /
> not-ready failures?

The diagnostic answer is nuanced:

- yes, deterministic role/quantity features separate at least 10 wrong
  mention choices at the slot level;
- no, those separable slot changes do not rescue any additional
  InstantiationReady queries under the current metric, because
  InstantiationReady is gated by coverage and type compatibility rather than
  numeric exactness.

Therefore this line is not a good next main-method implementation target for
the manuscript success gate.

## 2. Target Set

Reference baseline:

- 331 total `orig` NLP4LP queries.
- Fresh `tfidf_typed_greedy`: 257/331 InstantiationReady = `0.776435`.
- Schema-hit/not-ready queries: 54.

Targeted wrong assignments were restricted to slots satisfying all of:

1. query is schema-hit and not ready;
2. the current typed-greedy selected mention is not the gold numeric value;
3. the gold numeric value was extracted and present among candidates at that
   slot decision point;
4. more than one type-compatible candidate existed.

This produced 49 targeted wrong assignments.

## 3. Diagnostic Methodology

Implementation:

- `tools/role_quantity_stage_a_diagnostic.py`

Artifacts:

- `results/role_quantity_stage_a/per_slot.csv`
- `results/role_quantity_stage_a/targeted_failures.csv`
- `results/role_quantity_stage_a/separability.csv`
- `results/role_quantity_stage_a/cascade_analysis.csv`
- `results/role_quantity_stage_a/summary.json`

The script replays the current typed-greedy path using:

- `tools/nlp4lp_downstream_utility.py::_extract_num_tokens`
- `_expected_type`
- `_choose_token`
- `_is_type_match`
- TF-IDF retrieval from `retrieval.baselines`

It records diagnostic features separately and never feeds them back into the
production decision.

## 4. Feature Definitions

Mention-side deterministic features:

- quantity forms: `total`, `per_unit`, `rate`, `percent`, `currency`, `count`,
  `capacity`, `demand`, `cost`, `profit`, `bound`, `generic_scalar`;
- roles: `objective_coefficient`, `constraint_coefficient`, `rhs_capacity`,
  `lower_bound`, `upper_bound`, `cardinality`, `rate`, `unknown`;
- local context: sentence index, clause index, left/right/nearby tokens,
  entity anchor from nearby non-stopword tokens;
- cue families: `each`, `per`, `total`, `available`, `capacity`, `at least`,
  `at most`, `minimum`, `maximum`, cost/profit/revenue/currency/percent words.

Slot-side deterministic metadata:

- inferred coarse type from `_expected_type`;
- expected quantity forms and roles from slot-name tokens;
- slot entity tokens from camel-case splitting.

Separability score:

- +2 per matching quantity form;
- +2 per matching role;
- +1 per entity-token overlap;
- +1 for current coarse type compatibility.

An assignment is marked:

- `separable` if the gold candidate scores above the selected wrong candidate;
- `ambiguous` if equal;
- `not_separable` if below.

This is intentionally a diagnostic separability test, not a tuned scorer.

## 5. Case Taxonomy

Targeted wrong assignments: 49.

| Case type | Targeted | Separable |
|---|---:|---:|
| same_type_ambiguity | 21 | 10 |
| total_perunit | 9 | 6 |
| bound_role | 10 | 9 |
| objective_constraint | 6 | 3 |
| other_role_quantity | 3 | 0 |

Separability totals:

| Outcome | Assignments |
|---|---:|
| separable | 28 |
| ambiguous | 10 |
| not_separable | 11 |

## 6. Per-Case Summary

Representative separable cases:

- `nlp4lp_test_39`, `SubstituteShiftHours`: selected `1000`, gold `3`.
  The gold candidate has entity/role support for substitute shift context.
- `nlp4lp_test_46`, `CashMachinePaperRolls`: selected `90`, gold `4`.
  Count and entity/measure cues separate the candidates.
- `nlp4lp_test_66`, stamping-machine cases: bound/rate/usage slots show
  separable role and bound-polarity signals.

Representative non-separable/ambiguous cases:

- `nlp4lp_test_39`, shift pay slots: both selected and gold candidates carry
  similar cost/currency/per-unit role cues; deterministic role labels alone do
  not separate reliably.
- Some same-type symmetric/product-family cases require stronger entity or
  parallel-clause reasoning than this simple diagnostic feature set provides.

Full rows are in `results/role_quantity_stage_a/targeted_failures.csv`.

## 7. Separability Counts

The Stage-A assignment-level threshold was met:

- targeted wrong assignments: 49;
- deterministic role/quantity separable: 28;
- ambiguous: 10;
- not separable: 11.

However, this is not sufficient for manuscript-method implementation because
the query-level upper bound is zero.

## 8. Query-Level Upper Bound

Conservative upper bound:

- current ready queries: 257/331;
- potentially rescued queries after all separable targeted swaps: 0;
- projected ready: 257/331;
- projected InstantiationReady: `0.776435`;
- absolute gain: `+0.0000` percentage points.

Reason: the separable swaps mostly correct numeric value choices, but
InstantiationReady does not score numeric exactness. For the targeted
schema-hit/not-ready cases, these swaps did not change enough type-compatible
fills to cross the coverage/type gates.

This is the decisive no-go result.

## 9. Cascade Trigger Analysis

Broad pre-verification ambiguity triggers are not selective:

| Trigger | Triggered queries | Trigger rate | Correctable-failure recall | False trigger rate on ready queries |
|---|---:|---:|---:|---:|
| multiple compatible mentions | 295 | 0.891 | 1.000 | 0.996 |
| same-type multiplicity | 294 | 0.888 | 1.000 | 0.996 |
| role/quantity conflict | 229 | 0.692 | 0.952 | 0.778 |
| low assignment margin | 122 | 0.369 | 0.238 | 0.436 |
| low retrieval margin | 27 | 0.082 | 0.000 | 0.043 |
| verification failure | 74 | 0.224 | 1.000 | 0.000 |
| verification failure OR low retrieval margin | 85 | 0.257 | 1.000 | 0.043 |

The practical trigger would have to rely on verification failure, not raw
same-type multiplicity, if this family were revisited. But because the
query-level upper bound is zero, no Stage-B implementation is recommended.

## 10. API Oracle Analysis

`NOT_USED`.

Reason: deterministic analysis already answered the gating question. It found
28 separable targeted assignments but zero projected query-level rescues. An
API semantic-role oracle could label mentions more cleanly, but it would not
change the fundamental metric mismatch unless the oracle also changed coverage
or type compatibility. That would exceed the intended role-label control.

## 11. Limitations

- Separability labels are deterministic diagnostics, not a trained or tuned
  scorer.
- Gold-value presence uses numeric tolerance consistent with the repository's
  relative-error diagnostics; exact-value analysis may differ slightly.
- The script uses role/quantity cues from local text windows and slot names, not
  dependency parsing or full semantic parsing.
- Query-level rescue is measured against current InstantiationReady, which does
  not include numeric-value exactness. A future metric that rewards exact values
  would make role/quantity signal more valuable.

## 12. GO/NO-GO Decision

**Decision: `STAGE_A_NO_GO`.**

Rationale:

- assignment-level separability passes the raw threshold: 28 >= 10;
- query-level upper bound is 0 rescued queries;
- projected InstantiationReady remains `0.776435`;
- therefore the +2 pp final manuscript gate is not realistic for this TOP-1
  path as currently framed.

## 13. Minimal Stage-B Design

No Stage-B implementation is recommended for TOP-1.

If role/quantity features are ever reused, they should be scoped to a different
objective, such as Exact20/exact-value improvement, not the current
InstantiationReady gate.

## 14. Next Candidate

Move to the TOP-2 candidate from
`docs/METHOD_NOVELTY_EFFICIENCY_AUDIT_2026-08-13.md`:

**Selective top-k schema + grounding reranking for low retrieval-margin cases.**

Reason: Stage A showed role/quantity corrections do not move the current
readiness metric. The remaining plausible path to +2 pp InstantiationReady is
recovering schema/candidate-selection failures that affect coverage/type gates.
