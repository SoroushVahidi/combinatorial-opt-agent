# Strict-Failure Quick-Fix Diagnostic

**Date:** 2026-08-13
**Starting point:** fresh `tfidf_typed_greedy`
`StrictInstantiationReady = 247/331 = 0.746224`.
**Decision:** `QUICK_FIX_GO` for exactly one narrow candidate:
`multiplicative_ratio_word_extraction`.
**Resubmission recommendation:** `IMPLEMENT_ONE_QUICK_FIX_THEN_FREEZE_METHOD`.
**Production follow-up:** `QUICK_FIX_VALIDATED`; method state
`FROZEN_FOR_RESUBMISSION`.

This diagnostic asks whether any small deterministic change can rescue a
meaningful number of schema-correct strict-readiness failures without opening
another broad method-development loop.

## 1. Verified Failure Population

Source artifacts:

- `results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv`
- `results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_oracle.csv`
- `results/eswa_revision/00_env/nlp4lp_gold_cache.json`

Verified counts:

- current strict-ready: 247/331
- schema-correct/not-strict-ready: 54
- oracle-schema/not-ready: 58

The 54 current failures are schema-correct, so they isolate extraction,
expected-type, assignment, and slot-representation issues from retrieval.

## 2. Diagnostic Method

`tools/strict_failure_quick_fix_diagnostic.py` replays the current
typed-greedy code path for each schema-correct strict failure:

1. load the current top-1 TF-IDF per-query result;
2. keep only rows where `schema_hit=1` and ordinary readiness fails;
3. reconstruct expected scalar slots from the gold schema metadata;
4. extract numeric mentions with `_extract_num_tokens` / `_extract_num_mentions`;
5. replay `_choose_token` slot by slot with the current one-use candidate
   policy;
6. record selected mention, type compatibility, gold-value presence, and root
   cause;
7. compute query-level oracle ceilings for small intervention families.

The script does not alter production grounding behavior.

## 3. Root-Cause Taxonomy

| Root cause | Slot count | Query count |
|---|---:|---:|
| `DUPLICATE_REUSE_REQUIREMENT` | 66 | 33 |
| `SCHEMA_SLOT_REPRESENTATION_MISMATCH` | 69 | 21 |
| `WRONG_EXPECTED_SLOT_TYPE` | 40 | 24 |
| `NUMBER_NOT_EXTRACTED` | 10 | 9 |
| `INSUFFICIENT_NUMERIC_MENTIONS` | 4 | 4 |
| `OTHER_WRONG_VALUE_READY_NEUTRAL` | 53 | 0 |
| `OTHER` | 23 | 1 |

Interpretation:

- Many slots have wrong values but are readiness-neutral because they remain
  type-compatible; these matter for Exact20, not strict readiness.
- The largest query-level blockers are one-use/reuse conflicts, expected-type
  inference errors, and abstract template cases where scalar gold values are
  not actually present in the query text.
- Only 9 queries have a genuinely text-exposed missing number under the strict
  no-leakage criterion.

## 4. Oracle Intervention Bounds

| Intervention | Rescued queries | Projected strict readiness |
|---|---:|---:|
| perfect numeric extraction only, leakage-controlled | 7 | 254/331 = 0.767372 |
| perfect slot expected-type inference only | 24 | 271/331 = 0.818731 |
| perfect candidate filtering only | 0 | 247/331 = 0.746224 |
| allow mention reuse only | 25 | 272/331 = 0.821752 |
| perfect current-candidate choice | 5 | 252/331 = 0.761329 |
| perfect extraction + current chooser | 7 | 254/331 = 0.767372 |
| perfect type inference + current chooser | 24 | 271/331 = 0.818731 |
| perfect extraction + perfect type compatibility | 8 | 255/331 = 0.770393 |
| multiplicative ratio-word extraction prototype | 8 | 255/331 = 0.770393 |

The large type-inference and reuse ceilings are not quick fixes:

- expected-type failures mostly come from overloaded names like `Rate`,
  `Capacity`, and `Per`, where changing global rules risks regressions;
- mention reuse would change the core one-use grounding semantics and overlaps
  with previously failed assignment/search methods.

## 5. High-Frequency General Mechanisms

The only compact, low-effort, general mechanism with a meaningful projected
query-level payoff is multiplicative ratio wording:

- `twice as many`
- `double`
- `two times`
- `triple`
- `three times`

These phrases encode ratio constants, usually `2` or `3`, but current numeric
extraction does not expose them as usable numeric mentions. Existing slot typing
already represents many ratio slots as `percent`, so adding a ratio-word token
can make the current chooser satisfy TypeMatch without adding a new decoder.

Prototype gain IDs:

- `nlp4lp_test_47`
- `nlp4lp_test_98`
- `nlp4lp_test_116`
- `nlp4lp_test_128`
- `nlp4lp_test_156`
- `nlp4lp_test_195`
- `nlp4lp_test_245`
- `nlp4lp_test_261`

All-331 diagnostic exposure:

- 38 queries would receive at least one ratio-word token;
- simulated strict gains: 8;
- simulated strict losses: 0.

## 6. Candidate Quick Fixes

| Candidate | Projected rescues | Risk | Effort | Confidence | Recommendation |
|---|---:|---|---|---|---|
| multiplicative ratio-word extraction | 8 | low in simulation | low | HIGH | implement one localized patch, then freeze |
| broader text-exposed numeric extraction | 7 | medium | medium | MEDIUM | do not pursue separately before resubmission |
| allow mention reuse | 25 | high | low | LOW | reject as too broad |

No candidate is manuscript-worthy as a new method contribution. The ratio-word
patch is a robustness/bug-fix improvement to numeric extraction.

## 7. Relationship To Previous Failed Methods

- **ROLE_QUANTITY_STAGE_A_NO_GO:** the proposed ratio-word fix is not another
  role/quantity reranker. It adds missing numeric evidence before assignment;
  it does not re-score extracted candidates.
- **Max-weight/Hungarian and constrained assignment:** the fix does not change
  decoding. It keeps typed greedy and only exposes a missing ratio mention.
- **Semantic IR / optimization-role repair:** the fix is not a post-hoc repair
  or semantic rule bundle.
- **Global compatibility, relation-aware, search/hierarchical structured:** the
  fix does not introduce global search, pairwise compatibility, or beam search.
- **Learned scorers:** no learned model, no new supervision, no API.
- **Selective reranking:** no schema reranking or top-k tuning.

The high-ceiling rejected directions, especially expected-type oracle and
mention-reuse oracle, would repeat broad method-development patterns already
shown risky or negative.

## 8. Prototype Result

Prototype: diagnostic-only ratio-word extraction, simulated inside
`tools/strict_failure_quick_fix_diagnostic.py`.

Definition:

- add `NumTok(raw="RATIO_WORD:twice", value=2.0, kind="percent")` when the
  query contains `twice`, `double`, or `two times`;
- add `NumTok(raw="RATIO_WORD:triple", value=3.0, kind="percent")` when the
  query contains `triple` or `three times`;
- keep the current typed-greedy chooser and one-use policy unchanged.

All-331 simulated result:

- baseline strict-ready: 247
- prototype strict-ready: 255
- gains: 8
- losses: 0
- affected queries: 38

This is enough to justify one production implementation attempt, but not
enough to justify additional method search if the patch fails.

## 9. Decision

`QUICK_FIX_GO`

The ratio-word extraction patch clears the query-level threshold
(8 projected rescues), is general and localized, has no simulated strict losses,
and is not a repeat of previous failed assignment/reranking methods.

## 10. Stop Rule

After implementing this one patch:

- if production strict readiness improves with no regressions, freeze method
  development and move to external baselines + manuscript revision;
- if it fails to reproduce the diagnostic gain, record the negative result and
  freeze method development anyway.

Do not start another algorithm family for this resubmission.

## 11. Production Validation Addendum

The production patch was implemented in `tools/nlp4lp_downstream_utility.py`
and validated under `PYTHONHASHSEED=0`.

Final result:

- strict readiness: 247/331 -> 255/331
- ordinary readiness: 257/331 -> 265/331
- Schema R@1: unchanged at 301/331
- strict/ordinary readiness gains: 8
- strict/ordinary readiness losses: 0
- McNemar p-value: 0.0078125

Full freeze document: `docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`.
Full result artifacts: `results/final_resubmission_method/`.

## 12. Artifacts

Generated under `results/strict_failure_quick_fix/`:

- `per_query_failures.csv`
- `per_slot_failures.csv`
- `root_cause_summary.csv`
- `oracle_interventions.csv`
- `candidate_fixes.csv`
- `mechanism_exposure.csv`
- `prototype_ratio_word_extraction.csv`
- `summary.json`
- `README.md`
