# Method Freeze For Resubmission

**Date:** 2026-08-13
**Decision:** `FROZEN_FOR_RESUBMISSION`
**Final method state:** TF-IDF retrieval + typed greedy grounding with
deterministic multiplicative ratio-word numeric extraction.

This document freezes method development for the current resubmission cycle.
Do not start another algorithm-improvement track before resubmission.

## 1. Final Production Method Configuration

The final native method remains descriptively:

`TF-IDF + Typed Greedy`

Production revision:

`TF-IDF + Typed Greedy + multiplicative-expression extraction`

Pipeline:

1. retrieve the top-1 schema with TF-IDF;
2. extract numeric mentions from digit, written-number, fraction, currency,
   percent, and now multiplicative ratio expressions;
3. infer expected scalar-slot type from slot names;
4. assign extracted mentions with the unchanged typed-greedy chooser;
5. compute Coverage, TypeMatch, ordinary InstantiationReady, and
   StrictInstantiationReady.

No schema reranking, matching, search, learned scorer, external API, or solver
fallback is part of the final production method.

## 2. Ratio-Word Patch Outcome

Production extraction now exposes:

| Expression family | Numeric value | Token kind | Raw token |
|---|---:|---|---|
| `twice`, `double`, `two times` | 2.0 | `percent` | `RATIO_WORD:twice` |
| `triple`, `three times` | 3.0 | `percent` | `RATIO_WORD:triple` |

The patch is implemented in `tools/nlp4lp_downstream_utility.py` via
`_extract_multiplicative_ratio_tokens` and is consumed by both
`_extract_num_tokens` and `_extract_num_mentions`. It does not change
`_choose_token`, `_expected_type`, retrieval, schema selection, or evaluation
thresholds.

## 3. Fresh Metrics

Benchmark: NLP4LP `orig`, 331 queries, `PYTHONHASHSEED=0`.

| Method | Schema R@1 | Coverage | TypeMatch | InstantiationReady | StrictInstantiationReady | Exact5 | Exact20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre-patch `tfidf_typed_greedy` | 0.909366 | 0.879430 | 0.851545 | 257/331 = 0.776435 | 247/331 = 0.746224 | 0.219463 | 0.244888 |
| final patched method | 0.909366 | 0.888566 | 0.866549 | 265/331 = 0.800604 | 255/331 = 0.770393 | 0.235536 | 0.261391 |

Primary transition:

- strict-ready gains: 8
- strict-ready losses: 0
- gain IDs: `nlp4lp_test_47`, `nlp4lp_test_98`, `nlp4lp_test_116`,
  `nlp4lp_test_128`, `nlp4lp_test_156`, `nlp4lp_test_195`,
  `nlp4lp_test_245`, `nlp4lp_test_261`
- exact McNemar p-value: 0.0078125

Runtime:

- patched 331-query run: 1.09 seconds
- mean/query: about 3.29 ms
- overhead is negligible relative to the existing pipeline.

Full artifacts: `results/final_resubmission_method/`.

## 4. Strict Metric Definition

Ordinary InstantiationReady remains:

`Coverage >= 0.8 AND TypeMatch >= 0.8`

StrictInstantiationReady is:

`SchemaCorrect AND Coverage >= 0.8 AND TypeMatch >= 0.8`

StrictInstantiationReady should be the primary native end-to-end readiness
metric for manuscript revision. Ordinary InstantiationReady should be retained
as a predicted-schema diagnostic for historical comparability.

## 5. Changed-Query Audit

Changed-query counts:

- `CORRECT_MULTIPLICATIVE_RESCUE`: 8
- `CORRECT_EXTRACTION_BUT_NO_QUERY_FLIP`: 24
- `FALSE_POSITIVE_EXTRACTION`: 0
- `REGRESSION`: 4

The regressions are exact-value-only changes with no schema, ordinary-readiness,
or strict-readiness loss. Aggregate Exact5 and Exact20 both improve.

Exact-value impact:

- Exact5 mean delta: +0.016073
- Exact20 mean delta: +0.016503
- Exact5 per-query gains/losses: 22 / 6
- Exact20 per-query gains/losses: 23 / 6

## 6. Negative-Result Methods Not To Pursue

Do not restart any of the following method directions for this resubmission:

- role/quantity factorized scorer;
- max-weight/Hungarian matching;
- mention reuse;
- expected-type redesign;
- generic learned pair scorer;
- feature-augmented learned scorer;
- beam/search assignment with unchanged signals;
- global compatibility variants;
- relation-aware variants;
- semantic IR repair;
- optimization-role repair;
- acceptance/hierarchical reranking;
- selective top-k schema reranking as a main method.

The selective reranker remains a useful diagnostic and metric-artifact result,
not a final main method.

## 7. Remaining Work

Only these workstreams remain before manuscript submission:

1. external baseline empirical completion;
2. strict-metric manuscript integration;
3. literature update;
4. manuscript revision;
5. journal-specific formatting and framing.

`FROZEN_FOR_RESUBMISSION`
