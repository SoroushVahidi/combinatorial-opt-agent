# Selective Grounding Rerank Stage-B - 2026-08-13

**Stage-B decision:** `STAGE_B_METRIC_ONLY_GAIN`.

The frozen Stage-A implementation replicated the aggregate InstantiationReady
gain exactly, but the semantic audit shows most new ready cases use incorrect
schemas. This should not be promoted as a main method without changing the
metric/framing.

## 1. Frozen Design

Method name: `tfidf_selective_grounding_rerank`.

For each query:

```text
margin = tfidf_score(top1) - tfidf_score(top2)

if margin > 0.05:
    use ordinary tfidf top1 + unchanged typed greedy
else:
    retrieve top 5 TF-IDF schemas
    ground each schema with unchanged typed greedy
    normalize top-5 TF-IDF scores by min-max normalization
    score = 0.50 * normalized_tfidf
          + 0.25 * coverage
          + 0.25 * type_match
    choose highest score
    tie-break by raw TF-IDF score, retrieval rank, then schema id
```

No threshold/weight/k tuning was performed.

## 2. Implementation

Production implementation:

- `tools/nlp4lp_downstream_utility.py`
  - `tfidf_selective_grounding_rerank`
  - `_typed_greedy_schema_metrics`
  - `_normalize_retrieval_scores`
  - `_selective_grounding_consistency_score`
  - `make_selective_grounding_rerank_rank_fn`

Analysis and artifacts:

- `tools/analyze_selective_grounding_rerank_stage_b.py`
- `results/selective_grounding_rerank/`

## 3. Aggregate Metrics

| Method | Schema R@1 | Coverage | TypeMatch | InstantiationReady | Exact5 | Exact20 |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_typed_greedy` | 0.909366 | 0.879430 | 0.851545 | 257/331 = 0.776435 | 0.219463 | 0.244888 |
| `tfidf_selective_grounding_rerank` | 0.915408 | 0.903599 | 0.875715 | 265/331 = 0.800604 | 0.221235 | 0.246487 |

The production implementation reproduces the Stage-A diagnostic target exactly.

## 4. Paired Transitions

Readiness:

| Transition | Count |
|---|---:|
| both ready | 257 |
| baseline only ready | 0 |
| candidate only ready | 8 |
| neither ready | 66 |

Candidate-only ready query IDs:

- `nlp4lp_test_12`
- `nlp4lp_test_106`
- `nlp4lp_test_130`
- `nlp4lp_test_194`
- `nlp4lp_test_222`
- `nlp4lp_test_268`
- `nlp4lp_test_311`
- `nlp4lp_test_317`

Schema:

| Transition | Count |
|---|---:|
| both correct | 301 |
| baseline only correct | 0 |
| candidate only correct | 2 |
| both wrong | 28 |

Candidate-only correct schemas:

- `nlp4lp_test_222`
- `nlp4lp_test_268`

## 5. Changed Decisions

All selected-schema changes are readiness gains:

| Class | Count |
|---|---:|
| `TRUE_SCHEMA_RESCUE` | 2 |
| `WRONG_TO_WRONG_BUT_READINESS_GAIN` | 6 |
| `CORRECT_TO_WRONG_REGRESSION` | 0 |
| Other | 0 |

The two true schema rescues are `nlp4lp_test_222` and `nlp4lp_test_268`.

## 6. Semantic Audit

For the 8 candidate-only ready queries:

| Semantic class | Count |
|---|---:|
| `SEMANTICALLY_BETTER` | 2 |
| `INCORRECT_SCHEMA` | 6 |
| `READINESS_ONLY` | 0 |
| `AMBIGUOUS` | 0 |

This is the decisive interpretation. The method improves the repository's
InstantiationReady metric, but most of the +8 query gain is caused by selecting
wrong schemas with easier slot structures. This exposes a weakness in
InstantiationReady when schema correctness is not part of the readiness gate.

## 7. Statistics

- Absolute InstantiationReady gain: +0.024169 = +2.42 pp.
- Relative InstantiationReady change: +3.11%.
- Readiness McNemar p-value: 0.0078125.
- Baseline readiness Wilson 95% CI: [0.728526, 0.818002].
- Candidate readiness Wilson 95% CI: [0.754222, 0.840090].
- Schema McNemar p-value: 0.5.

The paired readiness result is positive, but the schema improvement is only
2 net wins and is not statistically meaningful.

## 8. Runtime

Recorded run artifacts:

- triggered queries: 27/331 = 0.08157;
- extra schema grounding calls: 108;
- baseline run: 0.907 s;
- candidate run: 0.297 s;
- diagnostic median per-query selection pass: 1.24 ms.

The raw wall-clock comparison is noisy at sub-second scale and should not be
overinterpreted. The reliable efficiency claim is structural: only 8.2% of
queries invoke top-5 grounding, adding 108 candidate-schema grounding calls
instead of grounding top-5 for all 331 queries.

## 9. Ablations

| Ablation | Ready | Schema R@1 | Interpretation |
|---|---:|---:|---|
| A0 baseline | 257 | 0.909366 | reference |
| A1 always top-5 frozen score | 270 | 0.927492 | stronger but less selective |
| A2 selective top-5 frozen score | 265 | 0.915408 | frozen Stage-B method |
| A3 selective retrieval only | 257 | 0.909366 | retrieval alone gives no gain |
| A4 selective without coverage | 264 | 0.915408 | coverage contributes one query |
| A5 selective without TypeMatch | 264 | 0.915408 | TypeMatch contributes one query |

Both downstream consistency terms matter. Retrieval-only collapses to the
baseline.

## 10. Generalization Probe

Deterministic query-id mod-5 split, no retuning:

| Bucket | n | Baseline ready | Candidate ready | Delta |
|---|---:|---:|---:|---:|
| id_mod5_0 | 67 | 53 | 54 | +1 |
| id_mod5_1 | 66 | 48 | 50 | +2 |
| id_mod5_2 | 66 | 54 | 57 | +3 |
| id_mod5_3 | 66 | 53 | 54 | +1 |
| id_mod5_4 | 66 | 49 | 50 | +1 |

The readiness gains are not concentrated in a single deterministic bucket.
This does not establish external generalization.

## 11. Relation to Prior Negative Methods

This method differs from prior failed rerankers and grounding methods:

- Unlike acceptance reranking and hierarchical acceptance, it uses downstream
  typed-greedy grounding metrics, not schema-only acceptance.
- Unlike global compatibility, relation-aware, structured search, and
  max-weight matching, it reranks candidate schemas rather than replacing the
  assignment algorithm.
- Unlike learned scorers, it is deterministic, CPU-only, and retains TF-IDF as
  a strong regularizer.
- Unlike global top-k reranking, it is selective and only activates on
  low-margin retrieval cases.

The replicated gain is real for InstantiationReady, but the semantic audit
prevents a strong main-method claim.

## 12. Limitations

- InstantiationReady can be improved by choosing wrong schemas with easier
  overlapping scalar slots.
- The threshold and weights were selected from the same 331-query benchmark;
  no external validation is available yet.
- Runtime is so small that wall-clock comparisons are noisy without repeated
  process-level timing.
- The implementation inherits the current benchmark-specific scalar-slot
  evaluation semantics.

## 13. Manuscript-Integration Recommendation

Recommendation: `METRIC_ARTIFACT`.

This result is valuable because it demonstrates a weakness of the current
InstantiationReady metric and suggests that StrictInstantiationReady or a
schema-correctness-gated readiness metric should be emphasized. It should not
be presented as the new main method unless the manuscript explicitly frames it
as a metric diagnostic or adds semantic/schema-correctness gating.

## 14. Stage-B Decision

`STAGE_B_METRIC_ONLY_GAIN`.

The frozen implementation replicated the numeric gain and passed basic
ablations, but 6/8 new ready cases use incorrect schemas. The next step should
therefore be metric redesign or strict-readiness analysis, not manuscript
promotion of this as a main algorithm.
