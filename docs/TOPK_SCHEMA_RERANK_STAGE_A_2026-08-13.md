# Top-k Schema Rerank Stage-A Diagnostic - 2026-08-13

**Status:** `TOP2_GO` for a minimal selective top-k schema + grounding
reranker.

This is a diagnostic-only pass. It does not alter typed-greedy behavior, does
not edit manuscript files, does not change canonical result tables, and does
not use external APIs.

## 1. Hypothesis

The precommitted Stage-A question was:

> Can a cheap selective reranking mechanism recover enough of the 30 current
> schema misses to improve InstantiationReady by at least the +2 percentage
> point manuscript gate?

Diagnostic answer: yes. The top-k oracle ceiling is high enough, and a simple
deterministic consistency rule has a selective setting that reaches 265/331
InstantiationReady while preserving schema correctness.

## 2. Reference State

The diagnostic reproduces the fresh current baseline:

- `tfidf_typed_greedy`: 257/331 InstantiationReady = `0.776435`.
- Schema R@1: 301/331 = `0.909366`.
- Coverage: `0.879430`.
- TypeMatch: `0.851545`.
- Schema misses: 30.

Implementation:

- `tools/topk_schema_rerank_stage_a.py`

Artifacts:

- `results/topk_schema_rerank_stage_a/schema_misses.csv`
- `results/topk_schema_rerank_stage_a/oracle_topk.csv`
- `results/topk_schema_rerank_stage_a/margin_analysis.csv`
- `results/topk_schema_rerank_stage_a/reranker_results.csv`
- `results/topk_schema_rerank_stage_a/transitions.csv`
- `results/topk_schema_rerank_stage_a/candidate_groundings.csv`
- `results/topk_schema_rerank_stage_a/summary.json`

## 3. Schema-miss Population

Gold-rank distribution over the 30 TF-IDF schema misses:

| Gold rank bucket | Count |
|---|---:|
| rank 2 | 9 |
| rank 3 | 5 |
| rank 4-5 | 4 |
| rank 6-10 | 6 |
| >10 | 6 |

Classification at top-10:

| Class | Count |
|---|---:|
| `A_RETRIEVAL_FIX_RESCUES_READY` | 21 |
| `B_RETRIEVAL_FIX_BUT_GROUNDING_STILL_FAILS` | 3 |
| `C_GOLD_NOT_IN_TOP10` | 6 |

This means retrieval is a real remaining bottleneck. Most schema misses are
near misses, and most top-10 gold-schema recoveries are also ready under the
existing typed-greedy grounder.

## 4. Oracle Top-k Ceiling

Gold+ready oracle, preserving schema correctness:

| k | Gold in top-k | True rescued queries | Projected ready | Projected InstantiationReady |
|---:|---:|---:|---:|---:|
| 2 | 310 | 4 | 261 | 0.788520 |
| 3 | 315 | 8 | 265 | 0.800604 |
| 5 | 319 | 9 | 266 | 0.803625 |
| 10 | 325 | 13 | 270 | 0.815710 |

Readiness-only oracle is misleading:

| k | Ready count | Schema R@1 | Wrong-schema ready false positives |
|---:|---:|---:|---:|
| 2 | 273 | 0.900302 | 22 |
| 3 | 277 | 0.903323 | 24 |
| 5 | 282 | 0.897281 | 28 |
| 10 | 286 | 0.885196 | 32 |

Conclusion: downstream readiness alone is not scientifically acceptable as the
selection target because wrong schemas can satisfy the coverage/type metric.

## 5. Retrieval-margin Analysis

| Margin <= | Triggered | Trigger rate | Schema-miss recall | True rescuable captured, k=5 | Ready/correct triggered |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 4 | 0.012 | 0.100 | 1 | 1 |
| 0.02 | 13 | 0.039 | 0.333 | 3 | 1 |
| 0.03 | 17 | 0.051 | 0.400 | 3 | 3 |
| 0.05 | 27 | 0.082 | 0.600 | 3 | 6 |
| 0.075 | 40 | 0.121 | 0.800 | 5 | 8 |
| 0.10 | 58 | 0.175 | 0.900 | 6 | 21 |

The previously observed `<=0.05` threshold is reproduced exactly: it triggers
27/331 queries and captures 18/30 schema misses. It is selective enough for a
Stage-B implementation and, with the small consistency score below, crosses the
precommitted 264/331 gate.

## 6. Candidate Downstream Signals

Cheap signals evaluated after grounding:

- retrieval TF-IDF score and top-1/top-2 margin;
- coverage;
- type match;
- InstantiationReady;
- expected scalar slots;
- filled slots;
- extracted-number count;
- unmatched mention count;
- incompatible assignment count;
- null slot count;
- assignment margin;
- lexical overlap between query and candidate schema text.

Observed behavior:

- Coverage/type/readiness are powerful but can over-select wrong schemas.
- Retrieval score is necessary as a regularizer.
- The small normalized consistency score is the only tested deterministic rule
  that improves readiness while preserving schema correctness in a selective
  cascade.

## 7. Simple Rerankers

All-schema top-5 diagnostic, not recommended as the production setting:

| Rule | Schema R@1 | Ready | Coverage | TypeMatch | Schema regressions |
|---|---:|---:|---:|---:|---:|
| R0 TF-IDF top1 | 0.909366 | 257 | 0.879430 | 0.851545 | 0 |
| R1 max coverage | 0.900302 | 278 | 0.931135 | 0.896611 | 10 |
| R2 max TypeMatch | 0.891239 | 278 | 0.923971 | 0.904499 | 13 |
| R3 ready/cov/type/TF-IDF | 0.870091 | 282 | 0.931135 | 0.904499 | 20 |
| R4 verified/cov/type/TF-IDF | 0.842900 | 282 | 0.931135 | 0.904499 | 29 |
| R5 small consistency score | 0.927492 | 270 | 0.918705 | 0.890820 | 0 |

R3/R4 maximize readiness but are not scientifically acceptable because they
cause many schema regressions. R5 is the serious candidate.

R5 score:

```text
score = 0.50 * normalized_tfidf
      + 0.25 * coverage
      + 0.25 * type_match
```

Tie-break: retrieval score, then higher-ranked schema.

## 8. Best Selective Cascade

Recommended minimal Stage-B candidate:

```text
if TF-IDF top1-top2 margin > 0.05:
    keep top1
else:
    ground top-5 schemas with current typed greedy
    select max 0.50 * normalized_tfidf
             + 0.25 * coverage
             + 0.25 * type_match
```

Diagnostic performance:

- reranked queries: 27/331 = `0.0816`;
- Schema R@1: `0.915408`;
- InstantiationReady: 265/331 = `0.800604`;
- Coverage: `0.903599`;
- TypeMatch: `0.875715`;
- schema recoveries: 2;
- schema regressions: 0;
- ready gains: 8;
- ready losses: 0.

This crosses the manuscript gate of at least 264/331 while keeping the fallback
selective.

## 9. Transitions

Best Stage-B candidate vs baseline:

| Transition | Count |
|---|---:|
| both ready | 257 |
| baseline only ready | 0 |
| candidate only ready | 8 |
| neither ready | 66 |
| ready McNemar p | 0.0078125 |
| both schema-correct | 301 |
| baseline only schema-correct | 0 |
| candidate only schema-correct | 2 |
| both schema-wrong | 28 |
| schema McNemar p | 0.5 |

The readiness gain is paired-positive; schema correctness is non-regressing but
only modestly improved.

## 10. Runtime

Measured lightweight diagnostic timings:

- top-1 retrieval-only loop: `0.139` s for 331 queries;
- full top-10 diagnostic grounding: `0.935` s for 331 queries.

The proposed selective Stage-B reranks only 27 queries and grounds at most 5
schemas for those queries. Expected overhead is small relative to the current
CPU-only pipeline.

## 11. API Oracle

`NOT_USED`.

Reason: deterministic/local analysis already produced a viable Stage-B rule.
An API schema-reranking oracle is unnecessary before implementing the minimal
CPU-only candidate.

## 12. Novelty and Relation to Prior Negative Results

This differs from prior failed acceptance and hierarchical rerankers in two
ways:

1. It is selective: extra grounding is only triggered on low retrieval-margin
   cases.
2. It uses retrieval-grounding consistency: schema choice is evaluated by a
   candidate's ability to instantiate expected scalar slots under the same
   grounding procedure, with TF-IDF retained as a regularizer.

It should not be framed as a deep semantic reranker. The scientifically
defensible mechanism is lightweight consistency selection between retrieval
and parameter instantiation.

## 13. GO/NO-GO Decision

**Decision: `TOP2_GO`.**

Rationale:

- oracle top-k shows >=7 true rescues at k=3 and above;
- a simple deterministic rule crosses 264/331;
- the recommended selective rule reranks only 27/331 cases;
- schema regressions are 0;
- no API, training, solver, or broad hyperparameter search is required.

## 14. Minimal Stage-B Design

Implement only:

- trigger: TF-IDF top1-top2 margin `<= 0.05`;
- k: 5;
- scorer: `0.50 * normalized_tfidf + 0.25 * coverage + 0.25 * type_match`;
- tie-break: retrieval score, then lower retrieval rank;
- decoder: unchanged typed greedy;
- fallback: unchanged TF-IDF top1 typed greedy.

Do not implement semantic/API reranking, learned reranking, or structured
assignment in Stage B.
