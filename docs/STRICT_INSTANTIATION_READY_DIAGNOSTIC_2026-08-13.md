# Strict Instantiation Ready Diagnostic

**Date:** 2026-08-13  
**Scope:** fresh current-code NLP4LP `orig` per-query artifacts, 331 queries.  
**Decision:** `STRICT_METRIC_RECOMMENDED`.

This diagnostic was triggered by the Stage-B selective reranker audit:
`tfidf_selective_grounding_rerank` improved ordinary `InstantiationReady`
from 257/331 to 265/331, but 6 of the 8 newly ready queries used incorrect
schemas. The result exposed a metric weakness, not a main-method improvement.

## 1. Current Metric Weakness

Current `InstantiationReady` is computed per query as:

```
Coverage(q) >= 0.8 AND TypeMatch(q) >= 0.8
```

where both terms are measured under the **predicted schema**. It does not
require:

- `predicted_schema == gold_schema`;
- exact numeric value correctness;
- solver feasibility or objective correctness;
- full structural/semantic equivalence.

A wrong schema can therefore pass when it has fewer or easier scalar slots, or
when its scalar slots are structurally similar enough that extracted numbers
satisfy Coverage and TypeMatch.

## 2. Strict Metric

Canonical diagnostic name:

```
strict_instantiation_ready
```

Human-readable aliases:

- `StrictInstantiationReady`
- `SchemaCorrectInstantiationReady`

Definition:

```
StrictInstantiationReady(q) =
  1[predicted_schema(q) == gold_schema(q)]
  AND 1[Coverage(q) >= 0.8]
  AND 1[TypeMatch(q) >= 0.8]
```

This is implemented only as a diagnostic/native evaluation metric. Historical
`InstantiationReady` tables are not renamed or overwritten.

## 3. Metric Hierarchy

| Level | Metric | Measures | Does not measure |
|---|---|---|---|
| 1 | `Schema R@1` | whether retrieval selects the gold catalog schema | grounding quality |
| 2 | `Coverage`, `TypeMatch`, ordinary `InstantiationReady` | predicted-schema scalar grounding readiness | schema correctness, exact values, solver correctness |
| 3 | `StrictInstantiationReady` | correct schema plus sufficient scalar coverage/type match | exact numeric values, solver correctness |
| 4 | `Exact5` / `Exact20` | value closeness for comparable scalar slots on schema hits | full model semantics; denominator is schema-hit comparable slots |
| 5 | solver-backed semantic correctness | executable/feasible/objective behavior on restricted subsets | unavailable for all 331 without additional solver work |

The manuscript should present this hierarchy to prevent overclaiming from a
single proxy metric.

## 4. Fresh Results

Source: `results/strict_instantiation_ready/`.

| Method | Schema R@1 | Ordinary Ready | Strict Ready | False Ready |
|---|---:|---:|---:|---:|
| `oracle_typed_greedy` | 1.0000 | 273/331 = 0.8248 | 273/331 = 0.8248 | 0 |
| `tfidf_selective_grounding_rerank` | 0.9154 | 265/331 = 0.8006 | 249/331 = 0.7523 | 16 |
| `tfidf_typed_greedy` | 0.9094 | 257/331 = 0.7764 | 247/331 = 0.7462 | 10 |
| `bm25_typed_greedy` | 0.8822 | 253/331 = 0.7644 | 241/331 = 0.7281 | 12 |
| `lsa_typed_greedy` | 0.8459 | 243/331 = 0.7341 | 233/331 = 0.7039 | 10 |

The selective reranker still improves strict readiness by 2 queries, but the
gain is small and not statistically supported by exact McNemar (`p=0.5`,
2 discordant wins, 0 losses). Its ordinary readiness gain (`+8`, `p=0.0078125`)
is mostly a wrong-schema artifact.

## 5. Baseline vs Selective Transitions

Strict readiness:

| Both strict-ready | Baseline only | Selective only | Neither |
|---:|---:|---:|---:|
| 247 | 0 | 2 | 82 |

Selective-only strict IDs:

- `nlp4lp_test_222`
- `nlp4lp_test_268`

Ordinary readiness:

| Both ordinary-ready | Baseline only | Selective only | Neither |
|---:|---:|---:|---:|
| 257 | 0 | 8 | 66 |

Selective-only ordinary-ready IDs:

- `nlp4lp_test_12`
- `nlp4lp_test_106`
- `nlp4lp_test_130`
- `nlp4lp_test_194`
- `nlp4lp_test_222`
- `nlp4lp_test_268`
- `nlp4lp_test_311`
- `nlp4lp_test_317`

Schema correctness:

| Both correct | Baseline only | Selective only | Both wrong |
|---:|---:|---:|---:|
| 301 | 0 | 2 | 28 |

## 6. False-Ready Taxonomy

For `tfidf_typed_greedy`, 10 ordinary-ready queries have wrong schemas:

- 9 `fewer_or_easier_scalar_slots`
- 1 `structurally_similar_schema`

For `tfidf_selective_grounding_rerank`, 16 ordinary-ready queries have wrong
schemas:

- 13 `fewer_or_easier_scalar_slots`
- 3 `structurally_similar_schema`

Across all evaluated fresh methods, the dominant false-ready mechanism is
wrong-schema selection with a smaller/easier scalar-slot set. This is not fixed
by raising Coverage/TypeMatch thresholds.

## 7. Threshold Diagnostic

False-ready counts persist under stricter thresholds:

| Method | Thresholds | Ordinary Ready | Strict Ready | False Ready |
|---|---:|---:|---:|---:|
| `tfidf_typed_greedy` | 0.8/0.8 | 257 | 247 | 10 |
| `tfidf_typed_greedy` | 0.9/0.9 | 206 | 196 | 10 |
| `tfidf_typed_greedy` | 1.0/1.0 | 202 | 192 | 10 |
| `tfidf_selective_grounding_rerank` | 0.8/0.8 | 265 | 249 | 16 |
| `tfidf_selective_grounding_rerank` | 0.9/0.9 | 214 | 198 | 16 |
| `tfidf_selective_grounding_rerank` | 1.0/1.0 | 210 | 194 | 16 |

The problem is missing schema correctness, not merely loose thresholds.

## 8. Exact-Metric Relationship

`Exact5` and `Exact20` are schema-gated by implementation: they are only
populated when `schema_hit=1` in `tools/nlp4lp_downstream_utility.py`.
However, they are value-closeness diagnostics over comparable scalar slots,
not full model correctness.

Fresh `tfidf_typed_greedy`:

- ordinary-ready: 257
- strict-ready: 247
- ordinary-ready wrong-schema: 10
- strict-ready with full `Exact20`: 7
- strict-ready with non-full `Exact20`: 240

Fresh `tfidf_selective_grounding_rerank`:

- ordinary-ready: 265
- strict-ready: 249
- ordinary-ready wrong-schema: 16
- strict-ready with full `Exact20`: 7
- strict-ready with non-full `Exact20`: 242

This shows strict readiness is still not exact grounding correctness; it is a
better end-to-end readiness proxy, not a final semantic metric.

## 9. Oracle Decomposition

Oracle typed greedy has perfect schema retrieval and reaches:

- ordinary/strict readiness: 273/331 = 0.8248

Current gaps:

- baseline strict gap to oracle: 26 queries
- selective strict gap to oracle: 24 queries

Thus the remaining strict-readiness failures are mixed:

- retrieval errors still matter: baseline has 30 wrong schemas, selective 28;
- grounding remains a large bottleneck even under oracle schema: 58 oracle
  schema-correct queries are not ready.

## 10. Major-Method Strict Ranking

Strict readiness changes magnitudes but not the central ranking: oracle is the
ceiling, fresh typed greedy remains the best semantically reliable non-oracle
baseline, and selective reranking is only a small retrieval-side diagnostic
gain.

| Method | Strict Ready |
|---|---:|
| `oracle_typed_greedy` | 273/331 = 0.8248 |
| `tfidf_selective_grounding_rerank` | 249/331 = 0.7523 |
| `tfidf_typed_greedy` | 247/331 = 0.7462 |
| `tfidf_acceptance_rerank` | 244/331 = 0.7372 |
| `bm25_typed_greedy` | 241/331 = 0.7281 |
| `tfidf_constrained` | 238/331 = 0.7190 |
| `tfidf_max_weight_matching` | 236/331 = 0.7130 |
| `tfidf_optimization_role_repair` | 234/331 = 0.7069 |
| `lsa_typed_greedy` | 233/331 = 0.7039 |
| `tfidf_semantic_ir_repair` | 230/331 = 0.6949 |
| `tfidf_hierarchical_acceptance_rerank` | 228/331 = 0.6888 |
| `tfidf_search_structured_grounding` | 223/331 = 0.6737 |
| `tfidf_hierarchical_structured_grounding` | 223/331 = 0.6737 |

## 11. Fresh Bottleneck Taxonomy

For `tfidf_typed_greedy`:

- schema wrong: 30
- schema correct but Coverage failure: 11
- schema correct but TypeMatch failure: 30
- schema correct but both Coverage and TypeMatch fail: 13
- schema correct and strict-ready but `Exact20` not full: 240
- schema correct, strict-ready, and full `Exact20`: 7

For `tfidf_selective_grounding_rerank`:

- schema wrong: 28
- schema correct but Coverage failure: 11
- schema correct but TypeMatch failure: 30
- schema correct but both Coverage and TypeMatch fail: 13
- schema correct and strict-ready but `Exact20` not full: 242
- schema correct, strict-ready, and full `Exact20`: 7

## 12. Historical Compatibility

The historical strict metric is reconstructable from
`results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv`:

| Historical method | Ordinary Ready | Strict Ready |
|---|---:|---:|
| TFIDF-TG | 0.5287 | 0.5045 |
| BM25-TG | 0.5196 | 0.4924 |
| LSA-TG | 0.5076 | 0.4864 |
| Oracle-TG | 0.5680 | 0.5680 |

These are historical 0.5287-era artifacts and must not be silently compared
against the fresh 247/331 strict metric.

## 13. Cross-Baseline Harness

`baselines/comparison/` now treats both `instantiation_ready` and
`strict_instantiation_ready` as **OUR-METHOD native metrics** only.
Neither is added to shared external baseline objective metrics because our
system performs fixed-catalog scalar grounding and does not generate/execute a
full optimization model.

## 14. Manuscript Metric Recommendation

Recommended primary metric hierarchy for a revision:

1. **Primary native end-to-end proxy:** `StrictInstantiationReady`.
2. **Required supporting metrics:** `Schema R@1`, `Coverage`, `TypeMatch`.
3. **Secondary diagnostic:** ordinary `InstantiationReady`, explicitly labeled
   as predicted-schema readiness.
4. **Value diagnostics:** `Exact20`/`Exact5`, with denominator and schema-hit
   conditioning stated.
5. **Solver-backed semantic validation:** restricted subset only, unless a
   broader solver comparison is run later.

Claims to weaken regardless:

- do not call ordinary `InstantiationReady` end-to-end correctness;
- do not treat the selective reranker's 265/331 ordinary result as a main
  method improvement;
- do not compare strict fresh metrics against historical non-strict metrics.

## 15. Method-Improvement Implications

Top next directions under strict readiness:

1. **Schema-correct retrieval/reranking:** improve schema selection without
   accepting wrong-schema readiness artifacts. The selective reranker suggests
   a small true retrieval benefit (+2 strict-ready), but any future reranker
   must optimize strict readiness and schema correctness jointly.
2. **Grounding under correct schema:** oracle strict readiness is only 273/331,
   leaving 58 schema-correct not-ready cases. TypeMatch failures (30) and
   coverage failures (11+13 mixed) are still substantial.
3. **Exact value correctness:** only 7 baseline strict-ready queries have full
   `Exact20`; improving true numeric assignments is a separate research target
   from readiness.

## 16. Decision

`STRICT_METRIC_RECOMMENDED`

Rationale: wrong-schema false-ready cases are meaningful, affect the
interpretation of the strongest new method candidate, persist under stricter
Coverage/TypeMatch thresholds, and are fixed directly by adding schema
correctness to the readiness metric.
