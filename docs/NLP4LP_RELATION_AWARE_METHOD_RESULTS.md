# NLP4LP Relation-Aware Method — Results

## How to produce results

Run downstream for orig (and optionally noisy/short) with the new assignment mode, then read the summary CSV:

```bash
# New method (orig)
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_relation_repair
python tools/nlp4lp_downstream_utility.py --variant orig --baseline oracle --assignment-mode optimization_role_relation_repair

# Optional: noisy and short
python tools/nlp4lp_downstream_utility.py --variant noisy --baseline tfidf --assignment-mode optimization_role_relation_repair
python tools/nlp4lp_downstream_utility.py --variant short --baseline tfidf --assignment-mode optimization_role_relation_repair
```

Summary is written to `results/paper/nlp4lp_downstream_summary.csv` (rows keyed by variant and effective_baseline).

## Comparison table (orig) — existing baselines + new method (run to fill new rows)

| Baseline | schema_R1 | param_coverage | type_match | key_overlap | exact5_on_hits | exact20_on_hits | instantiation_ready |
|----------|-----------|----------------|------------|-------------|----------------|-----------------|---------------------|
| tfidf | 0.9063 | 0.8222 | 0.2260 | 0.9188 | 0.2053 | 0.2330 | 0.0755 |
| tfidf_constrained | 0.9063 | 0.7720 | 0.1950 | 0.9188 | 0.2921 | 0.3250 | 0.0272 |
| tfidf_semantic_ir_repair | 0.9063 | 0.7783 | 0.2539 | 0.9188 | 0.2345 | 0.2614 | 0.0634 |
| tfidf_optimization_role_repair | 0.9063 | 0.8218 | 0.2427 | 0.9188 | 0.2514 | 0.2772 | 0.0604 |
| tfidf_hierarchical_acceptance_rerank | 0.8459 | 0.7771 | 0.2303 | 0.8592 | 0.1705 | 0.1965 | **0.0846** |
| **tfidf_optimization_role_relation_repair** | *(run cmd below)* | | | | | | |
| oracle | 1.0 | 0.8695 | 0.2401 | 0.9953 | 0.1824 | 0.2044 | 0.0816 |
| oracle_constrained | 1.0 | 0.8195 | 0.2092 | 0.9953 | 0.2938 | 0.3206 | 0.0211 |
| oracle_optimization_role_repair | 1.0 | 0.8691 | 0.2688 | 0.9953 | 0.2465 | 0.2702 | 0.0695 |
| **oracle_optimization_role_relation_repair** | *(run cmd below)* | | | | | | |

After running the commands in “How to produce results”, copy the new rows from `results/paper/nlp4lp_downstream_summary.csv` (variant=orig) into this table.

## Success criteria (from task)

- Improve InstantiationReady and/or TypeMatch over current strongest downstream methods.
- Improve exactness without collapsing coverage.
- Show whether the new method is better than optimization_role_repair and whether it closes the gap with hierarchical_acceptance_rerank.
