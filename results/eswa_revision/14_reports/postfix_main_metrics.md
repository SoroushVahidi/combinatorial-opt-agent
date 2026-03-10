# Post-Fix Downstream Benchmark Results (Measured)

**Date:** 2026-03-10T20:18:27Z  
**Source:** GitHub Actions run `22922351003` (commit `17e01d90`)  
**Status:** NEWLY MEASURED — real authenticated results, NOT manuscript-era estimates

## Summary by Variant

### ORIG

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf_typed_greedy | 0.8639 | 0.7513 | 0.1991 | 0.5257 |
| bm25_typed_greedy | 0.8509 | 0.7386 | 0.2057 | 0.5196 |
| lsa_typed_greedy | 0.8176 | 0.7028 | 0.2048 | 0.4985 |
| oracle_typed_greedy | 0.9151 | 0.8030 | 0.1882 | 0.5650 |
| tfidf_constrained | 0.8112 | 0.7113 | 0.3293 | 0.4230 |
| tfidf_semantic_ir_repair | 0.7817 | 0.7549 | 0.2843 | 0.4864 |
| tfidf_optimization_role_repair | 0.8248 | 0.7036 | 0.2847 | 0.4411 |
| tfidf_acceptance_rerank | 0.8332 | 0.7340 | 0.1994 | 0.5227 |
| tfidf_hierarchical_acceptance_rerank | 0.8121 | 0.7146 | 0.2003 | 0.5136 |

### NOISY

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf_typed_greedy | 0.7693 | 0.1414 | 0.1801 | 0.0393 |
| bm25_typed_greedy | 0.7620 | 0.1409 | 0.1817 | 0.0423 |
| lsa_typed_greedy | 0.7601 | 0.1415 | 0.1817 | 0.0423 |
| oracle_typed_greedy | 0.8196 | 0.1524 | 0.1797 | 0.0423 |
| tfidf_constrained | 0.4140 | 0.2335 | 0.2593 | 0.0363 |
| tfidf_semantic_ir_repair | 0.3465 | 0.0000 |  | 0.0000 |
| tfidf_optimization_role_repair | 0.7103 | 0.0000 |  | 0.0000 |
| tfidf_acceptance_rerank | 0.7312 | 0.1400 | 0.2006 | 0.0423 |
| tfidf_hierarchical_acceptance_rerank | 0.6956 | 0.1380 | 0.2118 | 0.0423 |

### SHORT

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf_typed_greedy | 0.1050 | 0.2286 | 0.2320 | 0.0151 |
| bm25_typed_greedy | 0.1062 | 0.2301 | 0.2298 | 0.0151 |
| lsa_typed_greedy | 0.1004 | 0.2165 | 0.2317 | 0.0121 |
| oracle_typed_greedy | 0.1258 | 0.2709 | 0.2317 | 0.0151 |
| tfidf_constrained | 0.1050 | 0.2915 | 0.2778 | 0.0151 |
| tfidf_semantic_ir_repair | 0.0318 | 0.0785 | 0.3333 | 0.0060 |
| tfidf_optimization_role_repair | 0.0318 | 0.0725 | 0.3939 | 0.0060 |
| tfidf_acceptance_rerank | 0.1130 | 0.2377 | 0.2650 | 0.0211 |
| tfidf_hierarchical_acceptance_rerank | 0.1062 | 0.2261 | 0.2826 | 0.0211 |

## Source

All numbers measured in this GitHub Actions run using the authenticated
`udell-lab/NLP4LP` dataset. No manuscript-era estimates are included.

Raw per-query CSVs: `results/eswa_revision/02_downstream_postfix/`  
Tables: `results/eswa_revision/13_tables/postfix_main_metrics.csv`
