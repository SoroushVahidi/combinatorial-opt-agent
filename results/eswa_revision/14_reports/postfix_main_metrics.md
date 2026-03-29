# Post-Fix Downstream Benchmark Results (Measured)

**Date:** 2026-03-10T21:05:34Z  
**Source:** GitHub Actions run `22924153330` (commit `17e01d90`)  
**Status:** NEWLY MEASURED — real authenticated results, NOT manuscript-era estimates

## Summary by Variant

### ORIG

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf_typed_greedy | 0.8609 | 0.7453 | 0.1834 | 0.5287 |
| bm25_typed_greedy | 0.8509 | 0.7336 | 0.1884 | 0.5196 |
| lsa_typed_greedy | 0.8267 | 0.7054 | 0.1822 | 0.5076 |
| oracle_typed_greedy | 0.9151 | 0.7998 | 0.1745 | 0.5680 |
| tfidf_constrained | 0.8082 | 0.7093 | 0.3239 | 0.4230 |
| tfidf_semantic_ir_repair | 0.7787 | 0.7529 | 0.2734 | 0.4864 |
| tfidf_optimization_role_repair | 0.8218 | 0.7016 | 0.3036 | 0.4411 |
| tfidf_acceptance_rerank | 0.8302 | 0.7261 | 0.1768 | 0.5257 |
| tfidf_hierarchical_acceptance_rerank | 0.8121 | 0.7097 | 0.1771 | 0.5196 |

### NOISY

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf_typed_greedy | 0.7697 | 0.1437 | 0.1982 | 0.0393 |
| bm25_typed_greedy | 0.7650 | 0.1431 | 0.2082 | 0.0423 |
| lsa_typed_greedy | 0.7658 | 0.1441 | 0.1999 | 0.0423 |
| oracle_typed_greedy | 0.8196 | 0.1537 | 0.2079 | 0.0423 |
| tfidf_constrained | 0.4154 | 0.2335 | 0.2578 | 0.0363 |
| tfidf_semantic_ir_repair | 0.3476 | 0.0000 |  | 0.0000 |
| tfidf_optimization_role_repair | 0.7100 | 0.0000 |  | 0.0000 |
| tfidf_acceptance_rerank | 0.7315 | 0.1385 | 0.2238 | 0.0423 |
| tfidf_hierarchical_acceptance_rerank | 0.6959 | 0.1357 | 0.2346 | 0.0423 |

### SHORT

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf_typed_greedy | 0.1065 | 0.2472 | 0.1909 | 0.0151 |
| bm25_typed_greedy | 0.1062 | 0.2487 | 0.2006 | 0.0151 |
| lsa_typed_greedy | 0.1005 | 0.2321 | 0.2067 | 0.0121 |
| oracle_typed_greedy | 0.1258 | 0.2880 | 0.2154 | 0.0151 |
| tfidf_constrained | 0.1065 | 0.2946 | 0.2411 | 0.0151 |
| tfidf_semantic_ir_repair | 0.0333 | 0.0816 | 0.2206 | 0.0060 |
| tfidf_optimization_role_repair | 0.0333 | 0.0755 | 0.2794 | 0.0060 |
| tfidf_acceptance_rerank | 0.1145 | 0.2543 | 0.2277 | 0.0211 |
| tfidf_hierarchical_acceptance_rerank | 0.1077 | 0.2412 | 0.2473 | 0.0211 |

## Source

All numbers measured in this GitHub Actions run using the authenticated
`udell-lab/NLP4LP` dataset. No manuscript-era estimates are included.

Raw per-query CSVs: `results/eswa_revision/02_downstream_postfix/`  
Tables: `results/eswa_revision/13_tables/postfix_main_metrics.csv`
