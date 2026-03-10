# Pre-Fix vs Post-Fix TypeMatch Ablation (Measured)

**Date:** 2026-03-10T20:18:27Z  
**Source:** GitHub Actions run `22922351003` (commit `17e01d90`)  
**Status:** NEWLY MEASURED — both pre-fix and post-fix values computed in this run

## Background

The pre-fix `_is_type_match` used strict equality (`expected == kind`).
The post-fix version additionally returns `True` for `(float, int)` pairs.
Both were run in this CI job: the pre-fix was simulated by patching the function
in-memory before each pre-fix run and restoring it after.

## Results by Variant

### ORIG

| Method | TM_pre | TM_post | TM_delta | Coverage_pre | Coverage_post |
|--------|--------|---------|----------|--------------|---------------|
| tfidf_typed_greedy | 0.2595 | 0.7513 | +0.4918 | 0.8639 | 0.8639 |
| tfidf_optimization_role_repair | 0.2716 | 0.7036 | +0.4320 | 0.8248 | 0.8248 |
| tfidf_hierarchical_acceptance_rerank | 0.2593 | 0.7146 | +0.4553 | 0.8121 | 0.8121 |
| oracle_typed_greedy | 0.2885 | 0.8030 | +0.5145 | 0.9151 | 0.9151 |

### NOISY

| Method | TM_pre | TM_post | TM_delta | Coverage_pre | Coverage_post |
|--------|--------|---------|----------|--------------|---------------|
| tfidf_typed_greedy | 0.0583 | 0.1414 | +0.0831 | 0.7693 | 0.7693 |
| tfidf_optimization_role_repair | 0.0000 | 0.0000 | +0.0000 | 0.7103 | 0.7103 |
| tfidf_hierarchical_acceptance_rerank | 0.0647 | 0.1380 | +0.0733 | 0.6956 | 0.6956 |
| oracle_typed_greedy | 0.0703 | 0.1524 | +0.0821 | 0.8196 | 0.8196 |

### SHORT

| Method | TM_pre | TM_post | TM_delta | Coverage_pre | Coverage_post |
|--------|--------|---------|----------|--------------|---------------|
| tfidf_typed_greedy | 0.0962 | 0.2286 | +0.1324 | 0.1050 | 0.1050 |
| tfidf_optimization_role_repair | 0.0438 | 0.0725 | +0.0287 | 0.0318 | 0.0318 |
| tfidf_hierarchical_acceptance_rerank | 0.1133 | 0.2261 | +0.1128 | 0.1062 | 0.1062 |
| oracle_typed_greedy | 0.1133 | 0.2709 | +0.1576 | 0.1258 | 0.1258 |

## Interpretation

TypeMatch_delta shows the net improvement from the `_is_type_match` fix.
The delta should be positive for all methods since integer tokens are now
correctly counted as matches for float-typed slots.

Tables: `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`
