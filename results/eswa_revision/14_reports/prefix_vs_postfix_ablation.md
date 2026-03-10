# Pre-Fix vs Post-Fix TypeMatch Ablation (Measured)

**Date:** 2026-03-10T21:05:34Z  
**Source:** GitHub Actions run `22924153330` (commit `17e01d90`)  
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
| tfidf_typed_greedy | 0.2497 | 0.7453 | +0.4956 | 0.8609 | 0.8609 |
| tfidf_optimization_role_repair | 0.2716 | 0.7016 | +0.4300 | 0.8218 | 0.8218 |
| tfidf_hierarchical_acceptance_rerank | 0.2478 | 0.7097 | +0.4619 | 0.8121 | 0.8121 |
| oracle_typed_greedy | 0.2777 | 0.7998 | +0.5221 | 0.9151 | 0.9151 |

### NOISY

| Method | TM_pre | TM_post | TM_delta | Coverage_pre | Coverage_post |
|--------|--------|---------|----------|--------------|---------------|
| tfidf_typed_greedy | 0.0520 | 0.1437 | +0.0917 | 0.7697 | 0.7697 |
| tfidf_optimization_role_repair | 0.0000 | 0.0000 | +0.0000 | 0.7100 | 0.7100 |
| tfidf_hierarchical_acceptance_rerank | 0.0576 | 0.1357 | +0.0781 | 0.6959 | 0.6959 |
| oracle_typed_greedy | 0.0620 | 0.1537 | +0.0917 | 0.8196 | 0.8196 |

### SHORT

| Method | TM_pre | TM_post | TM_delta | Coverage_pre | Coverage_post |
|--------|--------|---------|----------|--------------|---------------|
| tfidf_typed_greedy | 0.0957 | 0.2472 | +0.1515 | 0.1065 | 0.1065 |
| tfidf_optimization_role_repair | 0.0438 | 0.0755 | +0.0317 | 0.0333 | 0.0333 |
| tfidf_hierarchical_acceptance_rerank | 0.1088 | 0.2412 | +0.1324 | 0.1077 | 0.1077 |
| oracle_typed_greedy | 0.1143 | 0.2880 | +0.1737 | 0.1258 | 0.1258 |

## Interpretation

TypeMatch_delta shows the net improvement from the `_is_type_match` fix.
The delta should be positive for all methods since integer tokens are now
correctly counted as matches for float-typed slots.

Tables: `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`
