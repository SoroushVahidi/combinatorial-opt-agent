# Error Taxonomy

**Date:** 2026-03-10  
**Method:** tfidf_typed_greedy, orig, 331 queries  
**Source:** Code audit + documented failure family analysis (heuristic estimates)

## Labeling approach

All counts are **approximations** derived from:
1. **Direct code evidence:** schema retrieval miss count = 331 × (1 − 0.9063) = 31 queries.
2. **Structural analysis:** float-type mismatch count from _is_type_match structural analysis.
3. **Per-type TypeMatch:** integer TypeMatch ≈ 0.991, float ≈ 0.030, percent ≈ 0.484, currency ≈ 0.359.
4. **Failure family documentation** in `docs/NLP4LP_ANCHOR_BEAM_DELIVERABLES.md §5`.
5. **Bottleneck slice analysis** from `artifacts/learning_runs/bottleneck_slices/`.

These are NOT manually labeled; they are derived estimates. Treat as directional.

## Counts

| Error Type | Approx. Count (tfidf, orig) | Notes |
|-----------|----------------------------|-------|
| Wrong schema retrieval | 31 (9.4%) | Direct: 331 × (1−0.9063) |
| Float type mismatch (pre-fix) | ~230 | Structural: 79.7% of slots × 97.7% missing |
| Slot disambiguation error | ~50 | From TypeMatch_hits ≈ 0.237 |
| Total vs unit confusion | ~20 | Documented failure family |
| Float ambiguity (many values) | ~25 | Bottleneck slice: multiple_float_like |
| Percent vs absolute confusion | ~15 | Percent TypeMatch ≈ 0.484 → ~52% miss |
| Min/max inversion | ~10 | Documented failure family |
| Numeric extraction miss | ~5 | ~1.5% estimate |
| Unsupported schema | 0 | All queries have valid schema |

## Dominant failure mode

**Float type mismatch** is the largest single failure (~230 cases), and is the root cause
of the historically low TypeMatch (0.227). The `_is_type_match` fix directly targets this.

After the fix, the next dominant failures are:
- **Slot disambiguation** (~50 cases): multiple numeric tokens of the same type
  compete for the same slot (e.g., two integer tokens for two integer slots)
- **Total vs unit confusion** (~20 cases): global consistency methods target this
- **Float ambiguity** (~25 cases): many float-like values for a float-heavy schema

## What each method addresses

| Method | Main error(s) targeted |
|--------|----------------------|
| `tfidf_typed_greedy` | Baseline; addresses nothing specifically |
| `tfidf_constrained` | Slot disambiguation (1-to-1 constraint) |
| `tfidf_semantic_ir_repair` | Float confusion (semantic similarity) |
| `tfidf_optimization_role_repair` | Total vs unit, slot role disambiguation |
| `tfidf_hierarchical_acceptance_rerank` | Schema selection + all downstream |
| `global_consistency_grounding` | All of the above simultaneously (not yet benchmarked) |

## Files
`results/eswa_revision/13_tables/error_taxonomy_counts.csv`
`results/eswa_revision/12_figures/error_taxonomy_bar.png`
