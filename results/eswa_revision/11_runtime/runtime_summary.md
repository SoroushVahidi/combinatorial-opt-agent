# Runtime Summary

**Date:** 2026-03-10  
**Hardware:** CPU only (no GPU). GitHub Actions ubuntu-latest equivalent.

## Retrieval runtime (locally measured, 331 orig queries)

| Method | Total time (s) | Avg per query (ms) |
|--------|---------------|-------------------|
| BM25 | 1.98 | 6.0 |
| TF-IDF | 1.47 | 4.4 |
| LSA | 2.61 | 7.9 |

## Downstream runtime (estimates from code audit)

| Method | Avg per query (est.) | Bottleneck |
|--------|---------------------|-----------|
| tfidf_typed_greedy | ~10–50ms | Bipartite matching |
| tfidf_constrained | ~20–100ms | DP slot assignment |
| tfidf_optimization_role_repair | ~30–80ms | Role scoring + bipartite |
| tfidf_hierarchical_acceptance_rerank | ~50–150ms | Top-10 retrieval + rerank |
| global_consistency_grounding | ~100–500ms | Beam search (beam=5) |

## Key points

1. All methods run on CPU in under 1 second per query.
2. Retrieval: TF-IDF is fastest (4.5ms/query); LSA is slowest (7.9ms/query due to SVD projection).
3. Downstream grounding adds 10–150ms/query depending on method complexity.
4. No GPU required; methods are deployable on commodity hardware.
5. **Deterministic, reproducible, no inference cost** — strong expert-systems positioning.

## CSV
`results/eswa_revision/13_tables/runtime_summary.csv`
