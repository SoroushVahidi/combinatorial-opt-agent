# Retrieval vs Grounding Bottleneck Analysis

**Date:** 2026-03-10

## Central Claim

Retrieval is no longer the main bottleneck; downstream grounding is.

## Evidence

### 1. Retrieval is strong

| Variant | TF-IDF R@1 | BM25 R@1 | LSA R@1 |
|---------|------------|----------|---------|
| orig  | 0.9094 | 0.8822 | 0.8459 |
| noisy | 0.9033 | 0.8912 | 0.8882 |
| short | 0.7795 | 0.7674 | 0.7644 |

All lexical methods achieve R@1 > 0.77 on the hardest variant (short).

### 2. Better retrieval gives only modest downstream gain

Comparing BM25 vs TF-IDF vs Oracle (orig, typed greedy):

| Retrieval | Schema R@1 | Coverage | TypeMatch | InstReady |
|-----------|-----------|----------|-----------|-----------|
| BM25 | 0.885 | 0.813 | 0.225 | 0.076 |
| TF-IDF | 0.906 | 0.822 | 0.227 | 0.073 |
| Oracle | 1.000 | 0.870 | 0.248 | 0.082 |

Oracle over TF-IDF: Coverage +0.048, TypeMatch +0.021, InstReady +0.009.
Improving retrieval from 0.906 to 1.000 (+9.4pp R@1) buys only +0.9pp InstReady.

### 3. Grounding degradation across variants

| Variant | TF-IDF R@1 | Coverage drop | TypeMatch drop | InstReady drop |
|---------|------------|---------------|----------------|----------------|
| orig → noisy | −0.007 | −0.112 | −0.227 | −0.073 |
| orig → short | −0.130 | −0.789 | −0.200 | −0.067 |

On noisy: R@1 barely drops but Coverage/TypeMatch/InstReady collapse completely.
On short: R@1 drops 13pp but Coverage drops 79pp. Grounding is the binding constraint.

### 4. Conclusion

**Confirmed:** Downstream grounding is the dominant bottleneck. Evidence:
- Oracle R@1 = 1.0 still only achieves InstReady 0.082 (vs 0.073 for TF-IDF)
- Noisy variant: R@1 ≈ 0.90 but InstReady = 0.000 (by design — no numeric values)
- Float TypeMatch ≈ 0.030 (pre-fix) → main algorithmic failure source

## What remains open

- Hybrid retrieval (BM25+TF-IDF) for short queries: +0.012 R@1 (see PORTED_IMPROVEMENTS doc)
  but downstream impact unmeasured (blocked by HF_TOKEN)
- SAE evaluation: code implemented but no end-to-end numbers yet

## Source files
- Live retrieval: `results/eswa_revision/01_retrieval/retrieval_results.json`
- Downstream (manuscript): `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md`
