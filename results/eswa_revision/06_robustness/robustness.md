# Robustness Across Query Variants

**Date:** 2026-03-10

## Result summary (tfidf_typed_greedy)

| Metric | orig | noisy | noisy drop | short | short drop |
|--------|------|-------|-----------|-------|-----------|
| Schema R@1 | 0.9063 | 0.9033 | −0.3% | 0.7855 | −13.3% |
| Coverage | 0.8222 | 0.7100 | −13.6% | 0.0333 | −95.9% |
| TypeMatch | 0.2267 | 0.0000 | −100% | 0.0272 | −88.0% |
| InstReady | 0.0725 | 0.0000 | −100% | 0.0060 | −91.7% |

## Interpretation

### Noisy variant
- Retrieval: barely affected (−0.3% R@1). Schema text matches even without numbers.
- Grounding: **completely fails**. TypeMatch = 0, InstReady = 0 **by design**:
  the noisy variant replaces all numeric values with `<num>` placeholders that cannot
  be parsed as typed values. This is a structural property, not an algorithmic failure.
- **Manuscript note:** Noisy downstream metrics should be labeled "N/A — no numeric values"
  rather than "method fails on noisy queries."

### Short variant
- Retrieval: −13pp R@1. First-sentence queries often lack problem-specific terms.
- Grounding: **catastrophically low coverage** (0.033) because short queries contain almost
  no numeric parameters. Only ~3% of expected slots can be filled.
- Hybrid retrieval (BM25+TF-IDF) improves short R@1 by +0.012pp (see PORTED docs).
  Downstream impact of hybrid: not yet measured (needs HF_TOKEN).

### Stability recommendation
For the ESWA manuscript:
- Report noisy retrieval as the main noisy story (strong).
- Report noisy downstream as "numeric grounding blocked by design (no values in query)".
- Report short as "coverage collapses — method requires numeric information in query".
- These limitations are honest and strengthening: they tell reviewers exactly where the method
  works and where it doesn't, without hiding anything.

## Tables
`results/eswa_revision/13_tables/robustness_by_variant.csv`
`results/eswa_revision/13_tables/robustness_relative_drop.csv`
