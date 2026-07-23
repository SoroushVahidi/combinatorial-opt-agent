# Statistical Significance Summary

Bootstrap samples: B=1000, seed=42, 95% CIs (percentile method).
All tests are two-sided paired bootstrap tests on per-instance binary outcomes.
**Conservative interpretation**: p < 0.05 is noted but not over-interpreted;
overlapping CIs or p ≥ 0.05 are explicitly called out as non-significant.

---

## Confidence Intervals (orig variant, key methods)

| Method | Metric | Observed | 95% CI |
|--------|--------|----------|--------|
| tfidf | Schema_R@1 | 0.9063 | [0.8761, 0.9366] |
| tfidf | Coverage | 0.8609 | [0.8286, 0.8928] |
| tfidf | TypeMatch | 0.7453 | [0.7108, 0.7785] |
| tfidf | InstReady | 0.5287 | [0.4773, 0.5801] |
| bm25 | Schema_R@1 | 0.8852 | [0.8520, 0.9184] |
| bm25 | Coverage | 0.8509 | [0.8166, 0.8816] |
| bm25 | TypeMatch | 0.7336 | [0.6972, 0.7685] |
| bm25 | InstReady | 0.5196 | [0.4653, 0.5740] |
| lsa | Schema_R@1 | 0.8550 | [0.8187, 0.8912] |
| lsa | Coverage | 0.8267 | [0.7902, 0.8610] |
| lsa | TypeMatch | 0.7054 | [0.6686, 0.7416] |
| lsa | InstReady | 0.5076 | [0.4562, 0.5619] |
| oracle | Schema_R@1 | 1.0000 | [1.0000, 1.0000] |
| oracle | Coverage | 0.9151 | [0.8912, 0.9360] |
| oracle | TypeMatch | 0.7998 | [0.7710, 0.8285] |
| oracle | InstReady | 0.5680 | [0.5166, 0.6193] |
| tfidf_acceptance_rerank | Schema_R@1 | 0.8701 | [0.8369, 0.9063] |
| tfidf_acceptance_rerank | Coverage | 0.8302 | [0.7945, 0.8665] |
| tfidf_acceptance_rerank | TypeMatch | 0.7261 | [0.6891, 0.7631] |
| tfidf_acceptance_rerank | InstReady | 0.5257 | [0.4713, 0.5801] |
| tfidf_hierarchical_acceptance_rerank | Schema_R@1 | 0.8489 | [0.8097, 0.8882] |
| tfidf_hierarchical_acceptance_rerank | Coverage | 0.8121 | [0.7762, 0.8491] |
| tfidf_hierarchical_acceptance_rerank | TypeMatch | 0.7097 | [0.6708, 0.7491] |
| tfidf_hierarchical_acceptance_rerank | InstReady | 0.5196 | [0.4653, 0.5740] |

---

## Paired Significance Tests (orig variant)

| Comparison | Metric | A | B | Diff | 95% CI (diff) | p-value | Interpretation |
|------------|--------|---|---|------|---------------|---------|----------------|
| TFIDF vs BM25 (Schema_R@1) | Schema_R@1 | 0.9063 | 0.8852 | 0.0211 | [-0.0030, 0.0453] | 0.0880 | CI contains 0 (not significant) |
| TFIDF-TG vs BM25-TG (InstReady) | InstReady | 0.5287 | 0.5196 | 0.0091 | [-0.0121, 0.0302] | 0.5120 | CI contains 0 (not significant) |
| TFIDF-TG vs Oracle-TG (InstReady) | InstReady | 0.5287 | 0.5680 | -0.0393 | [-0.0665, -0.0151] | 0.0040 | p<0.01 (robust) |
| TFIDF-TG vs TFIDF-AR (InstReady) | InstReady | 0.5287 | 0.5257 | 0.0030 | [-0.0181, 0.0211] | 0.8900 | CI contains 0 (not significant) |
| TFIDF-TG vs TFIDF-HAR (InstReady) | InstReady | 0.5287 | 0.5196 | 0.0091 | [-0.0151, 0.0332] | 0.5800 | CI contains 0 (not significant) |
| TFIDF-TG vs GCG-Full (InstReady) | InstReady | 0.5287 | 0.4320 | 0.0967 | [0.0544, 0.1420] | 0.0000 | p<0.01 (robust) |
| TFIDF-TG vs RAL-Basic (InstReady) | InstReady | 0.5287 | 0.4985 | 0.0302 | [-0.0060, 0.0665] | 0.1500 | CI contains 0 (not significant) |
| TFIDF-TG vs RAL-Full (InstReady) | InstReady | 0.5287 | 0.4169 | 0.1118 | [0.0695, 0.1541] | 0.0000 | p<0.01 (robust) |
| TFIDF-TG vs AAG-Beam (InstReady) | InstReady | 0.5287 | 0.4230 | 0.1057 | [0.0544, 0.1541] | 0.0000 | p<0.01 (robust) |
| TFIDF-TG vs AAG-Full (InstReady) | InstReady | 0.5287 | 0.4199 | 0.1088 | [0.0604, 0.1571] | 0.0000 | p<0.01 (robust) |
| RAL-Basic vs Oracle-TG (InstReady) | InstReady | 0.4985 | 0.5680 | -0.0695 | [-0.1148, -0.0242] | 0.0060 | p<0.01 (robust) |
| TFIDF-TG vs AAG-Abstain (Coverage) | param_coverage | 0.8609 | 0.2207 | 0.6402 | [0.5985, 0.6780] | 0.0000 | p<0.01 (robust) |

---

## Notes and Caveats

- All bootstrap CIs are 95% percentile intervals from B bootstrap resamples.
- Paired tests resample instance indices jointly so each resample preserves instance-level pairing.
- Schema_R@1 and InstReady are binary 0/1 per-instance outcomes; Coverage and TypeMatch
  are continuous per-instance values (mean over slots per query).
- Exact20_on_hits has a conditional denominator (schema-hit queries only) and is not
  included in paired tests to avoid denominator instability.
- The 'orig' variant is the primary eval split; noisy/short results are in the full CSV.
- For the full table including noisy and short variants, see confidence_intervals.csv
  and paired_significance.csv.
