# Statistical Significance Summary

Bootstrap samples: B=1000, seed=42, 95% CIs (percentile method).
All tests are two-sided paired bootstrap tests on per-instance binary outcomes.
**Conservative interpretation**: p < 0.05 is noted but not over-interpreted;
overlapping CIs or p ≥ 0.05 are explicitly called out as non-significant.

---

## Confidence Intervals (orig variant, key methods)

| Method | Metric | Observed | 95% CI |
|--------|--------|----------|--------|
| tfidf | Schema_R@1 | 0.9094 | [0.8792, 0.9396] |
| tfidf | Coverage | 0.8639 | [0.8317, 0.8943] |
| tfidf | TypeMatch | 0.7513 | [0.7176, 0.7838] |
| tfidf | InstReady | 0.5257 | [0.4743, 0.5801] |
| bm25 | Schema_R@1 | 0.8822 | [0.8459, 0.9154] |
| bm25 | Coverage | 0.8509 | [0.8166, 0.8816] |
| bm25 | TypeMatch | 0.7386 | [0.7034, 0.7738] |
| bm25 | InstReady | 0.5196 | [0.4653, 0.5740] |
| lsa | Schema_R@1 | 0.8459 | [0.8097, 0.8822] |
| lsa | Coverage | 0.8176 | [0.7805, 0.8525] |
| lsa | TypeMatch | 0.7028 | [0.6663, 0.7397] |
| lsa | InstReady | 0.4985 | [0.4471, 0.5498] |
| oracle | Schema_R@1 | 1.0000 | [1.0000, 1.0000] |
| oracle | Coverage | 0.9151 | [0.8912, 0.9360] |
| oracle | TypeMatch | 0.8030 | [0.7749, 0.8307] |
| oracle | InstReady | 0.5650 | [0.5136, 0.6163] |
| tfidf_acceptance_rerank | Schema_R@1 | 0.8731 | [0.8399, 0.9094] |
| tfidf_acceptance_rerank | Coverage | 0.8332 | [0.7975, 0.8670] |
| tfidf_acceptance_rerank | TypeMatch | 0.7340 | [0.6973, 0.7693] |
| tfidf_acceptance_rerank | InstReady | 0.5227 | [0.4683, 0.5770] |
| tfidf_hierarchical_acceptance_rerank | Schema_R@1 | 0.8459 | [0.8066, 0.8852] |
| tfidf_hierarchical_acceptance_rerank | Coverage | 0.8121 | [0.7740, 0.8471] |
| tfidf_hierarchical_acceptance_rerank | TypeMatch | 0.7146 | [0.6768, 0.7541] |
| tfidf_hierarchical_acceptance_rerank | InstReady | 0.5136 | [0.4592, 0.5680] |

---

## Paired Significance Tests (orig variant)

| Comparison | Metric | A | B | Diff | 95% CI (diff) | p-value | Interpretation |
|------------|--------|---|---|------|---------------|---------|----------------|
| TFIDF vs BM25 (Schema_R@1) | Schema_R@1 | 0.9094 | 0.8822 | 0.0272 | [0.0060, 0.0514] | 0.0220 | p<0.05 (significant) |
| TFIDF-TG vs BM25-TG (InstReady) | InstReady | 0.5257 | 0.5196 | 0.0060 | [-0.0151, 0.0272] | 0.7120 | CI contains 0 (not significant) |
| TFIDF-TG vs Oracle-TG (InstReady) | InstReady | 0.5257 | 0.5650 | -0.0393 | [-0.0665, -0.0151] | 0.0040 | p<0.01 (robust) |
| TFIDF-TG vs TFIDF-AR (InstReady) | InstReady | 0.5257 | 0.5227 | 0.0030 | [-0.0181, 0.0211] | 0.8900 | CI contains 0 (not significant) |
| TFIDF-TG vs TFIDF-HAR (InstReady) | InstReady | 0.5257 | 0.5136 | 0.0121 | [-0.0121, 0.0363] | 0.3760 | CI contains 0 (not significant) |
| TFIDF-TG vs GCG-Full (InstReady) | InstReady | 0.5257 | 0.4320 | 0.0937 | [0.0483, 0.1420] | 0.0000 | p<0.01 (robust) |
| TFIDF-TG vs RAL-Basic (InstReady) | InstReady | 0.5257 | 0.4985 | 0.0272 | [-0.0091, 0.0665] | 0.2180 | CI contains 0 (not significant) |
| TFIDF-TG vs RAL-Full (InstReady) | InstReady | 0.5257 | 0.4169 | 0.1088 | [0.0695, 0.1541] | 0.0000 | p<0.01 (robust) |
| TFIDF-TG vs AAG-Beam (InstReady) | InstReady | 0.5257 | 0.4230 | 0.1027 | [0.0514, 0.1511] | 0.0000 | p<0.01 (robust) |
| TFIDF-TG vs AAG-Full (InstReady) | InstReady | 0.5257 | 0.4199 | 0.1057 | [0.0574, 0.1511] | 0.0000 | p<0.01 (robust) |
| RAL-Basic vs Oracle-TG (InstReady) | InstReady | 0.4985 | 0.5650 | -0.0665 | [-0.1118, -0.0211] | 0.0080 | p<0.01 (robust) |
| TFIDF-TG vs AAG-Abstain (Coverage) | param_coverage | 0.8639 | 0.2207 | 0.6432 | [0.6029, 0.6812] | 0.0000 | p<0.01 (robust) |

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
