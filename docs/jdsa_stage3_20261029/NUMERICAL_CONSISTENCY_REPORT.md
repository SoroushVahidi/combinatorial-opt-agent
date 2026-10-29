# Numerical and scientific consistency report

**PASS.** No experimental result, threshold, conditional denominator, population or statistical correction family changed.

## Authentication and checks

- Exact Stage 2 branch tip verified locally and at origin: `60785b0d144365413dd3b44b577bff901043ddd7`.
- GitHub evidence tag verified at `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`.
- Anonymous download of the release manifest at Hugging Face revision `f94012872211ab6a828147bb138fbe8ab695b982` is byte-identical to the local manifest; see [HF_AUTHENTICATION.json](HF_AUTHENTICATION.json).
- Unmodified frozen verifier: **9,737/9,737 checks**, including all 172 manifested payload files and saved-record arithmetic. [Receipt](FROZEN_REANALYSIS_VERIFICATION.json).
- Unmodified Stage 2 validator rerun on the baseline: **683 checks**, excluding the two optional final-build-log checks because that run did not supply a Stage 2 build directory. Original Stage 2 report bytes were restored after retaining the new receipt. [Baseline receipt](BASELINE_STAGE2_CHECKS.json).
- Stage 2 consistency procedures adapted only for isolated Stage 3 paths/output and the Stage 2 baseline: **711 checks**, including both final logs. [Receipt](NUMERICAL_CLAIM_CONSISTENCY.json).
- Two independent readiness-condition derivations and boundary tests pass; [DERIVED_DIAGNOSTICS.json](DERIVED_DIAGNOSTICS.json).
- All original numerical table bodies are preserved. A second per-source numeric-token comparison against Stage 2 passes after explicit non-scientific normalization of stage labels, layout commands, leading zeroes, `All24` spacing and SHA-256 spelling. This complements, rather than replaces, semantic review and authenticated evidence. [Final checks](FINAL_DOCUMENT_CHECKS.json).

## Protected conclusions

| Claim | Verified interpretation retained |
|---|---|
| Retrieval | TF–IDF 301/331 and BGE-M3 313/331; known exposed catalog of 335 schemas |
| Ready-conditioned agreement | 7/265 = 2.64%; 258/265 = 97.36% fail complete agreement; 7/255 = 2.75% under Strict readiness |
| Exact20 | Query macro-average conditional on comparable schema hits; headline denominators 291, 291, 303, 320; matched semantic-IR 285, other matched methods 291 |
| Complete agreement | All required gold scalar keys, correct schema, compatible types and stored-value tolerance; constrained assignment 40/331, not the historical eligible-only 41 |
| Frozen-score controls | 49 queries, 277 scalars, 24 cells; contextual exact-slot counts 106→136 and 100→133; full tolerance remains 5/49 |
| Statistics | Retrospective 16-test main family separate from 240-test factorial family; no new family or superiority test |
| Tolerance | Inclusive 0.20 scaled by max(1, absolute reference); not uniformly 20% relative error |
| OptMATH | Micro 23,241/28,799 = 0.8070; macro 0.8135; missing corrected per-instance bootstrap inputs remain acknowledged |
| PaMOP | Aggregate objective agreement withheld; display precision and 11 comparable/2 no-reference outcomes retained |
| Relationship study | 631/651 structurally valid judgments; 197 complete cases; 119/77/1 partition; agreement is not semantic accuracy |
| D1 | 12 pairs/24 assisted-author samples; no provider predictions or independent human gold |
| Historical investigations | Legacy, proxy, restricted execution, learning, unavailable and blocked studies retain their distinct evidence status |

## Coverage and limits

[CLAIM_LOCATION_CROSSWALK_V1.json](CLAIM_LOCATION_CROSSWALK_V1.json) maps all 37 original numbered objects, 183 numerical passages and 47 investigations, plus the unchanged Stage 2 derived claim. Final source lines, hashes and PDF object pages are refreshed; no immutable claim manifest was edited. [Citation locations](CITATION_LOCATIONS.json) record the current 41 cited keys.

Every numerical passage was read with its context. Authenticating a historical summary does not recover missing predictions or independently verify the underlying experiment. Original scalar maps, full 60/269/20 manifests/outcomes, full historical ranked lists, some prompts/runtime metadata, PaMOP precision and corrected OptMATH bootstrap inputs remain unavailable. Projections cannot establish source-semantic truth. The verifier's PASS is explicitly limited to supported saved-record reanalysis.

No model inference, decoder experiment, training or scientific solver run was performed.
