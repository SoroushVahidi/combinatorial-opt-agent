# Stage 2 scientific restructuring report

Status: **COMPLETED_SUCCESS** for manuscript restructuring and saved-record validation. This is not a claim of full experimental reproduction or independent semantic validation.

The revised Regular Paper organizes a single methodological contribution around three questions: target-schema retrieval in a fixed catalog (RQ1), failures hidden by structural readiness (RQ2), and what matched procedures versus frozen-score controls identify about grounding mechanisms (RQ3). The contribution is separating and jointly reporting retrieval, readiness, complete stored-reference scalar agreement and semantic-validation boundaries. It is not a new retrieval/matching algorithm, autonomous formulation system or independently established semantic gold standard.

## Authority and preservation

- Public main checked at start: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
- Isolated branch: `manuscript/jdsa-stage2-scientific-restructure-20261029`.
- Parent: Stage 1 report tip `3296f3bd20128077cd94a4eac82f0b5fe73e2946`.
- Frozen evidence: `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`, tag `jdsa-evidence-v2.0.0`.
- Frozen HF evidence: `f94012872211ab6a828147bb138fbe8ab695b982`.
- Original editorial source: `9d8c501c58415e766d0d5773cf7cbb576da66cfa`, preserved by the baseline publication.

Read in full: Stage 1 `FINAL_REPORT.md`; frozen `AUDIT.md`, `LIMITATIONS.md`, `claims.json`; and the baseline main/supplement with their included sources. All revisions are additive under `manuscript/jdsa-stage2/` and this report directory. Original manuscripts, release artifacts, historical observations, main and unrelated work remain untouched. No solver or model-inference experiment was run.

## Scientific organization

The Abstract, contributions, evaluation definitions, experimental setup, Results, Discussion and Conclusion now follow RQ1–RQ3. Main Tables 1–6 cover retrieval, headline grounding, new readiness-conditioned agreement, the exclusive residual partition, six matched grounding methods and the separate 49-query factorial control. Figure 1 and Algorithm 1 remain inspectable specifications.

The 331-query comparison holds query/retrieval inputs fixed but changes whole grounding procedures; it cannot identify search alone. The 49-query, 277-slot control fixes candidate pools and scores within decoder pairs. Its increased slot agreement without full-instance improvement is retained, including the separate 240-test Holm family and non-equivalence caveat. Candidate availability, pruning, scoring and selecting an incorrect optimum are separated in interpretation.

The 217-relationship study is used to demonstrate why output validity and inter-model agreement do not establish semantic accuracy. Detailed counts, provenance, witnesses and request recovery move to the supplement. The 12 author/AI-assisted D1 pairs illustrate semantic distinctions, not provider performance or independent blinded labels. Common-18, OptMATH and auxiliary positive/negative historical outcomes remain available in the supplement; the main explains their narrower evidential scope.

## New authenticated descriptive summary

Main Table 3 and Equations (9)–(10) add `STAGE2-D01`: complete all-required-key stored-reference scalar agreement conditional on ordinary or Strict readiness. The original 331-query frame remains fixed; conditional membership changes with method. Both independently implemented derivations authenticate the frozen input hashes and agree query by query. Five boundary tests check predicate behavior. The public JSON retains memberships and denominators for all ten inspected arms.

| Headline arm | Complete / ordinary ready | Complete / Strict ready |
|---|---:|---:|
| Baseline TF–IDF | 6/257 = 2.33% | 6/247 = 2.43% |
| Ratio-aware TF–IDF | 7/265 = 2.64% | 7/255 = 2.75% |
| BGE-M3 | 3/273 = 1.10% | 3/265 = 1.13% |
| Oracle | 8/281 = 2.85% | 8/281 = 2.85% |

Thus 258/265 = 97.36% of ratio-aware readiness-passing outputs fail complete stored-reference agreement. These are descriptive conditional fractions, not semantic precision, independent samples or a new corrected significance family. The main conclusion does not extrapolate them to unseen schemas.

## Novelty and retained qualifications

The related-work discussion compares structured schema/value grounding and Text2Opt-Bench's modeling/binding distinction. It explicitly avoids priority claims about discovering numerical binding failures. All 41 original citation keys remain cited across the two documents. Broader system context is consolidated in Supplement S11.

Correct conditional Exact20 populations, 40 all-key versus 41 eligible-only constrained successes, corrected OptMATH micro/macro definitions, PaMOP objective-agreement withholding, historical bootstrap procedures, separate Holm families, reference assistance and known-catalog restrictions are preserved. No newly exposed numerical inconsistency required a changed historical result. Existing unrecoverable evidence and ambiguous semantics remain limitations, rather than silently repaired records.

Availability statements now point to the exact GitHub release and HF revision, distinguishing public reanalysis, sanitized projections, historical HF v1.2, unreleased complete M5 and upstream-restricted content. See [REMAINING_SCIENTIFIC_WEAKNESSES.md](REMAINING_SCIENTIFIC_WEAKNESSES.md).

## Deliverables and evidence

- [Revised main PDF](../../manuscript/jdsa-stage2/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf) and [supplement PDF](../../manuscript/jdsa-stage2/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_Supplement.pdf).
- [Updated flat LaTeX source/build instructions](../../manuscript/jdsa-stage2/README.md).
- [Before/after section map](BEFORE_AFTER_SECTION_MAP.md), [redundancy audit](REDUNDANCY_REMOVAL_REPORT.md), [numerical consistency report](NUMERICAL_CLAIM_CONSISTENCY_REPORT.md).
- [Versioned claim crosswalk](CLAIM_LOCATION_CROSSWALK_V1.json), [compilation/validation report](COMPILATION_AND_VALIDATION_REPORT.md), [remaining weaknesses](REMAINING_SCIENTIFIC_WEAKNESSES.md).
- Final publication commit and independently downloaded artifact hashes are recorded in `PUBLICATION_VERIFICATION.json` after the artifact commit is pushed. Main is not merged; Stage 3 is not started.
