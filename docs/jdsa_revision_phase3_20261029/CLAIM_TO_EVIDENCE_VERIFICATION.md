# Phase 3 claim-to-evidence and preservation verification

This phase checks existing records and document preservation. It does not rerun experiments or infer missing observations. The authoritative scientific audit remains the [Phase 2 claim-to-evidence report](../jdsa_revision_phase2_20261029/CLAIM_TO_EVIDENCE_VERIFICATION.md), read together with the [Phase 1 register](../jdsa_revision_phase1_20261029/EXPERIMENT_EVIDENCE_REGISTER.md) and its `EVIDENCE_SOURCE_MANIFEST.csv`.

The existing Phase 2 offline verifier passed all 65 input-file hashes and its saved-record checks for extraction, common-18, relationships, D1 and preservation. Its generated tracked report was unchanged. Only document builds, metadata reads and deterministic checks of saved values were executed. The additional [Phase 3 checker](verify_phase3.py) compares the exact Phase 2 manuscript with the new package; its [output](DOCUMENT_CONSISTENCY_CHECKS.json) records all 35 tables, numerical sequences, equation semantics, unchanged Abstract/Introduction, keyword count and 47 destinations.

| Claim / endpoint | Preserved evidence and qualification |
|---|---|
| Schema retrieval | TF–IDF 301/331 and BGE-M3 313/331 in the 335-schema catalog; paired retrospective testing remains explicitly qualified. |
| Coarse versus complete agreement | TF–IDF ratio-aware 265/331 coarse-ready, 255/331 strict-ready and 7/331 full stored-reference scalar tolerance. These endpoints are not interchangeable. |
| Matched grounding | Full-success counts 7, 40, 37, 35, 46, 9 on the common 331-query frame; candidate construction, scoring, context, abstention and decoding can differ. Conditional Exact20 denominators remain 291 except semantic-IR 285. |
| Error partition | Ratio-aware mutually exclusive counts 30 + 31 + 49 + 214 + 7 = 331. Numerical failures dominate covered/type-compatible cases. |
| Controlled assignment | 49 queries/277 slots, fixed-score comparisons and 24 factorial cells; slot improvements do not become invented whole-query gains. Five of 49 full successes remains unchanged in the stated context comparison; the 240-test multiplicity qualification remains. |
| Joint/beam/exact diagnostics | Distinct 50- and 49-query populations and candidate-retention/selection endpoints remain separate. No decoder is inferred globally superior from archived finite controls. |
| Common-18 | OptMATH agreement 11/15 and GPT-5.4 12/16 use different valid denominators. PaMOP aggregate withheld for precision; ORLM execution unavailable. Historical terminal-objective correction is not new execution or semantic validation. |
| Relationship judgments | 217 targets per service, 651 total; 500 original valid + 59 offline repairs + 72 historically generated valid replacements = 631 final valid. 197 complete cases, Fleiss' kappa about 0.371. Recovery stages are retained and no independent human reference is claimed. |
| D1 | 12 pairs, 24 samples, eight categories; ten meaning-changing and two paraphrase pairs. All 12 summaries remain in S25–S26. Author-validated/LM-assisted construction; no provider-performance experiment. |
| External extraction | Raw 23805/31736 and text-supported 23805/29322; 591 complete instances. Corrected 23241/28799 and macro 0.8135 remain summary-supported, with the missing per-instance records/CI reconstruction boundary. The 150-case classifier check is distinct from the 1000-case denominator. |
| Feasibility and historical findings | G1–G3 remain methodological/identifiability audits. Archived negative, partial, blocked and unattempted studies retain their status; 47 groups are not described as 47 completed performance experiments. |

## Scope of the comparison

All original table rows, data cells and labels survive. Main scientific Sections 3–7 preserve their numerical-token sequence after removal of layout syntax and supplementary-resource designations. Equation expressions are compared while allowing line wrapping and the consistent indicator glyph. Supplementary inputs preserve quantitative sequences, allowing only explicit column widths/row-spacing changes. These mechanical checks support, but do not replace, the manual scientific and visual reading.

The preserved source evidence does not authenticate missing historical observations, new statistical superiority, autonomous formulation, human semantic correctness or public release of M5. No material measurement contradiction was silently corrected. See [remaining issues](REMAINING_ISSUES.md) for the inherited limitations and [reference verification](REFERENCE_VERIFICATION.md) for changes to prior-work attribution.
