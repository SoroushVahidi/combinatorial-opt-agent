# Stage 3 editorial-change report

**Completed final editorial pass; no new scientific restructuring.** Baseline `60785b0d144365413dd3b44b577bff901043ddd7`; new sources are in `manuscript/jdsa-stage3/`. The Stage 2 source tree and all published evidence remain unchanged.

## Scope and method

Read the complete main article, supplement and every included text file line by line. Checked prose against the frozen claim map and Stage 2 narrative. Reviewed every rendered page after compilation. The six main tables, thirty supplementary tables, one figure and one algorithm are retained; no empirical result or qualification was removed.

The machine-readable [edit ledger](EDITORIAL_CHANGES.json) records **92 targeted editorial/formatting edits** with exact before/after text and rationale. [Reference corrections](REFERENCE_CORRECTIONS.json) separately record **19 bibliography field edits**. These are edit operations, not a purported count of all changed words. Packaging and generated validation receipts are described separately.

## Editorial decisions

- Preserved RQ1–RQ3, section order and the distinction between the 331-query comparison and 49-query controls. The Introduction states the gap and methodological contribution; Results answer each RQ; Discussion interprets the diagnostic limits; Conclusion remains focused.
- Tightened transitions and redundant modifiers, improved grammatical constructions, and expanded abbreviations where first required. Standardized TF–IDF and explicit reference-label terminology. No arbitrary length target or new prose expansion was imposed.
- Clarified readiness, schema-gated readiness, complete scalar agreement and semantic fidelity without equating them. The Abstract contains 225 words and preserves the diagnostic findings and reconstruction caveats.
- Distinguished the selected mention symbol `tau_j*` from the query type score `t_i`; Equation (11) and Algorithm 1 now agree. Other scientific equations and numerical table cells are preserved. Corrected terminal equation punctuation and made Table 3's conditional-population headers explicit.
- Replaced two inaccurate supplementary destinations: selective-reranking history now points to Section S8.4, and the historical availability discussion for DeepOR/OR-R1 points to Section S5. Their original links landed in insufficient main-text summaries.
- Identified the supplement as Online Resource 1, added linked contents with adequate S10/S11 number spacing, aligned its affiliation with the article, and used `ESM_1.pdf` for source and PDF cross-links.
- Added descriptive PDF metadata, document language and numbered bookmarks. Removed eight redundant printed publisher URLs where a full DOI already identifies the same article. This also removed an almost empty bibliography spill page; the main article is now 15 pages.

## Traceability and unchanged scientific scope

No main/supplement table or equation numbering changed from Stage 2. PDF page positions and source lines did change; [the versioned crosswalk](CLAIM_LOCATION_CROSSWALK_V1.md) records both versions. The 25 main and 23 supplementary bibliography entries retain their compiled numbers; 41 distinct cited keys are unchanged. Historical evidence IDs are never renumbered in the release.

All local originals, historical versions, unexecuted studies and negative findings are preserved. No autonomous formulation, unseen-schema generalization, model-ranking superiority or independently established semantic labels were introduced. The 92-entry ledger includes small follow-up corrections made during the final render review, so its exact operations provide a more useful audit than a narrative word-reduction claim.

## File-level edit ledger

| Source | Recorded operations |
|---|---:|
| `manuscript/jdsa-stage3/additional_evidence.tex` | 5 |
| `manuscript/jdsa-stage3/common18_overview.tex` | 1 |
| `manuscript/jdsa-stage3/d1_contrasts.tex` | 1 |
| `manuscript/jdsa-stage3/grounding_diagnostics.tex` | 3 |
| `manuscript/jdsa-stage3/llm_evidence.tex` | 3 |
| `manuscript/jdsa-stage3/main.tex` | 50 |
| `manuscript/jdsa-stage3/preserved_results_details.tex` | 2 |
| `manuscript/jdsa-stage3/relationship_supplement.tex` | 5 |
| `manuscript/jdsa-stage3/reproduction_map.tex` | 4 |
| `manuscript/jdsa-stage3/supervision_diagnostics.tex` | 2 |
| `manuscript/jdsa-stage3/supplement.tex` | 12 |
| `manuscript/jdsa-stage3/supporting_core.tex` | 4 |

The complete before/after records are in the linked JSON rather than duplicated here. The source ZIP additionally includes compiled bibliography and auxiliary files for journal systems that process only the article entry point; its clean build regenerates those files.

Routine final trailing-space/EOF cleanup is excluded from the 92 targeted editorial operations. It changes no rendered content; the final clean-package build and crosswalk use the cleaned sources.
