# Stage 3 editorial-change report

The complete main article, all included supplementary text, equations, captions, tables, figure source and all 48 bibliography entries were reviewed line by line against Stage 2 commit `60785b0d144365413dd3b44b577bff901043ddd7`. Changes are confined to `manuscript/jdsa-stage3` and this report directory. All other tracked content is unchanged.

The final article remains a focused RQ1–RQ3 diagnostic study. No section was substantially restructured, no secondary study was deleted, no experimental result changed, and no new inference or solver experiment was performed. The original sources and immutable evidence package remain intact.

## Changes and rationale

67 recorded editorial/formatting operations are detailed in [EDITORIAL_CHANGES.json](EDITORIAL_CHANGES.json). Exact source differences are in [EDITORIAL_DIFF.patch](EDITORIAL_DIFF.patch); bibliography corrections are in [REFERENCE_CHANGES.json](REFERENCE_CHANGES.json).

- Corrected grammar, missing spaces, compressed slash constructions and opaque wording, including “All24”, calibration wording and duplicate-mapping terminology.
- Standardized TF–IDF labels, defined intermediate representation before IR, expanded the abstract's artificial-intelligence abbreviation, and used one complete-agreement symbol, $F^G$. The conditional table now explicitly uses set cardinalities.
- Corrected equation punctuation and distinguished the factorial's four endpoints from its 240-test family. No predicate, tolerance or denominator changed.
- Repaired the selective-reranking and historical availability cross-references, replaced an inaccurate “Appendix” designation with “Section”, and supplied explicit callouts for every numbered table, figure and algorithm.
- Added linked contents to Online Resource 1, including correctly sized S10/S11 number columns. Retained separate S-prefixed sections, tables and equations. Renamed the distributed supplement `ESM_1.pdf` and updated both directions of companion-PDF links.
- Filled PDF title, author, subject, keywords and language metadata. Retained the Springer class, main two-column layout, font sizes and margins. The supplementary reference-list gap is 0.8 em instead of 1 em, preventing a two-line final-page spillover.
- Removed redundant publisher URLs from 20 bibliography entries that already print full DOI links. Primary-source verification URLs remain in the report ledger and unchanged Stage 2 bibliography. Corrected names/diacritics, three proceedings titles and a verified page range.
- Split the dense OptMATH extraction discussion into sampling, raw metrics, calibrated micro/macro metrics and automated-validation paragraphs. Removed a redundant configuration enumeration already fully preserved in Table S23; this also corrected stretched justification.
- Updated manuscript/code and crosswalk links to the isolated Stage 3 branch. The evidence-release URLs and revision identifiers are unchanged.

## Narrative review

The Introduction states the measurement gap, the assisted fixed-catalog setting, three research questions and three bounded contributions. It explicitly declines novelty as a retrieval/matching algorithm and acknowledges prior modeling/binding work. Results 5.1–5.3 answer their corresponding research questions. The Discussion connects the endpoint discrepancy to attribution and validation requirements, retaining the semantic boundary. The two-paragraph Conclusion is concise and qualified; it was retained apart from a small wording review, rather than rewritten.

The reviewed text consistently distinguishes retrieval from grounding; readiness from complete scalar agreement; numerical agreement from semantic fidelity; whole-procedure comparisons from fixed-score decoder controls; and reanalysis from original experimental reproduction. No claim of autonomous formulation, unseen-schema generalization, adjusted model-ranking superiority or independent semantic ground truth was introduced.

## Coverage

All 19 Stage 2 `.tex` files have corresponding Stage 3 files. `pipeline_figure.tex`, `Fig1.tex`, `ready_conditioned_rows.tex`, `common18_overview.tex`, `reference_conventions.tex`, `relationship_evidence.tex`, `d1_contrasts.tex` and `related_context.tex` were inspected even where no prose change was warranted. All 37 original numbered objects, 183 numerical passages and 47 investigation IDs remain mapped; the Stage 2 conditional-agreement claim is retained as `STAGE2-D01`.

Author review remains required for the final wording, authorship/contact information and declarations. This report does not certify that the author has approved this new revision.
