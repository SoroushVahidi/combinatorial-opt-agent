# Phase 3 scientific and editorial change report

Date: 29 October 2026. Status: completed preparation for independent author review; **not author approval or journal submission**.

## Provenance and scope

The exact authoritative Phase 2 source is `manuscript/jdsa-phase2-scientific-consolidation-20261029` at `8fbf3d073c54a4103ef06263084ea7d3fded57b3`. The isolated revision branch is `manuscript/jdsa-phase3-journal-preparation-20261029`. Its editable package is [manuscript/jdsa-phase3](../../manuscript/jdsa-phase3/README.md). The historical packages remain unchanged; this phase adds a separate review package and reports.

The complete Phase 2 article, supplement, bibliography, revision/remaining-issues/claim-verification/section-map documents, Phase 1 evidence register and source manifest were read. Relevant M3, M5, D1, common-18, relationship, extraction and feasibility records informed the preservation checks. The downstream implementation was inspected for its parameter-inventory assistance; it was not executed.

## (a) Formatting changes

- Converted the main article to the verified Springer Nature `sn-jnl` `iicol,sn-basic,Numbered` configuration. Kept the supplement single-column for detailed evidence tables. The supplied class and bibliography style are unmodified vendor files.
- Reflowed the article instead of scaling the old pages. Tables 2–4, 6, 8 and 9 span the two columns; compact tables use concise labels. Main Tables 2–4 have enlarged readable text and row spacing. Table populations, denominators, notes and data are unchanged.
- Made Algorithm 1 and the vector pipeline figure full-width. Retained all 17 algorithm steps and the explicit reference-key input. Supplied a journal-named `Fig1.pdf` and editable figure source; the vector artwork is unchanged.
- Wrapped long equations using aligned/gathered layouts, preserving their mathematical expressions. All nine main and three supplementary displayed equations are numbered. The indicator glyph is consistently bold 1 in place of a poorly rendered blackboard 1; this is a typographic correction only.
- Reflowed supplementary table columns, enlarged the compact prior-work table, and improved the D1 category wrapping and row separation. Tables S1–S26, including all 12 D1 pair summaries, remain present.
- Added appropriate URL line breaking, normal microtypography and targeted float placement. No warning suppression or wholesale font shrinking was used. Some cover/float whitespace remains in the supplement.

## (b) Editorial changes

- Preserved the seven-section organization and the Phase 2 Abstract and Introduction verbatim. The abstract remains within the required 150–250-word interval and six keywords remain.
- Replaced the Related Work catalog with an analytical comparison organized around supplied versus generated structure, available slots, evaluated outputs and endpoints. Closest work receives explicit comparability boundaries; unexecuted published methods are not described as defeated baselines.
- Expanded specialized acronyms at relevant first uses and clarified abbreviated table headings/notes. Consolidated acknowledgement wording on the title page while retaining the named people, institutions and credit/support programs.
- Identified the supplement consistently as Online Resource 1, with journal/title/author/contact metadata and a descriptive ESM filename. The current review-package availability paragraph points to Phase 3; historical Phase 2 provenance descriptions remain historical.
- Kept the evidence-qualified Discussion and Conclusion and the existing separation of formulation, execution, objective agreement, extraction, relationship agreement and semantic correctness.

## (c) Citation and declaration corrections

- Audited all 44 cited keys against primary records or explicitly qualified publisher-deposited metadata. Corrected Chain-of-Experts' author spelling to Yuan Wang; represented Gao et al. as an arXiv preprint with author-reported acceptance rather than invented proceedings metadata. Preserved known venue/year distinctions for NL4Opt and SIRL.
- Moved arXiv identifiers into BibTeX `eprint` fields where the vendor style otherwise removed decimal points. No vendor-style edit was needed. Numbered square-bracket citations remain, with at most four keys per citation command.
- Separated the corresponding email from the ORCID. Used the class's ORCID hook with a linked identifier, since the distributed class expects an icon asset not included in the official bundle; did not place ORCID text in the email field.
- Used the requested **Statements and Declarations** heading and retained/organized funding, competing interests, author contributions, ethics, consent, data/code availability, AI and software assistance, and reproducibility qualifications. Publication consent is not described as author approval of this revision.
- Kept public GitHub/Hugging Face URLs and the distinction between historical public v1.2 and the unreleased local M5 candidate. Credits and acknowledgements require the author's factual confirmation; they are not inferred evidence of use in a particular campaign.

## (d) Scientific-text clarification

No measurements, denominators, statistical outcomes, scientific result rows or negative findings were changed. Related Work now distinguishes trained and untrained binding interventions in Gao et al., and explains why published formulation or equivalence work measures different endpoints. This changes comparative explanation, not the local experimental evidence. First-use notation and captions clarify existing definitions. No material contradiction requiring alteration of a measurement was found in this phase; inherited reproducibility gaps remain explicit.

## Validation and delivery

- [Document consistency checks](DOCUMENT_CONSISTENCY_CHECKS.json): 35 tables retain rows and numerical cells; scientific numerical sequences and equation semantics are preserved; Abstract/Introduction are identical; 47 investigation destinations remain.
- [Claim-to-evidence verification](CLAIM_TO_EVIDENCE_VERIFICATION.md): checks of existing saved records only. All 65 Phase 1 manifest inputs passed their existing hashes through the Phase 2 offline verifier.
- [Build/artifact verification](BUILD_AND_ARTIFACT_VERIFICATION.json): clean builds, 22 main pages and 32 supplement pages, embedded fonts, resolved internal/cross-document links, independently built flat archive with identical extracted text and 72-dpi page pixels.
- [Visual review](VISUAL_VERIFICATION_REPORT.md): every page inspected and difficult tables/figure/algorithm inspected at higher resolution.
- [Reference verification](REFERENCE_VERIFICATION.md), [journal checklist](JDSA_COMPLIANCE_CHECKLIST.md), [47-group destination map](SECTION_RESTRUCTURING_MAP.md), and [remaining decisions](REMAINING_ISSUES.md).

The source ZIP contains nine flat editable/build assets and independently builds both documents. It contains no datasets, predictions, private review conversations, local M5 evidence, auxiliary files or credentials. It reproduces the documents, not missing historical experiments.

No experiments, inference, training, solvers or Slurm jobs were run. No historical observations were reconstructed or replaced by later caches. No Hugging Face release, main-branch merge or journal submission was performed. **Actual journal submission must wait until withdrawal of the previous overlapping submission is confirmed**, as well as author review and resolution of applicable declaration/permission decisions.
