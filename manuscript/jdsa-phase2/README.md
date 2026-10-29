# JDSA Phase 2 scientific consolidation

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study**

[Complete manuscript PDF](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase2.pdf) · [Complete supplement PDF](Retrieval_Assisted_Scalar_Grounding_JDSA_Phase2_Supplement.pdf) · [Editable LaTeX](main.tex)

Starts exactly from Phase 1 commit `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650`. Seven main sections integrate complementary evaluation, Discussion, limitations and Conclusion. All nine original main tables and 24 supplementary tables are preserved. The Abstract and core Results are preserved; the Introduction changes only its roadmap. All 47 investigation groups retain destinations. No new experiments, inference, training, solver execution, dataset collection or Hugging Face updates were performed.

The complete package includes both LaTeX documents and inputs, bibliography, Springer class/style, figure assets and supporting tables. The historical Phase 1 package remains in `../jdsa-phase1/`. This is an author-review revision, not journal submission or a release of the local M5 evidence bundle.

[Scientific revision report](../../docs/jdsa_revision_phase2_20261029/SCIENTIFIC_REVISION_REPORT.md) · [Restructuring map](../../docs/jdsa_revision_phase2_20261029/SECTION_RESTRUCTURING_MAP.md) · [Claim verification](../../docs/jdsa_revision_phase2_20261029/CLAIM_TO_EVIDENCE_VERIFICATION.md) · [Remaining issues](../../docs/jdsa_revision_phase2_20261029/REMAINING_ISSUES.md)

## Build and verification

Install TeX Live with pdfLaTeX, BibTeX and the packages used in the sources. Run `sh build.sh` in this directory. It builds both documents in a clean temporary directory, resolves cross-document references, rejects unresolved citations/references, missing characters and overfull boxes, and writes only the two descriptive PDFs. No auxiliary files or deliverable named `main.pdf` are published. Keep both PDFs in the same directory for local cross-document links; browser PDF viewers may not support external PDF destinations.

Public saved-record checks: `python docs/jdsa_revision_phase1_20261029/verify_saved_metrics.py` and `python docs/jdsa_revision_phase2_20261029/verify_phase2.py` from the repository root. The second command explicitly skips local-only archives unless `--archive-parent` points to existing research worktrees. It does not download or regenerate missing data. The committed scientific-check report records the completed local archival check and its source hashes. Full two-column JDSA formatting remains a later phase.
