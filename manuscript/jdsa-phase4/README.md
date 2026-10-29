# JDSA Phase 4 — targeted corrections for independent review

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study**

- [Main manuscript PDF](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase4.pdf)
- [Online Resource 1](ESM_1_Retrieval_Assisted_Scalar_Grounding_JDSA_Phase4_Supplement.pdf)
- [Main LaTeX](main.tex), [supplement LaTeX](supplement.tex), [bibliography](references.bib)
- [Flat editable source archive](Retrieval_Assisted_Scalar_Grounding_JDSA_Phase4_Submission_Source.zip)
- [Phase 4 reports](../../docs/jdsa_revision_phase4_20261029/PHASE4_REVISION_REPORT.md)

Source: Phase 3 commit `49827d274cae995005681fb03a299a5363583861`. Revision branch: `manuscript/jdsa-phase4-final-consistency-20261029`. Earlier packages and branches remain unchanged.

## Document build

Run `sh build.sh` in this directory with TeX Live, pdfLaTeX and BibTeX installed. Standard dependencies include amsmath, graphicx, tabularx, booktabs, algorithm/algorithmicx/algpseudocode, xurl, microtype, xr-hyper, hyperref, multirow, placeins and TikZ. The original Springer class/style and LPPL orcidlink style are supplied. Builds run in fresh temporary directories and reject unresolved citations/references, missing glyphs and overfull boxes. `KEEP_BUILD=1 sh build.sh` retains build logs. Only descriptive PDFs are copied back.

The ZIP has a flat layout, with modular inputs inlined and the figure path flattened. Extract into an empty directory and run `sh build.sh` to build both complete documents. Its ten files include the main/supplement sources, bibliography, class/style, ORCID style, vector figure, editable figure source, script and instructions. Regenerate it using `python docs/jdsa_revision_phase4_20261029/package_submission.py` from the repository root. The pipeline figure need not be regenerated; its optional source uses TikZ/standalone.

Keep both PDFs together for relative cross-document links. Some browser viewers do not implement such navigation; printed section/table identifiers remain usable. The main article and supplement retain separate numbered reference lists. Unpublished work is discussed with direct source links in the text; its uncited BibTeX records are retained for provenance but not printed.

This package reproduces the documents, not missing historical predictions or execution environments. Historical public Hugging Face v1.2 remains distinct from the unreleased local M5 candidate. No experiments, model calls, dataset writes, main-branch changes or journal submission were performed. Established author information and acknowledgement wording are preserved without reopening verification requests.

Original third-party notices remain applicable. No datasets, restricted upstream corpora, private review logs or unreleased M5 evidence are included. Actual journal submission must await confirmation that the previous overlapping submission has been withdrawn. This revision awaits independent review; no subsequent phase has started.
