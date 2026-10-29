# JDSA Phase 3 — journal preparation for author review

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study**

- [Main manuscript PDF](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase3.pdf)
- [Online Resource 1: complete supplement](ESM_1_Retrieval_Assisted_Scalar_Grounding_JDSA_Phase3_Supplement.pdf)
- [Main source](main.tex), [supplement source](supplement.tex), [bibliography](references.bib)
- [Flat submission-source ZIP](Retrieval_Assisted_Scalar_Grounding_JDSA_Phase3_Submission_Source.zip)
- [Revision and verification reports](../../docs/jdsa_revision_phase3_20261029/SCIENTIFIC_EDITORIAL_CHANGE_REPORT.md)

Authoritative starting commit: `8fbf3d073c54a4103ef06263084ea7d3fded57b3`, Phase 2. The Phase 3 branch is `manuscript/jdsa-phase3-journal-preparation-20261029`. Earlier manuscript packages remain intact. The main article uses the official Springer Nature `sn-jnl` class with `iicol,sn-basic,Numbered`; the supplement remains single-column for table readability. This is a review package, not author approval, journal submission, or release of the local M5 evidence candidate.

## Document-only build

Install TeX Live (pdfLaTeX, BibTeX and the packages named in the sources; the checked environment was TeX Live 2023/Debian). Run `sh build.sh` here. Both documents build in a fresh temporary directory, with repeated passes resolving their cross-references. Final logs reject unresolved references/citations, missing glyphs, multiply defined labels and overfull boxes. Only descriptive PDFs are copied back. `KEEP_BUILD=1 sh build.sh` retains the temporary logs for inspection. No model, solver or experimental command is invoked.

Keep both PDFs together for external PDF links. Desktop readers can follow these relative links; GitHub/browser viewers may not implement external PDF destinations. Printed section/table references remain usable. The article and supplement have separate numbered bibliographies.

## Flat journal-upload source

The ZIP contains no subfolders. Its `main.tex` and `supplement.tex` inline all manuscript inputs; the figure path is flattened. Bibliography, original Springer class/style, vector figure, editable figure source, build script and instructions are included. Extract to an empty directory and run `sh build.sh`. Upload `main.tex` as the article source and the descriptively named ESM PDF as Online Resource 1 after author approval and completion of the submission requirements. Source filenames may be `main.tex`; deliverable PDFs are never named `main.pdf`.

The repository keeps modular editable sources and supporting assets. `python docs/jdsa_revision_phase3_20261029/package_submission.py` regenerates the ZIP. Figure regeneration, if needed, uses pdfLaTeX on `figures/Fig1.tex` (TikZ/standalone); the supplied vector PDF requires no regeneration for either document build.

The archive deliberately excludes datasets, predictions, local M5 material, private review records, logs and LaTeX auxiliary files. It reproduces the documents, not missing historical observations. Existing third-party notices remain applicable; no new blanket license is assigned to manuscript or third-party assets. Final manuscript licensing/permissions remain an author decision.

Actual submission must wait for confirmation that the previous overlapping submission has been withdrawn. That confirmation is not established by this package. The author must review the PDFs, funding/credit attribution, acknowledgements, declarations, metadata and remaining evidence-release/permission issues before submission. Hugging Face and GitHub main were not changed.
