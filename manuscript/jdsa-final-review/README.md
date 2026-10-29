# JDSA final editorial-polish package

Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study

- [Main manuscript PDF](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Final_Review.pdf)
- [Online Resource 1 PDF](ESM_1_Retrieval_Assisted_Scalar_Grounding_JDSA_Final_Review_Supplement.pdf)
- [Flat editable-source ZIP](Retrieval_Assisted_Scalar_Grounding_JDSA_Final_Review_Submission_Source.zip)
- [Exact editorial changes and verification](../../docs/jdsa_final_editorial_polish_20261029/FINAL_EDITORIAL_POLISH_REPORT.md)

Source: Phase 4 commit `cea99e86d33ad83219b9bf711c310e929ae18409`.
Branch: `manuscript/jdsa-final-editorial-polish-20261029`.

Run `sh build.sh` here with TeX Live, pdfLaTeX and BibTeX. The script builds
both documents in a clean temporary directory and rejects unresolved citations,
references, missing glyphs and overfull boxes. Use `KEEP_BUILD=1` to retain logs.
Standard dependencies and original Springer class/style are unchanged from Phase 4.
The supplied ORCID style and figure retain their original notices.

The flat ZIP contains all editable document inputs and the vector figure.
Extract it into an empty directory and run `sh build.sh` to compile independently.
Regenerate the archive from the repository root with
`python docs/jdsa_final_editorial_polish_20261029/package_submission.py`.
Keep both PDFs together for relative cross-document links; some browser viewers
do not implement those links, so printed section and table references are retained.

Historical public Hugging Face v1.2 remains distinct from local, unreleased M5
and M3 evidence. This package reproduces the documents, not missing experiments,
original scalar mappings, precision, prompts or environments. No dataset,
private annotation conversations or restricted upstream corpus is included.

No journal submission is made. The previous overlapping SN Computer Science
submission must be formally withdrawn before any new journal submission.
