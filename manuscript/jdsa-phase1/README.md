# JDSA Phase 1 evidence-preserving Results revision

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study**

[Download the complete manuscript](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase1.pdf) · [Supplement](Retrieval_Assisted_Scalar_Grounding_JDSA_Phase1_Supplement.pdf) · [LaTeX source](main.tex)

This Phase 1 revision starts from published Stage 1 commit `a3f0dde397341a41842a30ddb946e7966706351b`. It preserves the updated Abstract and Introduction verbatim, restructures Section 5 around retrieval, readiness and complete scalar agreement, and retains displaced details and additional archived investigations in the supplement. Sections 6–8 remain unchanged. No experiments or new predictions were generated. The original Stage 1 package remains in `../jdsa-stage1/`.

Review the [experimental evidence register](../../docs/jdsa_revision_phase1_20261029/EXPERIMENT_EVIDENCE_REGISTER.md), [restructuring map](../../docs/jdsa_revision_phase1_20261029/MANUSCRIPT_RESTRUCTURING_MAP.md), and [verification report](../../docs/jdsa_revision_phase1_20261029/VERIFICATION_REPORT.md). Historical SN Computer Science template comments remain; full JDSA formatting and later-section revision are deferred. This publication is for author review and is not journal submission.

The package includes the complete main and supplementary LaTeX sources, bibliography, Springer class and bibliography style, figure sources and PDFs, supporting tables, and both compiled documents. The supplement's external PDF links use the descriptive manuscript filename.

## Build

Install a TeX Live distribution with pdfLaTeX, BibTeX and the packages used by `main.tex` and `supplement.tex`, including `xr-hyper`, `algorithm`, `algorithmicx` and `algpseudocode`. From this directory run:

```sh
sh build.sh
```

The script builds in a temporary directory using pdfLaTeX, BibTeX for both documents, and repeated pdfLaTeX passes to resolve cross-document references. It writes the two descriptively named Phase 1 PDFs linked above. Auxiliary files and the intermediate `main.pdf` are not published.
