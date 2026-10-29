# JDSA Stage 1 manuscript

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study**

[Download the complete manuscript](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf) · [Supplement](supplement.pdf) · [LaTeX source](main.tex)

This is the latest Stage 1 revision for the International Journal of Data Science and Analytics (JDSA), with the updated Abstract and Introduction. The remaining scientific content, bibliography, figures and tables are preserved from the verified predecessor. Historical SN Computer Science template comments remain in the source. This repository publication is for author review and does not constitute journal submission.

The package includes the complete main and supplementary LaTeX sources, bibliography, Springer class and bibliography style, figure sources and PDFs, supporting tables, and both compiled documents. The supplement's external PDF links use the descriptive manuscript filename.

## Build

Install a TeX Live distribution with pdfLaTeX, BibTeX and the packages used by `main.tex` and `supplement.tex`, including `xr-hyper`, `algorithm`, `algorithmicx` and `algpseudocode`. From this directory run:

```sh
sh build.sh
```

The script builds in a temporary directory using pdfLaTeX, BibTeX for both documents, and repeated pdfLaTeX passes to resolve the cross-document references. It writes `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf` and `supplement.pdf` here. Auxiliary files and the intermediate `main.pdf` are not published.
