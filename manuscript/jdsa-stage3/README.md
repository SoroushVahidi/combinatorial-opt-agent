# JDSA Stage 3: author-review submission sources

Final editorial version of *Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study*, Soroush Vahidi.

- `manuscript.pdf`: main article, 15 pages, Springer two-column layout.
- `ESM_1.pdf`: Online Resource 1, 37 pages, with linked contents.
- `main.tex` and `supplement.tex`: document entry points.
- `references.bib`, `sn-jnl.cls`, `sn-basic.bst`, `orcidlink.sty`: shared bibliography and local styles.
- `Fig1.pdf`, `Fig1.tex`, `pipeline_figure.tex`: figure and editable TikZ source.
- Other `.tex` files: included text and retained conditional-table audit snippet.

The flat `JDSA_Stage3_Submission_Source.zip` also includes the final `.bbl` files for portability. No subdirectories, private records, model outputs, evidence-release changes or unrelated repository files are needed for document compilation.

From an extracted archive, run:

```sh
KEEP_BUILD=1 sh build.sh
```

The script creates a fresh temporary build, regenerates Fig1 from TikZ, runs pdfLaTeX and BibTeX for both documents, and performs paired passes for cross-document references. It publishes the two final PDFs only after the final logs pass citation, reference, missing-character and overflow checks. Keep `manuscript.pdf` and `ESM_1.pdf` together for local PDF links. Remote named-destination links require a PDF reader that permits opening companion files; browser viewers may restrict them. Printed section/table references remain usable.

Validated with pdfTeX 1.40.25 / TeX Live 2023 (Debian), BibTeX 0.99d. A standard full TeX Live installation supplies graphicx, amsmath, amssymb, booktabs, tabularx, multirow, algorithm, algorithmicx, algpseudocode, xurl, microtype, placeins, xr-hyper, hyperref, TikZ/PGF and standalone. The bundled class, BST and ORCID style retain their original licenses. These standard distribution packages are not vendored.

For the journal's source upload, use the flat files, identify `main.tex` as the primary manuscript, and designate `ESM_1.pdf` as Online Resource 1. Keep `supplement.tex` and its included files available as editable supplementary sources; do not select `Fig1.tex` as an article entry point. Supply this caption: “Online Resource 1: Secondary results, statistical and recovery protocols, evidence provenance, and twelve author-reviewed semantic contrast pairs.” No submission has been performed.

Baseline: `60785b0d144365413dd3b44b577bff901043ddd7` on `manuscript/jdsa-stage2-scientific-restructure-20261029`. Stage 3 branch: `manuscript/jdsa-stage3-submission-audit-20261029`. Immutable evidence: `jdsa-evidence-v2.0.0`; Hugging Face revision `f94012872211ab6a828147bb138fbe8ab695b982`.

[Final audit and author-review recommendation](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/manuscript/jdsa-stage3-submission-audit-20261029/docs/jdsa_stage3_20261029/FINAL_REPORT.md). The reports and crosswalk accompany the repository version, not the journal source ZIP. Saved-record verification does not imply original experimental reproduction or independent semantic validation.
