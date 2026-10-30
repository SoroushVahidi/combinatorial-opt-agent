# JDSA manuscript and editable submission package

*Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study* — Soroush Vahidi.

This is a pre-submission author-review manuscript, not an accepted journal article. Confirm that any overlapping prior submission has been formally withdrawn before submitting to a journal. No journal submission is performed by these scripts.

## Current files

- [Article PDF](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf)
- [Supplement: Online Resource 1](ESM_1_Retrieval_Assisted_Scalar_Grounding_Supplement.pdf)
- [Complete editable submission package](Retrieval_Assisted_Scalar_Grounding_JDSA_Submission_Source.zip)
- [SHA-256 checksums](SHA256SUMS.txt)
- [Versioned manuscript release v1.1.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.1.0)
- [Canonical repository directory](https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/main/manuscript/jdsa-stage3)

`main.tex` is the primary article entry point. `supplement.tex` builds the 38-page Online Resource 1. `Fig1.tex` and `pipeline_figure.tex` are editable TikZ figure sources; `Fig1.pdf` is the embedded vector figure, with embedded fonts. The remaining LaTeX inputs, bibliography, generated bibliographies, and class/style files are required source dependencies. Bundled class/style notices and licensing restrictions remain intact.

## Compile and upload

Install Python 3, pdfLaTeX/BibTeX, and the standard TeX Live packages graphicx, amsmath, amssymb, booktabs, tabularx, multirow, algorithm, algorithmicx, algpseudocode, xurl, microtype, placeins, xr-hyper, hyperref, TikZ/PGF, standalone, and orcidlink (bundled).

Run `KEEP_BUILD=1 sh build.sh` in this directory or an extracted source ZIP. It builds both documents and the figure in a fresh temporary directory, resolves cross-references, rejects serious LaTeX diagnostics, and copies the descriptive PDFs and generated bibliographies back. `KEEP_BUILD=1` retains diagnostic logs; omit it to remove the temporary build directory. Fixed timestamps permit byte-comparison between clean builds with the same toolchain. Python exports the small `main_crossrefs.tex` and `supplement_crossrefs.tex` companion-label dependencies; these allow either document to compile independently when a journal build does not generate the other document's auxiliary file. Fresh paired builds use the current auxiliary labels and refresh these exports.

Run `python3 package_source.py` after a successful build to regenerate the ZIP and checksums. The upload package is flat, as required by [JDSA's LaTeX instructions](https://link.springer.com/journal/41060/submission-guidelines). Select `main.tex` as the primary manuscript and upload the descriptive `ESM_1_...pdf` as Online Resource 1. Retain the supplementary editable sources and companion-label definitions. Do not choose `Fig1.tex` or `supplement.tex` as the main article. Keep the article and supplement PDFs together with their original names for companion-document links. Viewer support for such links varies.

The ZIP includes the matching PDF pair, vector figure, editable sources, dependencies, README, and build/package tools. It excludes temporary `.aux`/`.log` files, generic `main.pdf`, research datasets, caches, credentials, private records, and audit correspondence. Author contributions and competing interests must also be entered in the journal submission interface; the manuscript declarations are not a substitute for those fields.

## Public evidence and preservation

- [GitHub evidence release v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0)
- [Hugging Face evidence repository](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics)
- [Exact evidence revision and directory](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0)
- [Historical manuscript v1.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0)

Evidence revision `f94012872211ab6a828147bb138fbe8ab695b982`, evidence release `jdsa-evidence-v2.0.0`, and the historical manuscript release remain unchanged. The public evidence supports saved-record reanalysis, not full historical inference reproduction or independent semantic ground truth. Restricted originals, private reviews, and missing observations retain the boundaries documented in the article and supplement.

The four presentation phases preserve scientific results and qualifications from correction commit `d3583fa28fcdb87f7a446f764ca3225ab5f855d3`. Phase baselines and exact edits, numerical preservation checks, public-access checks, bibliography verification, and build results are recorded in `docs/jdsa_presentation_polish_phase1_20261029/` through `docs/jdsa_presentation_polish_phase4_20261029/` at the repository root. Historical source/PDF versions remain recoverable through Git and the unchanged releases.
