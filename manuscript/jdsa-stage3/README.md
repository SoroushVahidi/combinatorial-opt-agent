# JDSA manuscript and editable submission package

*Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study* — Soroush Vahidi.

This is a pre-submission author-review manuscript, not an accepted journal article. Confirm that any overlapping prior submission has been formally withdrawn before submitting to a journal. No journal submission is performed by these scripts.

## Current files

- [Article PDF](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf)
- [Supplement: Online Resource 1](ESM_1_Retrieval_Assisted_Scalar_Grounding_Supplement.pdf)
- [Portal-ready main-manuscript LaTeX ZIP](Retrieval_Assisted_Scalar_Grounding_JDSA_Manuscript_LaTeX_Submission.zip)
- [Portal ZIP file manifest and validation](JDSA_Manuscript_LaTeX_Submission_Manifest.json)
- [Complete research/source archive (article and supplement)](Retrieval_Assisted_Scalar_Grounding_JDSA_Submission_Source.zip)
- [SHA-256 checksums](SHA256SUMS.txt)
- [Versioned manuscript release v1.1.1](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.1.1)
- [Canonical repository directory](https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/main/manuscript/jdsa-stage3)

`main.tex` is the primary article entry point. `supplement.tex` builds the 40-page Online Resource 1. `Fig1.tex` and `pipeline_figure.tex` are editable TikZ figure sources; `Fig1.pdf` is the embedded vector figure, with embedded fonts. The remaining LaTeX inputs, bibliography, generated bibliographies, and class/style files are required source dependencies. Bundled class/style notices and licensing restrictions remain intact.

## Springer Nature JDSA manuscript upload

Upload **`Retrieval_Assisted_Scalar_Grounding_JDSA_Manuscript_LaTeX_Submission.zip`** as the portal's **Manuscript file**, selecting **`main.tex`** as the entry point if prompted. The nine-file, 114,396-byte ZIP has a flat layout, following [JDSA's LaTeX source-upload instructions](https://link.springer.com/journal/41060/submission-guidelines). Its SHA-256 is:

```text
205b2b6f7d27f8040701c34e2d1efc31a29786204f2233aed16df29ecb87e21d
```

This ZIP contains the unchanged v1.1.1 article source, Figure 1's vector PDF and editable TikZ fragment, bibliography database and generated bibliography, Springer class/bibliography style, ORCID style, and static supplementary cross-reference definitions. All tables and Algorithm 1 are editable within `main.tex`. Only `main.tex` declares a document; `pipeline_figure.tex` is an editable figure fragment, not another manuscript entry point. No article PDF, supplementary PDF, supplementary document sources, temporary build outputs, datasets, or project documentation are included. `supplement_crossrefs.tex` is the exact companion-label dependency needed for independent article compilation, not a supplementary manuscript.

Upload **`ESM_1_Retrieval_Assisted_Scalar_Grounding_Supplement.pdf`** separately as supplementary material / Online Resource 1. This unchanged 40-page PDF is not needed to compile the main manuscript. Keep it under its original filename when viewing companion-document links; viewer support varies. The canonical 17-page article PDF remains available separately if the portal requests a compiled PDF in addition to sources. Author contributions and competing interests must also be entered in the journal submission interface.

The existing **`Retrieval_Assisted_Scalar_Grounding_JDSA_Submission_Source.zip`** is the complete research/source archive for release v1.1.1. It includes both documents, their PDFs and editable sources, figure sources, README, and build/package tools. It is preserved unchanged and is distinct from the new manuscript-only portal upload ZIP. No new manuscript version or release is created by this packaging update.

### Verify the manuscript-only ZIP

Extract it into a new empty directory and run the following there, using installed pdfLaTeX/BibTeX and standard TeX Live packages:

```sh
export SOURCE_DATE_EPOCH=1793313920 FORCE_SOURCE_DATE=1
export TEXINPUTS=.: BIBINPUTS=.: BSTINPUTS=.:
pdflatex -no-shell-escape -recorder -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -no-shell-escape -recorder -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -recorder -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -recorder -interaction=nonstopmode -halt-on-error main.tex
```

No network downloads or shell escape are needed. Clean-extraction validation with pdfTeX 1.40.25 (TeX Live 2023/Debian) reproduced the canonical 17-page v1.1.1 PDF byte for byte, including all 140 numerical table cells and 15 bold highlights. The final LaTeX log had no warnings, unresolved references, missing citations, or overfull boxes; 57 inherited underfull notices do not affect content. Every rendered page was reviewed. The [manifest](JDSA_Manuscript_LaTeX_Submission_Manifest.json) records each file's size and checksum, compilation commands, dependency checks, and validation results. The temporary build's `main.pdf` is not part of the upload ZIP or committed deliverables.

### Rebuild the complete article and supplement

For the complete research/source archive, install Python 3, pdfLaTeX/BibTeX, and the standard TeX Live packages graphicx, amsmath, amssymb, booktabs, tabularx, multirow, algorithm, algorithmicx, algpseudocode, xurl, microtype, placeins, xr-hyper, hyperref, TikZ/PGF, standalone, and orcidlink (bundled).

Run `KEEP_BUILD=1 sh build.sh` in this repository directory or the extracted **complete** source archive. It builds both documents and the figure in a fresh temporary directory, resolves cross-references, rejects serious LaTeX diagnostics, and copies the descriptive PDFs and generated bibliographies back. `KEEP_BUILD=1` retains diagnostic logs; omit it to remove the temporary build directory. Fixed timestamps permit byte-comparison between clean builds with the same toolchain. Python exports `main_crossrefs.tex` and `supplement_crossrefs.tex` to support independent compilation. Fresh paired builds use the current auxiliary labels and refresh these exports.

`package_source.py` packages the complete research/source archive, not the manuscript-only portal ZIP. It rewrites `SHA256SUMS.txt` for its own outputs; preserve the portal ZIP and manifest checksum entries if using it for a later revision. Existing versioned releases are immutable.

## Public evidence and preservation

- [GitHub evidence release v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0)
- [Hugging Face evidence repository](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics)
- [Exact evidence revision and directory](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0)
- [Historical manuscript v1.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0)

Evidence revision `f94012872211ab6a828147bb138fbe8ab695b982`, evidence release `jdsa-evidence-v2.0.0`, and the historical manuscript release remain unchanged. The public evidence supports saved-record reanalysis, not full historical inference reproduction or independent semantic ground truth. Restricted originals, private reviews, and missing observations retain the boundaries documented in the article and supplement.

The four presentation phases preserve scientific results and qualifications from correction commit `d3583fa28fcdb87f7a446f764ca3225ab5f855d3`. Phase baselines and exact edits, numerical preservation checks, public-access checks, bibliography verification, and build results are recorded in `docs/jdsa_presentation_polish_phase1_20261029/` through `docs/jdsa_presentation_polish_phase4_20261029/` at the repository root. Historical source/PDF versions remain recoverable through Git and the unchanged releases.

The terminology-accessibility revision starts from manuscript release v1.1.0 (`f2b8a94ea01922618e36b667c1d9b5e2c08f893c`). Its complete terminology register, before/after locations, local change log, source checks, and validation report are in `docs/jdsa_terminology_accessibility_20261029/` at the repository root. The prior four editorial phases and all immutable evidence remain preserved.
