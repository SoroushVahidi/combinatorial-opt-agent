# Current corrected JDSA manuscript

*Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study*, Soroush Vahidi.

This is a public pre-submission author-review version, not an accepted or published journal article.
Scientific sources are taken from correction commit `d3583fa28fcdb87f7a446f764ca3225ab5f855d3`. Integration changes only the current-source link and companion PDF filename; findings, tables, citations and methodological qualifications are preserved.

- [Current manuscript](Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf)
- [Online Resource 1](ESM_1.pdf)
- [Complete editable sources](JDSA_Stage3_Submission_Source.zip)
- [SHA-256 checksums](SHA256SUMS.txt)
- [Scientific correction note](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/d3583fa28fcdb87f7a446f764ca3225ab5f855d3/docs/jdsa_attribution_correction_v1/CORRECTION_NOTE_V1.md)
- [Integration validation](../../docs/jdsa_main_finalization_20261029/FINALIZATION_REPORT.md)

Phase 1 presentation revision (2026-10-29): branch `manuscript/jdsa-presentation-polish-20261029`, based on verified main `1e90184923c753ed18652951f2037525fffad383`. Visible external links, bibliography presentation and the Newark postal-address order are polished; scientific content is unchanged. The PDF pair, generated bibliographies, source ZIP and checksums match these sources. This branch awaits Phase 4 integration into main. Validation and the exact change inventory are in `docs/jdsa_presentation_polish_phase1_20261029/` at the repository root.

Phase 2 editorial revision (2026-10-29): continues the same presentation-polish branch from Phase 1 commit `ae9eb9c14c53c687dafe20d9d774a419d198424d`. The abstract and main-text prose are polished, with every substantive rewrite logged in `docs/jdsa_presentation_polish_phase2_20261029/EDITORIAL_LOG.json`. Scientific definitions, results, research questions, tables, captions, bibliography, public destinations, acknowledgments and declarations remain unchanged. Phase 3 presentation work and Phase 4 integration are pending.

Phase 3 presentation revision (2026-10-29): continues from Phase 2 commit `70ff74c79628bdf5df2ae7e7375a0a0c3c6be240`. Tables use consistent descriptive highlighting, clearer captions and notes, and aligned numerical columns. Figure 1 has uniform node widths and improved spacing; Algorithm 1 has a concise caption, with its existing numbering and indentation retained. Every numerical table entry, algorithm operation, equation, and surrounding scientific paragraph is preserved. Online Resource 1 remains byte-identical to Phase 2. The numerical and rendered-font checks, change inventory, and independent source-package build results are recorded in `docs/jdsa_presentation_polish_phase3_20261029/`. Phase 4 review and integration remain pending.

## Clean build and source upload

Install pdfLaTeX/BibTeX and standard TeX Live packages: graphicx, amsmath, amssymb, booktabs, tabularx, multirow, algorithm, algorithmicx, algpseudocode, xurl, microtype, placeins, xr-hyper, hyperref, TikZ/PGF and standalone.
Run `KEEP_BUILD=1 sh build.sh` here or in an extracted source ZIP. The build regenerates the figure and bibliography in a fresh temporary directory and checks final logs before copying the named manuscript and `ESM_1.pdf`. Fixed timestamps permit byte-comparison with an independent build using the same TeX toolchain. `main.pdf` is a temporary intermediate only.

The ZIP is flat and self-contained: all LaTeX inputs, bibliography and generated bibliographies, class/style files, editable figure and figure PDF, build/package scripts, this README, and the matching PDF pair. Run `python3 package_source.py` after a successful build to regenerate it and `SHA256SUMS.txt`. For journal source upload select `main.tex` as the primary entry point, and `ESM_1.pdf` as Online Resource 1. Keep supplementary sources; do not select `Fig1.tex` as an article entry point. No experimental datasets or private records are included.

Download both PDFs with their original filenames into one directory for companion links. PDF-viewer support for remote-document links varies. Internal links and printed references remain usable.

## Immutable historical artifacts

[Manuscript v1.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0) is an earlier immutable manuscript release, not the current corrected version. Its PDFs and source ZIP are preserved on that release. [Evidence v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0) and [its exact Hugging Face revision](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0) are unchanged.

Saved-record reanalysis does not reproduce all original experiments or establish independent semantic ground truth. Gated/restricted originals, private reviews and missing historical records remain unavailable as documented. Bundled class/style notices and upstream licensing restrictions are preserved. No DOI, journal submission or new release is created by this integration.
