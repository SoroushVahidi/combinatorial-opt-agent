# Manuscript v1.0.0: editable submission sources

*Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study*, Soroush Vahidi.

This is a public pre-submission research manuscript. It does not imply journal acceptance or publication, and no journal submission was performed by this release task.

## Permanent research artifacts

- [Public project repository](https://github.com/SoroushVahidi/combinatorial-opt-agent).
- [Immutable verification code and evidence release, jdsa-evidence-v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0).
- [Exact-version Hugging Face evidence](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0).
- [Manuscript v1.0.0 release: article, Online Resource 1, editable sources and checksums](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0). Its release notes link the versioned audit and claim-location crosswalk.

Saved-record reanalysis does not reproduce the original inference experiments or establish independent semantic ground truth. Restricted/gated source corpora, raw provider conversations, private reviews and missing historical records are not supplied. Author-owned evidence licensing does not relicense third-party content. See the unchanged availability and reproducibility limitations in both PDFs.

## Source contents and clean build

`main.tex` and `supplement.tex` are the document entry points. The included `.tex` files, `references.bib`, both `.bbl` files, `sn-jnl.cls`, `sn-basic.bst`, `orcidlink.sty`, `Fig1.pdf`, and its editable `Fig1.tex`/`pipeline_figure.tex` sources are supplied in a flat archive. The bundled class/styles retain their original licenses.

Run from an extracted archive:

```sh
KEEP_BUILD=1 sh build.sh
```

The script builds in a fresh temporary directory, regenerates the figure, runs pdfLaTeX/BibTeX with paired cross-document passes, and publishes `manuscript.pdf` and `ESM_1.pdf` only after citation/reference/missing-character/overflow checks pass. Standard TeX Live packages are dependencies, not vendored source: graphicx, amsmath, amssymb, booktabs, tabularx, multirow, algorithm, algorithmicx, algpseudocode, xurl, microtype, placeins, xr-hyper, hyperref, TikZ/PGF and standalone. Validation used pdfTeX 1.40.25 / TeX Live 2023 (Debian) and BibTeX 0.99d.

The public release uses these descriptive names for the same PDF bytes:

- `manuscript.pdf` → `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf`.
- `ESM_1.pdf` → `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_Supplement.pdf`.

Cross-document PDF links use the permanent online asset URLs, so they do not depend on local filenames or a temporary branch. Opening them requires network access and a PDF viewer that supports named destinations; printed section/table references remain usable offline. Internal links and contents remain local.

For the journal's flat source upload, identify `main.tex` as the primary manuscript and `ESM_1.pdf` as Online Resource 1. Retain the editable supplementary sources and do not select `Fig1.tex` as an article entry point. Suggested supplementary caption: “Online Resource 1: Secondary results, statistical and recovery protocols, evidence provenance, and twelve author-reviewed semantic contrast pairs.” Review the submission system's generated proof before submission.

Exact prior Stage 3 baseline: `e12c413c18e6d1f068d29509e7f9108e7ef51f65`. The prior version remains available at that commit. No scientific result, numerical value, statistical test or evidence-release payload was changed in this link finalization. No DOI is assigned or invented.
