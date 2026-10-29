> Historical publication record for `d3071ba31eaf7babdf1e7fe234d8527191ed93de`. Its old PDF paths refer to [that immutable tree](https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/d3071ba31eaf7babdf1e7fe234d8527191ed93de/manuscript/jdsa). The current corrected package is [Stage 3](../../manuscript/jdsa-stage3/README.md); the original report below is retained as historical evidence.

# Canonical JDSA publication verification

Paper: **Retrieval-Assisted Scalar Grounding for Natural-Language Optimization**
(the verified PDF retains the subtitle “A Diagnostic Study”).
Target: **International Journal of Data Science and Analytics**.
Canonical public directory: [`manuscript/jdsa/`](../../manuscript/jdsa/).

## Authority and integration

Authoritative final-review commit: `9d8c501c58415e766d0d5773cf7cbb576da66cfa`.
Fetched main baseline: `a3f0dde397341a41842a30ddb946e7966706351b`, unchanged
from the user's observed baseline. Work was prepared on the isolated branch
`manuscript/jdsa-canonical-main-20261029`, based directly on that main commit.
Only this package, its verification documentation, the two repository navigation
files, and the 25 identified Stage 1 paths are in scope. Push must be a normal
fast-forward after refreshing origin/main; no history rewrite is authorized.

## Necessary PDF changes

Simply renaming the two PDFs would break 36 cross-document links. The canonical
PDFs therefore preserve the entire original PDF byte sequence and append an
incremental update to link annotations: 26 main-to-supplement targets, 10
supplement-to-main targets, and two main-document annotations for the same
repository source URL (split across lines). The source URL now points to
`main/manuscript/jdsa/source`. PDF serialization/trailers change, but no original
page content stream changes. [PDF_LINK_CHANGES.json](PDF_LINK_CHANGES.json)
identifies every updated annotation object.

The only manuscript-source edits are the two external-document PDF names and
that repository URL, shown completely in [SOURCE_LINK_DIFF.patch](SOURCE_LINK_DIFF.patch).
Build scripts and ZIP packaging use the canonical output names. No prose,
declarations, equations, results, bibliography records, algorithms or conclusions
were edited. No experiments, model calls, paid inference, Hugging Face writes,
authorship changes or journal submission were performed.

## Verification results

- Both PDFs open and render: main **22 pages**, supplement **32 pages**.
- All **54 pages** have identical extracted text and **144-dpi rendered pixels**
  across authoritative PDFs, canonical PDFs, independent flat-ZIP build and
  modular-source build. The authoritative visual review thus remains applicable;
  no new clipping or visible change is introduced. No text extends outside pages.
- All **35 table bodies**, **12 equations**, algorithms, inline mathematics,
  citation commands, labels, section headings, **48 bibliography records**, D1
  source and associated methodological qualifications remain unchanged.
- All **47 investigation-group destinations** are retained byte-for-byte in
  [INVESTIGATION_DESTINATIONS.json](INVESTIGATION_DESTINATIONS.json).
- All **138 internal** and **36 cross-document** PDF links resolve to existing
  pages/named destinations. Other web-link targets remain unchanged except for
  the canonical source URL; this task does not re-audit third-party web availability.
- Clean independent ZIP and modular builds pass: no undefined citations/references,
  duplicate labels, missing glyphs, unresolved `??` markers or overfull boxes.
  The main retains the previously verified harmless hyperref bookmark-level
  adjustment warning. No warnings are suppressed.
- The flat ZIP extracts and compiles without a repository checkout. Its exact
  ten-file allowlist consists solely of editable manuscript/supplement sources,
  figure PDF/source, bibliography, class/style files, build script and README.
  Source members match the public final inputs (apart from documented link/build
  packaging changes); private-key/token patterns are absent. No dataset, private
  annotations, unreleased M5 evidence, credentials or unrelated research artifacts
  are included.
- Both published PDFs have descriptive filenames. No `main.pdf` is present in the
  canonical package or presented as its current paper.

Checksums and detailed results: [VERIFICATION.json](VERIFICATION.json),
[MODULAR_BUILD_VERIFICATION.json](MODULAR_BUILD_VERIFICATION.json).
The document-only checker is [verify_package.py](verify_package.py); it requires
Python with PyMuPDF and the TeX dependencies in the [build guide](../../manuscript/jdsa/BUILD.md).
The source ZIP can be regenerated with `python manuscript/jdsa/package_source.py`.
The original final review receipts are preserved verbatim under
[`docs/jdsa_final_editorial_polish_20261029/`](../jdsa_final_editorial_polish_20261029/).
Their relative paths and verification scripts describe the historical checkout;
use commit `9d8c501c58415e766d0d5773cf7cbb576da66cfa` to reproduce that earlier review.
They are historical evidence, not additional current submission packages.

## Exact deletion scope and lineage

The Stage 1 README names the same Scalar Grounding paper. Main commit `a3f0dde`
cherry-picked its publication from `67fb2b5`. The Phase 1 verification report at
`a09032a:docs/jdsa_revision_phase1_20261029/VERIFICATION_REPORT.md` explicitly names
`manuscript/jdsa-stage1/main.tex` as its predecessor. The direct revision chain is
`a3f0dde → a09032a → 8fbf3d0 → 49827d2 → cea99e8 → df4fa55 → 9d8c501`.
Every removed path belongs to that tracked 25-file Stage 1 package. All removals
used an explicit Git path list. Historical commits and branches remain intact.

| Deleted path | Reason |
|---|---|
| `manuscript/jdsa-stage1/.gitignore` | Build support for the superseded Stage 1 package; canonical build support replaces it. |
| `manuscript/jdsa-stage1/README.md` | Stage 1 README explicitly identifies this earlier JDSA revision as current. |
| `manuscript/jdsa-stage1/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf` | Compiled Stage 1 document, superseded by the final-review PDF. |
| `manuscript/jdsa-stage1/build.sh` | Build support for the superseded Stage 1 package; canonical build support replaces it. |
| `manuscript/jdsa-stage1/closed_audits.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/figures/pipeline_corrected.pdf` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/figures/pipeline_corrected.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/figures/pipeline_m1.pdf` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/figures/pipeline_m1.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/grounding_diagnostics.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/grounding_summary.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/llm_evidence.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/main.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/pipeline_figure.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/reference_conventions.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/references.bib` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/relationship_evidence.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/relationship_supplement.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/reproduction_map.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/sn-basic.bst` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/sn-jnl.cls` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/supervision_diagnostics.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/supplement.pdf` | Compiled Stage 1 document, superseded by the final-review PDF. |
| `manuscript/jdsa-stage1/supplement.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |
| `manuscript/jdsa-stage1/tables/corrected_significance_rows.tex` | Tracked Stage 1 document source or dependency; the corresponding authoritative final-review source/asset is retained in jdsa/source/. |

## Preserved materials and full change inventory

`manuscript/main.tex`, `manuscript/main.pdf`, `manuscript/submission_package/`,
`KAIS_submission_source.zip`, the KAIS cover letter, and associated root assets
are positively identified as KAIS materials by their title, journal comments,
cover letter and manuscript preparation record. They are intentionally retained,
not treated as JDSA predecessors to delete. `manuscript/sncs/` and `manuscript/dke/`
remain byte-identical. The root EAAI archive is likewise preserved. There are no
ambiguous deletions. No Phase 1–4 or final-review duplicate package existed on main
and none was imported. Only the final package is published at `manuscript/jdsa/`.

All **1,430** other baseline tracked files remain unchanged, including research
results, experiment logs, datasets, source code and historical investigation
records. Only README.md and manuscript/MANUSCRIPT_README.md change among retained
pre-existing files, to identify the current JDSA package and clearly scope the
historical journal material.

[CHANGE_INVENTORY.json](CHANGE_INVENTORY.json) lists every changed path, including
new canonical build/navigation support, relocated final documents and sources,
repackaged source archive, deleted Stage 1 files, updated navigation, and preserved
review/new validation evidence. No other changes are included.

No outstanding technical packaging issue was identified. Public remote commit,
tree and unauthenticated download verification is performed after the push and
reported with the resulting main commit.
