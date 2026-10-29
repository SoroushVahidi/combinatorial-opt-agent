# Correction v1: validation and author-review status

Result: **PASS for the bounded correction; author review remains required before submission.** This is not a new manuscript release.

## Source and scientific preservation

- Exact baseline: `00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc`; corrected sources are `manuscript/jdsa-stage3/`.
- Only the enumerated attribution, terminology, literature-positioning, provenance and companion-link replacements are permitted. The validator reconstructs each changed scientific source from the exact baseline plus `EXACT_SOURCE_CHANGES.json`.
- All main/supplement table bodies are byte-identical except the explicitly authorized `P@5` → `Hit@5` heading. Main-document scientific numeral sequence and all source labels are preserved.
- Core/matched gates were checked using the existing independent set and row-indicator functions, without calling their file-writing main routine. No original evaluator or expected numerical value changed.
- Protected results include 7/331, 7/265, 7/255, the 30+31+49+214+7 partition, 7/40/37/35/46/9 matched counts, and constrained eligible-only 41 versus full-key 40. Frozen checks also cover conditional Exact20 populations and the separate statistical correction families.
- Original oracle candidate CSV, oracle CSV, summary and implementation were read from commit `2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd`. No diagnostic, retriever, grounder or solver was executed.
- Historical source packages, release PDFs, `.bbl` files, `manuscript/jdsa/source/`, scientific results, datasets, code/evaluators and the entire evidence directory are unchanged against the authoritative baseline.

## Commands and results

```sh
KEEP_BUILD=1 OUTPUT_DIR="$PWD/manuscript/jdsa-attribution-correction-v1" sh manuscript/jdsa-stage3/build.sh
python -I reproducibility/jdsa-v2.0.0/verify.py --output /tmp/jdsa-attribution-correction-v1-frozen.json
python -B docs/jdsa_attribution_correction_v1/validate_correction.py /tmp/jdsa-build.YFzxBq
python -B docs/jdsa_attribution_correction_v1/build_location_overlay.py /tmp/jdsa-build.YFzxBq
git diff --check
```

The build creates a fresh temporary directory and runs pdfLaTeX/BibTeX with paired cross-document passes. Final retained build directory: `/tmp/jdsa-build.YFzxBq`. No old bibliography output is used. Environment: Python 3.12, NumPy 2.3.5, pdfTeX 1.40.25 / TeX Live 2023 (Debian).

- Unchanged frozen verifier: **9,737/9,737 PASS**. Its report hash is in `VALIDATION_RESULTS.json`; no release payload was written.
- Correction validator: **402/402 PASS**. This adds source-diff, preserved-table, oracle-identity, existing-metric-function, PDF link/font/bounds and page-number checks. It does not redefine scientific expectations.
- Archived text-log copies have only trailing whitespace removed; their substantive output is unchanged. Original logs remain in the temporary build directory.
- Final logs: zero undefined citations/references, missing characters, multiply-defined labels or overfull boxes; no BibTeX warnings/errors.
- 31 corrected companion destinations resolve in the matching PDF pair. All inspected internal named destinations resolve. The former online-URL destination override was removed for standard local companion links; this corrected a double-encoded destination observed in the first draft.
- All PDF fonts embedded; no extracted text outside page bounds; numbered pages present throughout. Visual review covers main pp.2,3,11,14,15,16 and supplement pp.10,26,28,29. Other pages received programmatic checks, not a claimed new visual review.

| PDF | Pages | Companion links | Underfull-box notices | SHA-256 |
|---|---:|---:|---:|---|
| manuscript.pdf | 16 | 20 | 60 | `2d16b5aae4bebcd82d64316d513251217817e7cb01129cdb38b80e5ddd1f96e9` |
| ESM_1.pdf | 38 | 11 | 39 | `7cdb128b534cd6738561d5330716480685ff8aefb9c12ffeb1fa681301438d22` |

## Remaining typesetting notices

The main log retains the baseline hyperref bookmark-level warning (`Difference (4) between bookmark levels...`). Underfull-box notices concern loose line/page composition and are not clipping/overflow errors: baseline counts were 56 main and 38 supplement; current counts are shown above. Their counts can change as added text reflows. Affected pages were visually inspected. These notices are disclosed rather than described as a warning-free build.

## Links, metadata and provenance

The immutable Phase 1 register and baseline crosswalk resolve through the GitHub connector. The large claims manifest required the raw immutable GitHub endpoint; its bytes match the local frozen manifest. Official PMLR pages verify OPT-Engine (306:13807–13834), OptMATH and Autoformulation. Refai–Ahmed metadata and component metrics are supported by the authors' institutional publication record; publisher full text was inaccessible (HTTP 403), so full-method details remain UNVERIFIED. No incorrectness is inferred from that access failure.

`CLAIM_LOCATION_OVERLAY_V1.json` maps all 37 original numbered objects, 183 numerical passages and 47 investigations to current labels/source locations and compiled pages, with current citation locations. It is an overlay, not a rewritten frozen claims manifest or claim that every paragraph remains verbatim. The E35 and Hit@5 qualifications are explicit.

Only new corrected sources were searched for stale attributions: `P@5` remains solely in the historical-field explanation; the unconditional gold-schema-oracle-270 sentence is absent. Earlier manuscript copies, reports and released evidence intentionally preserve historical wording. Old release URLs remain links to baseline artifacts, explicitly distinguished from this unreleased correction.

## Preservation and publication boundary

Remote main and permanent release refs were recorded in `REMOTE_REFS_BEFORE.txt`. Public release asset identities are recorded separately. No existing branch/worktree, main, PR #31, tag, dataset or public release was modified by the correction work. No private/local review report, raw provider response, credential or restricted corpus was added. Historical uncommitted reports are not attributed to their base SHA.

The branch is intended for author review. Download both revised PDFs into one directory; cross-document links require a PDF viewer that supports external named destinations. The old source ZIP intentionally remains the v1.0.0 package. No new submission ZIP, release, PR or journal submission is part of this correction.

Scientifically consistent within the existing reference-assisted, exposed-population and incomplete-archive limits. Independent semantic correctness and full experimental reproduction remain unestablished.
