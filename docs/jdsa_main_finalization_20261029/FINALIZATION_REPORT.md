# Final corrected JDSA manuscript on main

## Source and integration scope

Remote-main baseline: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`. Scientific source: `d3583fa28fcdb87f7a446f764ca3225ab5f855d3` on `manuscript/jdsa-attribution-corrections-20261029`. No newer canonical Stage 3 source correction was found among fetched manuscript refs. The isolated integration branch is `manuscript/jdsa-finalize-main-20261029`; its commit is intended for a normal fast-forward push directly to remote main. The original primary checkout and its uncommitted directories are deliberately untouched.

Only the canonical Stage 3 package is imported, rather than merging unrelated research-branch changes. Its `.tex`, `.bib`, `.cls`, `.bst` and `.sty` inputs are byte-identical to the corrected scientific source except two recorded navigation edits: the supplement's companion PDF filename and a link identifying the current corrected source directory. Funding, acknowledgments, conflicts, scientific prose, numerical tables and evaluation expectations are unchanged. `SOURCE_LINK_CHANGES.json` records the exact two replacements.

The stage3 PDFs/ZIP at the correction commit were explicitly historical v1.0.0 copies. They were not reused. The final pair and flat source ZIP were rebuilt from the corrected sources; generated bibliographies and figure are refreshed from those inputs. The existing `manuscript/jdsa/` sources, scripts and ZIP are retained but prominently retired from current submission use.

## Removed standalone PDFs

- `manuscript/jdsa/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf`
- `manuscript/jdsa/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_Supplement.pdf`

Both are recoverable from `d3071ba31eaf7babdf1e7fe234d8527191ed93de`. `PDF_INVENTORY_BEFORE.json` records all 23 baseline tracked PDFs, identities, hashes, page counts and dispositions. All other baseline PDFs remain byte-identical, including separate SNCS/DKE/KAIS manuscripts and JDSA figure assets. Historical release PDFs, source ZIPs and tags are unchanged. No obsolete JDSA PDF pair from the correction branch's historical packaging was imported into main.

## Validation

- Clean paired pdfLaTeX/BibTeX builds: PASS, 16 manuscript pages and 38 supplement pages.
- Independent extraction/build of the flat ZIP: PASS; both resulting PDFs match the final pair byte-for-byte.
- Integration validator: 410 checks PASS; exact corrected source identity, archive-member equality, source dependencies, all PDF page numbers, embedded fonts, page bounds, internal named destinations and all 31 companion links checked.
- Final logs: zero undefined references/citations, missing characters, multiply-defined labels or overfull boxes; no BibTeX warnings/errors.
- Existing unchanged evidence verifier: 9,737 checks PASS, executed against the corrected worktree's immutable evidence package with output outside it. This is frozen-record verification, not new inference or experimental reproduction.
- Scientific identity preserves oracle A 268 versus oracle C 270 (including C's two wrong-schema-ready cases), Hit@5, corrected related-work attribution, 7/331, 7/265, 7/255, the 30+31+49+214+7 partition, 7/40/37/35/46/9 and constrained 41-versus-40 semantics.
- Every supplement page and main pages 1–13 render identically to the corrected author-review PDFs. Main pages 14–16 reflow because of the current-source navigation sentence. Those three pages and supplement p.28 were visually inspected and have readable, unclipped content. No scientific change was made to resolve layout.
- Git whitespace inspection flags only inherited whitespace in unchanged upstream class/style files and two unchanged supplement lines; these are preserved byte-for-byte rather than cosmetically rewriting the scientific baseline. All newly authored edits pass the scoped whitespace check.
- Retained nonfatal typesetting notices: inherited main hyperref bookmark-level warning; 60 main and 39 supplement underfull-box notices. These are not undefined links or clipped content.
- Active README/index links resolve to actual final paths. Historical publication documentation is explicitly marked historical; its removed PDF paths are recoverable at the pinned baseline commit.
- Release IDs/asset digests and the exact public Hugging Face revision were checked read-only; `PUBLIC_ARTIFACT_IDENTITIES.json` records them. No release operation or dataset mutation occurred.

## PDF hashes

- `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf`: `9e7815c8fd66748bd41bf739577573697be8fb707b4ec744160533dddd2ed17b`
- `ESM_1.pdf`: `d90456a7ac71cae6f7600964edb5fe6f2b62e08c80082b858801d237e80d7311`

## Commands and records

`KEEP_BUILD=1 sh manuscript/jdsa-stage3/build.sh`; `python3 manuscript/jdsa-stage3/package_source.py`; clean extraction followed by `KEEP_BUILD=1 sh build.sh`; unchanged `python3 -I reproducibility/jdsa-v2.0.0/verify.py --output /tmp/jdsa-main-finalization-frozen.json`; `validate_finalization.py <source-build-dir> <independent-archive-dir> <frozen-report>`; Git diff/inventory checks. Build locations are recorded separately for this session. No experiments, training, paid inference, solver runs, Slurm jobs, journal submission or new release.

## Publication boundary

This committed report records validation completed before the push. Remote commit and independent PDF downloads are verified after the push and reported in the final task response; this report does not claim an unperformed push. The current manuscript remains a public pre-submission author-review document, not an accepted journal publication.
