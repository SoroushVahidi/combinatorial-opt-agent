# JDSA reproducibility stage — final report

**Status: COMPLETED_SUCCESS for the bounded public saved-record reanalysis release.** Full original experimental reproduction remains unavailable where historical observations or permissions are missing. No manuscript narrative, main branch, original evidence, unrelated work or existing branch was changed or deleted.

## Branch, commits and public releases

- Isolated branch: `reproducibility/jdsa-evidence-v2-20261029`.
- Evidence commit and GitHub release tag target: `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`.
- Public main, verified unchanged: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
- Original local main, unchanged: `a3f0dde397341a41842a30ddb946e7966706351b`.
- Authoritative editorial origin: `9d8c501c58415e766d0d5773cf7cbb576da66cfa`. Canonical publication changes only document crosslinks/source URL; scientific text is preserved.
- [GitHub release jdsa-evidence-v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0), release ID `428219113`.
- [Hugging Face evidence directory](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0), revision `f94012872211ab6a828147bb138fbe8ab695b982`, tag `jdsa-v2.0.0`.
- Historical HF v1.2 revision `24a047fa5d661be6cfcf75980ac8f62e6c942243`, now also tagged `v1.2`. All29 prior non-card files, including16 dataset configurations, retain their original Git/LFS content identities. The old card is archived byte-identically at `historical/v1.2/README.md`; the new overview clearly distinguishes historical v1.2 from curated v2 and the unreleased complete M5.

This report and the independent remote receipt are committed after the immutable release commit. The branch tip consequently includes documentation beyond the release tag; the release payload itself is unchanged.

## Exact published artifact list

[PUBLISHED_ARTIFACTS.tsv](PUBLISHED_ARTIFACTS.tsv) enumerates all172 payload files with repository-relative paths, byte sizes and SHA256s. The same172 files are mirrored under HF `releases/jdsa-evidence-v2.0.0/`. The [release manifest](../../reproducibility/jdsa-v2.0.0/release_manifest.json) authenticates all payload entries except itself and its outer checksum file. The checksum file authenticates the manifest.

The six GitHub release assets are:

1. `jdsa-evidence-v2.0.0.zip` —172 files, deterministic1,154,571-byte archive; SHA256 `de7f968bd4dbf4dfc9efa67818c461c0aa48b2993e704823bbf1b321ce44bd37`.
2. `release_manifest.json` —SHA256 `ac20ad695b7ceea866890376ef5e08751921bbcb520c017624727524e79b8b38`.
3. `CHECKSUMS.sha256`.
4. `verification.json` —complete9,737-check reanalysis report; SHA256 `1bacf16801ae0a3ab10e175a8f41965c6318a2b43d33abee1d66717d2a93350b`.
5. `verification-receipt.json` —clean-environment/testing and scope receipt.
6. `ASSET_CHECKSUMS.sha256` —hashes of the other five assets.

HF additionally contains the identical archive at `archives/jdsa-evidence-v2.0.0.zip`, three receipt/checksum files under `releases/jdsa-evidence-v2.0.0-publication/`, the archived historical card, and the revised root overview. Its commit comprises178 operations:177 additions and the root card update. No historical data file was overwritten.

The payload includes97 frozen metric input CSVs;61 projected assignment checkpoints; three217-target relationship validity/label stages; common18 outcome records;24 original D1 fixtures and12 final pair decisions; pilot, factorial, reference-sensitivity and G3 counts; historical aggregate summaries; original-source inventories/hashes; deterministic verifiers and statistical analysis; dependency/environment information; licenses and explicit limitations. The entire33,618,383-byte M5 candidate ZIP and private archive remain local, preserved.

## Main-paper reproducibility classification

| Object | Classification | Verified boundary |
|---|---|---|
| Table1: retrieval | Independently reanalyzable | Twelve saved top-1 cells/IDs; generation environment and full rankings are not reconstructed. |
| Table2: headline grounding | Independently reanalyzable | Four-arm query metrics and denominators; original per-slot assignments unavailable. |
| Table3: grounding alternatives | Independently reanalyzable | Six saved configurations, including40 all-key versus41 eligible-only constrained successes. |
| Table4: error partition | Independently reanalyzable | Ordered per-query residual partition; no new source-semantic adjudication. |
| Table5: extraction ablation | Independently reanalyzable | Saved pre/post metrics and paired transitions. |
| Table6: significance | Independently reanalyzable |327 frozen groups; B10000/seed42 percentile intervals; exact group swaps; Holm16. |
| Table7: Strict readiness | Independently reanalyzable | Saved schema and coverage/type gates. |
| Table8: common18 | Projected-record reanalysis | Saved dispositions and numerical objective tokens; excluded raw logs/programs prevent full source validation. PaMOP aggregate accuracy withheld. |
| Table9: relationships | Projected-record reanalysis | Validity/label/request ledgers, counts/agreement; raw quote/schema checks passed locally but raw prose is excluded publicly. |
| Figure1: pipeline | Independently inspectable specification | Canonical editable diagram source; no empirical figure data or new figure build. |
| Algorithm1 | Independently inspectable specification | Versioned algorithm and implementation; historical runtime/assignment replay is incomplete. |

All26 supplementary tables,183 numeral-containing source passages and47 investigation groups are mapped in [claims.json](../../reproducibility/jdsa-v2.0.0/claims.json). The map uses paragraph/table-block granularity, retaining multiple evidence families for mixed passages rather than asserting an unsupported one-to-one observation mapping. [AUDIT.md](../../reproducibility/jdsa-v2.0.0/AUDIT.md) explains each numbered object's classification. No entire inference experiment is labeled fully reproduced by this stage.

## Verification results

- Fresh Python3.12.3 virtual environment, NumPy2.3.5 only; copied standalone package outside every repository worktree; cleared credentials/environment.
- 9,737 verifier checks PASS. Six tests cover intact inputs, changed/missing/extra files, manifest tampering and statistical edge cases.
- System-call tracing across child processes: zero private-path file accesses, zero network socket/connect calls. OS namespace isolation could not be enabled (bubblewrap network/UID mapping denied); no namespace isolation is claimed.
- Exact historical factorial bootstrap procedure reproduced (Python Random42/B2000/order-statistic interval); all saved paired intervals match. Separate240-test Holm family and primary16-test family remain separate.
- All172 Git blobs match the reviewed local bytes. Original CSV line endings were intentionally preserved; whitespace checking with `cr-at-eol` passes.
- Original local M5 checks:2,756 included payload hashes matched; plus three archive infrastructure members. Raw local relationship validator passed5,120 checks; common18 validator passed350 checks across108 rows/72 saved generated cases; ten core arms verified.
- Anonymous independent remote downloads: all six GitHub assets and all172 HF payload files match. Reanalysis from each separately downloaded package passes9,737 checks and produces the same verification-report hash.
- Historical HF file identities and both version tags checked; public main unchanged. See [REMOTE_PUBLICATION_VERIFICATION.json](REMOTE_PUBLICATION_VERIFICATION.json) for exact paths, hashes, asset IDs and timestamps.
- Zero new inference calls, paid API calls or scientific solver experiments.

## Missing evidence and qualifications

Original headline scalar mappings,60/269/20 subset manifests and per-case outcomes, full historical rendered prompts/provider defaults, complete BGE hash-seed/device/environment, ambiguous PaMOP full precision and corrected per-instance OptMATH bootstrap inputs were not recovered. Full authenticated historical higher-rank lists are also unavailable. No observations were reconstructed or replaced with newer outputs.

The final main numerical results agree with the supported saved-record calculations. Retain the already-corrected40-versus41 constrained distinction, legacy175-versus-obsolete174 distinction, conditional Exact20 denominators, and corrected OptMATH micro/macro definitions. Historical OptMATH JSON names reverse the estimands; the corrected micro ratio is23241/28799, whereas0.8135 is a macro mean. Its preserved corrected interval cannot be independently re-bootstrapped from the missing corrected per-instance inputs.

Stage2 should qualify availability text: the curated v2 subset is public, while complete M5, raw assignment source fields and raw provider responses remain unreleased. Claims that all candidate identities or raw validation evidence are publicly inspectable would be too broad. Reanalysis is not full reproduction; relationship agreement is not accuracy; D1's author/AI-assisted labels are not independent blinded gold; common18 methods received unequal model/prompt/solver treatment. Historical subset studies do not establish complete formulation correctness or disjoint sampling.

## Publication blockers and preservation

The old candidate ZIP contained gated-source context/value traces and source quotations, so it was not uploaded wholesale. The release omits raw NLP4LP data, license-unclear OptMATH source/program material, provider response prose/account metadata, private author conversations, and third-party weights. Account-specific provider redistribution permissions remain unverified. These block broader raw-content publication, not the coherent scoped numeric/label/author-fixture release. The review does not claim that all derived data are automatically free to redistribute; rights are scoped by file/family and upstream source content is excluded.

The original local evidence, manifests and unrelated worktrees were preserved. No branches were deleted or merged, and the three previously untracked audit/manuscript directories in the original checkout remain untouched. The manuscript narrative was not rewritten.

## Stage2 recommendation

**READY for careful scientific-narrative clarification based on this bounded release.** Use the frozen evidence tag/revision and preserve every missing-evidence/permission limitation. Do not describe the package as complete experimental reproduction, silently revise observations, or promote summary-only or unexecuted investigations into supported outcomes.
