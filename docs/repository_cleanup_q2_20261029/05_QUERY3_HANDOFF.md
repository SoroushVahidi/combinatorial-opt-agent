# Query 3 handoff

Baseline: `d3071ba31eaf7babdf1e7fe234d8527191ed93de` (verified current `origin/main`). Cleanup branch: `cleanup/repository-q2-safe-files-20261029`.


**QUERY3_READINESS: READY** for the bounded structural/documentation work below, with all conditional deletion gates repeated at execution time. No branch/worktree deletion is approved by this report.

## Current inventory and preservation

Before the cleanup worktree was added: 34 local branches, 31 remote branches, 54 distinct names; 27 KEEP, 21 historical conditional deletion-candidate and six investigation names; 31 worktrees, 13 dirty and two active-observed. The cleanup branch adds one distinct/local/remote name after push and one worktree: delivery totals are 35 local branches, 32 remote branches, 55 distinct names and 32 worktrees. All original refs/worktrees are retained. The following table is the pre-cleanup snapshot; the new cleanup branch points to the delivery commit returned with this report.

| Scope | Branch | Observed tip | Current handoff class |
|---|---|---|---|
| local | `audit/d1-final-controlled-pairs-20261029` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `audit/nl4opt-relationship-qc-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | REQUIRES_INVESTIGATION |
| local | `audit/sncs-metric-definitions-20261027` | `5fe08d9e23d0d534b023ff60eed4299648092cd0` | KEEP |
| local | `audit/supervised-grounding-feasibility-20261027` | `bd5e04c2aae50acf7f6f3d907f89e92542ad7c0a` | KEEP |
| local | `experiments/d1-semantic-failure-diagnostics-20261029` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `main` | `a3f0dde397341a41842a30ddb946e7966706351b` | KEEP |
| local | `manuscript/jdsa-abstract-introduction-stage1-20261029` | `67fb2b51700429e0338d2ac63eeebd321095846e` | KEEP |
| local | `manuscript/jdsa-canonical-main-20261029` | `d3071ba31eaf7babdf1e7fe234d8527191ed93de` | CONDITIONAL Q3 CANDIDATE |
| local | `manuscript/jdsa-final-editorial-polish-20261029` | `9d8c501c58415e766d0d5773cf7cbb576da66cfa` | KEEP |
| local | `manuscript/jdsa-permanent-release-v1-20261029` | `0db80afe6122b1b20fd28d4f3a440001e645cdf1` | KEEP |
| local | `manuscript/jdsa-phase1-evidence-results-20261029` | `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650` | CONDITIONAL Q3 CANDIDATE |
| local | `manuscript/jdsa-phase2-scientific-consolidation-20261029` | `8fbf3d073c54a4103ef06263084ea7d3fded57b3` | CONDITIONAL Q3 CANDIDATE |
| local | `manuscript/jdsa-phase3-journal-preparation-20261029` | `49827d274cae995005681fb03a299a5363583861` | CONDITIONAL Q3 CANDIDATE |
| local | `manuscript/jdsa-phase4-final-consistency-20261029` | `cea99e86d33ad83219b9bf711c310e929ae18409` | CONDITIONAL Q3 CANDIDATE |
| local | `manuscript/jdsa-stage2-scientific-restructure-20261029` | `60785b0d144365413dd3b44b577bff901043ddd7` | KEEP |
| local | `manuscript/jdsa-stage3-final-editorial-20261029` | `65c4b00f99ed73e32e063b303d3d2fd91ba7876b` | KEEP |
| local | `manuscript/jdsa-stage3-submission-audit-20261029` | `e12c413c18e6d1f068d29509e7f9108e7ef51f65` | KEEP |
| local | `manuscript/m1-scientific-restructure-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/m3-scientific-corrections-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/m4-final-editorial-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/m5-llm-open-reproducibility-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/scientific-revision-phase1-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/scientific-revision-phase2-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/scientific-revision-phase3-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `manuscript/sncs-evidence-corrections-20261027` | `64aa80670ace66c158ef9976e69d2cfd6b6b05ba` | KEEP |
| local | `manuscript/sncs-grounding-attribution-notes-20261027` | `a2391c0747a67242586695bcca63ef8a262aed24` | KEEP |
| local | `reproducibility/jdsa-evidence-v2-20261029` | `3296f3bd20128077cd94a4eac82f0b5fe73e2946` | KEEP |
| local | `research/constraint-grounding-pilot-20261027` | `df408d347d732a87730ee819063d7a79f059d85e` | KEEP |
| local | `research/g2-gold-eligibility-ablation-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `research/grounding-followup-20261027` | `47c4733cd200b68a26077c8cc442e93d72b9a79f` | KEEP |
| local | `research/nl4opt-human-annotation-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | REQUIRES_INVESTIGATION |
| local | `research/nl4opt-label-recovery-v2-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP |
| local | `research/nl4opt-three-model-labeling-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | REQUIRES_INVESTIGATION |
| local | `research/wulver-grounding-attribution-20261027` | `45dd183cd2624c63625f5198370676c088002b03` | KEEP |
| remote | `SoroushVahidi-patch-1` | `70913652bd8ff042667fa29746b62c24a5bc84c2` | CONDITIONAL Q3 CANDIDATE |
| remote | `codex/connect-codex-to-another-repository` | `5e4730c419000b3ed24175bbec37844816013938` | CONDITIONAL Q3 CANDIDATE |
| remote | `codex/normalize-datasets-and-build-schema-catalog` | `0a2e79548fe9c74026a685cbf478b385f1b83788` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/analyze-combinatorial-optimization-bot` | `007bdf9c2d4723b8da2606dfde3c20677f15daa0` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/clean-repo-for-manuscript-support` | `37ffd55c9beb68cfd48289a6fe72535c12e086c4` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/cleanup-repository-for-public-quality` | `da32805d24e53094e93966dbf99e5cb0ca41d558` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/create-instances-set-cover-problem` | `18ec926b8f05aabbec0bbbf212332361ff1efbd3` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/experiment-overview` | `4f025995e44338ff4688c6f331e0b6011ebab40e` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/group-1-hard-family-improvements` | `b3db74c9406165a063f781a48179d60414b98329` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/identify-program-weakpoint` | `c15dd9d9f6fd6cdd9cf355748dd706558f786c84` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/inspect-downstream-slot-grounding` | `f0982d5c23b89e8067b5592fcb43b5bf28d4ad9f` | REQUIRES_INVESTIGATION |
| remote | `copilot/main-branch-description` | `a4bcc8a57af8c40e46a0cb674ee5e74450049b28` | KEEP |
| remote | `copilot/normalize-data-sources-audit` | `31baa4aee9804f4b5aef90f17e7a667107d7feb3` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/polish-clean-reorganize-repo` | `a74d1b6893903d912417192e1565442ea4b3e6e7` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/prepare-research-codebase-eaai-paper` | `6291c76a570722a4cf77440f41ece4ea238561da` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/second-hard-family-pass-problem-7` | `f12dc0231d99c34ba0d0256bbc362fb7cb38d3df` | REQUIRES_INVESTIGATION |
| remote | `copilot/setup-project-scaffolding-ai-agent` | `7b9ea2bc19c781d0327022c41f38358e7862b2a2` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/test-program-on-iphone` | `086e1de529c758f2230c33e671906ea335591bca` | CONDITIONAL Q3 CANDIDATE |
| remote | `copilot/update-repository-for-eaai-manuscript` | `1584a4f6cd1036e6a540527df953a36aea2ebf8f` | CONDITIONAL Q3 CANDIDATE |
| remote | `kais-final-submission-prep` | `2822908c2677ffb6649530335608b41f4ef221f9` | REQUIRES_INVESTIGATION |
| remote | `main` | `d3071ba31eaf7babdf1e7fe234d8527191ed93de` | KEEP |
| remote | `manuscript/jdsa-final-editorial-polish-20261029` | `9d8c501c58415e766d0d5773cf7cbb576da66cfa` | KEEP |
| remote | `manuscript/jdsa-permanent-release-v1-20261029` | `0db80afe6122b1b20fd28d4f3a440001e645cdf1` | KEEP |
| remote | `manuscript/jdsa-phase1-evidence-results-20261029` | `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650` | CONDITIONAL Q3 CANDIDATE |
| remote | `manuscript/jdsa-phase2-scientific-consolidation-20261029` | `8fbf3d073c54a4103ef06263084ea7d3fded57b3` | CONDITIONAL Q3 CANDIDATE |
| remote | `manuscript/jdsa-phase3-journal-preparation-20261029` | `49827d274cae995005681fb03a299a5363583861` | CONDITIONAL Q3 CANDIDATE |
| remote | `manuscript/jdsa-phase4-final-consistency-20261029` | `cea99e86d33ad83219b9bf711c310e929ae18409` | CONDITIONAL Q3 CANDIDATE |
| remote | `manuscript/jdsa-stage2-scientific-restructure-20261029` | `60785b0d144365413dd3b44b577bff901043ddd7` | KEEP |
| remote | `manuscript/jdsa-stage3-final-editorial-20261029` | `65c4b00f99ed73e32e063b303d3d2fd91ba7876b` | KEEP |
| remote | `manuscript/jdsa-stage3-submission-audit-20261029` | `e12c413c18e6d1f068d29509e7f9108e7ef51f65` | KEEP |
| remote | `reproducibility/jdsa-evidence-v2-20261029` | `3296f3bd20128077cd94a4eac82f0b5fe73e2946` | KEEP |

New delivery ref: local and remote `cleanup/repository-q2-safe-files-20261029`; preserve pending review/merge authorization. New worktree: `/home/soroush/combinatorial-opt-agent-cleanup-q2`.

| Existing worktree | HEAD | Dirty files | Ignored entries | Local-use observation | Commits absent from remote heads |
|---|---|---:|---:|---|---:|
| `/home/soroush/combinatorial-opt-agent` | `a3f0dde397341a41842a30ddb946e7966706351b` | 123 | 47 | processes observed | 0 |
| `/home/soroush/combinatorial-opt-agent-d1-final-pair-audit` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 25 | 0 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-d1-semantic-diagnostics` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 653 | 15 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-g2-gold-eligibility-ablation` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 16 | 0 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-grounding-followup` | `47c4733cd200b68a26077c8cc442e93d72b9a79f` | 0 | 8 | not established inactive | 15 |
| `/home/soroush/combinatorial-opt-agent-jdsa-evidence-v2` | `3296f3bd20128077cd94a4eac82f0b5fe73e2946` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-final-polish` | `9d8c501c58415e766d0d5773cf7cbb576da66cfa` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-main-publish` | `d3071ba31eaf7babdf1e7fe234d8527191ed93de` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase1` | `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase2` | `8fbf3d073c54a4103ef06263084ea7d3fded57b3` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase3` | `49827d274cae995005681fb03a299a5363583861` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase4` | `cea99e86d33ad83219b9bf711c310e929ae18409` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-release-v1` | `0db80afe6122b1b20fd28d4f3a440001e645cdf1` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage1` | `67fb2b51700429e0338d2ac63eeebd321095846e` | 26 | 11 | not established inactive | 19 |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage2` | `60785b0d144365413dd3b44b577bff901043ddd7` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage3` | `65c4b00f99ed73e32e063b303d3d2fd91ba7876b` | 0 | 1 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage3-audit` | `e12c413c18e6d1f068d29509e7f9108e7ef51f65` | 0 | 0 | not established inactive | 0 |
| `/home/soroush/combinatorial-opt-agent-joint-grounding` | `df408d347d732a87730ee819063d7a79f059d85e` | 0 | 18 | not established inactive | 6 |
| `/home/soroush/combinatorial-opt-agent-manuscript-attribution-notes` | `a2391c0747a67242586695bcca63ef8a262aed24` | 0 | 1 | not established inactive | 9 |
| `/home/soroush/combinatorial-opt-agent-manuscript-m1` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 120 | 10 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-manuscript-m3` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 37 | 17 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-manuscript-m4` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 236 | 23 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-manuscript-m5` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 1131 | 38 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-metric-audit` | `5fe08d9e23d0d534b023ff60eed4299648092cd0` | 0 | 8 | not established inactive | 1 |
| `/home/soroush/combinatorial-opt-agent-nl4opt-label-audit` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 618 | 12 | processes observed | 18 |
| `/home/soroush/combinatorial-opt-agent-scientific-revision-phase1` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 47 | 5 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-scientific-revision-phase2` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 142 | 5 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-scientific-revision-phase3` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 75 | 5 | not established inactive | 18 |
| `/home/soroush/combinatorial-opt-agent-sncs-corrections` | `64aa80670ace66c158ef9976e69d2cfd6b6b05ba` | 0 | 13 | not established inactive | 16 |
| `/home/soroush/combinatorial-opt-agent-supervised-feasibility` | `bd5e04c2aae50acf7f6f3d907f89e92542ad7c0a` | 0 | 3 | not established inactive | 17 |
| `/home/soroush/combinatorial-opt-agent-wulver-attribution` | `45dd183cd2624c63625f5198370676c088002b03` | 0 | 6 | not established inactive | 11 |

The six historical worktree candidates remain the final-polish, main-publish and phase1–phase4 worktrees. The other 25 pre-existing worktrees remain preservation targets; the new cleanup worktree is also retained. Newly created Stage 2/Stage 3/manuscript-release/evidence-release worktrees are not added to any deletion allowlist.

## Exact evidence required before future deletion

For every one of the 21 historical candidate branch names above: re-query live local/remote SHAs and open PRs; verify its full tip ancestry is still reachable from retained main or the final-review anchor; check workflow/configuration references and linked worktrees; establish owner/cluster inactivity and explicit authorization. All 25 candidate refs passed ancestry again in Query 2. Phase branches require retained `manuscript/jdsa-final-editorial-polish-20261029` at `9d8c501c58415e766d0d5773cf7cbb576da66cfa`, not an assumption that main contains their reports. No current open-PR clearance is asserted here; it must be checked immediately before a future deletion.

For each of the six worktree candidates: require expected HEAD, clean tracked/untracked state, no ignored payload, live local/editor/cluster inactivity and remote preservation. Use normal worktree removal only after those conditions and authorization; never force removal or run a broad clean/prune. No process observed is not proof of inactivity.

Preserve all six investigation branch names: `audit/nl4opt-relationship-qc-20261027`, `copilot/inspect-downstream-slot-grounding`, `copilot/second-hard-family-pass-problem-7`, `kais-final-submission-prep`, `research/nl4opt-human-annotation-20261027`, `research/nl4opt-three-model-labeling-20261027`. They need owner/patch/content decisions. `copilot/main-branch-description` stays because the downstream workflow still references it. All permanent release tags, reachable commits and the manuscript report commit remain protected regardless of any future branch consolidation.

For files: no additional deletion is authorized. The 19 investigation files need producer/consumer/provenance/licensing evidence and a separately approved disposition. Empty evaluation data, unique negative results and historical scripts are not disposable. A future private backup must include ignored and untracked originals; a Git bundle alone is insufficient.

## Remaining documentation and engineering work

1. Add a current/historical availability index and a JDSA claim-to-artifact map using the permanent manuscript/evidence releases and pinned Hugging Face revision. Preserve old correction history. The old audit predates the v2 release: sanitized saved-record evidence is now public, complete local M5 originals are not. Do not reuse obsolete blanket availability claims. Do not merge or overwrite the final manuscript into main without explicit authorization.

2. Repair the one existing historical relative link to `docs/archive/EXPERIMENT_AUDIT.md`. Address stale EAAI/SNCS-centric navigation and the six-row/four-row README description through a separately reviewed documentation change, not a source-directory migration.

3. Repair the Cardinal downloader coherently with fixture tests; do not merely remove its unmatched brace. Make network/model tests opt-in, include baseline test suites in documented test scope, and document optional dependencies without inventing historical environments.

4. Review manual workflows that automatically commit results or upload processed data; migrate the obsolete branch default before considering removal. Do not trigger acquisition, inference, training or solvers as routine CI. The collection workflow is limited to main pushes touching collectors/pipeline; this cleanup branch does not trigger that route.

5. Add verified manuscript citation metadata if requested, describing a pre-submission manuscript, with no invented DOI, acceptance or publication. Address public-source access/provenance questions separately; root MIT does not relicense templates, models or datasets.

## Suggested organization

Keep the existing `retrieval/`, `tools/`, `baselines/`, `src/`, `training/`, `scripts/`, `batch/`, data and results layouts. Improve navigation through current/historical and public/restricted indexes; do not move scientific evidence to make the tree look smaller. Add lightweight offline validation and explicit artifact review. Avoid blanket extension ignores and mass normalization of frozen CSV/log/source bytes. Historical venue packages remain distinct, even where names or selected outputs coincide.

Completed Query 2 work is limited to the six deletions, three relocation notices, exact ignore policy and reports. Every item in this handoff is a future recommendation or deletion prerequisite, not a completed Query 3 action.
