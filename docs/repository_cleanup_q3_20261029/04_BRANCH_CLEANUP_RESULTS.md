# Branch consolidation results

BRANCHES_DELETED: 0. Every preexisting branch and tag is preserved.

The historical 21 candidate names correspond to 25 local/remote ref observations.
All candidates still resolve to their audited SHA and have zero unique commits
relative to their retained anchor. This is necessary but insufficient for deletion:
owner/cluster inactivity is not established. All were skipped under the user's rule.
No bulk prune, ref deletion, force push or unrelated branch update was performed.

JDSA phase 1–4 are ancestors of retained final-editorial-polish commit
`9d8c501c58415e766d0d5773cf7cbb576da66cfa`, not assumed to be ancestors of main.
The other candidates use `d3071ba31eaf7babdf1e7fe234d8527191ed93de`. Both local and remote final-editorial-polish
anchors remain intact. [Machine inventory](CONSOLIDATION_INVENTORY.json) records
current SHA, live remote SHA, exact anchor, uniqueness, worktree association,
operational reference files and open-PR associations for every ref.

Open PR [#30](https://github.com/SoroushVahidi/combinatorial-opt-agent/pull/30) uses investigation branch
`kais-final-submission-prep`; it is preserved. All six investigation branch names,
all KEEP branches, release branches, Query 2 and Query 3 remain protected.
Repository/workflow reference searches covered tracked operational directories and
documentation. No claim of definitive owner inactivity follows from these searches.

No deletion/recovery action was necessary. For a future separately verified ref
removal, record its exact SHA and retained anchor, verify the SHA is still reachable,
then record a recovery command such as `git branch <exact-name> <recorded-sha>`
(and ordinary push for a remote ref). This is a recommendation, not executed work.

| Ref | Current SHA | Historical classification | Current checks; decision is PRESERVE for every row |
|---|---|---|---|
| local: `audit/d1-final-controlled-pairs-20261029` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `audit/nl4opt-relationship-qc-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | REQUIRES_INVESTIGATION | open PRs: none; worktrees: 0; operational references: 0 |
| local: `audit/sncs-metric-definitions-20261027` | `5fe08d9e23d0d534b023ff60eed4299648092cd0` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `audit/supervised-grounding-feasibility-20261027` | `bd5e04c2aae50acf7f6f3d907f89e92542ad7c0a` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `cleanup/repository-q2-safe-files-20261029` | `2ed093db6c31b905d533693c8f6fe8ef70fe3833` | KEEP (new since Query 2 inventory) | open PRs: none; worktrees: 1; operational references: 0 |
| local: `cleanup/repository-q3-final-polish-20261029` | `2ed093db6c31b905d533693c8f6fe8ef70fe3833` | KEEP (new since Query 2 inventory) | open PRs: none; worktrees: 0; operational references: 0 |
| local: `experiments/d1-semantic-failure-diagnostics-20261029` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `main` | `a3f0dde397341a41842a30ddb946e7966706351b` | KEEP | open PRs: none; worktrees: 1; operational references: 114 |
| local: `manuscript/jdsa-abstract-introduction-stage1-20261029` | `67fb2b51700429e0338d2ac63eeebd321095846e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-canonical-main-20261029` | `d3071ba31eaf7babdf1e7fe234d8527191ed93de` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-final-editorial-polish-20261029` | `9d8c501c58415e766d0d5773cf7cbb576da66cfa` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-permanent-release-v1-20261029` | `0db80afe6122b1b20fd28d4f3a440001e645cdf1` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-phase1-evidence-results-20261029` | `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-phase2-scientific-consolidation-20261029` | `8fbf3d073c54a4103ef06263084ea7d3fded57b3` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-phase3-journal-preparation-20261029` | `49827d274cae995005681fb03a299a5363583861` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-phase4-final-consistency-20261029` | `cea99e86d33ad83219b9bf711c310e929ae18409` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-stage2-scientific-restructure-20261029` | `60785b0d144365413dd3b44b577bff901043ddd7` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-stage3-final-editorial-20261029` | `65c4b00f99ed73e32e063b303d3d2fd91ba7876b` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/jdsa-stage3-submission-audit-20261029` | `e12c413c18e6d1f068d29509e7f9108e7ef51f65` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/m1-scientific-restructure-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/m3-scientific-corrections-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/m4-final-editorial-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/m5-llm-open-reproducibility-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/scientific-revision-phase1-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/scientific-revision-phase2-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/scientific-revision-phase3-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/sncs-evidence-corrections-20261027` | `64aa80670ace66c158ef9976e69d2cfd6b6b05ba` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `manuscript/sncs-grounding-attribution-notes-20261027` | `a2391c0747a67242586695bcca63ef8a262aed24` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `reproducibility/jdsa-evidence-v2-20261029` | `3296f3bd20128077cd94a4eac82f0b5fe73e2946` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `research/constraint-grounding-pilot-20261027` | `df408d347d732a87730ee819063d7a79f059d85e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `research/g2-gold-eligibility-ablation-20261028` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `research/grounding-followup-20261027` | `47c4733cd200b68a26077c8cc442e93d72b9a79f` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `research/nl4opt-human-annotation-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | REQUIRES_INVESTIGATION | open PRs: none; worktrees: 0; operational references: 0 |
| local: `research/nl4opt-label-recovery-v2-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| local: `research/nl4opt-three-model-labeling-20261027` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | REQUIRES_INVESTIGATION | open PRs: none; worktrees: 0; operational references: 0 |
| local: `research/wulver-grounding-attribution-20261027` | `45dd183cd2624c63625f5198370676c088002b03` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `SoroushVahidi-patch-1` | `70913652bd8ff042667fa29746b62c24a5bc84c2` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `cleanup/repository-q2-safe-files-20261029` | `2ed093db6c31b905d533693c8f6fe8ef70fe3833` | KEEP (new since Query 2 inventory) | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `codex/connect-codex-to-another-repository` | `5e4730c419000b3ed24175bbec37844816013938` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `codex/normalize-datasets-and-build-schema-catalog` | `0a2e79548fe9c74026a685cbf478b385f1b83788` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/analyze-combinatorial-optimization-bot` | `007bdf9c2d4723b8da2606dfde3c20677f15daa0` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/clean-repo-for-manuscript-support` | `37ffd55c9beb68cfd48289a6fe72535c12e086c4` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/cleanup-repository-for-public-quality` | `da32805d24e53094e93966dbf99e5cb0ca41d558` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/create-instances-set-cover-problem` | `18ec926b8f05aabbec0bbbf212332361ff1efbd3` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/experiment-overview` | `4f025995e44338ff4688c6f331e0b6011ebab40e` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/group-1-hard-family-improvements` | `b3db74c9406165a063f781a48179d60414b98329` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/identify-program-weakpoint` | `c15dd9d9f6fd6cdd9cf355748dd706558f786c84` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/inspect-downstream-slot-grounding` | `f0982d5c23b89e8067b5592fcb43b5bf28d4ad9f` | REQUIRES_INVESTIGATION | open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/main-branch-description` | `a4bcc8a57af8c40e46a0cb674ee5e74450049b28` | KEEP | open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/normalize-data-sources-audit` | `31baa4aee9804f4b5aef90f17e7a667107d7feb3` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/polish-clean-reorganize-repo` | `a74d1b6893903d912417192e1565442ea4b3e6e7` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/prepare-research-codebase-eaai-paper` | `6291c76a570722a4cf77440f41ece4ea238561da` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/second-hard-family-pass-problem-7` | `f12dc0231d99c34ba0d0256bbc362fb7cb38d3df` | REQUIRES_INVESTIGATION | open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/setup-project-scaffolding-ai-agent` | `7b9ea2bc19c781d0327022c41f38358e7862b2a2` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/test-program-on-iphone` | `086e1de529c758f2230c33e671906ea335591bca` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `copilot/update-repository-for-eaai-manuscript` | `1584a4f6cd1036e6a540527df953a36aea2ebf8f` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 0; operational references: 0 |
| remote: `kais-final-submission-prep` | `2822908c2677ffb6649530335608b41f4ef221f9` | REQUIRES_INVESTIGATION | open PRs: [30]; worktrees: 0; operational references: 0 |
| remote: `main` | `d3071ba31eaf7babdf1e7fe234d8527191ed93de` | KEEP | open PRs: none; worktrees: 1; operational references: 114 |
| remote: `manuscript/jdsa-final-editorial-polish-20261029` | `9d8c501c58415e766d0d5773cf7cbb576da66cfa` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-permanent-release-v1-20261029` | `0db80afe6122b1b20fd28d4f3a440001e645cdf1` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-phase1-evidence-results-20261029` | `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-phase2-scientific-consolidation-20261029` | `8fbf3d073c54a4103ef06263084ea7d3fded57b3` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-phase3-journal-preparation-20261029` | `49827d274cae995005681fb03a299a5363583861` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-phase4-final-consistency-20261029` | `cea99e86d33ad83219b9bf711c310e929ae18409` | SAFE_TO_DELETE | ancestry PASS; unique commits versus anchor 0; open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-stage2-scientific-restructure-20261029` | `60785b0d144365413dd3b44b577bff901043ddd7` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-stage3-final-editorial-20261029` | `65c4b00f99ed73e32e063b303d3d2fd91ba7876b` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `manuscript/jdsa-stage3-submission-audit-20261029` | `e12c413c18e6d1f068d29509e7f9108e7ef51f65` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
| remote: `reproducibility/jdsa-evidence-v2-20261029` | `3296f3bd20128077cd94a4eac82f0b5fe73e2946` | KEEP | open PRs: none; worktrees: 1; operational references: 0 |
