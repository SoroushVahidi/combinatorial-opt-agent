# Worktree consolidation results

WORKTREES_REMOVED: 0. All 32 preexisting worktrees and the new Query 3 worktree remain.

The exact six candidates from Query 2 (JDSA final-polish, main-publish and phases
1–4) still have expected HEADs, no tracked changes, no untracked files and no ignored
files. Their history is preserved by main/final-editorial-polish retained anchors.
Nevertheless, owner/editor/cluster inactivity cannot be positively established.
Local process working-directory observations alone are insufficient. `squeue` is
not installed locally; no cluster activity attestation or owner release was obtained.
Every candidate was therefore skipped. No forced removal or manual directory
removal was attempted. Primary and active research worktrees were never modified.

Before/after comparison verified all preexisting HEADs, branches, dirty/untracked
paths and content hashes, and ignored paths/sizes. Thirteen worktrees are dirty;
948 ignored-file observations remain identical. Ignored bytes were not exhaustively
hashed, so this is not a claim of full ignored-payload authentication. See the
[worktree preservation receipt](WORKTREE_PRESERVATION.json). No private file bodies
or provider conversations are included in these reports.

| Exact path | HEAD | Changed/untracked | Ignored | Observed local processes | Decision |
|---|---|---|---|---|---|
| `/home/soroush/combinatorial-opt-agent` | `a3f0dde397341a41842a30ddb946e7966706351b` | 123 | 438 | 2 | KEEP |
| `/home/soroush/combinatorial-opt-agent-cleanup-q2` | `2ed093db6c31b905d533693c8f6fe8ef70fe3833` | 0 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-d1-final-pair-audit` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 25 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-d1-semantic-diagnostics` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 653 | 18 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-g2-gold-eligibility-ablation` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 16 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-grounding-followup` | `47c4733cd200b68a26077c8cc442e93d72b9a79f` | 0 | 20 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-jdsa-evidence-v2` | `3296f3bd20128077cd94a4eac82f0b5fe73e2946` | 0 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-jdsa-final-polish` | `9d8c501c58415e766d0d5773cf7cbb576da66cfa` | 0 | 0 | 0 | Candidate: SKIP |
| `/home/soroush/combinatorial-opt-agent-jdsa-main-publish` | `d3071ba31eaf7babdf1e7fe234d8527191ed93de` | 0 | 0 | 0 | Candidate: SKIP |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase1` | `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650` | 0 | 0 | 0 | Candidate: SKIP |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase2` | `8fbf3d073c54a4103ef06263084ea7d3fded57b3` | 0 | 0 | 0 | Candidate: SKIP |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase3` | `49827d274cae995005681fb03a299a5363583861` | 0 | 0 | 0 | Candidate: SKIP |
| `/home/soroush/combinatorial-opt-agent-jdsa-phase4` | `cea99e86d33ad83219b9bf711c310e929ae18409` | 0 | 0 | 0 | Candidate: SKIP |
| `/home/soroush/combinatorial-opt-agent-jdsa-release-v1` | `0db80afe6122b1b20fd28d4f3a440001e645cdf1` | 0 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage1` | `67fb2b51700429e0338d2ac63eeebd321095846e` | 26 | 11 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage2` | `60785b0d144365413dd3b44b577bff901043ddd7` | 0 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage3` | `65c4b00f99ed73e32e063b303d3d2fd91ba7876b` | 0 | 2 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-jdsa-stage3-audit` | `e12c413c18e6d1f068d29509e7f9108e7ef51f65` | 0 | 0 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-joint-grounding` | `df408d347d732a87730ee819063d7a79f059d85e` | 0 | 38 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-manuscript-attribution-notes` | `a2391c0747a67242586695bcca63ef8a262aed24` | 0 | 10 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-manuscript-m1` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 120 | 25 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-manuscript-m3` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 37 | 57 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-manuscript-m4` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 236 | 63 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-manuscript-m5` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 1131 | 109 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-metric-audit` | `5fe08d9e23d0d534b023ff60eed4299648092cd0` | 0 | 41 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-nl4opt-label-audit` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 618 | 29 | 3 | KEEP |
| `/home/soroush/combinatorial-opt-agent-scientific-revision-phase1` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 47 | 5 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-scientific-revision-phase2` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 142 | 5 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-scientific-revision-phase3` | `7324dfdacef5e102d57a2e64288e824fb267a44e` | 75 | 5 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-sncs-corrections` | `64aa80670ace66c158ef9976e69d2cfd6b6b05ba` | 0 | 36 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-supervised-feasibility` | `bd5e04c2aae50acf7f6f3d907f89e92542ad7c0a` | 0 | 7 | 0 | KEEP |
| `/home/soroush/combinatorial-opt-agent-wulver-attribution` | `45dd183cd2624c63625f5198370676c088002b03` | 0 | 29 | 0 | KEEP |
