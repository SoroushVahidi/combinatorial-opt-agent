# Final GitHub readiness

STATUS: COMPLETED_WITH_ISSUES (local implementation complete; publication receipt follows).
BASELINE_SHA: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
QUERY2_SHA: `2ed093db6c31b905d533693c8f6fe8ef70fe3833`.
QUERY3_BRANCH: `cleanup/repository-q3-final-polish-20261029`.
QUERY3_FINAL_SHA: available as the containing Git commit after the final receipt is committed; the exact verified remote SHA is recorded in the PR description and final delivery message.
MAIN_SHA_BEFORE: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
MAIN_SHA_AFTER: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
PULL_REQUEST_URL: PR creation has not yet occurred at this pre-publication snapshot.
MERGE_STATUS: NOT_MERGED; two baseline-confirmed aggregate regression failures require review.

Implementation is complete on an isolated branch. Separate commits cover engineering
repairs, documentation/navigation, and test/CI safeguards. Final publication will
include all eight reports and preservation receipts; no manuscript-release branch
is merged into main. Do not interpret this snapshot as proof of a push or PR.

BRANCHES_DELETED: 0. BRANCHES_PRESERVED: all preexisting refs.
BRANCHES_REQUIRING_INVESTIGATION: all six historical names preserved (see report 04).
WORKTREES_REMOVED: 0. WORKTREES_PRESERVED: 32 original plus the new Query 3 worktree.
FILES_DELETED: 0 in Query 3; six approved Query 2 deletions in cumulative PR.

TEST_RESULTS: minimal/offline/adapters pass with explicit skips; two independently
confirmed baseline regression failures remain. DOCUMENTATION_RESULTS: relative
links pass; permanent artifact links authenticated. MANUSCRIPT_RELEASE_PRESERVED,
EVIDENCE_RELEASE_PRESERVED, HUGGING_FACE_PRESERVED, SCIENTIFIC_DATA_PRESERVED and
UNPUBLISHED_WORK_PRESERVED: YES, with observation boundaries in report 06.

## Remaining issues

- Resolve/review the two historical aggregate test expectations separately; do not
  change scientific numbers to obtain a green maintenance run.
- Owner/cluster activity is not established; every conditional branch/worktree
  deletion was skipped. Investigation branches and open PR #30 remain intact.
- Existing third-party redistribution questions need source-rights review and a
  preservation plan before removal or further raw-data publication.
- The other 18 unresolved Query 2 file candidates remain untouched; empty evaluation
  placeholders, legacy destructive cleanup script and demo helper are not silently fixed.
- Live network/provider/model/solver and Cardinal schema validation were not run.
- Main's active ruleset includes code-quality errors and deletion protection; no
  bypass or protection change is permitted or attempted.

[Branch](https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/cleanup/repository-q3-final-polish-20261029) · [Validation](07_VALIDATION_RESULTS.md) ·
[Preservation](06_REPRODUCIBILITY_PRESERVATION.md).
