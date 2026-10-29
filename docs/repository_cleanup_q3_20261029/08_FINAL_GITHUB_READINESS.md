# Final GitHub readiness

STATUS: COMPLETED_WITH_ISSUES.
BASELINE_SHA: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
QUERY2_SHA: `2ed093db6c31b905d533693c8f6fe8ef70fe3833`.
QUERY3_BRANCH: `cleanup/repository-q3-final-polish-20261029`.
QUERY3_FINAL_SHA: the exact verified delivery SHA is recorded in the [PR #31 description](https://github.com/SoroushVahidi/combinatorial-opt-agent/pull/31); the containing Git commit identifies this final receipt without a self-referential hash.
VERIFIED_IMPLEMENTATION_AND_REPORT_SHA: `410928bf0856acc69d5c3656e41d6eff603f2a99` (final receipt is a following documentation-only commit).
MAIN_SHA_BEFORE: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
MAIN_SHA_AFTER: `d3071ba31eaf7babdf1e7fe234d8527191ed93de`.
PULL_REQUEST_URL: https://github.com/SoroushVahidi/combinatorial-opt-agent/pull/31
MERGE_STATUS: NOT_MERGED; PR is open and ready for review.

The isolated branch was pushed and independently matched with `git ls-remote`.
PR #31 targets main and includes cumulative Query 2 + Query 3 changes. The initial
implementation/report commit passed both push and pull-request maintenance CI:
[push run](https://github.com/SoroushVahidi/combinatorial-opt-agent/actions/runs/40962701350),
[PR run](https://github.com/SoroushVahidi/combinatorial-opt-agent/actions/runs/40962752659). GitHub reported CLEAN merge state at that
snapshot. This receipt changes documentation only; final-head CI status is recorded
in the PR and final delivery response rather than treating an earlier run as final-head evidence.

No merge was attempted: the two baseline-confirmed dataset-backed regressions still
fail when explicitly enabled. The user's all-required-tests-passing integration
condition is therefore not met. Main's active ruleset includes code-quality errors
and deletion protection. No protection bypass, force update or rule change occurred.

## Scope and preservation

FILES_MODIFIED: 23 versus Query 2.
FILES_ADDED: 22 versus Query 2.
FILES_DELETED: 0 in Query 3; six approved Query 2 removals in the cumulative PR.
BRANCHES_DELETED: 0. BRANCHES_PRESERVED: all preexisting local/remote refs and tags.
BRANCHES_REQUIRING_INVESTIGATION: all six historical names preserved:
`audit/nl4opt-relationship-qc-20261027`, `copilot/inspect-downstream-slot-grounding`,
`copilot/second-hard-family-pass-problem-7`, `kais-final-submission-prep`,
`research/nl4opt-human-annotation-20261027`, `research/nl4opt-three-model-labeling-20261027`.
WORKTREES_REMOVED: 0. WORKTREES_PRESERVED: 32 originals plus the new Query 3 worktree.

TEST_RESULTS: 38 minimal tests passed; default root suite 1,610 passed/9 skipped;
adapter suite 218 passed/7 skipped. Explicit hash-seeded regressions: 8 passed/2 failed
identically at Query 2 and Query 3. 404 Python and 62 shell/SLURM syntax checks passed.
DOCUMENTATION_RESULTS: 317 repository-relative destinations checked with zero errors
before this final receipt; rerun after receipt recorded in final delivery validation.

MANUSCRIPT_RELEASE_PRESERVED: YES.
EVIDENCE_RELEASE_PRESERVED: YES.
HUGGING_FACE_PRESERVED: YES.
SCIENTIFIC_DATA_PRESERVED: YES.
UNPUBLISHED_WORK_PRESERVED: YES, within the observation boundaries in report 06.

Both public releases and all ten assets were independently authenticated before and
after implementation. All 172 evidence archive entries and the exact HF manifest
match; five source ZIPs pass integrity checks. Scientific payloads remain unchanged.
The original checkout, dirty files and 948 ignored-file observations were preserved.

## Remaining issues

- Resolve/review the two historical aggregate test expectations separately; no
  numerical expectations or scientific records were changed to obtain green tests.
- Owner/cluster inactivity is not established. All 21 conditional candidate branch
  names (25 ref observations) and six candidate worktrees were preserved. Open PR #30
  and all investigation/KEEP/release branches remain intact.
- Existing third-party redistribution questions require source-rights review and a
  preservation plan before any removal or additional raw-data publication.
- The other 18 unresolved Query 2 file candidates remain unchanged. Empty evaluation
  placeholders, the legacy destructive cleanup script and demo helper are retained.
- Live provider/model/solver and Cardinal upstream-schema validation were not run.
  Historical external endpoints and ignored-file bytes were not exhaustively audited.

## Exact Query 3 file scope

| Change | Path |
|---|---|
| M | `.github/workflows/check-hf-access.yml` |
| M | `.github/workflows/collect_data.yml` |
| M | `.github/workflows/downstream_benchmark.yml` |
| A | `.github/workflows/maintenance.yml` |
| M | `.github/workflows/nlp4lp.yml` |
| M | `CONTRIBUTING.md` |
| M | `README.md` |
| M | `baselines/pamop/tests/test_data.py` |
| A | `conftest.py` |
| M | `docs/CURRENT_STATUS.md` |
| A | `docs/DATA_ACCESS_AND_LICENSING.md` |
| A | `docs/JDSA_REPRODUCIBILITY.md` |
| M | `docs/README.md` |
| M | `docs/REPO_STRUCTURE.md` |
| M | `docs/REVIEWER_GUIDE.md` |
| A | `docs/TESTING.md` |
| A | `docs/repository_cleanup_q3_20261029/01_FINAL_REPOSITORY_STRUCTURE.md` |
| A | `docs/repository_cleanup_q3_20261029/02_CODE_AND_TEST_REPAIRS.md` |
| A | `docs/repository_cleanup_q3_20261029/03_DOCUMENTATION_AND_LINK_AUDIT.md` |
| A | `docs/repository_cleanup_q3_20261029/04_BRANCH_CLEANUP_RESULTS.md` |
| A | `docs/repository_cleanup_q3_20261029/05_WORKTREE_CLEANUP_RESULTS.md` |
| A | `docs/repository_cleanup_q3_20261029/06_REPRODUCIBILITY_PRESERVATION.md` |
| A | `docs/repository_cleanup_q3_20261029/07_VALIDATION_RESULTS.md` |
| A | `docs/repository_cleanup_q3_20261029/08_FINAL_GITHUB_READINESS.md` |
| A | `docs/repository_cleanup_q3_20261029/CONSOLIDATION_INVENTORY.json` |
| A | `docs/repository_cleanup_q3_20261029/INTEGRITY_RESULTS.json` |
| A | `docs/repository_cleanup_q3_20261029/RELEASE_PRESERVATION.json` |
| A | `docs/repository_cleanup_q3_20261029/WORKTREE_PRESERVATION.json` |
| M | `manuscript/MANUSCRIPT_README.md` |
| M | `pytest.ini` |
| A | `results/README.md` |
| M | `results/eswa_revision/14_reports/FINAL_ESWA_EXPERIMENT_SUMMARY.md` |
| M | `scripts/check_docs_integrity.py` |
| A | `scripts/check_repository_syntax.py` |
| M | `scripts/get_cardinal_nl4opt.py` |
| M | `tests/conftest.py` |
| M | `tests/test_baselines.py` |
| A | `tests/test_cardinal_downloader.py` |
| M | `tests/test_catalog_enrichment.py` |
| A | `tests/test_documentation_integrity.py` |
| A | `tests/test_offline_policy.py` |
| M | `tests/test_selective_grounding_rerank_production.py` |
| M | `tests/test_short_query.py` |
| M | `tests/test_topk_schema_rerank_stage_a.py` |
| M | `tools/run_eaai_final_solver_attempt.py` |

[Branch](https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/cleanup/repository-q3-final-polish-20261029) · [Validation](07_VALIDATION_RESULTS.md) · [Preservation](06_REPRODUCIBILITY_PRESERVATION.md).
