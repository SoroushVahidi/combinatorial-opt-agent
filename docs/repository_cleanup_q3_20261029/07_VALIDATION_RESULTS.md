# Validation results

## Completed checks

| Command/check | Result |
|---|---|
| `python scripts/check_repository_syntax.py` | PASS: 404 Python files, 62 shell/SLURM scripts; warnings treated as errors |
| `python scripts/check_docs_integrity.py` | PASS: zero broken tracked destinations/conflict markers; final count recorded in readiness receipt |
| Minimal five-file suite documented in TESTING.md | PASS: 38 tests, including 14 Cardinal fixtures |
| `python -m pytest -q` | 1,610 passed, 9 skipped; no provider/model/solver calls |
| `python -m pytest -q baselines/ --import-mode=importlib` | 218 passed, 7 skipped; live dataset tests not enabled |
| Final dataset-capability marker tests (PaMOP data and two aggregate-regression modules) | 31 passed, 9 skipped; skips are not passes |
| `PYTHONHASHSEED=0` targeted aggregate regressions | 8 passed, 2 failed, identically on Query 2 and Query 3 |
| Import smoke checks | Cardinal downloader, documentation checker, syntax checker and retrieval/data-adapter imports exercised without downloads |
| Source ZIP integrity | PASS: four tracked archives and final public source ZIP, CRC and unchanged hashes |
| Git integrity: `git fsck --full --no-reflogs` | Exit 0; existing dangling objects retained, no pruning |
| `git diff --check` and exact Query 2 comparison | PASS; no scientific payload changes or Query 3 deletions |
| Release/HF preservation | PASS; all assets independently downloaded twice and authenticated |
| Existing-worktree preservation | PASS within recorded HEAD/status/hash/path-size scope; original checkout untouched |
| Workflow YAML/triggers/permissions | Parsed all five workflows; read-only contents; research dispatch only; no generated-result pushes/raw uploads |

Python 3.12.3, pytest 9.0.3 were used locally. These describe this maintenance run,
not historical scientific execution versions. No large dependency installation,
model inference, training, external data acquisition or scientific solver run occurred.
The existing dataset-backed CPU regression tests use local records and temporary
outputs only. Network accesses were limited to public artifact/GitHub verification,
official workflow documentation, fetch/push and PR operations.

## Existing failures and limitations

Two existing tests require `PYTHONHASHSEED=0` to exercise their full assertions:

1. `tests/test_selective_grounding_rerank_production.py::test_production_candidate_regression`:
   observed `0.824773413897281`, expected `265 / 331`.
2. `tests/test_topk_schema_rerank_stage_a.py::test_full_diagnostic_preserves_current_baseline`:
   observed `current_ready == 265`, expected `257`.

The same tests were run against an extracted exact Query 2 Git archive. Only the
new offline test policy/helper was supplied to prevent baseline collection-time
network probes; scientific code, fixtures and data remained at Query 2. Both
failures and values reproduced identically. No numerical expectations or scientific
implementation were changed to make them pass. With final policy, use
`PYTHONHASHSEED=0 python -m pytest --run-research-data -q` and these two test paths
to reproduce them. These baseline failures block automatic integration pending review.

The default root suite's nine skips comprise seven network/model tests and these
two dataset-backed aggregate tests. Adapter skips are seven live dataset fetches.
Live Cardinal input/schema, provider credentials, model downloads, solver execution,
cluster inactivity and exhaustive historical external-link validity are unverified.
No skipped capability is claimed as tested. The socket guard is a Python test
policy, not an OS/network sandbox for arbitrary subprocesses.

Raw local test logs remain under `/tmp/cleanup-q3/`; no private dataset content is
copied into the public report. Machine receipts retain only permitted metadata.
