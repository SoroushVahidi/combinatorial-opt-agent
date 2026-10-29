# Query 2 validation results

Baseline: `d3071ba31eaf7babdf1e7fe234d8527191ed93de` (verified current `origin/main`). Cleanup branch: `cleanup/repository-q2-safe-files-20261029`.


**Query 2 scope checks pass; pre-existing unrelated defects remain.** No dependency installation, model/solver API call, scientific solver invocation, training or expensive historical experiment was used.

| Check | Result |
|---|---|
| Audit recovery | All eight report hashes recorded; all 1,486 baseline path/size/SHA-256 rows match current public main |
| Exact diff | Six deletions, four authorized baseline-file modifications, only Query 2 documentation additions |
| Preserved files | 1,480 baseline paths survive; 1,476 unchanged bytes, including all 19 investigation files and all scientific/source/test/config records |
| Frozen ignore allowlist | All 72 exact existing paths allowed (71 results plus raw placeholder); 21 new private/run/output probes blocked; canonical PDF/ZIP paths remain trackable |
| Existing repository hygiene tests | 15 passed before changes |
| Relevant existing tests after cleanup | 22 passed: `tests/test_repo_hygiene.py` and `tests/test_consistency_benchmark.py`; synthetic outputs use pytest temporary directories |
| Imports | `validation.solve_and_verify` and `src.analysis.consistency_benchmark` import successfully; no solve function invoked |
| Python syntax | 398/399 pass; the same baseline `scripts/get_cardinal_nl4opt.py:121` unmatched brace remains; unchanged escape warning in `tools/run_eaai_final_solver_attempt.py` |
| Shell syntax | All 62 `.sh`/`.slurm`/`.sbatch` files pass `bash -n` via stdin; none executed |
| Existing documentation checker | `python -B scripts/check_docs_integrity.py` passes before and after cleanup; its limited canonical-doc scope does not detect the wider historical-link defect |
| Tracked Markdown target scan | One missing relative target before and after, no new missing target: `results/eswa_revision/14_reports/FINAL_ESWA_EXPERIMENT_SUMMARY.md` → `../../../docs/EXPERIMENT_AUDIT.md` |
| Source archives | All four existing ZIP CRC checks pass; all archive bytes unchanged |
| Git integrity | `git fsck --full --no-reflogs --no-dangling` passes |
| Releases | Anonymous before/after downloads: four manuscript assets and six evidence assets; metadata/tag targets/digests identical; all 172 evidence-archive files authenticate |
| Hugging Face | Exact-version tree returns HTTP 200; downloaded manifest is byte-identical to the GitHub evidence archive before/after |
| Worktree/ref preservation | All 31 existing worktree HEADs, statuses, changed-file hashes and ignored listings match the initial snapshot; existing refs/tags are preserved |

## Commands and comparison method

Safe fetch: `git -c fetch.prune=false -c fetch.pruneTags=false fetch --no-prune --no-tags origin 'refs/heads/*:refs/remotes/origin/*'`. Read-only inventories used `git status`, `git for-each-ref`, `git ls-remote`, `git worktree list --porcelain`, and `git ls-tree`/`git cat-file --batch`. `GIT_OPTIONAL_LOCKS=0` was used while inspecting existing worktrees. A single `/proc/*/cwd` snapshot identified observed local use; it does not establish remote cluster inactivity.

Tests used `python -B`, disabled external pytest plugin autoload and the cache provider, and patched `socket.create_connection` to raise an offline error before calling `pytest.main`. This prevents the existing conftest's unconditional reachability probe from making a network connection. The arguments were `-q -o addopts= -p no:cacheprovider tests/test_repo_hygiene.py tests/test_consistency_benchmark.py`. No new tests mirror the file removals.

Python parsing used `ast.parse` on tracked Python bytes; the failing source was compared byte-for-byte with the baseline. Shell checks used `bash -n` on source text. ZIP checks used `zipfile.testzip()`. The Markdown comparison resolves inline relative file/directory links across tracked Markdown and excludes fenced code, external URLs and anchors; it does not certify all possible Markdown syntax. Current scientific results were protected by exact blob/hash preservation, not regenerated.

Release checks used unauthenticated HTTPS GETs, SHA-256, ZIP manifest/checksum validation and exact remote refs. The evidence archive has **170 manifest payload entries plus `release_manifest.json` and `CHECKSUMS.sha256` = 172 files**. No release or Hugging Face write was made. The frozen arithmetic verifier was not rerun for this file-only cleanup: its entire published code/input package is authenticated unchanged.

[Machine-readable validation](VALIDATION_RECEIPT.json), [release verification](RELEASE_PRESERVATION.json), [audit recovery](AUDIT_RECOVERY.json), [branch/worktree inventory](HANDOFF_INVENTORY.json), [worktree preservation](WORKTREE_PRESERVATION.json).

## Existing failures versus new failures

New failures: **none**. The Cardinal syntax error, historical missing link and invalid-escape warning predate Query 2 and remain outside its repair scope. The correct existing link target is `docs/archive/EXPERIMENT_AUDIT.md`; repairing that unrelated historical result-report link is deferred to Query 3. Historical workflow/data-access/environment issues remain recommendations, not completed fixes.
