# Query 2 execution report

**STATUS: COMPLETED_WITH_ISSUES.** The approved six-file cleanup and its navigation/ignore prerequisites are complete. Existing unrelated syntax and documentation defects remain for Query 3; no new failure was introduced.

Baseline: `d3071ba31eaf7babdf1e7fe234d8527191ed93de` (verified current `origin/main`). Cleanup branch: `cleanup/repository-q2-safe-files-20261029`.


## Recovered audit and current state

All eight original reports were found in the primary checkout's untracked `docs/repository_cleanup_audit_20261029/` directory. Their hashes are preserved in [AUDIT_RECOVERY.json](AUDIT_RECOVERY.json). The full 1,486-row inventory was parsed and every SHA-256 rechecked against current public main; all match. The exact six-path allowlist was recovered from report 06 and cross-checked against report 01's six SAFE_REMOVE rows and report 02's per-file reasons. No replacement list was invented. The original eight reports remain untouched locally; this change does not publish their detailed private-worktree inventories.

The historical 1,486 / 6 / 19 / 1,461 file counts still describe the baseline. Live refs/worktrees have changed: before this cleanup there were 34 local branches, 31 remote branches, 54 distinct names, and 31 worktrees. Historical classifications revalidate to 27 KEEP, 21 conditional cleanup-candidate and six investigation names. Thirteen worktrees have tracked/untracked changes; processes were observed in the primary and NL4Opt-label-audit worktrees. These observations do not prove inactivity elsewhere or describe remote cluster jobs.

No Query 2 report was found at any current branch tip or in any existing worktree; the all-ref subject search found no prior completion commit. All six files still existed with their original hashes. Local main remains `a3f0dde397341a41842a30ddb946e7966706351b`; it was not updated to the newer remote main. A new isolated worktree at `/home/soroush/combinatorial-opt-agent-cleanup-q2` was created from the verified public baseline.

## Exact changes and reasons

- Removed only the four empty placeholders and two exact duplicate SNCS PDFs listed in [02_REMOVED_FILES.md](02_REMOVED_FILES.md). The six remove 702,564 bytes from the current tree, not from Git history. Every parent directory retains real tracked files.
- Added dated relocation notices with a working retained-PDF link and unchanged SHA-256 to `manuscript/sncs/SUBMISSION_METADATA.md`, `manuscript/sncs/FINAL_UPLOAD_MANIFEST.md`, and `docs/SNCS_STAGE4_FINAL_SUBMISSION_AUDIT.md`. All original historical text remains byte-for-byte after the inserted notice.
- Updated `.gitignore` only as report 06 directs: exact allowances for 71 existing frozen evidence paths plus `data/raw/.gitkeep`; deny unreviewed files under those six families; ignore only the two removed generic SNCS build PDFs. A short comment policy requires provenance/public-private review and preserves scientific logs/negative results. The misleading blanket claim that raw data can always be regenerated was replaced with a conservative local-data rule.
- Added the five requested Query 2 reports and supporting metadata-only receipts in this directory. No raw private records or provider conversations are included.

## Scope and preservation

The deletion set is exactly the approved six paths; the only modified baseline files are the three approved navigation documents and `.gitignore`. Of 1,480 surviving baseline files, 1,476 are byte-identical; four contain only the authorized changes. All 19 investigation files, all result/data/source/test/config/environment records, the retained SNCS PDF, all four source archives and all 35 existing main-branch JDSA files are preserved.

Both new permanent releases were independently downloaded and authenticated. Their tags, metadata, assets and checksums are protected, including the 172-file evidence archive (170 manifested payloads plus manifest and checksum). The exact Hugging Face manifest matches that archive. The finalized 15/37-page manuscript remains on its tagged release; main's older JDSA package remains unchanged. No manuscript merge, source migration, dependency addition, scientific recomputation, experiment, branch/tag/worktree deletion, force operation or journal submission occurred.

## Remaining issues

The pre-existing Cardinal downloader syntax error, one historical Markdown link and one Python escape warning remain. Nineteen investigation files and six investigation branches remain unresolved and preserved. Full historical reproduction, raw-source redistribution questions and remote-job inactivity are not established by this cleanup. [Validation](04_VALIDATION_RESULTS.md) distinguishes these limits from successful Query 2 checks; [Query 3 handoff](05_QUERY3_HANDOFF.md) gives the bounded next actions.
