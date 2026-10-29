# Preserved files and unresolved candidates

Baseline: `d3071ba31eaf7babdf1e7fe234d8527191ed93de` (verified current `origin/main`). Cleanup branch: `cleanup/repository-q2-safe-files-20261029`.


No member of the six-file allowlist had to be rejected: all passed independent revalidation. No additional deletion was attempted. The following 19 prior investigation files remain excluded by the execution plan and byte-identical to baseline; their questions are not resolved by this cleanup.

| Path | Verified issue / remaining question | Query 2 disposition |
| --- | --- | --- |
| `scripts/get_cardinal_nl4opt.py` | SyntaxError at line 121 (unmatched brace), with duplicated/incompatible implementation tail. Needed dataset adapter path: repair and validate, do not delete. | PRESERVE; excluded from deletion allowlist |
| `scripts/cleanup_old_artifacts.sh` | Uses missing docs/REPO_CLEANUP_DELETED_FILES_MANIFEST.txt; manifest moved to docs/archive/. Script recursively deletes paths, so do not repoint/run during audit. Decide archival retirement separately. | PRESERVE; excluded from deletion allowlist |
| `launch_and_capture_url.py` | Documented demo helper terminates the app process after capturing its URL. Likely unusable persistent-link behavior; owner intent and integration test needed. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nl4opt_comp_eval.jsonl` | Zero-byte evaluation file; empty is not a usable evaluation dataset. Identify intended generator/consumers and mark unavailable or regenerate only with separate authorization. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nl4opt_family_eval_test.jsonl` | Zero-byte evaluation file; empty is not a usable evaluation dataset. Identify intended generator/consumers and mark unavailable or regenerate only with separate authorization. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nl4opt_family_eval_test_masked.jsonl` | Zero-byte evaluation file; empty is not a usable evaluation dataset. Identify intended generator/consumers and mark unavailable or regenerate only with separate authorization. | PRESERVE; excluded from deletion allowlist |
| `data/processed/resocratic_eval.jsonl` | Zero-byte evaluation file; empty is not a usable evaluation dataset. Identify intended generator/consumers and mark unavailable or regenerate only with separate authorization. | PRESERVE; excluded from deletion allowlist |
| `results/eswa_revision/00_env/nlp4lp_gold_cache.json` | 331 reference records include parameters, problem_info, optimus_code and solution. Contradicts broad no-raw-cache redistribution wording; preserve privately pending access/license review, do not unilaterally delete or republish. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nlp4lp_eval.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nlp4lp_eval_orig.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nlp4lp_eval_noisy.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nlp4lp_eval_short.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nlp4lp_eval_noentity.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/nlp4lp_eval_nonum.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/catalogs/nlp4lp_catalog.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/baselines/pamop/nlp4lp_pamop_subset.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/all_problems.json` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/processed/all_problems_extended.json` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |
| `data/external_real_examples/collected_examples.jsonl` | Contains source-derived catalog/query/problem material. Inventory is not proof of redistribution rights; review source provenance and public/private boundary. Preserve until resolved. | PRESERVE; excluded from deletion allowlist |

## Important unpublished and historical evidence

Preserve local M2/M3 reports in the primary checkout; M3 corrected evaluators; M4/M5 original records, prompts, manifests, environments, recovery/validity ledgers and author review; D1/G2 diagnostic work; NL4Opt label recovery/review; joint/factorial/enumeration records and ignored debugging traces; and all scientific-revision drafts. The current worktree inventory records dirty/ignored counts without republishing private bodies. No original worktree was edited.

The published v2 evidence release now provides sanitized saved-record material; this supersedes the old audit's statement that no public M5-derived release exists. It does not publish complete local M5 originals or resolve source-access/redistribution rights. A public summary cannot substitute for originals. Negative/equal-result experiments, statistical inputs, historical predictions, solver configurations, provenance and environment records all remain protected even when uncited in the current manuscript.

## Protected permanent anchors

- Manuscript tag `jdsa-manuscript-v1.0.0` → `00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc`.
- Manuscript report commit `0db80afe6122b1b20fd28d4f3a440001e645cdf1`, retained on its existing remote branch.
- Evidence tag `jdsa-evidence-v2.0.0` → `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`.
- Hugging Face revision `f94012872211ab6a828147bb138fbe8ab695b982`, including the exact published version directory.
- All other existing tags, branch tips and their reachable histories remain intact. No branch or worktree is deleted in Query 2.

The retained historical SNCS PDF and every JDSA/main source-package byte remain unchanged. The 71 frozen results newly allowed by `.gitignore` remain tracked without content changes; new unreviewed run/raw files stay ignored. The original eight cleanup-audit reports and all untracked research artifacts remain in their original worktrees.
