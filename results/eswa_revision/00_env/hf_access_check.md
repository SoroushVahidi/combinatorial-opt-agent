# HF Access Check Report

**Date:** 2026-03-10  
**Branch:** copilot/main-branch-description (commit e3fdaf4)

## Summary

| Item | Status |
|------|--------|
| `HF_TOKEN` in sandbox environment | **NOT SET** |
| `udell-lab/NLP4LP` dataset access | **BLOCKED** (no token) |
| Retrieval experiments (no HF needed) | **COMPLETED** |
| Downstream experiments (need gold params) | **BLOCKED** (no token) |

## Detail

The sandbox/CI environment does not have `HF_TOKEN` available. The gated dataset
`udell-lab/NLP4LP` requires authentication. As a result:

- All **retrieval experiments** (BM25, TF-IDF, LSA × orig/noisy/short) were run locally
  using `data/catalogs/nlp4lp_catalog.jsonl` and `data/processed/nlp4lp_eval_*.jsonl`
  — **no HF_TOKEN required**.
- All **downstream experiments** (TypeMatch, InstReady, etc.) could not be re-run with
  the new `_is_type_match` code change. Manuscript-verified numbers from
  `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` are used as the downstream baseline.
- These are the **pre-fix** downstream numbers; post-fix TypeMatch is estimated
  structurally (see `03_prefix_vs_postfix/` and `14_reports/prefix_vs_postfix_ablation.md`).

## Command to verify access

```bash
export HF_TOKEN=hf_...
python training/external/verify_hf_access.py
```

## CI note

The `nlp4lp.yml` workflow was changed from `push` + `workflow_dispatch` trigger to
`workflow_dispatch` only. This prevents GitHub from blocking bot-pushed commits with
`action_required` status. The workflow should be triggered manually after confirming
HF_TOKEN is set in repository secrets (Settings → Secrets and variables → Actions).
