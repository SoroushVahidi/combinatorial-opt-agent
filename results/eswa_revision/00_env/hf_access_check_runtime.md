# HF Access Check — Runtime Status

**Date:** 2026-03-10  
**Branch:** copilot/main-branch-description (commit e6e7a59)  
**Status:** ⏳ AWAITING GITHUB ACTIONS TRIGGER

## Summary

| Check | Result |
|-------|--------|
| `huggingface.co` DNS in sandbox | **BLOCKED** — no address associated with hostname |
| GitHub API `actions/dispatches` endpoint in sandbox | **BLOCKED** — DNS monitoring proxy, HTTP 403 |
| `HF_TOKEN` in sandbox environment | **NOT SET** |
| `HF_TOKEN` in GitHub Actions secrets | **CONFIGURED** (per repo owner's statement) |
| Benchmark code committed to PR branch | **YES** — commit e6e7a59 |
| `nlp4lp.yml` workflow updated with downstream benchmark | **YES** |
| `downstream_benchmark.yml` standalone workflow added | **YES** |

## Why experiments did not run in the Copilot sandbox

The Copilot SWE agent runs in a restricted sandbox environment. Two independent
firewall policies block the required external access:

1. **HuggingFace blocked**: `huggingface.co` DNS lookup returns
   `[Errno -5] No address associated with hostname`. The `udell-lab/NLP4LP`
   gated dataset is not reachable from this environment.

2. **GitHub Actions API blocked**: `api.github.com/actions/*/dispatches`
   (workflow_dispatch endpoint) returns HTTP 403 "Blocked by DNS monitoring proxy".
   This prevents programmatic triggering of the workflow from inside the sandbox.

## What HAS been prepared (all code is ready)

### Pipeline script
`training/external/run_full_downstream_benchmark.py`:
- Verifies HF access and writes runtime env report
- Loads gold parameters from `udell-lab/NLP4LP` (cached as JSON after first load)
- Runs **9 deterministic methods × 3 variants = 27 settings** + **3 random controls** = **30 total**
- Simulates pre-fix behavior (patches `_is_type_match` in memory) for ablation
- Writes results to `results/eswa_revision/` with `source: measured` label
- All results labeled as "newly measured" — NO manuscript-era estimates

### GitHub Actions workflow (`.github/workflows/nlp4lp.yml`)
Updated to run in 3 phases:
1. Verify HF token + dataset access
2. Build NLP4LP eval sets (existing)  
3. **Run full downstream benchmark** (all 30 settings + pre/post-fix ablation)
4. Commit measured results back to branch (`contents: write` permission)
5. Upload all artifacts (90-day retention)

## HOW TO TRIGGER (repo owner action required)

**Steps:**
```
1. Go to: https://github.com/SoroushVahidi/combinatorial-opt-agent/actions
2. Click "NLP4LP benchmark" workflow in the left sidebar
3. Click "Run workflow" button (top right of the run list)
4. In the dropdown, select branch: copilot/main-branch-description
   ^^^ IMPORTANT: must select THIS branch (not main) to use the updated workflow
5. Click the green "Run workflow" button
```

The workflow will:
- Verify HF_TOKEN access to udell-lab/NLP4LP
- Run all 30 experiment settings (estimated ~2 hours)
- Commit the measured results directly to this branch
- Upload artifacts with 90-day retention

## Expected outputs after successful run

All results will have `source: measured` (no estimates):
- `results/eswa_revision/02_downstream_postfix/` — per-query CSVs + JSON (30+ files)
- `results/eswa_revision/13_tables/postfix_main_metrics.csv` — aggregate table
- `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` — ablation
- `results/eswa_revision/14_reports/postfix_main_metrics.md` — markdown report
- `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md` — ablation
- `results/paper/nlp4lp_downstream_summary.csv` — master summary CSV
- `results/paper/nlp4lp_downstream_types_summary.csv` — per-type breakdown
