# HF Access Check — Runtime Status

**Date:** 2026-03-10  
**Branch:** copilot/main-branch-description  
**Commit:** 8aa9507 (run_full_downstream_benchmark.py + downstream_benchmark.yml added)

## Status: ⏳ AWAITING GITHUB ACTIONS TRIGGER

The Copilot SWE agent sandbox environment cannot access HuggingFace or the
GitHub Actions API (both blocked by the sandbox DNS monitoring proxy). However,
ALL code is committed and ready to run in GitHub Actions where HF_TOKEN is available.

## Sandbox environment diagnostics

| Check | Result |
|-------|--------|
| `huggingface.co` DNS | **BLOCKED** (No address associated with hostname) |
| `api.github.com/actions` dispatch | **BLOCKED** (DNS monitoring proxy, HTTP 403) |
| HF_TOKEN in sandbox env | **NOT SET** |
| All benchmark code committed | **YES** (commit 8aa9507) |
| GitHub Actions HF_TOKEN secret | **CONFIGURED** (per repo owner) |

## What has been prepared

1. **`training/external/run_full_downstream_benchmark.py`** — complete pipeline:
   - Verifies HF access before starting
   - Loads gold parameters from `udell-lab/NLP4LP` via HF_TOKEN (cached to JSON)
   - Runs 10 methods × 3 variants = 30 settings + 3 random controls
   - Runs pre-fix vs post-fix ablation (in-memory patch simulation)
   - Writes results to `results/eswa_revision/` paths
   - Commits results back to the branch

2. **`.github/workflows/nlp4lp.yml`** — updated to run the full downstream benchmark:
   - Phase 0: Verify HF access
   - Phase 1: Build NLP4LP eval sets  
   - Phase 2: Run all downstream experiments
   - Phase 3: Commit measured results back to branch

3. **`.github/workflows/downstream_benchmark.yml`** — standalone workflow for benchmark only

## How to run (repo owner action required)

**Option A (recommended): Trigger the updated nlp4lp.yml**

```
GitHub.com → SoroushVahidi/combinatorial-opt-agent
→ Actions tab
→ "NLP4LP benchmark" workflow
→ "Run workflow" button (top right)
→ Select branch: copilot/main-branch-description
→ Click "Run workflow"
```

**Option B: Trigger downstream_benchmark.yml**

```
GitHub.com → Actions → "NLP4LP downstream benchmark (authenticated)"
→ Run workflow → branch: copilot/main-branch-description
```

## Expected outputs after successful run

- `results/eswa_revision/02_downstream_postfix/` — per-query CSVs + JSON (30 files)
- `results/eswa_revision/13_tables/postfix_main_metrics.csv` — aggregate metrics
- `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` — ablation
- `results/eswa_revision/14_reports/postfix_main_metrics.md` — markdown report
- `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md` — ablation report
- `results/paper/nlp4lp_downstream_summary.csv` — master summary
- All results labeled `source: measured` (NOT manuscript estimates)
