# How to Run the NLP4LP Benchmark

> **TL;DR — The "Run workflow" button is missing because the workflow files are not on the
> `main` branch yet. Merge this PR first, then the button appears.**

---

## Why the button is missing

GitHub only shows the **"Run workflow"** button for `workflow_dispatch` workflows that exist on
the repository's **default branch** (`main`).

Current state:
- `main` has **no `.github/workflows/` directory**
- All 3 workflow files live only on `copilot/main-branch-description` (this PR branch)
- Result: the button never appears, regardless of which branch you are viewing

---

## Step 1 — Merge this PR

1. Go to https://github.com/SoroushVahidi/combinatorial-opt-agent/pull/3
2. Click **"Ready for review"** (converts from draft)
3. Click **"Merge pull request"** → **"Confirm merge"**

Once merged, the 3 workflow files will be on `main` and GitHub will immediately show the
**"Run workflow"** button for all three workflows.

---

## Step 2 — Verify HF token (quick test, ~60 seconds)

Before running the multi-hour benchmark, confirm your `HF_TOKEN` secret is correct:

1. Go to https://github.com/SoroushVahidi/combinatorial-opt-agent/actions
2. In the left sidebar, click **"Check HF access"**
3. Click the **"Run workflow"** button (top-right of the run list)
4. Leave branch as `main`, click the green **"Run workflow"** button
5. Wait ~60 seconds — look for a green ✅

**If it fails (red ✗):** Go to  
Settings → Secrets and variables → Actions → **New repository secret**  
Name: `HF_TOKEN`  
Value: your token from https://huggingface.co/settings/tokens (needs read access to `udell-lab/NLP4LP`)

---

## Step 3 — Run the full benchmark (~2–3 hours)

1. Go to https://github.com/SoroushVahidi/combinatorial-opt-agent/actions
2. In the left sidebar, click **"NLP4LP benchmark"**
3. Click the **"Run workflow"** button
4. Leave branch as `main`, click the green **"Run workflow"** button
5. Wait 2–3 hours — the job runs 30 experiment settings × ~4 minutes each
6. Results are automatically committed back to the branch

---

## Alternative: trigger via GitHub CLI (no merge required)

If you cannot merge the PR yet, you can trigger any workflow directly on the feature branch
using the GitHub CLI:

```bash
# Install GitHub CLI if needed: https://cli.github.com/

# Quick HF access check (~60s):
gh workflow run check-hf-access.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref copilot/main-branch-description

# Full benchmark (~2-3 hours):
gh workflow run nlp4lp.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref copilot/main-branch-description
```

---

## What happens when the benchmark runs

The workflow (`nlp4lp.yml`) executes these steps in sequence:

| Step | Script | Duration |
|------|--------|----------|
| Verify HF token | `verify_hf_access.py` | ~5s |
| Build NLP4LP eval sets | `build_nlp4lp_benchmark.py` | ~10 min |
| Run downstream benchmark (10 methods × 3 variants × 3 runs) | `run_full_downstream_benchmark.py` | ~2 h |
| Upload artifacts | GitHub Actions artifact upload | ~1 min |
| Commit results to branch | `git push` | ~10s |

Result files written:
- `results/eswa_revision/02_downstream_postfix/` — per-query CSVs
- `results/eswa_revision/13_tables/postfix_main_metrics.csv` — aggregate metrics table
- `results/eswa_revision/14_reports/postfix_main_metrics.md` — human-readable report
- `results/paper/nlp4lp_downstream_summary.csv` — paper-ready summary
