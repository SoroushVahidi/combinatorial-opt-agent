# How to Run the NLP4LP Benchmark via GitHub Actions

This document explains how to trigger the NLP4LP benchmark CI workflow.
For local reproduction steps, see **[`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md)**.

---

## Prerequisites

The GitHub Actions workflow requires a `HF_TOKEN` repository secret with read access
to the gated `udell-lab/NLP4LP` dataset.

### Add the HF_TOKEN secret

1. Go to **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HF_TOKEN`
3. Value: your token from <https://huggingface.co/settings/tokens>
   (requires approved access to `udell-lab/NLP4LP`)

---

## Step 1 — Verify HF token (~60 seconds)

Before running the full benchmark, confirm your token works:

1. Go to <https://github.com/SoroushVahidi/combinatorial-opt-agent/actions>
2. In the left sidebar, click **"Check HF access"**
3. Click **"Run workflow"** → leave branch as `main` → click the green **"Run workflow"** button
4. Wait ~60 seconds — look for a green ✅

---

## Step 2 — Run the full benchmark (~2–3 hours)

1. Go to <https://github.com/SoroushVahidi/combinatorial-opt-agent/actions>
2. In the left sidebar, click **"NLP4LP benchmark"**
3. Click **"Run workflow"** → leave branch as `main` → click the green **"Run workflow"** button
4. Wait 2–3 hours (30 experiment settings × ~4 minutes each)
5. Results are automatically committed back to the branch

---

## Alternative: trigger via GitHub CLI

```bash
# Quick HF access check (~60s):
gh workflow run check-hf-access.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref main

# Full benchmark (~2–3 hours):
gh workflow run nlp4lp.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref main
```

---

## What the benchmark workflow produces

| Step | Script | Duration |
|------|--------|----------|
| Verify HF token | `verify_hf_access.py` | ~5 s |
| Build NLP4LP eval sets | `build_nlp4lp_benchmark.py` | ~10 min |
| Run downstream benchmark | `run_full_downstream_benchmark.py` | ~2 h |
| Upload artifacts | GitHub Actions artifact upload | ~1 min |
| Commit results to branch | `git push` | ~10 s |

Result files written to `results/paper/` and committed back to the branch.
