# How to run the NLP4LP benchmark via GitHub Actions

This document explains how to trigger the **optional** NLP4LP downstream benchmark CI workflow (utility CSVs and related paths). It is **not** the same artifact as the **camera-ready EAAI tables** in `results/paper/eaai_camera_ready_tables/` — see **[`RESULTS_PROVENANCE.md`](RESULTS_PROVENANCE.md)** for which numbers are manuscript-authoritative.

For **local** reproduction of EAAI subset experiments, see **[`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md)**.

---

## Prerequisites

The workflow needs a **`HF_TOKEN`** repository secret with read access to the gated **`udell-lab/NLP4LP`** dataset.

1. **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HF_TOKEN`
3. Value: token from https://huggingface.co/settings/tokens (with dataset access approved)

---

## Step 1 — Verify HF token (~1 minute)

1. https://github.com/SoroushVahidi/combinatorial-opt-agent/actions
2. Workflow **“Check HF access”** (`check-hf-access.yml`)
3. **Run workflow** on `main` (or your branch)

---

## Step 2 — Run the NLP4LP benchmark workflow

1. Actions → **“NLP4LP benchmark”** (`nlp4lp.yml`)
2. **Run workflow** on the desired branch

**Runtime:** The workflow header documents **roughly a few minutes** on a typical runner (dataset fetch + `training/external/run_full_downstream_benchmark.py`). It is **not** a multi-hour job unless the runner or network is unusually slow.

**Scripts involved (authoritative paths in the YAML):**

| Step | Script |
|------|--------|
| Verify access | `training/external/verify_hf_access.py` |
| Build eval JSONL | `training/external/build_nlp4lp_benchmark.py` |
| Downstream loop | `training/external/run_full_downstream_benchmark.py` |

**Outputs:** May update `results/paper/nlp4lp_downstream_summary.csv`, `results/paper/nlp4lp_downstream_types_summary.csv`, and paths under `results/eswa_revision/` as configured in the workflow commit step. **Compare** any new numbers to **`RESULTS_PROVENANCE.md`** before treating them as manuscript headlines.

---

## GitHub CLI

```bash
gh workflow run check-hf-access.yml --repo SoroushVahidi/combinatorial-opt-agent --ref main
gh workflow run nlp4lp.yml --repo SoroushVahidi/combinatorial-opt-agent --ref main
```

---

## Optional LLM baselines (OpenAI / Gemini)

Not driven by this Actions workflow. Use **`tools/nlp4lp_downstream_utility.py`** and Slurm batch scripts under **`batch/learning/`**. Gemini infrastructure is documented in **[`GEMINI_RERUN_REPORT.md`](GEMINI_RERUN_REPORT.md)**.
