# How to Reproduce Paper Results

This document provides canonical commands for reproducing the EAAI manuscript results.

---

## Prerequisites

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for running tests
```

### 2. HuggingFace token (required for downstream metrics only)

Retrieval metrics can be reproduced without a HF token.
Downstream grounding metrics (TypeMatch, Exact20, InstantiationReady) require
approved access to the gated `udell-lab/NLP4LP` dataset.

```bash
cp .env.example .env
# Edit .env and set:  HF_TOKEN=hf_...
# Get a read-only token at: https://huggingface.co/settings/tokens
# Request access at: https://huggingface.co/datasets/udell-lab/NLP4LP
export $(grep -v '^#' .env | xargs)
```

---

## Step 1 — Validate repo integrity (no HF token needed)

```bash
python scripts/paper/run_repo_validation.py
```

This checks that all required tables, figures, analysis reports, and canonical
documents are present.  Expected: all checks pass.

---

## Step 2 — Reproduce retrieval metrics (no HF token needed)

```bash
python -m training.run_baselines \
  --splits data/processed/splits.json --split test \
  --regenerate --num 331 --seed 999 --k 10 \
  --baselines bm25 tfidf lsa --results-dir results
```

Expected output:
- `results/baselines_test.csv` — per-query results
- `results/nlp4lp_retrieval_summary.csv` — aggregated R@1, R@5, MRR

Canonical numbers (Table 1):

| Method | Variant | Schema R@1 |
|--------|---------|-----------|
| TF-IDF | orig | 0.9094 |
| TF-IDF | noisy | 0.9033 |
| TF-IDF | short | 0.7855 |
| BM25 | orig | 0.8822 |

---

## Step 3 — Reproduce downstream grounding metrics (HF token required)

```bash
# Engineering structural subset — Table 2 (60 instances)
python tools/run_eaai_engineering_subset_experiment.py

# Executable-attempt study — Table 3 (269 instances)
python tools/run_eaai_executable_subset_experiment.py

# Solver-backed restricted subset — Table 4 (20 instances, SciPy HiGHS shim)
python tools/run_eaai_final_solver_attempt.py
```

Expected canonical results:
- Table 2: structural consistency breakdown for 60 instances
- Table 3: executable-attempt rates with documented blockers for 269 instances
- Table 4: TF-IDF achieves **80% feasible** on the 20-instance solver subset

---

## Step 4 — Regenerate camera-ready figures

```bash
pip install "Pillow>=9.0.0"
python tools/build_eaai_camera_ready_figures.py
```

Figures are written to `results/paper/eaai_camera_ready_figures/`.

---

## Step 5 — Run the test suite

```bash
python -m pytest tests/ -q
```

Expected: ~1 400 tests pass on a CPU-only environment (no HF token, no GPU needed).
Tests tagged `requires_network` are automatically skipped when `HF_TOKEN` is absent.

---

## Blocked reproducibility paths

| Path | Blocker | Fallback |
|------|---------|---------|
| Full downstream rerun | `udell-lab/NLP4LP` gated on HuggingFace | Pre-computed results in `results/paper/` |
| Learned model training | GPU + `torch` required | Inspect `src/learning/` and `training/`; no checkpoint produced |
| Full benchmark-wide solver validation | External LP/ILP solver | SciPy HiGHS shim covers the 20-instance restricted subset |

---

## Canonical artifact locations

| Artifact | Location |
|---------|---------|
| Main benchmark tables (Tables 1–5) | `results/paper/eaai_camera_ready_tables/` |
| Camera-ready figures (Figures 1–5) | `results/paper/eaai_camera_ready_figures/` |
| Experiment reports | `analysis/eaai_*_report.md` |
| Results provenance | `docs/RESULTS_PROVENANCE.md` |
| Paper framing authority | `docs/EAAI_SOURCE_OF_TRUTH.md` |

---

## External validation datasets (Text2Zinc & CP-Bench) — staging only

These steps **do not** reproduce EAAI Tables 1–5. They only **stage** optional corpora for future evaluation design.

**CP-Bench (DCP-Bench-Open, public):**

```bash
python scripts/datasets/get_cp_bench_open.py
```

**Text2Zinc (gated Hugging Face):** request access to `skadio/text2zinc`, set `HF_TOKEN`, then:

```bash
python scripts/get_text2zinc.py
```

**Capability smoke check (any registered adapter with local splits):**

```bash
python tools/run_dataset_benchmarks.py --dataset cp_bench   # after staging CP-Bench JSONL
python tools/run_dataset_benchmarks.py --dataset text2zinc # after Text2Zinc JSONL exists
```

See [`DATASET_EXPANSION_STATUS.md`](DATASET_EXPANSION_STATUS.md) for what is and is not benchmarked.

---

## Optional — full downstream utility CSVs and LLM baselines

The **camera-ready** headline row for typed greedy + TF-IDF is in **`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`**, not necessarily identical to every column in **`results/paper/nlp4lp_downstream_summary.csv`** (different metric definitions — see **`docs/RESULTS_PROVENANCE.md`**).

To run the **interactive** downstream utility locally (all variants / methods):

```bash
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf
```

**Optional LLM baselines** require API keys and use **`batch/learning/run_*_llm_baselines.sbatch`** on Slurm or direct CLI with `--method openai` / `gemini` / `mistral`. Gemini: **`docs/GEMINI_RERUN_REPORT.md`**. Mistral (orig-first batch, preflight): **`docs/MISTRAL_RERUN_REPORT.md`**.
