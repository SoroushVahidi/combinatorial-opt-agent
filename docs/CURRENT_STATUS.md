# Current repository status (reviewer-facing)

**Last updated:** 2026-04-03 (Mistral batch defaults + rerun docs)

This file is the **single concise summary** of what this codebase claims today, what is validated,
and where the evidence lives. It complements **[`docs/EAAI_SOURCE_OF_TRUTH.md`](EAAI_SOURCE_OF_TRUTH.md)**
(the manuscript authority document).

---

## Benchmark scope (official)

| Item | Detail |
|------|--------|
| **Dataset** | NLP4LP (`udell-lab/NLP4LP` on HuggingFace — **gated**; approval + token required) |
| **Primary test split** | `orig` variant, **331** queries (fixed catalog retrieval + downstream grounding) |
| **Paper framing** | Companion repo for an **EAAI** manuscript on retrieval-assisted schema grounding + deterministic scalar instantiation |

---

## Canonical headline metrics (public reporting)

**For manuscript-aligned, camera-ready numbers, use:**

`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`

**Table 1 row (TF-IDF + typed greedy, “core” group)** — values copied from that file:

| Field | Value |
|------|------:|
| Schema retrieval R@1 | **0.9094** |
| Coverage (paper table) | **0.8639** |
| Type match (paper table) | **0.7513** |
| Instantiation ready (paper table) | **0.5257** |

**Provenance chain:** Table 1 CSV points at `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` (see `analysis/eaai_tables_build_report.md` for how tables were reconciled).

**Important:** Other CSVs under `results/paper/` (e.g. `nlp4lp_downstream_summary.csv`) use **pipeline-specific column definitions** and are **not** interchangeable with Table 1 without reading the corresponding build notes. Do not mix headline numbers from different files without checking definitions.

---

## What is validated vs not validated

| Area | Status | Notes |
|------|--------|--------|
| **Schema retrieval + deterministic grounding (paper core)** | **Paper-validated** | Tables 1–5, figures, and EAAI experiment scripts under `tools/run_eaai_*.py` |
| **Structural LP checks (no solver)** | **Reproducible** | `formulation/verify.py`; used in engineering subset |
| **Solver-backed subset (SciPy HiGHS shim)** | **Paper-validated (20 instances)** | Table 4; not full benchmark |
| **Web demo (`app.py`)** | **Demo / UX** | Not the main evaluated pipeline; may log queries locally |
| **Open-domain / LLM formulation paths** | **Demo / auxiliary** | Outside main NLP4LP benchmark unless explicitly documented otherwise |
| **Learned retrieval fine-tuning** | **Experimental** | Does not beat rule baseline on held-out eval (see `KNOWN_ISSUES.md`) |
| **Optional LLM baselines (OpenAI / Gemini / Mistral)** | **Optional tooling** | Requires API keys; **not** in camera-ready Tables 1–5. OpenAI downstream artifacts exist under `results/paper/` for some variants; Gemini rerun infra — [`docs/GEMINI_RERUN_REPORT.md`](GEMINI_RERUN_REPORT.md); **Mistral** wiring (preflight + Slurm + `--method mistral`) — [`docs/MISTRAL_RERUN_REPORT.md`](MISTRAL_RERUN_REPORT.md). Do not assume completion without matching `results/rerun/…` or documented output artifacts. |
| **Text2Zinc & CP-Bench (external validation)** | **Integration / adapters only** | Staging + `InternalExample` adapters; **not** paper headline metrics. Text2Zinc: gated HF. CP-Bench: public DCP-Bench-Open JSONL. See [`DATASET_EXPANSION_STATUS.md`](DATASET_EXPANSION_STATUS.md). |

---

## Current limitations (honest)

- **Gated data:** Full NLP4LP gold-parameter evaluation needs HuggingFace access.
- **Not all instances are solver-executable** end-to-end; the paper reports **restricted subsets** for execution-oriented claims.
- **Gurobi is not required** for paper results; the solver-backed study uses a **SciPy HiGHS shim** on a small subset.
- **Metric names differ across artifacts** (paper tables vs downstream utility CSVs); always check the source file and `analysis/eaai_tables_build_report.md`.

---

## Manuscript vs repo-only reruns

- **Canonical for the submitted manuscript story:** camera-ready artifacts in `results/paper/eaai_camera_ready_tables/` plus **`docs/EAAI_SOURCE_OF_TRUTH.md`**.
- **Intermediate / audit documents** (manuscript comparisons, internal decision logs) live under **`docs/archive_internal_status/`** — useful provenance, **not** the headline source of truth.

---

## Quick links

| Need | Location |
|------|----------|
| Reviewer orientation | [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md) |
| Full paper artifact list | [`docs/EAAI_SOURCE_OF_TRUTH.md`](EAAI_SOURCE_OF_TRUTH.md) |
| How to rerun EAAI experiments | [`README.md`](../README.md) section “How to reproduce the main paper artifacts” |
| Known issues (retrieval training, etc.) | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |
| Repo layout | [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md) |
| External datasets (Text2Zinc, CP-Bench) | [`DATASET_EXPANSION_PLAN.md`](DATASET_EXPANSION_PLAN.md), [`DATASET_EXPANSION_STATUS.md`](DATASET_EXPANSION_STATUS.md) |
