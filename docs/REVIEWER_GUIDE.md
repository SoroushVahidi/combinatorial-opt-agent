# Reviewer Guide

A concise entry point for reviewers evaluating this companion research repository.

---

## What this repository is

This is the companion codebase for the EAAI manuscript:

> **"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"**

It implements and evaluates a pipeline that retrieves a compatible optimization schema
from a fixed catalog and then deterministically instantiates scalar parameters from
numeric evidence in a natural-language problem description.

The primary evaluation is on the **NLP4LP** benchmark (331 test queries, `orig` variant).

---

## The paper-core pipeline (what is benchmarked)

```
NL query
  → Schema Retrieval   (TF-IDF / BM25 / LSA; catalog of 331 NLP4LP schemas)
  → Deterministic Scalar Grounding   (typed greedy; tools/nlp4lp_downstream_utility.py)
  → Structural LP Check   (formulation/verify.py; no live solver)
  → [Restricted] Solver Execution   (SciPy HiGHS shim; 20-instance subset only)
```

Everything else (Gradio app, LLM baselines, learned retrieval fine-tuning) is
outside the paper-evaluated scope unless explicitly stated.

---

## Canonical files — start here

| File | Purpose |
|------|---------|
| `docs/CURRENT_STATUS.md` | Single reviewer-facing status page — headline metrics, validated vs auxiliary |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Manuscript authority — paper framing, what IS and IS NOT claimed |
| `docs/RESULTS_PROVENANCE.md` | Canonical metrics + full provenance chain (which number came from where) |
| `results/paper/eaai_camera_ready_tables/` | Camera-ready tables (Tables 1–5); **do not edit** |
| `results/paper/eaai_camera_ready_figures/` | Camera-ready figures (Figures 1–5); **do not edit** |
| `analysis/eaai_*_report.md` | EAAI experiment reports (canonical) |

---

## Official headline metrics

All values are from `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
(provenance: `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`).

**NLP4LP `orig` variant — 331 test queries — TF-IDF + typed greedy:**

| Metric | Value |
|--------|------:|
| Schema R@1 | **0.9094** |
| Coverage | **0.8639** |
| TypeMatch | **0.7513** |
| InstantiationReady | **0.5257** |

**Restricted solver-backed subset (Table 4) — 20 instances — SciPy HiGHS shim:**

| Metric | TF-IDF |
|--------|--------|
| Feasible | **0.80** |

Do **not** mix these with columns from other CSVs without reading the corresponding
provenance notes in `docs/RESULTS_PROVENANCE.md`.

---

## What requires gated NLP4LP access

The dataset `udell-lab/NLP4LP` (HuggingFace) is **gated**.
A HuggingFace account with approved access and a personal access token (`HF_TOKEN`)
is required to re-run downstream grounding metrics (TypeMatch, Exact20, InstantiationReady).

**Retrieval metrics (Schema R@1) can be reproduced without HF access** using the local
catalog files in `data/processed/`.

Pre-computed camera-ready results are committed under `results/paper/` and do not
require re-running.

See `HOW_TO_REPRODUCE.md` for step-by-step commands.

---

## What is outside paper scope

| Component | Status |
|-----------|--------|
| Gradio web app (`app.py`) | Demo / UX only; not the benchmarked pipeline |
| LLM baselines (OpenAI / Gemini via `tools/llm_baselines.py`) | Optional tooling; not in paper tables |
| Learned retrieval fine-tuning (`training/`, `src/learning/`) | Experimental; does not beat rule baseline |
| GAMSPy / Pyomo / PuLP / Gurobi solver paths | Demo code only; not used for paper results |
| GCG and extended grounding modes | Exploratory; not in main paper tables |
| Dense retrieval (SBERT, E5, BGE) | Supplementary; TF-IDF is the paper's primary baseline |

---

## Where limitations are documented

| Limitation | Location |
|------------|---------|
| ~47% of queries not fully instantiation-ready | `KNOWN_ISSUES.md` §1 and `docs/CURRENT_STATUS.md` |
| Gated dataset dependency | `KNOWN_ISSUES.md` §2.1 and `HOW_TO_REPRODUCE.md` |
| Solver execution restricted to 20 instances | `KNOWN_ISSUES.md` §1.3 and `docs/paper_vs_demo_scope.md` |
| Learned model does not beat rule baseline | `KNOWN_ISSUES.md` §3 and `docs/learning_runs/README.md` |
| Short-query grounding near zero | `KNOWN_ISSUES.md` §1.2 |
| Metric definitions differ across artifact files | `docs/RESULTS_PROVENANCE.md` legacy note |

---

## Archive / provenance folders (not headline sources)

| Folder | Contents |
|--------|---------|
| `docs/archive/` | Historical dev notes, method iteration logs. Not authoritative. |
| `docs/archive_internal_status/` | Internal audits, go/no-go decision logs. Provenance only. |
| `docs/eswa_revision/` | ESWA-era revision materials. Superseded by EAAI framing. |
| `analysis/archive/` | Non-EAAI analysis files (exploratory reports). Not headline source. |
| `results/eswa_revision/` | Earlier experiment runs. `13_tables/` is a provenance source for Table 1. |
