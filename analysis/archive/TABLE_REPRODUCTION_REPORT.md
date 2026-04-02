# Table Reproduction Report

**Date:** 2026-03-30
**Status:** ✅ All 5 paper tables verified present

> **Note:** This is the top-level table status report for the current polish pass.
> For full build provenance, see `analysis/eaai_tables_reproduction_report.md`
> and `analysis/eaai_tables_build_report.md`.

---

## Summary

All five EAAI camera-ready tables are present in
`results/paper/eaai_camera_ready_tables/` and have been verified against their
authoritative source reports. No tables were missing; no regeneration was required.

---

## Tables Found

| Table File | Description | Data Rows | Status |
|-----------|-------------|-----------|--------|
| `table1_main_benchmark_summary.csv` | Main NLP4LP benchmark results (TF-IDF, BM25, Oracle, Best) | 4 | ✅ Present & verified |
| `table2_engineering_structural_subset.csv` | Engineering structural subset (60 instances, 3 baselines) | 3 | ✅ Present & verified |
| `table3_executable_attempt_with_blockers.csv` | Executable-attempt blocker study (269 instances, 3 baselines) | 3 | ✅ Present & verified |
| `table4_final_solver_backed_subset.csv` | Final solver-backed subset (20 instances, 2 baselines) | 2 | ✅ Present & verified |
| `table5_failure_taxonomy.csv` | Failure taxonomy (7 failure categories) | 7 | ✅ Present & verified |

Human-readable summary: `results/paper/eaai_camera_ready_tables/camera_ready_tables.md`

---

## Table Provenance

### Table 1 — Main Benchmark Summary

- **Source file:** `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`
- **Experiment:** Full NLP4LP benchmark, `orig` variant, 331 test queries
- **Authoritative report:** `analysis/eaai_tables_build_report.md`
- **Build script:** `tools/make_nlp4lp_paper_artifacts.py`
- **Key values:**

| Method | Schema R@1 | Coverage | TypeMatch | InstReady |
|--------|-----------|----------|-----------|-----------|
| TF-IDF | 0.9094 | 0.8639 | 0.7513 | 0.5257 |
| BM25 | 0.8822 | 0.8509 | 0.7386 | 0.5196 |
| Oracle | 1.0000 | 0.9151 | 0.8030 | 0.5650 |

### Table 2 — Engineering Structural Subset

- **Source report:** `analysis/eaai_engineering_subset_report.md`
- **Experiment script:** `tools/run_eaai_engineering_subset_experiment.py`
- **Subset:** 60 NLP4LP instances with 2–8 scalar parameters
- **Key values:** TF-IDF structural valid = 0.75, inst. complete = 0.75

### Table 3 — Executable-Attempt with Blockers

- **Source report:** `analysis/eaai_executable_subset_report.md`
- **Experiment script:** `tools/run_eaai_executable_subset_experiment.py`
- **Subset:** All 269 NLP4LP rows with non-empty `optimus_code`
- **Key values:** All executable/solver/feasible rates = 0.0 (gurobipy blocker)

### Table 4 — Final Solver-Backed Subset

- **Source report:** `analysis/eaai_final_solver_attempt_report.md`
- **Experiment script:** `tools/run_eaai_final_solver_attempt.py`
- **Subset:** 20 instances, deterministically filtered, SciPy HiGHS MILP shim
- **Key values:** TF-IDF feasible = 0.80, Oracle feasible = 0.75

### Table 5 — Failure Taxonomy

- **Source report:** `analysis/eaai_engineering_subset_report.md`
- **Derived from:** Engineering and executable subset experiments combined
- **Key values:** 7 failure categories covering schema miss, incomplete instantiation,
  type mismatch, structural invalidity, gurobipy absence, shim incompatibility, and
  infeasibility

---

## Conflicts Resolved

Two potential conflicts were investigated during table construction and resolved:

| Conflict | Resolution |
|----------|-----------|
| **ESWA revision tables vs EAAI camera-ready tables:** Multiple `results/eswa_revision/` subdirectories exist. | `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` is the canonical upstream source for Table 1. All other ESWA revision artifacts are historical context only. Documented in `analysis/eaai_tables_build_report.md`. |
| **"Best method" label in Table 1:** The `best` row in `table1_main_benchmark_summary.csv` has the same values as the TF-IDF row. | `best` maps to `tfidf_typed_greedy`, which is indeed the best-performing method. This is a deliberate inclusion to mark the best row; no conflict with paper text. Documented in `analysis/eaai_tables_build_report.md §Conflict 2`. |

---

## Commands to Reproduce

To rebuild all camera-ready tables from authoritative source data:

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild tables
python tools/make_nlp4lp_paper_artifacts.py
```

To re-run the underlying experiments (requires HuggingFace NLP4LP dataset access):

```bash
# Engineering structural subset (Table 2)
python tools/run_eaai_engineering_subset_experiment.py

# Executable-attempt study (Table 3)
python tools/run_eaai_executable_subset_experiment.py

# Final solver-backed subset (Table 4)
python tools/run_eaai_final_solver_attempt.py
```

> **Blocker:** Full experiment re-runs require access to the gated
> `udell-lab/NLP4LP` HuggingFace dataset. The camera-ready CSVs in
> `results/paper/eaai_camera_ready_tables/` contain the pre-computed authoritative
> results and can be cited directly without re-running experiments.
