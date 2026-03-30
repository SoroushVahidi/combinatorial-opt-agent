# EAAI Tables Reproduction Report

**Generated:** 2026-03-30  
**Status:** ✅ All tables present and verified

---

## Summary

All five EAAI camera-ready tables were present in `results/paper/eaai_camera_ready_tables/` and have been verified against their authoritative source reports in `analysis/`.

---

## Were Tables Missing?

**No.** All tables were already present before this reproducibility pass.

| File | Status | Size |
|------|--------|------|
| `table1_main_benchmark_summary.csv` | ✅ Present | 723 bytes |
| `table2_engineering_structural_subset.csv` | ✅ Present | 312 bytes |
| `table3_executable_attempt_with_blockers.csv` | ✅ Present | 622 bytes |
| `table4_final_solver_backed_subset.csv` | ✅ Present | 494 bytes |
| `table5_failure_taxonomy.csv` | ✅ Present | 1,245 bytes |
| `camera_ready_tables.md` | ✅ Present | 4,791 bytes |

---

## Table-by-Table Provenance

### Table 1 — Main Benchmark Summary

- **Source:** `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`
- **Authoritative report:** Pre-existing ESWA revision metrics (Schema R@1, Coverage, TypeMatch, InstReady)
- **Verification:** Consistent with EXPERIMENTS.md Section 2 and the main NLP4LP benchmark results

**Key values:**

| Method | Schema R@1 | Coverage | TypeMatch | InstReady |
|--------|-----------|----------|-----------|-----------|
| TF-IDF | 0.9094 | 0.8639 | 0.7513 | 0.5257 |
| BM25 | 0.8822 | 0.8509 | 0.7386 | 0.5196 |
| Oracle | 1.0000 | 0.9151 | 0.8030 | 0.5650 |

### Table 2 — Engineering Structural Subset (60 instances)

- **Source:** `analysis/eaai_engineering_subset_report.md`
- **Experiment:** End-to-end structural validation on 60 NLP4LP instances with 2–8 scalar parameters
- **Verification:** Consistent with `tools/run_eaai_engineering_subset_experiment.py` output format

**Key values:**

| Baseline | Schema Hit | Structural Valid | Inst. Complete |
|----------|-----------|-----------------|----------------|
| tfidf | 0.9333 | 0.7500 | 0.7500 |
| bm25 | 0.9000 | 0.7333 | 0.7333 |
| oracle | 1.0000 | 0.7667 | 0.7833 |

### Table 3 — Executable-Attempt Study (269 instances)

- **Source:** `analysis/eaai_executable_subset_report.md`
- **Experiment:** All 269 NLP4LP rows with non-empty `optimus_code`
- **Dominant blocker:** `ModuleNotFoundError: gurobipy` on 805/807 baseline-instance rows
- **Verification:** Consistent with `tools/run_eaai_executable_subset_experiment.py` output format

**Key values:** All executable/solver/feasible rates = 0.0 due to gurobipy blocker.

### Table 4 — Final Solver-Backed Subset (20 instances)

- **Source:** `analysis/eaai_final_solver_attempt_report.md`
- **Experiment:** 20 deterministically selected instances, solved via SciPy HiGHS MILP compatibility shim
- **Key achievement:** Real nonzero solver outcomes (TF-IDF: 80% feasible; Oracle: 75% feasible)
- **Verification:** Consistent with `tools/run_eaai_final_solver_attempt.py` output format

**Key values:**

| Baseline | Executable | Solver Success | Feasible | Objective |
|----------|-----------|----------------|----------|-----------|
| tfidf | 0.95 | 0.80 | 0.80 | 0.80 |
| oracle | 0.95 | 0.75 | 0.75 | 0.75 |

### Table 5 — Failure Taxonomy

- **Source:** Consolidated from all three EAAI experiment reports
- **Verification:** Counts and scope descriptions match reports in `analysis/`

---

## Conflicts Found

**None.** All table values are consistent with their cited EAAI experiment reports. No stale or conflicting values were found in the camera-ready tables.

---

## Commands Run

No table regeneration was needed. To regenerate tables from scratch:

```bash
# Re-run individual experiments (requires HF token and gurobipy or scipy):
python tools/run_eaai_engineering_subset_experiment.py   # → analysis/eaai_engineering_subset_report.md
python tools/run_eaai_executable_subset_experiment.py    # → analysis/eaai_executable_subset_report.md
python tools/run_eaai_final_solver_attempt.py            # → analysis/eaai_final_solver_attempt_report.md
```

Note: The engineering and executable subset experiments require HuggingFace dataset access (`udell-lab/NLP4LP`, gated). The final solver attempt requires `scipy>=1.9.0`.

---

## Blockers for Full Table Regeneration

| Blocker | Affected scripts |
|---------|-----------------|
| HuggingFace gated dataset (`udell-lab/NLP4LP`) requires `HF_TOKEN` | All three experiment scripts |
| `gurobipy` not available in CI/sandbox | `run_eaai_executable_subset_experiment.py` (expected; this is the subject of Table 3) |
| `scipy>=1.9.0` must be installed | `run_eaai_final_solver_attempt.py` (now added to `requirements.txt`) |
