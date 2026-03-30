# EAAI Camera-Ready Figures Reproduction Report

**Generated:** 2026-03-30  
**Status:** ✅ All figures successfully reproduced

---

## Summary

All five EAAI camera-ready figures were **missing** from the repository (only source CSV files existed). This report documents the successful reproduction of all PNG and PDF figure files.

---

## Were Figures Missing?

**Yes.** Prior to reproduction, `results/paper/eaai_camera_ready_figures/` contained only:

| File | Status Before |
|------|---------------|
| `figure1_pipeline_overview_source.csv` | ✅ Present (156 bytes) |
| `figure2_main_benchmark_source.csv` | ✅ Present (173 bytes) |
| `figure3_engineering_validation_source.csv` | ✅ Present (107 bytes) |
| `figure4_solver_subset_source.csv` | ✅ Present (105 bytes) |
| `figure5_failure_breakdown_source.csv` | ✅ Present (135 bytes) |
| `figure1_pipeline_overview.png` | ❌ **Missing** |
| `figure1_pipeline_overview.pdf` | ❌ **Missing** |
| `figure2_main_benchmark_comparison.png` | ❌ **Missing** |
| `figure2_main_benchmark_comparison.pdf` | ❌ **Missing** |
| `figure3_engineering_validation_comparison.png` | ❌ **Missing** |
| `figure3_engineering_validation_comparison.pdf` | ❌ **Missing** |
| `figure4_final_solver_backed_subset.png` | ❌ **Missing** |
| `figure4_final_solver_backed_subset.pdf` | ❌ **Missing** |
| `figure5_failure_breakdown.png` | ❌ **Missing** |
| `figure5_failure_breakdown.pdf` | ❌ **Missing** |

---

## Figures Rebuilt

All 5 figures were rebuilt (PNG + PDF) from verified authoritative table sources.

### Figure 1 — Pipeline Overview

- **Description:** Schematic of the six-step EAAI pipeline
- **Steps depicted:** NL query → Schema retrieval → Scalar grounding → Structural validation → Executable attempt → Solver-backed subset validation
- **Source:** Hardcoded step labels in `tools/build_eaai_camera_ready_figures.py`
- **Source CSV:** `figure1_pipeline_overview_source.csv`
- **Output:** `figure1_pipeline_overview.png`, `figure1_pipeline_overview.pdf`

### Figure 2 — Main Benchmark Comparison

- **Description:** Grouped bar chart comparing TF-IDF, BM25, and Oracle across 4 metrics
- **Metrics:** Schema retrieval R@1, coverage, type match, instantiation ready
- **Source table:** `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` (rows where `group == "core"`)
- **Source CSV:** `figure2_main_benchmark_source.csv`
- **Key values:** TF-IDF schema R@1=0.9094, coverage=0.8639, type_match=0.7513, inst_ready=0.5257
- **Output:** `figure2_main_benchmark_comparison.png`, `figure2_main_benchmark_comparison.pdf`

### Figure 3 — Engineering Validation Subset

- **Description:** Grouped bar chart for engineering subset (60 instances)
- **Metrics:** Structural valid rate, instantiation complete rate
- **Source table:** `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv`
- **Source CSV:** `figure3_engineering_validation_source.csv`
- **Key values:** TF-IDF structural=0.75, inst_complete=0.75; BM25=0.7333/0.7333; Oracle=0.7667/0.7833
- **Output:** `figure3_engineering_validation_comparison.png`, `figure3_engineering_validation_comparison.pdf`

### Figure 4 — Final Solver-Backed Subset

- **Description:** Grouped bar chart for final 20-instance solver-backed subset
- **Metrics:** Executable rate, solver success, feasible, objective produced
- **Source table:** `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv`
- **Source CSV:** `figure4_solver_subset_source.csv`
- **Key values:** TF-IDF exec=0.95, solver=0.80, feasible=0.80, objective=0.80; Oracle exec=0.95, solver=0.75
- **Output:** `figure4_final_solver_backed_subset.png`, `figure4_final_solver_backed_subset.pdf`

### Figure 5 — Failure Breakdown

- **Description:** Failure category counts from executable-attempt study
- **Source:** Failure counts from `analysis/eaai_executable_subset_report.md` (hardcoded in build script)
- **Categories:** gurobipy missing (805), incomplete instantiation (267), type mismatch (82), missing scalar slots (69), schema miss (37)
- **Source CSV:** `figure5_failure_breakdown_source.csv`
- **Output:** `figure5_failure_breakdown.png`, `figure5_failure_breakdown.pdf`

---

## Source Files Used

| Figure | Primary Source |
|--------|----------------|
| Figure 1 | `tools/build_eaai_camera_ready_figures.py` (hardcoded step labels) |
| Figure 2 | `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` |
| Figure 3 | `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv` |
| Figure 4 | `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv` |
| Figure 5 | `analysis/eaai_executable_subset_report.md` (counts hardcoded in build script) |

All table sources were verified against the corresponding EAAI experiment reports in `analysis/`.

---

## Exact Command Run

```bash
# Install missing dependencies first
pip install Pillow>=9.0.0

# Build all figures
python tools/build_eaai_camera_ready_figures.py
```

The script exited with code 0 (success). All 10 output files (5 PNG + 5 PDF) were created in `results/paper/eaai_camera_ready_figures/`.

---

## Dependency Note

The figure build script requires `Pillow` (PIL). This was not previously listed in `requirements.txt` and was added as `Pillow>=9.0.0` during this reproducibility pass.

`scipy>=1.9.0` was also added to `requirements.txt` for `tools/run_eaai_final_solver_attempt.py` which uses `scipy.optimize.milp`.

---

## Blockers

None. All figures reproduced successfully.

---

## Post-Reproduction File Inventory

| File | Size |
|------|------|
| `figure1_pipeline_overview.png` | ~19 KB |
| `figure1_pipeline_overview.pdf` | ~48 KB |
| `figure2_main_benchmark_comparison.png` | ~41 KB |
| `figure2_main_benchmark_comparison.pdf` | ~81 KB |
| `figure3_engineering_validation_comparison.png` | ~28 KB |
| `figure3_engineering_validation_comparison.pdf` | ~65 KB |
| `figure4_final_solver_backed_subset.png` | ~29 KB |
| `figure4_final_solver_backed_subset.pdf` | ~67 KB |
| `figure5_failure_breakdown.png` | ~35 KB |
| `figure5_failure_breakdown.pdf` | ~78 KB |
