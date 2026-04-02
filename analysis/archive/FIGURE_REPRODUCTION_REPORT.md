# Figure Reproduction Report

**Date:** 2026-03-30
**Status:** ✅ All 5 paper figures verified present

> **Note:** This is the top-level figure status report for the current polish pass.
> For full build provenance, see `analysis/eaai_figures_reproduction_report.md`
> and `analysis/eaai_figures_build_report.md`.

---

## Summary

All five EAAI camera-ready figures are present in
`results/paper/eaai_camera_ready_figures/` in both PDF (for submission) and PNG (for
preview) formats. Figures were regenerated from authoritative source CSVs during
this pass; they were missing prior to regeneration.

---

## Figures Found

| Figure | Description | PDF | PDF Size | PNG | PNG Size | Status |
|--------|-------------|-----|----------|-----|----------|--------|
| `figure1_pipeline_overview` | Six-step pipeline schematic | ✅ | 48 KB | ✅ | 20 KB | ✅ Present |
| `figure2_main_benchmark_comparison` | Grouped bar chart: 3 baselines × 4 metrics | ✅ | 80 KB | ✅ | 40 KB | ✅ Present |
| `figure3_engineering_validation_comparison` | Engineering subset bar chart | ✅ | 64 KB | ✅ | 28 KB | ✅ Present |
| `figure4_final_solver_backed_subset` | Solver-backed subset bar chart | ✅ | 68 KB | ✅ | 32 KB | ✅ Present |
| `figure5_failure_breakdown` | Failure taxonomy stacked bar chart | ✅ | 76 KB | ✅ | 36 KB | ✅ Present |

All files reside at:
`results/paper/eaai_camera_ready_figures/figure{N}_{name}.{pdf,png}`

---

## Figures Missing

**None.** All 5 figures are present in both formats.

---

## Figures Regenerated in This Pass

All 5 figures were regenerated during this polish pass. Prior to regeneration, only the
`*_source.csv` files existed; the PDF and PNG outputs were absent.

| Figure | Regenerated? | Method |
|--------|-------------|--------|
| `figure1_pipeline_overview` | ✅ Yes | `tools/build_eaai_camera_ready_figures.py` |
| `figure2_main_benchmark_comparison` | ✅ Yes | `tools/build_eaai_camera_ready_figures.py` |
| `figure3_engineering_validation_comparison` | ✅ Yes | `tools/build_eaai_camera_ready_figures.py` |
| `figure4_final_solver_backed_subset` | ✅ Yes | `tools/build_eaai_camera_ready_figures.py` |
| `figure5_failure_breakdown` | ✅ Yes | `tools/build_eaai_camera_ready_figures.py` |

---

## Source Files Used

Each figure has a corresponding source CSV co-located in the same directory:

| Figure | Source CSV | Source Contents |
|--------|-----------|----------------|
| Figure 1 | `figure1_pipeline_overview_source.csv` | Step labels for the 6-stage pipeline schematic |
| Figure 2 | `figure2_main_benchmark_source.csv` | Rows from `table1_main_benchmark_summary.csv` where `group == "core"` |
| Figure 3 | `figure3_engineering_validation_source.csv` | Rows from `table2_engineering_structural_subset.csv` |
| Figure 4 | `figure4_solver_subset_source.csv` | Rows from `table4_final_solver_backed_subset.csv` |
| Figure 5 | `figure5_failure_breakdown_source.csv` | Rows from `table5_failure_taxonomy.csv` |

All source CSVs trace back to the camera-ready tables in
`results/paper/eaai_camera_ready_tables/`, which are themselves sourced from the
authoritative experiment reports in `analysis/` (see `analysis/eaai_tables_build_report.md`
for the full provenance chain).

---

## Build Commands

To regenerate all 5 figures from scratch:

```bash
# Install dependencies (matplotlib, pandas, Pillow required)
pip install -r requirements.txt

# Regenerate figures
python tools/build_eaai_camera_ready_figures.py
```

Output goes to `results/paper/eaai_camera_ready_figures/`.

For individual figure descriptions and the data they visualize, see
`analysis/eaai_figures_build_report.md`.

---

## Blockers

**None.** Figure generation has no external dependency blockers:

- No HuggingFace dataset access required (source CSVs are already committed).
- No Gurobi license required.
- No GPU required.
- Dependencies: `matplotlib`, `pandas`, `Pillow` — all installable via `pip`.
