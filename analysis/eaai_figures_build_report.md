# EAAI camera-ready figures build report

## Goal
Built publication-ready figures (PNG + PDF) from verified repository artifacts under:
- `results/paper/eaai_camera_ready_figures/`

Each figure has a matching source CSV in the same folder.

## Existing pipeline figure check (Figure 1 requirement)
Existing file found:
- `figures/nlp4lp_instantiation_pipeline_v2.png`

Assessment:
- Useful historical diagram, but **not sufficient** for final EAAI narrative because it does not explicitly include the executable-attempt stage and the final solver-backed subset validation stage.

Action:
- Created a new Figure 1 schematic matching final pipeline:
  `NL query -> retrieval -> grounding -> structural validation -> executable attempt -> solver-backed subset validation`.

## Figures created

1. **Figure 1 — Pipeline overview**
   - `figure1_pipeline_overview.png`
   - `figure1_pipeline_overview.pdf`
   - Source CSV: `figure1_pipeline_overview_source.csv`

2. **Figure 2 — Main benchmark comparison (orig)**
   - `figure2_main_benchmark_comparison.png`
   - `figure2_main_benchmark_comparison.pdf`
   - Source CSV: `figure2_main_benchmark_source.csv`
   - Source metrics from: `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`

3. **Figure 3 — Engineering validation comparison**
   - `figure3_engineering_validation_comparison.png`
   - `figure3_engineering_validation_comparison.pdf`
   - Source CSV: `figure3_engineering_validation_source.csv`
   - Source metrics from: `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv`

4. **Figure 4 — Final solver-backed subset**
   - `figure4_final_solver_backed_subset.png`
   - `figure4_final_solver_backed_subset.pdf`
   - Source CSV: `figure4_solver_subset_source.csv`
   - Source metrics from: `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv`

5. **Figure 5 — Failure breakdown**
   - `figure5_failure_breakdown.png`
   - `figure5_failure_breakdown.pdf`
   - Source CSV: `figure5_failure_breakdown_source.csv`
   - Source metrics from verified executable-attempt failure counts in `analysis/eaai_executable_subset_report.md`.

## Build command

```bash
python tools/build_eaai_camera_ready_figures.py
```

## Main-paper vs appendix recommendation

### Main paper
- Figure 1 (pipeline overview)
- Figure 2 (main benchmark)
- Figure 3 (engineering validation)
- Figure 4 (final solver-backed subset)

### Appendix
- Figure 5 (failure breakdown)
  - Useful and evidence-backed, but more diagnostic/detail-oriented than core narrative.

## Notes
- Figures are intentionally simple and publication-friendly for rapid camera-ready use.
- If journal style requires vector-native editing, the generated PDFs can be used directly as base assets.
