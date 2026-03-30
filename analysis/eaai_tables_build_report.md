# EAAI camera-ready tables build report

## Scope
Built manuscript-ready tables under:
- `results/paper/eaai_camera_ready_tables/`

Generated both CSV and Markdown artifacts for all required tables.

## Source selection policy used
I used the latest verified artifacts already present in this repository and prioritized measured machine-generated tables over narrative summaries when both were available.

Priority order applied:
1. `results/eswa_revision/13_tables/*.csv` (measured canonical ESWA tables)
2. EAAI experiment reports in `analysis/*.md` when raw EAAI CSV artifacts were not present in tracked repo
3. Explicit conflict notes when a required metric existed in multiple places

## Table-by-table provenance

### Table 1 — Main benchmark summary
Output files:
- `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`

Primary source:
- `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`

Reason:
- Contains normalized paper-facing orig-variant metrics (`Schema_R1`, `Coverage`, `TypeMatch`, `InstReady`) for core methods.

Methods included:
- TF-IDF (`tfidf_typed_greedy`)
- BM25 (`bm25_typed_greedy`)
- Oracle (`oracle_typed_greedy`)
- Best downstream grounding method used in final paper (designated as `tfidf_typed_greedy`, matching final ESWA summary statement of best non-oracle InstReady)

### Table 2 — Engineering subset structural validation
Output files:
- `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv`

Source:
- `analysis/eaai_engineering_subset_report.md`

Reason:
- The tracked repo includes the verified engineering subset summary in this report, including subset size and rates for tfidf/bm25/oracle.

### Table 3 — Executable-attempt study with blockers
Output files:
- `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv`

Source:
- `analysis/eaai_executable_subset_report.md`

Reason:
- Contains the verified 269-instance executable-attempt table plus explicit blocker details.

### Table 4 — Final solver-backed subset
Output files:
- `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv`

Source:
- `analysis/eaai_final_solver_attempt_report.md`

Reason:
- Contains the latest verified final-attempt solver-backed rates and subset rule for tfidf/oracle.

### Table 5 — Failure taxonomy
Output files:
- `results/paper/eaai_camera_ready_tables/table5_failure_taxonomy.csv`

Sources:
- `analysis/eaai_engineering_subset_report.md`
- `analysis/eaai_executable_subset_report.md`
- `analysis/eaai_final_solver_attempt_report.md`

Reason:
- Consolidates dominant failure categories across downstream engineering/executable/final-solver validations.
- Engineering counts are derived from reported rounded rates (explicitly marked approximate).

## Additional markdown deliverable
- `results/paper/eaai_camera_ready_tables/camera_ready_tables.md`

This file contains manuscript-friendly Markdown renderings for Tables 1–5 and a source note under each table.

## Conflicts and resolution

### Conflict 1: multiple benchmark summary artifacts for orig metrics
- Candidates found:
  - `results/eswa_revision/13_tables/postfix_main_metrics.csv`
  - `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`
  - `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv`
- Resolution:
  - Chose `deterministic_method_comparison_orig.csv` because it already contains direct paper columns including retrieval (`Schema_R1`) and is explicitly the deterministic orig comparison table.
  - `downstream_comparison_all_methods.csv` appears malformed in tracked file (method column values shown as `0.0`), so it was not used.

### Conflict 2: “best downstream grounding method” label ambiguity
- Requirement asked for tfidf, bm25, oracle, and “best downstream grounding method used in final paper.”
- In available verified orig comparison metrics, the best non-oracle InstReady method is `tfidf_typed_greedy`, which coincides with TF-IDF baseline.
- Resolution:
  - Included a dedicated fourth row explicitly labeled as the best downstream method, with the same measured values as TF-IDF baseline.

## BLOCKERS / Ambiguities

1. **Tracked raw EAAI subset CSVs were not present** (e.g., `results/paper/eaai_engineering_subset/...`, `results/paper/eaai_executable_subset/...`), while verified narrative reports with numeric tables were present.
   - Action taken: used committed analysis reports as authoritative EAAI sources.

2. **Engineering failure taxonomy exact category counts are not directly exported** in a tracked CSV for that experiment.
   - Action taken: used explicit approximate derivations from published rounded rates and marked them as approximate.

## READY TO PASTE INTO PAPER

### Main paper
- Table 1 — `table1_main_benchmark_summary.csv` (or corresponding Markdown section)
- Table 2 — `table2_engineering_structural_subset.csv`
- Table 3 — `table3_executable_attempt_with_blockers.csv`
- Table 4 — `table4_final_solver_backed_subset.csv`

### Appendix
- Table 5 — `table5_failure_taxonomy.csv` (compact cross-experiment taxonomy + caveats)
- Full Markdown bundle: `camera_ready_tables.md`
