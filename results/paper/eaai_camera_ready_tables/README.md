# Camera-ready tables — staleness note (2026-08-11)

`table1_main_benchmark_summary.csv` contains **stale** values for the core
downstream metrics (Coverage, TypeMatch, InstantiationReady). `manuscript/main.tex`
documents that this table was populated from a stale intermediate significance
snapshot and was regenerated from live per-query artifacts during final KAIS
preparation (corrections up to 0.009 per metric; no qualitative change to the
paper's conclusions).

**Current, corrected numbers:** see [`../../../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md)
§3, sourced from `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv`
and `results/eswa_revision/13_tables/postfix_main_metrics.csv`.

`table2_engineering_structural_subset.csv`, `table3_executable_attempt_with_blockers.csv`,
`table4_final_solver_backed_subset.csv`, and `table5_failure_taxonomy.csv` were not
found stale in this pass but were also not independently re-verified line-by-line
against the manuscript.

This file was left in place (not deleted or overwritten) pending a proper
regeneration pass — see `PROJECT_STATUS.md` §13 (P0).
