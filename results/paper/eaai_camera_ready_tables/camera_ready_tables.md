# EAAI Camera-Ready Tables

## Table 1 — Main benchmark summary

| method_label | method_key | group | schema_retrieval_r1 | coverage_metric | type_match_metric | instantiation_ready | source_file |
|---|---|---|---|---|---|---|---|
| TF-IDF | tfidf_typed_greedy | core | 0.9094 | 0.8639 | 0.7513 | 0.5257 | results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv |
| BM25 | bm25_typed_greedy | core | 0.8822 | 0.8509 | 0.7386 | 0.5196 | results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv |
| Oracle schema | oracle_typed_greedy | core | 1.0 | 0.9151 | 0.803 | 0.565 | results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv |
| Best downstream grounding used in final paper (TF-IDF + typed_greedy) | tfidf_typed_greedy | best_downstream | 0.9094 | 0.8639 | 0.7513 | 0.5257 | results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv |

Source: `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`; best-downstream designation follows final report statement that TF-IDF typed_greedy is top non-oracle InstReady.


## Table 2 — Engineering subset structural validation

| baseline | subset_size | schema_hit_rate | structural_valid_rate | instantiation_complete_rate | source_file |
|---|---|---|---|---|---|
| tfidf | 60 | 0.9333 | 0.75 | 0.75 | analysis/eaai_engineering_subset_report.md |
| bm25 | 60 | 0.9 | 0.7333 | 0.7333 | analysis/eaai_engineering_subset_report.md |
| oracle | 60 | 1.0 | 0.7667 | 0.7833 | analysis/eaai_engineering_subset_report.md |

Source: `analysis/eaai_engineering_subset_report.md` (reported 60-instance engineering subset table).


## Table 3 — Executable-attempt study with blockers

| baseline | subset_size | schema_hit_rate | structural_valid_rate | instantiation_complete_rate | executable_model_rate | solver_success_rate | feasible_rate | objective_rate | blocker_note | source_file |
|---|---|---|---|---|---|---|---|---|---|---|
| tfidf | 269 | 0.9368 | 0.8141 | 0.6654 | 0.0 | 0.0 | 0.0 | 0.0 | Missing gurobipy runtime (ModuleNotFoundError). | analysis/eaai_executable_subset_report.md |
| bm25 | 269 | 0.9257 | 0.8104 | 0.658 | 0.0 | 0.0 | 0.0 | 0.0 | Missing gurobipy runtime; plus 2 schema-miss rows with missing_optimus_code. | analysis/eaai_executable_subset_report.md |
| oracle | 269 | 1.0 | 0.8253 | 0.684 | 0.0 | 0.0 | 0.0 | 0.0 | Missing gurobipy runtime (ModuleNotFoundError). | analysis/eaai_executable_subset_report.md |

Source: `analysis/eaai_executable_subset_report.md` (269-instance executable-eligible subset).


## Table 4 — Final solver-backed subset

| baseline | subset_rule | subset_size | executable_rate | solver_success_rate | feasible_rate | objective_produced_rate | source_file |
|---|---|---|---|---|---|---|---|
| tfidf | orig eval rows with non-empty gold optimus_code and static shim-compatibility filter; deterministic cap limit=20 | 20 | 0.95 | 0.8 | 0.8 | 0.8 | analysis/eaai_final_solver_attempt_report.md |
| oracle | orig eval rows with non-empty gold optimus_code and static shim-compatibility filter; deterministic cap limit=20 | 20 | 0.95 | 0.75 | 0.75 | 0.75 | analysis/eaai_final_solver_attempt_report.md |

Source: `analysis/eaai_final_solver_attempt_report.md` (20-instance deterministic shim-compatible subset).


## Table 5 — Failure taxonomy

| failure_category | scope | count_or_rate | evidence | source_file |
|---|---|---|---|---|
| Schema retrieval miss | Engineering subset (60 x 3 baselines) | ~10 cases total (derived from rounded schema-hit rates) | tfidf 4 + bm25 6 + oracle 0 | analysis/eaai_engineering_subset_report.md |
| Incomplete instantiation | Engineering subset (60 x 3 baselines) | ~44 cases total (derived from rounded rates) | instantiation_complete deficits across tfidf/bm25/oracle | analysis/eaai_engineering_subset_report.md |
| Missing scalar slots | Executable-attempt subset (269) | 69 total counts | tfidf 24 + bm25 24 + oracle 21 | analysis/eaai_executable_subset_report.md |
| Type mismatch | Executable-attempt subset (269) | 82 total counts | tfidf 27 + bm25 28 + oracle 27 | analysis/eaai_executable_subset_report.md |
| Incomplete instantiation | Executable-attempt subset (269) | 267 total counts | tfidf 90 + bm25 92 + oracle 85 | analysis/eaai_executable_subset_report.md |
| Missing runtime dependency: gurobipy | Executable-attempt subset (269) | 805/807 baseline-instance rows | all tfidf/oracle rows, 267/269 bm25 rows | analysis/eaai_executable_subset_report.md |
| Indexing incompatibility in shim path | Final solver-backed subset (20 x 2 baselines) | 2/40 rows | IndexError: list index out of range | analysis/eaai_final_solver_attempt_report.md |

Source: combined from engineering/executable/final solver reports listed in `source_file` column.
