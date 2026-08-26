# SN Computer Science — Reproducibility Map

Current authoritative manuscript: [`manuscript/sncs/main.tex`](../manuscript/sncs/main.tex) (migrated 2026-08-27 from the corrected [`manuscript/dke/main.tex`](../manuscript/dke/main.tex)). Machine-readable companion: [`SNCS_RESULT_MANIFEST_2026-08-27.json`](SNCS_RESULT_MANIFEST_2026-08-27.json).

This table traces every major table/claim in the manuscript to its authoritative source artifact and (where applicable) a committed, deterministic reproduction script.

| Manuscript table/section | Result | Authoritative artifact | Reproduction script | Expected output |
|---|---|---|---|---|
| Table 3 (Schema R@1) | TF-IDF/BM25/LSA/BGE-M3 retrieval, orig/noisy/short | `results/dense_retrieval_bge_m3/retrieval_metrics.json` | — (frozen retrieval run; BGE-M3 requires GPU + `sentence-transformers`) | JSON with per-method, per-variant Schema R@1 |
| Table 4 (main downstream) | Coverage/TypeMatch/Exact20/InstReady/Strict, 4 configs | `results/final_resubmission_method/metrics.json`, `results/dense_retrieval_bge_m3/downstream_metrics.json`, `results/oracle_recomputation_2026-08-15/oracle_frozen_verification.json` | `tools/compute_uniform_exact20.py` (Exact20 column only; other columns are the frozen aggregates themselves) | `results/final_resubmission_method/exact20_uniform_2026-08-27.json` |
| Matched same-task grounding baselines (`tab:nlp4lp-grounding-baselines`) | Typed greedy vs constrained / max-weight / opt-role / search-structured / semantic-IR under fixed TF-IDF | `results/stage6_matched_grounding_baselines_2026-08-27/matched_grounding_baselines_summary.json` | `tools/run_stage6_matched_grounding_baselines.py` (`PYTHONHASHSEED=0`, `NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json`) | same summary JSON + per-query CSVs |
| Residual-error decomposition | 5-category mutually exclusive partition of 331 queries | `results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv` | `tools/recompute_residual_error_analysis.py` | `results/final_resubmission_method/residual_error_analysis_2026-08-27.json` |
| Numeric-extraction ablation | Coverage/TypeMatch/Exact20/InstReady/Strict, baseline vs ratio-aware | `results/final_resubmission_method/metrics.json` (both rows) | `tools/compute_uniform_exact20.py` (Exact20); rest are the frozen metrics.json fields | as above |
| Significance tests | 5 paired-bootstrap/McNemar rows | `results/dense_retrieval_bge_m3/significance_tests.json` (2 rows), `results/final_resubmission_method/summary.json` (McNemar), and the 3 previously-unreproduced rows | `tools/recompute_dke_significance.py` | `results/final_resubmission_method/significance_recomputation_2026-08-27.json` (matches manuscript diff/CI/p exactly) |
| Table 8 (overlap/sanitization) | TF-IDF/BM25/LSA retrieval under number/stopword removal | `results/eswa_revision/17_overlap_analysis/retrieval_overlap_ablation.csv` (pre-DKE, historical, values unchanged since verified) | — | — |
| Table 9 (Strict vs ordinary readiness) | 3-row comparison, $n_{\text{differ}}$ | `results/final_resubmission_method/summary.json` | — | — |
| Computational Complexity and Runtime (new) | 1.09s / 3.29ms/query / 202508 KB | `results/final_resubmission_method/runtime.json` | — (measured artifact) | — |
| StrictInstantiationReady motivation (new) | 257→265 ordinary-ready, 6/8 wrong-schema | `docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md`, `results/strict_instantiation_ready/` | — | — |
| Table (engineering structural, 60 inst.) | Schema-hit/structural-valid/inst-complete | `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv` | `tools/run_eaai_engineering_subset_experiment.py` | — |
| Table (extended structural, 269 inst.) | Same fields | `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv` | `tools/run_eaai_executable_subset_experiment.py` | — |
| Table (solver-backed, 20 inst.) | Executable/Success/Feasible/Objective | `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv` | `tools/run_eaai_final_solver_attempt.py` | — |
| Tables (external baseline, 18-inst. shared subset) | PaMOP/ORLM/OptMATH/Generic-LLM outcomes | `results/external_baseline_comparison/comparison.json`, `results/optmath/`, `results/orlm/`, `results/pamop/`, `results/generic_llm/` | see `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` | — |
| §4.9 (OptMATH-Train numeric extraction) | Raw/text-supported/audit-calibrated recall | `results/external_validation/optmath/final_verification/verified_metrics.json`, `classifier_validation.json` | `scripts/verify_optmath_external.py` | — |
| DeepOR/OR-R1 provenance | Code/checkpoint availability | `docs/DEEPOR_PROVENANCE.md`, `docs/ORR1_PROVENANCE.md` | — | — |
| Exact20 denominator root cause | 0.2527 vs 0.2614 explanation | `results/final_resubmission_method/exact20_denominator_audit.json` | `tools/audit_exact20_denominator.py` | — |

## How to re-run the deterministic verification scripts

All four scripts below are pure Python, CPU-only, read-only over already-committed CSV/JSON artifacts, and require no gated dataset access, no GPU, and no external API calls:

```bash
python3 tools/audit_exact20_denominator.py
python3 tools/compute_uniform_exact20.py
python3 tools/recompute_residual_error_analysis.py
python3 tools/recompute_dke_significance.py
```

Each prints its output to stdout and writes the corresponding JSON file listed in the table above (already committed; re-running should reproduce them byte-for-byte given the same Python version, since no randomness affects any value except the bootstrap CIs in `recompute_dke_significance.py`, which fix `seed=42`).

## What requires gated data access

Reproducing the underlying frozen per-query CSVs from scratch (rather than verifying claims computed from them) requires an approved Hugging Face account with access to `udell-lab/NLP4LP`; see [`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md) (EAAI-era, historical) and `manuscript/sncs/main.tex`'s Data availability declaration. Reproducing BGE-M3 retrieval additionally requires a GPU and the `sentence-transformers` package. Reproducing the external-baseline (PaMOP/OptMATH/Generic-LLM) rows requires Azure OpenAI API access and, for OptMATH/Generic-LLM, a local Gurobi license.

## Where historical results live

Pre-DKE (EAAI/KAIS-era) results and documentation are marked with a historical banner (added 2026-08-27) and listed in [`DKE_SOURCE_OF_TRUTH.md`](DKE_SOURCE_OF_TRUTH.md); they remain in the repository for provenance/audit-trail purposes but must not be cited as current evidence for the SN Computer Science submission.
