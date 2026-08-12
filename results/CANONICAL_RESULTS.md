# Canonical Results Manifest

**Purpose:** the single place a new agent checks to find the authoritative
source for any result family, without reading old submission documents.
Companion to [`canonical_results_manifest.json`](canonical_results_manifest.json)
(machine-readable) and [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) (headline
summary). Verified 2026-08-11 by direct comparison against `manuscript/main.tex`
unless noted otherwise.

Do not duplicate every row of every CSV here — this file identifies *where
the truth lives*, not what every number is.

---

## A. Schema retrieval

- **Authoritative artifact:** `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` (`Schema_R1` column)
- **Generator:** historical run, no longer reproducible via a single committed script (predates the current `training/external/run_full_downstream_benchmark.py`); values verified unchanged against `manuscript/main.tex` (TF-IDF 0.9094, BM25 0.8822, LSA 0.8459, Oracle 1.0)
- **Benchmark variant / denominator:** `orig`, 331 queries
- **Superseded artifact(s):** none for this specific column
- **Notes:** a **disclosed, separate, small offset** exists in some diagnostic contexts (TF-IDF 0.9063, BM25 0.8852, LSA 0.8550 — used in `results/eswa_revision/18_strict_instready/` and `results/eswa_revision/15_significance/`). The manuscript attributes this to a 335- vs 331-document catalog-vintage difference between the diagnostic script and the canonical retrieval run. **Both values are legitimate for their respective contexts** — do not silently pick one.

## B. Typed-greedy downstream (Coverage / TypeMatch / Exact20 / InstantiationReady)

- **Authoritative artifact:** `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` (core rows only; regenerated 2026-08-11)
- **Generator:** `tools/build_camera_ready_table1.py`, reading `results/eswa_revision/13_tables/postfix_main_metrics.csv` (Coverage/TypeMatch/Exact20/InstReady) and `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` (Schema_R1)
- **Underlying full-precision source:** `results/eswa_revision/13_tables/postfix_main_metrics.csv`, produced by `training/external/run_full_downstream_benchmark.py`
- **Benchmark variant / denominator:** `orig` (primary), also `noisy`/`short` in the same CSV
- **Superseded artifact(s):** `table1_main_benchmark_summary.csv`'s pre-2026-08-11 content (Coverage 0.8639, TypeMatch 0.7513, InstReady 0.5257 for TF-IDF) — stale, populated from a pre-correction intermediate snapshot per `manuscript/main.tex`'s own documented correction
- **Verified values (orig):** see `PROJECT_STATUS.md` §3

## C. Richer deterministic grounding methods

- **Authoritative artifact:** `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv` (all methods, all variants) and `results/eswa_revision/13_tables/postfix_main_metrics.csv` (orig/noisy/short, non-JSON form)
- **Per-method raw output:** `results/eswa_revision/02_downstream_postfix/nlp4lp_downstream_{variant}_{method}.json` (+ `_per_query_*.csv`)
- **Generator:** `training/external/run_full_downstream_benchmark.py` drives `tools/nlp4lp_downstream_utility.py` (assignment modes) plus `tools/relation_aware_linking.py` / `tools/ambiguity_aware_grounding.py`
- **Method inventory with CANONICAL/NEGATIVE_RESULT/EXPERIMENTAL status:** see `docs/METHOD_INVENTORY.md`
- **Superseded artifact(s):** `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv.stale`, `results/eswa_revision/15_significance/confidence_intervals.csv.stale`, `results/eswa_revision/15_significance/paired_significance.csv.stale` (already marked `.stale` by prior work)

## D. StrictInstantiationReady

- **Authoritative artifact:** `results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv` and `strict_vs_standard_significance.csv`
- **Generator:** `tools/run_strict_instantiation_ready.py` (per `manuscript/main.tex`'s own citation of this script)
- **Benchmark variant / denominator:** `orig`, 331 queries, paired bootstrap (B=1000, seed=42)
- **Notes:** uses the 0.9063-family Schema_R1 (see §A note); this is internally consistent within this specific artifact, not a new discrepancy

## E. Paired significance tests

- **Authoritative artifact:** `results/eswa_revision/15_significance/SIGNIFICANCE_SUMMARY.md` (+ `confidence_intervals.csv`, `paired_significance.csv`)
- **Method:** two-sided paired bootstrap, B=1000, seed=42, 95% percentile CIs, per-instance paired resampling
- **Key results:** see `docs/NEGATIVE_RESULTS.md` for the grounding-method comparisons; TF-IDF vs BM25 retrieval difference is *not* significant (p=0.088); TF-IDF-TG vs Oracle-TG InstReady difference *is* significant (p=0.004) — this is the paper's central bottleneck evidence

## F. Sanitization / lexical-overlap analysis

- **Authoritative artifact:** `results/eswa_revision/17_overlap_analysis/OVERLAP_ANALYSIS.md`, `retrieval_overlap_ablation.csv`, `lexical_overlap_stats.csv`
- **Key finding:** retrieval performance is preserved or improved after removing numeric tokens/stopwords from queries (LSA improves from 0.8459 to 0.9184 under stopword removal), indicating retrieval success is driven by structural/domain-term overlap, not query-specific numeric-value leakage. 327/331 `orig` queries fall in the "high lexical overlap" bucket by design (symbolic NLP4LP schema parameter names reduce raw lexical overlap, per the file's own note) — the "medium" bucket (4 queries) is too small for a separate quantitative claim.

## G. Structural 60-instance subset

- **Authoritative artifact:** `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv`
- **Generator:** `tools/run_eaai_engineering_subset_experiment.py`
- **Status:** **CURRENT** — verified byte-for-byte against `manuscript/main.tex` Table (schema-hit/structural-valid/instantiation-complete: TF-IDF 0.9333/0.75/0.75, BM25 0.9/0.7333/0.7333, Oracle 1.0/0.7667/0.7833)

## H. 269-instance executable-attempt subset

- **Authoritative artifact:** `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv`
- **Generator:** `tools/run_eaai_executable_subset_experiment.py`
- **Status:** **CURRENT** — verified against manuscript (TF-IDF 0.9368/0.8141/0.6654, BM25 0.9257/0.8104/0.6580, Oracle 1.0/0.8253/0.6840; executable/solver/feasible/objective rates uniformly 0.0 due to a missing `gurobipy` runtime in the evaluation environment used for this specific historical run — documented as an environment blocker, not a method failure)

## I. Final solver-backed 20-instance subset

- **Authoritative artifact:** `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv`
- **Generator:** `tools/run_eaai_final_solver_attempt.py`
- **Status:** **CURRENT** — verified against manuscript (TF-IDF 0.95/0.80/0.80/0.80, Oracle 0.95/0.75/0.75/0.75; solver = SciPy HiGHS shim, not Gurobi)

## J. PaMOP reproduction / pilot / forensics

- **Authoritative artifact:** `results/pamop/forensics_targeted/summary.json` (post-bug-fix forensics pass), `results/pamop/pilot/summary.json` (earlier pilot)
- **Generator:** `baselines/pamop/` pipeline (see `baselines/pamop/README.md`)
- **Status:** **RUNNING/IN PROGRESS** — 6-problem deterministic pilot subset (IDs 14, 23, 34, 72, 84, 88), verified: initial execution 2/6, final execution 6/6, semantic correctness 1/6, total tokens 24,194, decision gate `"A. PROCEED TO LARGER RUN"`. Not a full-scale PaMOP reproduction.
- **Superseded:** none — this is the current state, not a correction of a prior claim

## K. Baseline-comparison artifacts (external LLM baselines: OpenAI/Gemini/Mistral)

- **Authoritative artifact:** `results/paper/nlp4lp_downstream_*_openai.json` (historical, OpenAI only) plus infra docs `docs/GEMINI_RERUN_REPORT.md`, `docs/MISTRAL_RERUN_REPORT.md`
- **Status:** **AUXILIARY, NOT PAPER-CORE** — infrastructure exists for Gemini/Mistral reruns but no committed `results/rerun/{gemini,mistral}/` artifacts prove a completed full benchmark rerun as of this pass; do not assert completion without checking those directories directly
- **Not to be confused with:** the baseline-*roadmap* in `PROJECT_STATUS.md` §9 (ORLM/OptMATH/DeepOR/OR-R1/PaMOP), which is about implementing entirely different published methods as comparison baselines, not about optional LLM backends for our own retrieval pipeline

## L. Newly-evaluated grounding methods (max-weight matching, search-structured, hierarchical-structured) -- 2026-08-12

- **Authoritative artifact:** `results/unevaluated_methods_evaluation/` (`nlp4lp_downstream_orig_tfidf_{method}.json`, per-query CSVs, `significance.json`)
- **Generator:** unmodified canonical CLI, `python3 -m tools.nlp4lp_downstream_utility --variant orig --baseline tfidf --assignment-mode {max_weight_matching,search_structured_grounding,hierarchical_structured_grounding}`
- **Benchmark variant / denominator:** `orig`, full 331 queries (not a subsample)
- **Status:** **NEWLY VALIDATED, STRONG POSITIVE RESULT.** `max_weight_matching` reaches InstantiationReady **0.7432** (+0.2145 over typed greedy's 0.5287, p<0.001); `search_structured_grounding` and `hierarchical_structured_grounding` each reach **0.7039** (+0.1752, p<0.001). All three exceed Oracle-TG (0.5680), the previous highest value recorded anywhere in this repository's evaluated-method history.
- **Not yet in:** `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` or the manuscript -- this is a repository-internal finding pending a deliberate future manuscript-integration decision (see `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` P1).
- **Superseded artifact:** none -- this is new evidence, not a correction.

## M. P0 learned local grounding scorer -- 2026-08-12

- **Authoritative artifact:** `results/learned_grounding_p0/` (`test_results.csv`, `significance.json`, `ablation_results.csv`, `error_analysis.csv`, `split_metadata.json`)
- **Generator:** `scripts/learning/{build_p0_corpus,train_p0_classifier,eval_p0_grounding}.py` (see `docs/LEARNED_GROUNDING_P0.md` "Training Procedure" for exact commands)
- **Benchmark variant / denominator:** `orig`, oracle-schema 50-instance test subsample (NOT the full 331-query retrieval-conditioned benchmark -- see caveat in `docs/LEARNED_GROUNDING_P0.md` "Evaluation Protocol")
- **Status:** **NEGATIVE RESULT, decision gate C.** No P0 configuration beat the canonical oracle+typed-greedy baseline on the same subsample (0.80 best vs. 0.86); the gap is not statistically significant at n=50 (p=0.44). See `docs/NEGATIVE_RESULTS.md` NR11 for the full ledger entry.
- **Superseded artifact:** none -- this is a new, documented negative result, not a correction of a prior claim.

---

## Regenerating table1

```bash
python tools/build_camera_ready_table1.py
```

Reads the two source CSVs in §B above and writes
`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
deterministically. Re-run this if `postfix_main_metrics.csv` or
`deterministic_method_comparison_orig.csv` are ever regenerated with new
values, rather than hand-editing the table.
