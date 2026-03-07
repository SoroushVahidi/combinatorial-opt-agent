# NLP4LP Experiment Verification Report

Full audit of experiment numbers recomputable from code and data: source files, formulas, assumptions, and reproducibility status.

---

## 1. Verification table (per result group)

### 1.1 Retrieval metrics (Schema R@1 / Recall@1 by variant)

| Result group / table | Metric | Recomputed value / formula | Source result file(s) | Source script/function | Assumptions | Status |
|---------------------|--------|----------------------------|------------------------|------------------------|-------------|--------|
| **nlp4lp_retrieval_main_table** | orig_schema_R1 (tfidf) | 0.9063 (4 dp) | `results/nlp4lp_retrieval_summary.csv` | `make_nlp4lp_paper_artifacts.py` §4i: reads `Recall@1` for variant=orig, baseline=tfidf | Recall@1 = P@1 from retrieval | **Reproducible** |
| Same | noisy_schema_R1 (tfidf) | 0.9033 | `nlp4lp_retrieval_summary.csv` | Same | variant=noisy | **Reproducible** |
| Same | short_schema_R1 (tfidf) | 0.7855 | Same | Same | variant=short | **Reproducible** |
| Same | random (all variants) | 0.0030 (4 dp) | **None** (hardcoded) | `make_nlp4lp_paper_artifacts.py`: `r1_random = 1.0 / 331` | N=331 eval queries; theoretical random | **Reproducible** (note: differs from downstream random) |
| **Retrieval pipeline** | Recall@1, Recall@5, MRR@10, nDCG@10 | P@1, P@5, MRR@k, nDCG@k from `compute_metrics()` | `results/nlp4lp_retrieval_metrics_{orig,noisy,short,noentity,nonum}.json` | `training/run_baselines.py` (writes `--out`), `tools/summarize_nlp4lp_results.py` (aggregates to CSV) | Eval = `data/processed/nlp4lp_eval_<variant>.jsonl`; k=10 | **Reproducible** |

- **Retrieval Recall@1 formula**: For each query, 1 if expected problem in top-1, else 0; aggregate = mean over all eval instances. Implemented as `P@1` in `training/metrics.py` (`precision_at_k(..., 1)`), denominator = number of eval pairs.
- **Random for retrieval**: **Theoretical** 1/N (N=331). No random baseline is run in retrieval; the artifact script injects 1/331 for the retrieval main table only.

---

### 1.2 Downstream summary (orig / noisy / short)

| Result group / table | Metric | Recomputed value / formula | Source result file(s) | Source script/function | Assumptions | Status |
|---------------------|--------|----------------------------|------------------------|------------------------|-------------|--------|
| **nlp4lp_downstream_section_table** (main downstream) | schema_R1 | hits/n_queries | `results/paper/nlp4lp_downstream_summary.csv` | `nlp4lp_downstream_utility.run_one()` → `run_setting()` → artifact §4j | n = 331 | **Reproducible** |
| Same | param_coverage | mean over queries of (n_filled / n_expected_scalar) per query | Same | Same | 0 when n_expected_scalar=0 | **Reproducible** |
| Same | type_match | mean over queries of (type_matches / n_filled) per query | Same | Same | empty when n_filled=0 | **Reproducible** |
| Same | exact5_on_hits / exact20_on_hits | mean over **schema-hit** queries with non-empty comparable_errs | Same | Same | denominator = count of hit queries with comparable_errs | **Reproducible** |
| Same | instantiation_ready | fraction of queries with param_coverage≥0.8 and type_match≥0.8 | Same | Same | denominator = n_queries | **Reproducible** |
| Same | key_overlap | per query: \|pred_scalar ∩ gold_scalar\| / \|gold_scalar\|; aggregate = mean over all | Same | Same | denominator per query = \|gold_scalar\| | **Reproducible** |

- **Variants**: `variant` column in `nlp4lp_downstream_summary.csv`: `orig`, `noisy`, `short`. Eval files: `nlp4lp_eval_orig.jsonl`, `nlp4lp_eval_noisy.jsonl`, `nlp4lp_eval_short.jsonl` (same 331 query IDs per variant, different query text).
- **Rounding**: Artifact uses `_fmt4(v)` → `f"{float(v):.4f}"` for section table, variant table, error tables.

---

### 1.3 Downstream – random baseline

| Result group | Metric | Recomputed value | Source | Source script/function | Assumptions | Status |
|--------------|--------|------------------|--------|------------------------|-------------|--------|
| **Downstream (all tables using summary)** | random schema_R1 | 0.006042… (2/331) | `nlp4lp_downstream_summary.csv` | `nlp4lp_downstream_utility.run_one()` with baseline="random" | Single run, deterministic | **Reproducible** |
| Same | random param_coverage, type_match, etc. | As in CSV | Same | Same | pred_id = `random.Random(_md5_seed(qid)).randrange(len(doc_ids))` | **Reproducible** |

- **Random baseline**: **Deterministic single run**: one prediction per query via `random.Random(_md5_seed(query_id)).randrange(len(catalog))`. So schema_R1 is **empirical** (2 hits in current run), not expectation 1/331. **Inconsistency**: retrieval main table uses 1/331 (0.0030); downstream tables use empirical random (0.0060). Both are reproducible but **use different definitions**.

---

### 1.4 Downstream – oracle

| Result group | Metric | Recomputed value | Source | Source script/function | Assumptions | Status |
|--------------|--------|------------------|--------|------------------------|-------------|--------|
| **All downstream** | oracle schema_R1 | 1.0 | Same summary | `run_one()`: for baseline "oracle", pred_id = gold_id | Oracle = correct schema always | **Reproducible** |
| Same | oracle param_coverage, type_match, etc. | As in CSV | Same | Same downstream pipeline (extraction, typing, comparable_err) | Oracle subset = full eval set with pred=gold | **Reproducible** |

- **Oracle definition**: No retrieval; predicted schema = gold schema for every query. Same denominator (n_queries) and same extraction/typing pipeline as other baselines.

---

### 1.5 Schema-hit vs schema-miss (hit/miss subsets)

| Result group | Metric | Recomputed value / formula | Source | Source script/function | Assumptions | Status |
|--------------|--------|----------------------------|--------|------------------------|-------------|--------|
| **nlp4lp_downstream_hitmiss_table_orig** / **nlp4lp_error_hitmiss_table** | param_coverage_hits | Mean of per-query param_coverage over queries where pred_id == gold_id | `nlp4lp_downstream_summary.csv` (variant=orig, baseline in {lsa,bm25,tfidf}) | `run_one()`: cov_hits = list of param_coverage for schema_hit; agg = mean(cov_hits) | Hit = schema hit; denominator = number of hit queries | **Reproducible** |
| Same | param_coverage_miss | Mean over queries where pred_id != gold_id | Same | Same | Miss = schema miss | **Reproducible** |
| Same | type_match_hits / type_match_miss, key_overlap_hits / key_overlap_miss | Same pattern: mean over hit subset, mean over miss subset | Same | Same | Consistent hit/miss split by schema_hit | **Reproducible** |

- **Hit/miss consistency**: Hit = `(predicted_doc_id == gold_doc_id)`; miss = complement. Same definition in `nlp4lp_downstream_utility.py` and in artifact (artifact reads pre-aggregated _hits/_miss from summary).

---

### 1.6 Per-type metrics

| Result group | Metric | Recomputed value / formula | Source | Source script/function | Assumptions | Status |
|--------------|--------|----------------------------|--------|------------------------|-------------|--------|
| **nlp4lp_downstream_types_summary** | param_coverage (per type) | type_filled_total[t] / type_expected_total[t] | Written by `run_one()` → types_agg | `nlp4lp_downstream_utility.run_one()`: sums over all queries | **Micro** (total filled / total expected per type) | **Reproducible** |
| Same | type_match (per type) | type_correct_total[t] / type_filled_total[t] | Same | Same | **Micro** | **Reproducible** |
| Same | exact5_on_hits / exact20_on_hits (per type) | Over schema-hit queries only; denominator = type_exact5_den[t] etc. | Same | Same | Per-type denominators from hit queries with comparable_errs | **Reproducible** |
| **nlp4lp_downstream_types_table_orig** / **nlp4lp_error_types_table** | (same, pivoted) | Pivoted from types_summary for tfidf, oracle, random (orig) | `nlp4lp_downstream_types_summary.csv` | `make_nlp4lp_paper_artifacts.py` §4g, ESWA error-types block | 4 dp | **Reproducible** |

- **Macro vs micro**: Per-type metrics are **micro-averaged**: numerator and denominator are summed over **all queries** for that type (and baseline/variant), not “average of per-query type_match per type”.

---

### 1.7 Error ablation (typed vs untyped)

| Result group | Metric | Recomputed value | Source | Source script/function | Assumptions | Status |
|--------------|--------|------------------|--------|------------------------|-------------|--------|
| **nlp4lp_error_ablation_table** | param_coverage, type_match, exact20_on_hits, instantiation_ready for tfidf / tfidf_untyped / oracle / oracle_untyped | From downstream summary (orig) | `nlp4lp_downstream_summary.csv` | Artifact §4j: by_b from rows_orig; baselines tfidf, tfidf_untyped, oracle, oracle_untyped | Same n_queries; untyped = type-agnostic assignment | **Reproducible** |

---

### 1.8 Denominators (consistency)

| Metric | Denominator | Notes |
|--------|-------------|--------|
| schema_R1 | n_queries (331) | All queries |
| param_coverage (aggregate) | n_queries; per-query denom = n_expected_scalar | 0 when no expected scalar |
| type_match (aggregate) | n_queries; per-query denom = n_filled | |
| exact5_on_hits / exact20_on_hits | Number of schema-hit queries with non-empty comparable_errs | Not n_queries |
| instantiation_ready | n_queries | |
| key_overlap | n_queries; per-query denom = \|gold_scalar\| | |
| param_coverage_hits / _miss | Number of hit / miss queries | |
| type_match_hits / _miss | Same | |
| Per-type param_coverage | type_expected_total[t] (over all queries) | Micro |
| Per-type type_match | type_filled_total[t] (over all queries) | Micro |

---

### 1.9 Rounding rules

| Output | Rule | Where |
|--------|------|--------|
| Downstream section table, variant table, error tables, section note, error note | 4 decimal places (`_fmt4`: `f"{float(v):.4f}"`) | `make_nlp4lp_paper_artifacts.py` |
| Retrieval main table | 4 decimals for schema_R1; random = 1/331 → 0.0030 | Same |
| Retrieval note (e.g. schema_R1=… vs random≈…) | 3 decimals in text | `.3f` in f-strings |
| Main retrieval table (variant comparison) from retrieval_summary | 4 dp when written from artifact | Same |
| Dataset characterization | round(..., 2) for mean/std; percentiles as-is | `nlp4lp_dataset_characterization.py` |

---

## 2. Special checks summary

- **Random baseline**
  - **Retrieval**: Not run; artifact uses **theoretical** 1/331 (0.0030). Single value for all variants.
  - **Downstream**: **Deterministic single run** (seed from MD5 of query_id). Empirical schema_R1 = 2/331 ≈ 0.0060. Reproducible for same catalog and eval.

- **Oracle**
  - **Definition**: pred_id = gold_id for every query. No retrieval step; same extraction/typing/eval pipeline.
  - **Subset**: All 331 queries with “correct” schema by construction.

- **Schema-hit / schema-miss**
  - **Definition**: hit = (pred_id == gold_id); miss = (pred_id != gold_id). Used consistently in downstream script and artifact; hit/miss metrics are means over these subsets.

- **Per-type**
  - **Averaging**: **Micro**: per-type param_coverage = (sum over queries of filled for type t) / (sum over queries of expected for type t); per-type type_match = (sum correct for t) / (sum filled for t). Not macro (average of per-query type-level metrics).

- **Variants**
  - **orig**: `nlp4lp_eval_orig.jsonl`, `variant=orig` in summary CSV and types CSV.
  - **noisy**: `nlp4lp_eval_noisy.jsonl`, `variant=noisy`.
  - **short**: `nlp4lp_eval_short.jsonl`, `variant=short`.
  - All use same 331 query IDs; only query text differs.

---

## 3. Files and functions that generate each main result group

| Result group (intended table / output) | Files that produce it | Functions / scripts |
|---------------------------------------|------------------------|----------------------|
| **Retrieval metrics by variant** (Recall@1, etc.) | `results/nlp4lp_retrieval_metrics_{orig,noisy,short,noentity,nonum}.json` | `training/run_baselines.py` (main, with `--out results/nlp4lp_retrieval_metrics_<v>.json`) |
| **Retrieval summary CSV** | `results/nlp4lp_retrieval_summary.csv` | `tools/summarize_nlp4lp_results.py` (main) |
| **Retrieval main table (paper)** | `results/paper/nlp4lp_retrieval_main_table.csv` (.tex) | `tools/make_nlp4lp_paper_artifacts.py` §4i |
| **Downstream summary** | `results/paper/nlp4lp_downstream_summary.csv` | `tools/nlp4lp_downstream_utility.run_setting()` → `run_one()` per (variant, baseline) |
| **Downstream types summary** | `results/paper/nlp4lp_downstream_types_summary.csv` | Same: `run_one()` appends per-type rows |
| **Downstream section table** | `results/paper/nlp4lp_downstream_section_table.csv` (.tex) | `make_nlp4lp_paper_artifacts.py` §4j |
| **Downstream variant table** | `results/paper/nlp4lp_downstream_variant_table.csv` (.tex) | Same §4j |
| **Downstream main table orig** | `results/paper/nlp4lp_downstream_main_table_orig.csv` (.tex) | `make_nlp4lp_paper_artifacts.py` §4d |
| **Downstream hit/miss table orig** | `results/paper/nlp4lp_downstream_hitmiss_table_orig.csv` (.tex) | Same (block that builds hm_csv from dsum, variant=orig, baselines lsa,bm25,tfidf) |
| **Error hit/miss table** | `results/paper/nlp4lp_error_hitmiss_table.csv` (.tex) | Same block (err_hm_csv) |
| **Per-type table orig** | `results/paper/nlp4lp_downstream_types_table_orig.csv` (.tex) | Same §4g (types_csv_out from types_csv) |
| **Error types table** | `results/paper/nlp4lp_error_types_table.csv` (.tex) | Same (err_types_csv from types_csv_out) |
| **Error ablation table** | `results/paper/nlp4lp_error_ablation_table.csv` (.tex) | Same §4j (by_b for tfidf, tfidf_untyped, oracle, oracle_untyped) |
| **Section / error notes** | `nlp4lp_downstream_section_note.txt`, `nlp4lp_error_analysis_note.txt`, etc. | Same script (derived from same CSVs) |
| **Retrieval note** | `results/paper/nlp4lp_retrieval_note.txt` | Same §4i |

---

## 4. Ambiguities that cannot be resolved without the manuscript

- Whether the **manuscript** uses “random” for schema retrieval as **theoretical 1/N** or **empirical** (single run). The code uses theoretical for the retrieval table and empirical for downstream.
- Which **exact table numbers** are printed in the paper (e.g. 3 vs 4 decimal places) and which table corresponds to which figure/section title.
- Whether “schema R@1” in the text is always **retrieval** Recall@1 or sometimes **downstream** schema_R1 (they match when the same eval set is used, but random differs).
- Whether per-type results are described as “micro” or “macro” in the text.

---

## 5. Values or metrics to manually cross-check in the manuscript

- **Random baseline**: If the paper reports one number for “random schema retrieval”, confirm whether it is ~0.003 (1/331) or ~0.006 (2/331).
- **Oracle instantiation_ready**: Should be &lt; 0.1 on orig (e.g. 0.0816); confirm wording “even oracle stays below 0.1”.
- **TF-IDF vs BM25 vs LSA** on orig: schema_R1 order tfidf ≥ bm25 ≥ lsa; instantiation_ready and param_coverage values in section table.
- **Variant table**: orig vs noisy vs short for schema_R1 and instantiation_ready (noisy/short instantiation_ready 0 for tfidf/bm25/lsa in current outputs).
- **Hit/miss**: Coverage on hits ~0.88–0.90, on misses ~0.20–0.24; type_match higher on hits than misses but &lt; 0.3 on hits.
- **Per-type**: Integer easiest (type_match high), float hardest for type_match; currency/percent in between.
- **Ablation**: Typed vs untyped: type_match and instantiation_ready higher for typed (tfidf, oracle).

---

## 6. Suggested mapping: code outputs → manuscript tables

| Code output (file) | Likely manuscript table / use |
|--------------------|-------------------------------|
| `results/paper/nlp4lp_retrieval_main_table.csv` (.tex) | Table: Schema retrieval (Recall@1) by variant (orig / noisy / short) and baseline (random, LSA, BM25, TF-IDF). |
| `results/paper/nlp4lp_downstream_section_table.csv` (.tex) | Table: Downstream utility on NLP4LP (orig): schema_R1, param_coverage, type_match, key_overlap, exact5/20 on hits, instantiation_ready. |
| `results/paper/nlp4lp_downstream_variant_table.csv` (.tex) | Table: Cross-variant summary (orig/noisy/short schema_R1 and instantiation_ready per baseline). |
| `results/paper/nlp4lp_downstream_hitmiss_table_orig.csv` (.tex) or `nlp4lp_error_hitmiss_table.csv` (.tex) | Table: Schema-hit vs schema-miss downstream behavior (coverage, type match, key overlap on hit vs miss). |
| `results/paper/nlp4lp_downstream_types_table_orig.csv` (.tex) or `nlp4lp_error_types_table.csv` (.tex) | Table: Per-type downstream behavior (param_coverage, type_match by param type for TF-IDF, Oracle, Random). |
| `results/paper/nlp4lp_error_ablation_table.csv` (.tex) | Table: Effect of type-aware assignment (typed vs untyped for TF-IDF and Oracle). |
| `results/paper/nlp4lp_downstream_main_table_orig.csv` / `nlp4lp_downstream_final_table_orig.csv` | Alternative orig downstream table (same metrics, possibly different baselines or ordering). |
| `results/nlp4lp_retrieval_summary.csv` | Underlying retrieval numbers; artifact builds retrieval main table from this (plus 1/331 for random). |
| `results/paper/nlp4lp_downstream_summary.csv` | Canonical downstream aggregates; all downstream paper tables derive from this (and types_summary). |

---

**Status legend**

- **Reproducible**: Value or formula is fixed by code and data; rerunning the pipeline reproduces the number.
- **Unclear**: Definition or source could match multiple interpretations (e.g. random 0.003 vs 0.006 depending on context).
- **Missing pipeline link**: No script found that produces that exact output from raw data (none identified in this audit).
