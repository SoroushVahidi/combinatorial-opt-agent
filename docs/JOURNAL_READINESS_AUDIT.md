# Journal Readiness Audit — Combinatorial Opt Agent (NLP4LP)

**Date:** 2025-03-07  
**Scope:** Current repository state only; evidence-driven diagnostic for paper revision.  
**Manuscript:** Not inspected (as per instructions).

---

## 1. Executive summary

The repository implements an **NL-to-optimization** pipeline with two main evaluation tracks: (1) **retrieval** of problem schemas from natural-language queries (BM25, TF-IDF, LSA on a general catalog and on the NLP4LP benchmark), and (2) **NLP4LP downstream** parameter instantiation (schema selection → numeric extraction → typed/constrained mention–slot assignment → metrics). The **current method is deterministic** (no LLMs in the instantiation path): regex-based numeric extraction, heuristic typing (percent/currency/int/float), and either greedy typed assignment or global constrained (DP) assignment with interpretable scoring.

**Strengths:** Retrieval is strong (TF-IDF Recall@1 ~0.91 on orig, ~0.90 on noisy; short ~0.79). Downstream metrics are well-defined and documented; leak-free splits exist for the general catalog; NLP4LP experiment verification is documented. Constrained assignment improves **Exact20_on_hits** and per-slot type correctness at the cost of coverage and InstantiationReady.

**Weaknesses:** Typed assignment and numeric extraction remain the main bottlenecks: type_match and instantiation_ready are low (~0.23 and ~0.07 for best TF-IDF orig). Noisy variant has type_match 0 and instantiation_ready 0 (`<num>` placeholders). Oracle-vs-non-oracle gap is moderate (e.g. coverage 0.87 vs 0.82), so retrieval is not the only limiter. Several inconsistencies exist: random baseline definition differs between retrieval (1/331) and downstream (empirical 2/331); some paper tables (e.g. final_table_orig) omit constrained/repair baselines; empty eval files (nl4opt_family, resocratic) and duplicate/legacy scripts need cleanup. The project is **defensible for a journal submission** if claims are carefully scoped to retrieval + diagnostic downstream utility and caveats (no solver validation, scalar-only, heuristic-only) are stated; it is **not** ready to claim full NL-to-instantiation automation.

---

## 2. Current pipeline and code map

### 2.1 End-to-end flow

1. **Data:** Catalog from `pipeline/run_collection.py` (NL4Opt + OptMATH + merge) → `data/processed/all_problems.json`. For NLP4LP: `training/external/build_nlp4lp_benchmark.py` → `data/catalogs/nlp4lp_catalog.jsonl` (331 docs) and `data/processed/nlp4lp_eval_{orig,noisy,short,nonum,noentity}.jsonl` (331 queries per variant).
2. **Retrieval:** Query → baseline (BM25/TF-IDF/LSA) or oracle → top-1 (or top-k for rerank) doc_id.
3. **Schema/slots:** Gold schema and expected scalar parameters from HuggingFace `udell-lab/NLP4LP` test split (by query_id).
4. **Numeric extraction:** Regex + context → `NumTok` (value, kind: percent|currency|int|float|unknown); mentions with context/cue words.
5. **Assignment:** Typed greedy, untyped greedy, constrained (DP), semantic_ir_repair, or optimization_role_repair.
6. **Evaluation:** schema_R1, param_coverage, type_match, exact5/20_on_hits, key_overlap, instantiation_ready; optional per-type and hit/miss breakdowns.

### 2.2 Top-level components

| Role | Path | Key functions/classes | Status |
|------|------|------------------------|--------|
| **Data loading / preprocessing** | `pipeline/run_collection.py` | `main()`; calls collectors + `scripts/merge_catalog` | Active |
| | `training/external/build_nlp4lp_benchmark.py` | `build_nlp4lp_benchmark()`, `_query_variant()` | Active |
| | `retrieval/search.py` | `_load_catalog()`, `_searchable_text()`, `build_index()`, `search()` | Active |
| **Retrieval / schema prediction** | `retrieval/baselines.py` | `TfidfBaseline`, `BM25Baseline`, `SBERTBaseline`, `LSABaseline`, `get_baseline()` | Active |
| | `training/run_baselines.py` | `main()`, `_load_catalog()`, `_generate_eval_instances()`, `_load_eval_instances()` | Active (retrieval eval + optional `--out` for NLP4LP) |
| **Numeric extraction** | `tools/nlp4lp_downstream_utility.py` | `_parse_num_token()`, `_extract_num_tokens()`, `_extract_num_mentions()`, `NumTok`, `MentionRecord` | Active |
| **Slot/assignment** | `tools/nlp4lp_downstream_utility.py` | `_score_mention_slot()`, `_constrained_assignment()`, `_run_semantic_ir_repair()`, `_run_optimization_role_repair()`, `SlotRecord` | Active |
| **Evaluation** | `training/metrics.py` | `precision_at_k`, `reciprocal_rank_at_k`, `ndcg_at_k`, `coverage_at_k`, `compute_metrics()` | Active |
| | `tools/nlp4lp_downstream_utility.py` | `run_one()`, aggregation to summary CSV/types CSV; exact5/20, param_coverage, type_match, instantiation_ready | Active |
| **Result aggregation / tables / plots** | `tools/summarize_nlp4lp_results.py` | Reads `nlp4lp_retrieval_metrics_*.json` → `nlp4lp_retrieval_summary.csv` | Active |
| | `tools/make_nlp4lp_paper_artifacts.py` | Builds paper CSV/TeX/plots from retrieval + downstream summaries and stratified metrics | Active |

### 2.3 Main entry points (currently used)

- **Retrieval (general):**  
  `python -m training.run_baselines --splits data/processed/splits.json --split test --regenerate --num 500 --seed 999 --k 10 --baselines bm25 tfidf sbert --results-dir results`  
  (See `docs/BASELINE_TABLE_CLI.md`, `docs/PATCH_LEAK_FREE_EVAL.md`.)

- **NLP4LP retrieval (per variant):**  
  Run `run_baselines` with `--eval-file data/processed/nlp4lp_eval_<variant>.jsonl` and `--out results/nlp4lp_retrieval_metrics_<variant>.json` (or equivalent); then `python tools/summarize_nlp4lp_results.py` to get `results/nlp4lp_retrieval_summary.csv`.

- **NLP4LP downstream (single setting):**  
  `python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode constrained`  
  Writes/updates `results/paper/nlp4lp_downstream_summary.csv` and per-query JSON/CSV.

- **Build NLP4LP benchmark:**  
  `python -m training.external.build_nlp4lp_benchmark --split test --variants orig,nonum,short,noentity,noisy`.

- **Paper artifacts:**  
  `python tools/make_nlp4lp_paper_artifacts.py` (defaults: `results/nlp4lp_retrieval_summary.csv`, `results/nlp4lp_stratified_metrics.csv`, `results/paper/nlp4lp_downstream_summary.csv`).

- **Catalog build:**  
  `python pipeline/run_collection.py`.

---

## 3. Current method actually implemented

### 3.1 Pipeline steps (input problem text → evaluated output)

1. **Query** from eval JSONL (query_id, query, relevant_doc_id).
2. **Retrieval:** Baseline ranks catalog by query; top-1 doc_id chosen (or oracle uses gold relevant_doc_id).
3. **Gold schema:** From HF `udell-lab/NLP4LP` test: `parameters` (gold scalar keys) and `problem_info`; expected scalar list derived from gold parameters.
4. **Numeric extraction:**  
   - Regex `NUM_TOKEN_RE` for numbers; `_parse_num_token()` assigns kind (percent if `%` or context “percent”/“percentage”; currency if `$` or context “budget”/“cost”/etc.; else int/float).  
   - Mentions built with context window, sentence tokens, and cue words (CUE_WORDS set).
5. **Slot records:** From expected scalar names: `_expected_type(name)`, `_slot_aliases(name)`, normalized tokens.
6. **Assignment (by mode):**  
   - **typed:** Greedy best score per slot (mention–slot score with type compatibility, lexical overlap, cue overlap, operator/unit bonuses).  
   - **untyped:** Same but type not used in scoring.  
   - **constrained:** DP over slot subsets; at most one mention per slot, one slot per mention; same score function.  
   - **semantic_ir_repair / optimization_role_repair:** IR-based mention/slot representation + validation/repair.
7. **Metrics:** schema_R1 (pred_id == gold_id), param_coverage (mean over queries of n_filled/n_expected_scalar), type_match (mean over queries of type_correct/n_filled), exact5_on_hits / exact20_on_hits (mean over schema-hit queries with comparable_errs: fraction of slots with rel_err ≤ 5% / 20%), key_overlap (mean over queries of |pred∩gold|/|gold|), instantiation_ready (fraction of queries with param_coverage ≥ 0.8 and type_match ≥ 0.8).

### 3.2 Heuristics, rules, thresholds

- **Type incompatibility:** percent ↔ currency hard-blocked in `_is_type_incompatible()`.
- **ASSIGN_WEIGHTS:** type_match_bonus 3.0, type_mismatch -4.0, lex_context_overlap 0.7, cue_overlap 1.5, operator/unit bonuses 1–2, weak_match_penalty -1.0.
- **Exact5/20:** Relative error `_rel_err(pred, gold)`; exact5 = fraction ≤ 0.05, exact20 = fraction ≤ 0.20; only on schema-hit queries with comparable gold scalar values.
- **Instantiation_ready:** Thresholds 0.8 and 0.8 (param_coverage and type_match per query).

### 3.3 Ablations / alternative modes

- **Baselines:** bm25, tfidf, lsa, oracle, random; tfidf_acceptance_rerank, tfidf_hierarchical_acceptance_rerank (and bm25 variants).
- **Assignment:** typed, untyped, constrained, semantic_ir_repair, optimization_role_repair → effective baseline names e.g. tfidf_constrained, oracle_constrained.

### 3.4 Brittle / ad hoc spots

- **Catalog format:** Downstream expects `data/catalogs/nlp4lp_catalog.jsonl` with doc_id/text; retrieval baselines use `all_problems.json` with id/name/description/aliases. Two catalog worlds.
- **Gold source:** HF test split loaded at runtime; query_id must match `nlp4lp_test_{i}` convention.
- **Noisy variant:** &lt;num&gt; in query → no concrete numeric value → type_match and instantiation_ready collapse to 0; coverage still computed (slot filled with placeholder).
- **Short variant:** Very low coverage (e.g. ~0.03) because first-sentence-only often has few numbers.

---

## 4. Current datasets and evaluation setup

### 4.1 Datasets

| Name | Location | Size | Fields consumed | Role |
|------|----------|------|-----------------|------|
| **NLP4LP (HF test)** | HuggingFace `udell-lab/NLP4LP` split=test | 331 | description, parameters, problem_info → doc_id, passage, query variants | Main NLP4LP eval |
| **nlp4lp_catalog** | `data/catalogs/nlp4lp_catalog.jsonl` | 331 | doc_id, text, meta | Schema retrieval for NLP4LP |
| **nlp4lp_eval_orig/noisy/short/nonum/noentity** | `data/processed/nlp4lp_eval_*.jsonl` | 331 each | query_id, query, relevant_doc_id | Eval instances per variant |
| **all_problems** | `data/processed/all_problems.json` | ~1.1k+ | id, name, description, aliases | General retrieval catalog |
| **splits** | `data/processed/splits.json` | train/dev/test problem IDs | Problem IDs per split | Leak-free retrieval eval (general) |
| **eval_test / eval_dev** | `data/processed/eval_test.jsonl`, etc. | 500 or split size | query, problem_id | General retrieval eval |
| **nl4opt_type_eval_test** | `data/processed/nl4opt_type_eval_test.jsonl` | ~few | — | Auxiliary; small |
| **nl4opt_family_eval_***, resocratic_eval** | `data/processed/` | 0 (empty files) | — | Placeholder / broken |

### 4.2 Main evaluation datasets

- **NLP4LP:** 331 test queries, 5 variants (orig, nonum, short, noentity, noisy). Primary for paper: retrieval R@1/R@5/MRR@10/nDCG@10 and downstream schema_R1, param_coverage, type_match, exact5/20_on_hits, instantiation_ready.
- **General retrieval:** test split from splits.json, eval_test.jsonl; P@1, P@5, MRR@10, nDCG@10, Coverage@10.

### 4.3 Noisy/short variants

- **Noisy:** `_query_variant(..., "noisy")`: lowercase, numbers → &lt;num&gt;, drop stopwords, 10% token drop (seed from query_id). Type_match and instantiation_ready are 0 because values are not recoverable.
- **Short:** First sentence only (first [.!?] not after Mr/Mrs/Ms/Dr). Low coverage.

---

## 5. Current latest results and what they show

### 5.1 Retrieval (from `results/nlp4lp_retrieval_summary.csv`)

| Variant | TF-IDF R@1 | BM25 R@1 | LSA R@1 |
|---------|------------|----------|---------|
| orig    | 0.9063     | 0.8852   | 0.8550  |
| noisy   | 0.9033     | 0.8943   | 0.8912  |
| short   | 0.7855     | 0.7734   | 0.7704  |
| nonum   | 0.9063     | 0.8973   | 0.8550  |
| noentity| 0.9063     | 0.8882   | 0.8489  |

Retrieval is strong; short is the hardest.

### 5.2 Downstream (from `results/paper/nlp4lp_downstream_summary.csv`)

**Orig, main baselines (331 queries):**

- **schema_R1:** tfidf 0.9063, oracle 1.0, random ~0.006.
- **param_coverage:** tfidf 0.8222, oracle 0.8695, tfidf_constrained 0.7720.
- **type_match:** tfidf 0.2267, oracle 0.2401 (oracle_constrained 0.2092), tfidf_constrained 0.1950.
- **exact20_on_hits:** tfidf 0.2140, tfidf_constrained 0.3279, oracle 0.1871, oracle_constrained 0.3206.
- **instantiation_ready:** tfidf 0.0755, oracle 0.0816, tfidf_constrained 0.0272.

**Noisy:** type_match 0, instantiation_ready 0 for all; coverage ~0.71 (tfidf) vs ~0.29 (tfidf_constrained).

**Short:** Very low coverage (~0.03); exact20_on_hits improves with constrained (e.g. 0.0588 → 0.3088 for tfidf vs tfidf_constrained).

### 5.3 What improved

- Constrained assignment improves **Exact20_on_hits** and per-type type_match on filled slots, at the cost of coverage and instantiation_ready (documented in `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md`).
- Acceptance rerank and hierarchical acceptance rerank are in the summary; semantic_ir_repair and optimization_role_repair show modest gains in type_match/exact20 in some settings.

### 5.4 Where the pipeline still breaks

- **Noisy:** No real numeric values → type_match and instantiation_ready uninformative (0).
- **Short:** Too little text → very low coverage.
- **Typed assignment:** Still low type_match (~0.23) and instantiation_ready (~0.07); many slots wrong type or unfilled.
- **Integer/float:** Per-type tables show float parameters hardest by type_match.

---

## 6. Main bottlenecks and failure modes

1. **Retrieval/schema selection:** Strong (R@1 ~0.91 orig). Not the main bottleneck.
2. **Numeric extraction:** Rule-based; &lt;num&gt; in noisy variant yields no value; percent/currency context can be wrong.
3. **Typed assignment / value–slot matching:** Main bottleneck: type_match and instantiation_ready low; constrained improves precision on filled slots but reduces coverage.
4. **Metrics that remain weak:** type_match (~0.23), instantiation_ready (~0.07), exact5 (stricter than exact20).
5. **Noisy-query results:** Poor for downstream because placeholders are not resolved; retrieval is still good.
6. **Oracle vs non-oracle:** Oracle coverage 0.87 vs tfidf 0.82; oracle type_match ~0.24 vs tfidf ~0.23. So retrieval helps but is not the only limiter; extraction/assignment matter a lot.
7. **Overfitting:** Single benchmark (NLP4LP 331 test); no external held-out set. Possible dataset-specific bias.
8. **Contradictions:** `nlp4lp_downstream_final_table_orig.csv` omits constrained/repair baselines; random = 1/331 in retrieval main table vs 2/331 empirical in downstream (documented in verification report).

---

## 7. Code and evaluation risks and inconsistencies

- **Random baseline:** Retrieval table uses theoretical 1/331; downstream uses one deterministic run (2/331). Same term “random” used for both → reviewer confusion.
- **Denominators:** exact5/20_on_hits use “schema-hit queries with non-empty comparable_errs”; instantiation_ready uses all n_queries. Documented but easy to misread.
- **Two catalogs:** General (all_problems.json) vs NLP4LP (nlp4lp_catalog.jsonl); different schemas and uses.
- **Empty files:** `nl4opt_family_eval_test.jsonl`, `resocratic_eval.jsonl`, `nl4opt_comp_eval.jsonl` are 0 bytes; scripts that expect them can fail or be skipped.
- **Magic numbers:** ASSIGN_WEIGHTS, SEMANTIC_IR_WEIGHTS, 0.8/0.8 for instantiation_ready; not centralized.
- **Duplicate/legacy:** `scripts/evaluate_retrieval.py` vs `training/evaluate_retrieval.py`; ensure one is the canonical entry.
- **Train/test:** General retrieval uses splits.json (disjoint problem IDs); NLP4LP uses HF test split only (no train in downstream path). No leakage in NLP4LP downstream from retrieval training because downstream does not train.

Evaluation could mislead if: (1) “random” is reported without specifying which definition; (2) exact20_on_hits is reported without “on schema-hit with comparable errors”; (3) instantiation_ready is presented as “solver-ready” without caveat (no solver run).

---

## 8. Likely reviewer criticisms

- **Single benchmark:** Only NLP4LP test (331); need clarity that this is one benchmark, not “NL-to-optimization in general.”
- **No solver validation:** Instantiation_ready does not check feasibility or optimality; reviewer may ask for solver runs or explicit caveats.
- **Noisy variant:** Downstream metrics are 0 by design; either explain clearly or add a separate “noisy retrieval only” narrative.
- **Random baseline inconsistency:** Different definitions for retrieval vs downstream; should be aligned or explicitly labeled (e.g. “random (theoretical)” vs “random (empirical)”).
- **Low absolute numbers:** type_match ~0.23, instantiation_ready ~0.07; reviewers may question practical impact; response: diagnostic value and improvement from constrained/repair.
- **Heuristic-only:** No learned component in extraction/assignment; some reviewers may want learning-based baselines or discussion of limitations.

---

## 9. Priority action plan

### A. Highest-impact fixes for method/results

1. **Align or document random baseline**  
   - **Why:** Removes inconsistency between retrieval (1/331) and downstream (2/331).  
   - **Where:** `tools/make_nlp4lp_paper_artifacts.py` (retrieval table), `tools/nlp4lp_downstream_utility.py` (random baseline).  
   - **Change:** Either use same definition (e.g. run random retrieval and report empirical R@1) or label both in tables/notes (“random (theoretical)” vs “random (empirical, 1 run)”).

2. **Noisy variant narrative**  
   - **Why:** type_match/instantiation_ready = 0 by design; avoid “method fails on noisy.”  
   - **Where:** Paper text and `results/paper/nlp4lp_downstream_caveats_for_paper.txt`.  
   - **Change:** State explicitly that noisy queries use `<num>` placeholders and downstream numeric metrics are N/A; report retrieval-only for noisy and optionally “coverage with placeholders” for downstream.

3. **Single benchmark / generalizability**  
   - **Why:** Addresses “only one dataset.”  
   - **Where:** Intro / experimental section.  
   - **Change:** Frame as “benchmark study on NLP4LP” and add a short limitation: no second benchmark yet; future work on cross-dataset or external eval.

### B. Highest-impact fixes for evaluation credibility

4. **Centralize and document denominators**  
   - **Why:** Reduces risk of misreporting.  
   - **Where:** `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` (already good); add one “Metrics quick reference” table in README or docs listing each metric, denominator, and filter (e.g. “exact20_on_hits: mean over schema-hit queries with ≥1 comparable slot”).

5. **Final table vs full summary**  
   - **Why:** `nlp4lp_downstream_final_table_orig.csv` omits constrained/repair; reviewers may think they are not compared.  
   - **Where:** `tools/make_nlp4lp_paper_artifacts.py` (section that writes final_table).  
   - **Change:** Either include tfidf_constrained, oracle_constrained, and optionally one repair variant in the “final” table, or add a short note in the paper that the main table is a subset and full table is in appendix/supplement.

6. **Empty eval files**  
   - **Why:** Avoid silent skips or failures.  
   - **Where:** `data/processed/nl4opt_family_eval_test.jsonl`, `resocratic_eval.jsonl`, `nl4opt_comp_eval.jsonl`.  
   - **Change:** Either generate minimal valid eval files or remove/guard references in scripts and document which evals are “active” vs “placeholder.”

### C. Cleanup and documentation

7. **Canonical retrieval eval entry point**  
   - **Where:** `training/evaluate_retrieval.py` vs `scripts/evaluate_retrieval.py`.  
   - **Change:** Prefer one (e.g. `training/`) and point all docs/README to it; deprecate or remove the other.

8. **Constants / thresholds**  
   - **Where:** `tools/nlp4lp_downstream_utility.py` (ASSIGN_WEIGHTS, 0.8/0.8).  
   - **Change:** Move to a small config (e.g. `schema/` or `training/config.py`) or document in verification report so they are auditable.

9. **Reproducibility script**  
   - **Where:** `docs/BASELINE_TABLE_CLI.md`, `docs/PATCH_LEAK_FREE_EVAL.md`, `results/paper/nlp4lp_reproducibility.md`.  
   - **Change:** Single “Reproduce paper results” section or script that lists: (1) build catalog, (2) build NLP4LP benchmark, (3) run retrieval baselines per variant, (4) summarize retrieval, (5) run downstream for each (variant, baseline, assignment_mode), (6) run make_nlp4lp_paper_artifacts.

---

## 10. Appendix: file-by-file evidence list

| Evidence | Path |
|----------|------|
| Project layout | `find` (directories), `results/`, `data/processed/` listings |
| Retrieval search and catalog | `retrieval/search.py`, `retrieval/baselines.py` |
| Retrieval eval runner | `training/run_baselines.py` |
| Metrics definitions | `training/metrics.py` |
| NLP4LP downstream pipeline | `tools/nlp4lp_downstream_utility.py` (extraction, assignment, run_one, main) |
| NLP4LP benchmark build | `training/external/build_nlp4lp_benchmark.py` |
| Splits and leakage tests | `training/splits.py`, `tests/test_no_leakage.py` |
| Schema | `schema/problem_schema.json` |
| Pipeline entry | `pipeline/run_collection.py` |
| Retrieval summary (current) | `results/nlp4lp_retrieval_summary.csv` |
| Downstream summary (current) | `results/paper/nlp4lp_downstream_summary.csv` |
| Paper tables | `results/paper/nlp4lp_retrieval_main_table.csv`, `results/paper/nlp4lp_downstream_final_table_orig.csv` |
| Verification and denominators | `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` |
| Constrained assignment interpretation | `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md` |
| Claims and caveats | `results/paper/nlp4lp_downstream_claims_for_paper.txt`, `results/paper/nlp4lp_downstream_caveats_for_paper.txt` |
| Baseline table CLI | `docs/BASELINE_TABLE_CLI.md`, `docs/PATCH_LEAK_FREE_EVAL.md` |
| Artifact generator | `tools/make_nlp4lp_paper_artifacts.py`, `tools/summarize_nlp4lp_results.py` |
| Eval file sizes and sample | `wc -l` on nlp4lp_eval_*.jsonl; `head` on nlp4lp_eval_orig.jsonl |
| Catalog sizes | `data/catalogs/nlp4lp_catalog.jsonl` (331 lines), `data/processed/all_problems.json` |
| Splits content | `data/processed/splits.json` |
