# Strict current-state audit

**Date:** 2025-03-08  
**Scope:** Current repository state only. Evidence from file contents and code paths; shell returned empty for `git status`, `stat`, and `ls logs/` so timestamps and log presence are inferred from content where possible.

---

## 1. Recently changed files and whether results are up to date

**Git status:** Not available (command returned empty). Branch: `main` (from `.git/HEAD`).

**Evidence from content:**

- **Downstream summary vs paper tables:**  
  `results/paper/nlp4lp_downstream_summary.csv` contains **current** values for orig/tfidf: `exact5_on_hits=0.20531573278052154`, `exact20_on_hits=0.23302949007174353`.  
  `results/paper/nlp4lp_downstream_final_table_orig.csv` and `results/paper/nlp4lp_downstream_section_table.csv` contain **older** values for the same: `exact5_on_hits=0.1876`, `exact20_on_hits=0.2140`.  
  **Conclusion:** The summary was updated after the last run of `make_nlp4lp_paper_artifacts.py`. Paper tables are **stale** relative to the summary.

- **Generation path:**  
  Downstream summary is updated by `tools/nlp4lp_downstream_utility.py` (`run_setting` → `_upsert_summary_row`) each time a downstream run completes.  
  Paper tables are written only when `tools/make_nlp4lp_paper_artifacts.py` runs; they read `results/paper/nlp4lp_downstream_summary.csv` at that time.  
  So: **result summaries (downstream_summary.csv) are the current source of truth; derived paper tables (final_table_orig, section_table, main_table_orig) are stale.**

- **Retrieval:**  
  `results/nlp4lp_retrieval_summary.csv` is produced by `tools/summarize_nlp4lp_results.py` from `results/nlp4lp_retrieval_metrics_<variant>.json`. No value mismatch was checked; if retrieval was re-run and summarize was run, retrieval summary is current. Paper retrieval table is produced by the artifact script from that summary.

---

## 2. Authoritative current scripts and commands

| Purpose | Script (exact path) | Command pattern | Status |
|--------|----------------------|------------------|--------|
| Build datasets/catalogs (general) | `pipeline/run_collection.py` | `python pipeline/run_collection.py` | Active |
| Build NLP4LP benchmark (catalog + eval JSONL) | `training/external/build_nlp4lp_benchmark.py` | `python -m training.external.build_nlp4lp_benchmark --split test --variants orig,nonum,short,noentity,noisy` | Active |
| Run retrieval experiments (NLP4LP) | `training/run_baselines.py` | `python -m training.run_baselines --eval-file data/processed/nlp4lp_eval_<variant>.jsonl --out results/nlp4lp_retrieval_metrics_<variant>.json --catalog data/catalogs/nlp4lp_catalog.jsonl --baselines bm25 tfidf lsa` (and optionally `--splits`/`--split` for general catalog) | Active |
| Run downstream/instantiation | `tools/nlp4lp_downstream_utility.py` | `python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf [--assignment-mode typed|untyped|constrained|semantic_ir_repair|optimization_role_repair]` | Active |
| Aggregate retrieval results | `tools/summarize_nlp4lp_results.py` | `python tools/summarize_nlp4lp_results.py` (reads `results/nlp4lp_retrieval_metrics_*.json`, writes `results/nlp4lp_retrieval_summary.csv`) | Active |
| Generate paper artifacts | `tools/make_nlp4lp_paper_artifacts.py` | `python tools/make_nlp4lp_paper_artifacts.py` (reads retrieval summary, stratified CSV, `results/paper/nlp4lp_downstream_summary.csv`; writes all CSVs/TeX/plots in `results/paper/`) | Active |

**Legacy/duplicate:**  
- `scripts/evaluate_retrieval.py` exists alongside `training/evaluate_retrieval.py`; docs point to `training/run_baselines.py` for paper-ready retrieval. Canonical retrieval runner for NLP4LP is `training/run_baselines.py` with `--out` and NLP4LP eval/catalog.

**Job on compute node:**  
- `jobs/run_paper_artifacts.slurm` runs `python tools/make_nlp4lp_paper_artifacts.py` on a compute node. Submit with `sbatch jobs/run_paper_artifacts.slurm` from project root.

---

## 3. Exact current method implemented

**Input format:**  
- Eval: JSONL with `query_id`, `query`, `relevant_doc_id` per line (`data/processed/nlp4lp_eval_<variant>.jsonl`).  
- Catalog: JSONL with `doc_id`, `text`, `meta` (`data/catalogs/nlp4lp_catalog.jsonl`) for NLP4LP; or JSON with `id`, `name`, `description`, `aliases` for general retrieval.

**Pipeline (downstream in `tools/nlp4lp_downstream_utility.py`):**

1. **Retrieval**  
   - **File:** `retrieval/baselines.py`.  
   - **Methods:** `TfidfBaseline`, `BM25Baseline`, `LSABaseline` (fit on catalog, rank(query, top_k) → [(problem_id, score)]).  
   - **Schema selection:** Top-1 doc_id from baseline rank, or oracle: `pred_id = gold relevant_doc_id`.  
   - **Entry:** `run_setting` calls `rank_fn(query)` (or oracle path); k=1 only.

2. **Gold schema**  
   - **Source:** HuggingFace `udell-lab/NLP4LP` split=test, loaded in `_load_hf_gold(split="test")`.  
   - **Consumed:** `parameters` (gold scalar keys), `problem_info`; expected scalar list from gold.

3. **Numeric extraction**  
   - **File:** `tools/nlp4lp_downstream_utility.py`.  
   - **Functions:** `_parse_num_token()` (regex `NUM_TOKEN_RE` for numbers), `_extract_num_tokens()`, `_extract_num_mentions()` (context, sentence tokens, CUE_WORDS).  
   - **Output:** `NumTok(raw, value, kind)` with kind in `percent|currency|int|float|unknown`; `MentionRecord` (index, tok, context_tokens, sentence_tokens, cue_words).  
   - **Type inference:** In `_parse_num_token`: `%` or context "percent"/"percentage" → percent; `$` or money context → currency; else int/float from value.

4. **Slot representation**  
   - **Function:** `_build_slot_records(expected_scalar)` → `SlotRecord(name, norm_tokens, expected_type, aliases, alias_tokens)`.  
   - **Helpers:** `_expected_type(name)`, `_slot_aliases(name)` (file-local).

5. **Slot/value assignment**  
   - **Modes (argument `--assignment-mode`):**  
     - **typed:** Greedy per-slot best score using `_score_mention_slot()` (same as below).  
     - **untyped:** Same scoring with type not used.  
     - **constrained:** `_constrained_assignment(mentions, slots)` — DP over slot subsets; at most one mention per slot, one slot per mention; score from `_score_mention_slot()`.  
     - **semantic_ir_repair:** `_run_semantic_ir_repair()` — IR-based mention/slot representation + validation/repair.  
     - **optimization_role_repair:** `_run_optimization_role_repair()` — optimization-role-aware assignment.  
   - **Scoring (constrained/typed):** `_score_mention_slot(MentionRecord, SlotRecord)` → (score, features).  
     - Hard: `_is_type_incompatible(expected, kind)` → percent vs currency blocked (-1e9).  
     - Weights: `ASSIGN_WEIGHTS` (type_match_bonus 3.0, lex_context_overlap 0.7, cue_overlap 1.5, operator/unit bonuses, weak_match_penalty -1.0).  
     - Tie-break: implicit in score ordering; DP picks best global assignment.

6. **Evaluation logic**  
   - **File:** same; inside `run_one()` / aggregation.  
   - **Metrics:** schema_R1 (pred_id == gold_id), param_coverage (mean over queries of n_filled/n_expected_scalar), type_match (mean of type_correct/n_filled per query), exact5_on_hits / exact20_on_hits (mean over schema-hit queries with non-empty comparable_errs: fraction of slots with rel_err ≤ 0.05 / 0.20), key_overlap (mean |pred∩gold|/|gold|), instantiation_ready (fraction of queries with param_coverage ≥ 0.8 and type_match ≥ 0.8).  
   - **Relative error:** `_rel_err(pred, gold)`; comparable only when both scalar.

**Constants / thresholds:**  
- `ASSIGN_WEIGHTS` (lines 55–65): type_match_bonus 3.0, type_mismatch_penalty -4.0, lex_context_overlap 0.7, lex_sentence_overlap 0.3, cue_overlap 1.5, operator_min/max_bonus 1.0, unit_percent/currency_bonus 2.0, weak_match_penalty -1.0.  
- Instantiation_ready: 0.8 and 0.8 (param_coverage and type_match per query).  
- Exact5/20: rel_err ≤ 0.05 and ≤ 0.20.

**Retrieval metrics (`training/metrics.py`):**  
- P@1, P@5, MRR@k, nDCG@k, Coverage@k; denominator = number of eval instances.  
- `compute_metrics(results, k=10)`.

**Baselines vs preferred:**  
- Retrieval: bm25, tfidf, lsa (all active); tfidf is typically reported as best.  
- Downstream: typed (default), untyped (ablation), constrained (improves exact20, lowers coverage), semantic_ir_repair, optimization_role_repair. Oracle = gold schema every query.

---

## 4. Exact current datasets used

| Dataset | Path/source | Size | Fields | Consumed | Status |
|--------|-------------|------|--------|----------|--------|
| NLP4LP catalog | `data/catalogs/nlp4lp_catalog.jsonl` | 331 lines | doc_id, text, meta | doc_id, text (and meta if used) | Active, main for NLP4LP |
| NLP4LP eval (per variant) | `data/processed/nlp4lp_eval_{orig,noisy,short,nonum,noentity}.jsonl` | 331 lines each | query_id, query, relevant_doc_id | all | Active |
| Gold schema/params | HuggingFace `udell-lab/NLP4LP` split=test | 331 | parameters, problem_info, description, etc. | parameters, problem_info (and description for query variants) | Active, at runtime |
| General catalog | `data/processed/all_problems.json` | ~1.1k+ entries | id, name, description, aliases, formulation, ... | id, name, description, aliases for retrieval | Active for general retrieval |
| Splits (general) | `data/processed/splits.json` | train/dev/test problem IDs | train, dev, test | For leak-free retrieval eval | Active |
| Eval (general) | `data/processed/eval_test.jsonl` etc. | 500 or split size | query, problem_id | For run_baselines with --splits | Active |

**Query variants:**  
- **orig:** raw description.  
- **nonum:** numbers replaced with `<num>`.  
- **short:** first sentence only (first [.!?] not after Mr/Mrs/Ms/Dr).  
- **noentity:** remove Mr/Mrs/Ms/Dr + following word; drop standalone capitalized at sentence start.  
- **noisy:** lowercase, numbers→`<num>`, drop stopwords, 10% token drop (seed=query_id).  
Defined in `training/external/build_nlp4lp_benchmark.py` `_query_variant()`.

**Main benchmark for current reported results:**  
- NLP4LP test, 331 queries, 5 variants. Eval files: `data/processed/nlp4lp_eval_<variant>.jsonl`. Catalog: `data/catalogs/nlp4lp_catalog.jsonl`. Gold: HF `udell-lab/NLP4LP` test.

**Auxiliary/placeholder:**  
- `data/processed/nl4opt_family_eval_test.jsonl`, `resocratic_eval.jsonl`, `nl4opt_comp_eval.jsonl` referenced in docs; if 0-byte or missing, not used for current NLP4LP results.

---

## 5. Latest trusted results

**Retrieval (trusted source: `results/nlp4lp_retrieval_summary.csv`):**  
- orig: tfidf R@1 0.9063, bm25 0.8852, lsa 0.8550.  
- noisy: tfidf 0.9033, bm25 0.8943, lsa 0.8912.  
- short: tfidf 0.7855, bm25 0.7734, lsa 0.7704.  
- nonum, noentity: same as orig for tfidf (0.9063).  
Denominator: 331 per variant. Generated by `training/run_baselines.py` → `results/nlp4lp_retrieval_metrics_<variant>.json` → `tools/summarize_nlp4lp_results.py`.

**Downstream (trusted source: `results/paper/nlp4lp_downstream_summary.csv`):**  
- **orig (331):** schema_R1 tfidf 0.9063, oracle 1.0, random 0.0060; param_coverage tfidf 0.8222, oracle 0.8695, tfidf_constrained 0.7720; type_match tfidf 0.2260, oracle 0.2401; exact20_on_hits tfidf **0.2330**, tfidf_constrained 0.3250, oracle 0.2044; instantiation_ready tfidf 0.0755, oracle 0.0816.  
- **noisy:** type_match 0, instantiation_ready 0 for all (expected; `<num>` placeholders).  
- **short:** very low param_coverage (~0.03); exact20_on_hits higher with constrained.  
Generated by repeated runs of `tools/nlp4lp_downstream_utility.py`; each run updates one or more rows in the summary via `_upsert_summary_row`.

**Stale (do not treat as latest):**  
- `results/paper/nlp4lp_downstream_final_table_orig.csv`  
- `results/paper/nlp4lp_downstream_main_table_orig.csv`  
- `results/paper/nlp4lp_downstream_section_table.csv`  
They contain tfidf exact5=0.1876, exact20=0.2140 instead of 0.2053 and 0.2330. They were produced by an earlier run of `make_nlp4lp_paper_artifacts.py` and have not been regenerated since the summary was last updated.

**Incomplete/broken:**  
- No evidence of partial runs in the summary; n=331 for all reported rows. Empty eval files (e.g. nl4opt_family, resocratic) are not used for these numbers.

---

## 6. Whether paper artifacts are current and consistent

**Paper artifact job:**  
- `logs/` was listed; output was empty (no log files found or listing failed). So it could not be confirmed from logs whether `jobs/run_paper_artifacts.slurm` has run or whether `make_nlp4lp_paper_artifacts.py` completed.

**Consistency (content-based):**  
- **Not consistent.** The downstream summary has newer values (orig tfidf exact20=0.2330, exact5=0.2053) than the paper tables (0.2140, 0.1876). So either the artifact job has not been run after the last downstream runs, or it ran earlier and the summary was updated later.  
- **Conclusion:** Paper artifacts (final_table_orig, section_table, main_table_orig, and any other downstream-derived tables/plots) are **not** current with `results/paper/nlp4lp_downstream_summary.csv`. To make them consistent, run `python tools/make_nlp4lp_paper_artifacts.py` (e.g. via `sbatch jobs/run_paper_artifacts.slurm` on a compute node).

---

## 7. Main remaining weaknesses and risks

- **Strongest:** Retrieval (R@1 ~0.91 orig, ~0.90 noisy); clear pipeline and metrics; leak-free splits for general catalog; multiple assignment modes and ablations.  
- **Weakest:** type_match and instantiation_ready still low (~0.23 and ~0.07); noisy downstream uninformative (type_match/instantiation_ready 0 by design).  
- **Main bottleneck:** Typed/constrained assignment and numeric extraction (slot filling and type correctness), not retrieval.  
- **Retrieval:** Still strong.  
- **Numeric extraction:** Rule-based; weak on noisy (`<num>` → no value).  
- **Noisy-query results:** Downstream metrics 0 for type_match/instantiation_ready because queries use `<num>` placeholders; retrieval remains strong.  
- **Oracle vs non-oracle:** Oracle coverage 0.87 vs tfidf 0.82; oracle type_match ~0.24 vs tfidf ~0.23. So retrieval helps but extraction/assignment are the main limiters.  
- **Risks:** Random baseline definition differs (retrieval 1/331 vs downstream empirical); paper tables stale; single benchmark (NLP4LP 331); no solver validation; denominators for exact5/20 (on schema-hit with comparable_errs) must be stated clearly.

---

## 8. What to trust right now vs what not to trust yet

**Trust:**  
- `results/paper/nlp4lp_downstream_summary.csv` as the **current** downstream source of truth.  
- `results/nlp4lp_retrieval_summary.csv` as the current retrieval summary (unless retrieval was re-run and summarize not re-run).  
- Code paths and constants as documented in §3 and §4.  
- NLP4LP eval files and catalog as the data underlying the reported n=331 results.

**Do not trust as “latest”:**  
- `results/paper/nlp4lp_downstream_final_table_orig.csv`  
- `results/paper/nlp4lp_downstream_main_table_orig.csv`  
- `results/paper/nlp4lp_downstream_section_table.csv`  
and any other paper CSVs/TeX/plots that are derived from the downstream summary by `make_nlp4lp_paper_artifacts.py`, until that script is re-run and the outputs are regenerated from the current summary.

**Verify before citing:**  
- That retrieval metrics JSONs and retrieval summary were regenerated after any change to retrieval or eval.  
- That paper tables and figures were regenerated after any change to the downstream summary or to the artifact script.

---

## 9. Appendix: evidence list

| Evidence | Path / source |
|----------|----------------|
| Downstream summary (current values) | `results/paper/nlp4lp_downstream_summary.csv` (orig tfidf exact5=0.2053, exact20=0.2330) |
| Final table (stale values) | `results/paper/nlp4lp_downstream_final_table_orig.csv` (tfidf exact5=0.1876, exact20=0.2140) |
| Section table (stale) | `results/paper/nlp4lp_downstream_section_table.csv` (same stale values) |
| Retrieval summary | `results/nlp4lp_retrieval_summary.csv` |
| Downstream pipeline and assignment | `tools/nlp4lp_downstream_utility.py` (ASSIGN_WEIGHTS, _score_mention_slot, _constrained_assignment, run_setting, main) |
| Artifact script (reads/writes) | `tools/make_nlp4lp_paper_artifacts.py` (downstream_csv = out_dir / "nlp4lp_downstream_summary.csv", order for final_table line 299) |
| Retrieval baselines | `retrieval/baselines.py` (TfidfBaseline, BM25Baseline, LSABaseline, get_baseline) |
| Retrieval runner | `training/run_baselines.py` (_load_catalog, --out, eval-file) |
| Retrieval metrics | `training/metrics.py` (compute_metrics, P@1, MRR@k, nDCG@k) |
| Summarize retrieval | `tools/summarize_nlp4lp_results.py` (reads nlp4lp_retrieval_metrics_*.json, writes nlp4lp_retrieval_summary.csv) |
| Build NLP4LP benchmark | `training/external/build_nlp4lp_benchmark.py` (_query_variant, build_nlp4lp_benchmark) |
| Catalog build | `pipeline/run_collection.py` |
| Splits | `data/processed/splits.json` |
| Git HEAD | `.git/HEAD` (ref: refs/heads/main) |
| Paper artifact job | `jobs/run_paper_artifacts.slurm` |
| Logs | `logs/` (listing returned empty; no log files inspected) |

**Timestamps:** `stat` and `ls -la` on key files returned empty in this environment; recency and staleness are inferred from **value comparison** (summary vs paper tables) and **generation path** (summary updated by downstream runs; paper tables by artifact script only).
