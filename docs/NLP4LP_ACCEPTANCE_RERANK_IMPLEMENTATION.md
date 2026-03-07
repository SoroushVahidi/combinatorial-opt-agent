# NLP4LP Acceptance Reranking Implementation

## Where the reranking plugs in

- **File:** `tools/nlp4lp_downstream_utility.py`. All reranking logic lives in this file (no separate module), reusing extraction and gold/schema helpers and avoiding circular imports.
- **Entry:** `main()` builds `rank_fn` from the chosen `--baseline`. For baselines ending in `_acceptance_rerank` or `_hierarchical_acceptance_rerank`, `rank_fn` is `make_rerank_rank_fn(base.rank, gold_by_id, catalog, k_retrieval=--acceptance-k, use_hierarchy=..., variant=args.variant)`.
- **Usage:** `run_setting(..., rank_fn=rank_fn, ...)` calls `rank_fn(query, top_k=1)` for each eval query. The reranker internally calls the base retriever with `top_k=k_retrieval` (default 10), scores each candidate with schema acceptance, combines with retrieval score, and returns the top-1 (or top_k) schema id.
- **Downstream:** The returned schema id is used unchanged: `pred = gold_by_id.get(pred_id)`, `expected_params` from `problem_info`/`parameters`, then the existing assignment modes (typed, constrained, semantic_ir_repair, optimization_role_repair) run as before.

No changes were made to:
- `retrieval/baselines.py`
- Benchmark splits or metric definitions
- Existing baseline names (bm25, tfidf, lsa, oracle) or assignment modes

## Files / functions changed

- **tools/nlp4lp_downstream_utility.py**
  - **Added:** `ACCEPTANCE_RERANK_WEIGHTS` (config at top of acceptance block).
  - **Added:** `FAMILY_KEYWORDS`, `_schema_family()`, `_schema_subgroup()`, `_query_family_hints()`, `_query_subgroup_hints()` for hierarchy.
  - **Added:** `_extract_query_acceptance_features(query, variant)` for query-side evidence.
  - **Added:** `_get_expected_scalar_for_schema()`, `_build_schema_acceptance_profile()` for schema profiles.
  - **Added:** `_acceptance_score()` for additive acceptance score and debug breakdown.
  - **Added:** `make_rerank_rank_fn()` returning a `rank_fn(query, top_k)`.
  - **Changed:** `main()`: new `--acceptance-k` (default 10); baseline can be `tfidf_acceptance_rerank`, `tfidf_hierarchical_acceptance_rerank`, `bm25_acceptance_rerank`, `bm25_hierarchical_acceptance_rerank`; construction of `rank_fn` for these via `make_rerank_rank_fn`.

## Family / subgroup design

- **Coarse families (rule-based):** Keywords in schema id + description + slot names map to: `allocation`, `production`, `transportation`, `scheduling`, `packing`, `covering`, `network`, `inventory`, `ratio`, `generic_lp`. First match by keyword count wins; default `generic_lp`.
- **Subgroups (optional):** Inferred from slot names only: `total_budget_per_unit_profit`, `capacity_demand`, `min_max_bounds`, `ratio_fraction`, `fixed_cost_penalty`, `time_resource`, `item_count`, or empty.
- **Query hints:** `_query_family_hints(query)` and `_query_subgroup_hints(query)` return sets of family/subgroup tags present in the query text; used for consistency bonus/penalty in acceptance scoring.

## Query acceptance features

- **type_counts:** percent, currency, int, float, unknown (from `_extract_num_tokens` + kind).
- **role_evidence:** budget, cost, profit, demand, capacity, min_max, ratio, fixed_penalty, time, quantity (binary from keyword presence).
- **operator_evidence:** min_like, max_like, per_unit, total_like (from OPERATOR_* and phrase cues).
- **structural:** objective_like, constraint_like, resource_like.
- **n_numeric:** count of numeric tokens; **cue_words:** query tokens that are in CUE_WORDS.

## Schema acceptance profile

- **slot_names:** scalar parameter names from `problem_info["parameters"]` or `parameters`, filtered by scalar gold values.
- **type_expectations:** counts per percent/currency/int/float from `_expected_type(slot_name)`; **expects_percent** / **expects_currency** / **expects_count** booleans.
- **role_expectations:** booleans from slot name substrings (budget, demand, capacity, objective_coeff, ratio, fixed_penalty, time).
- **structural:** objective_like, bound_heavy, total_budget_like from schema text and slot names.
- **family,** **subgroup:** from `_schema_family` and `_schema_subgroup`.

## Scoring formula

- **Acceptance score (additive):**
  - Type: bonus if query has numeric types the schema expects; penalty if schema expects percent/currency but query has none.
  - Role: bonus for each schema-expected role present in query evidence.
  - Operator: bonus when min/max cues in query align with min/max slots.
  - Fillability: bonus if `n_numeric >= 0.5 * n_slots`; penalty if schema has slots but query has no numerics.
  - Family: bonus if schema family in query family hints; penalty if query has specific family and schema family does not match.
  - Subgroup: bonus if schema subgroup in query subgroup hints.
  - Weak fit: penalty if total acceptance ≤ 0 and schema has slots.
- **Final score:** `0.5 * normalized_retrieval_score + 0.5 * scaled_acceptance`, with acceptance shifted to a non-negative scale. If `use_hierarchy=True`, an extra penalty is applied when schema family is not in query family hints (and not generic_lp).

Weights are in `ACCEPTANCE_RERANK_WEIGHTS` (type_coverage_bonus, type_missing_penalty, role_coverage_bonus, operator_compat_bonus, fillability_bonus, family_consistency_bonus, family_mismatch_penalty, missing_critical_penalty, weak_fit_penalty).

## Method names used in outputs

- **tfidf_acceptance_rerank** — TF-IDF top-k + acceptance reranking (no hierarchy bonus/penalty).
- **tfidf_hierarchical_acceptance_rerank** — TF-IDF top-k + family/subgroup consistency + acceptance reranking.
- **bm25_acceptance_rerank** — BM25 top-k + acceptance reranking.
- **bm25_hierarchical_acceptance_rerank** — BM25 top-k + hierarchy + acceptance reranking.

Outputs are written to the same summary and per-query CSVs as other baselines, with these baseline names, so they do not overwrite existing results (new rows are upserted by variant+baseline).
