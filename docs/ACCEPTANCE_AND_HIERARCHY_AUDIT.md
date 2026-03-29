# Audit: “Problem Accepts Statement” and Hierarchical Grouping — Current Repo State

**Date:** 2025-03-08  
**Scope:** Verify whether the two requested ideas (Idea 1: problem accepts statement / acceptance scoring; Idea 2: hierarchical grouping) are implemented, active, and present in current results. Evidence from repo files only.

---

## 1. Verdict on Idea 1

**Idea 1 (“problem accepts statement”) is fully implemented and active.** The repo implements candidate-side acceptance scoring: for each candidate schema, it builds a profile of what evidence the schema expects (types, roles, structure), extracts query-side evidence (types, roles, operators, fillability), and scores how well the schema “accepts” the query. This is used to **rerank** retrieval candidates (separate from plain lexical retrieval). It is exposed as baselines `tfidf_acceptance_rerank`, `tfidf_hierarchical_acceptance_rerank`, `bm25_acceptance_rerank`, `bm25_hierarchical_acceptance_rerank` and is reachable from the CLI. Current result files contain rows for the TF-IDF acceptance variants (orig, and per-query JSON/CSV). It is **retrieval-side only** (reranking); downstream assignment modes run afterward unchanged.

---

## 2. Evidence for Idea 1 in code

**Single file:** `tools/nlp4lp_downstream_utility.py`. All acceptance logic lives here (no separate retrieval module).

| Component | Location | What it does |
|-----------|----------|---------------|
| **Weights** | `ACCEPTANCE_RERANK_WEIGHTS` (lines 1753–1762) | type_coverage_bonus, type_missing_penalty, role_coverage_bonus, operator_compat_bonus, **fillability_bonus**, family_consistency_bonus, family_mismatch_penalty, missing_critical_penalty, weak_fit_penalty |
| **Query-side features** | `_extract_query_acceptance_features(query, variant)` (1850–1902) | type_counts (percent/currency/int/float), role_evidence (budget, cost, profit, demand, capacity, min_max, ratio, …), operator_evidence (min_like, max_like, per_unit, total_like), structural (objective_like, constraint_like, resource_like), n_numeric, cue_words |
| **Schema profile** | `_get_expected_scalar_for_schema()` (1903–1915), `_build_schema_acceptance_profile()` (1917–1975) | For each candidate schema: slot_names, type_expectations, expects_percent/currency/count, role_expectations (budget, demand, capacity, objective_coeff, ratio, …), structural (objective_like, bound_heavy, total_budget_like), family, subgroup |
| **Acceptance score** | `_acceptance_score(query_features, schema_profile, query_family_hints, query_subgroup_hints)` (1977–2002) | Additive score: type coverage (bonus/penalty), **role coverage** (bonus), **operator compatibility** (min/max), **fillability** (bonus if n_numeric ≥ 0.5×n_slots; penalty if slots but no numerics), family/subgroup consistency, weak_fit penalty. Returns (score, debug). |
| **Reranker** | `make_rerank_rank_fn(base_rank_fn, gold_by_id, catalog, k_retrieval=10, use_hierarchy=False, variant)` (2005–2094) | Gets top-`k_retrieval` from base retriever; for each candidate builds schema profile, gets query features, computes acceptance score; combines `0.5 * norm_retrieval + 0.5 * scaled_acceptance`; optional hierarchy penalty; sorts and returns top_k. |

**Answers to the five Idea-1 questions:**

1. **Logic that computes candidate-side features and scores how well schema accepts the query?** Yes. `_build_schema_acceptance_profile` (schema) and `_extract_query_acceptance_features` (query) feed `_acceptance_score`, which implements type coverage, role coverage, operator compatibility, fillability, and family/subgroup consistency.
2. **Separate from plain lexical retrieval?** Yes. Base retriever (tfidf/bm25) returns top-k by lexical similarity; `make_rerank_rank_fn` then scores each candidate by acceptance and reranks; final rank is by combined score.
3. **Used only in retrieval reranking, only in downstream assignment, or both?** **Retrieval reranking only.** Downstream assignment (typed, constrained, semantic_ir_repair, optimization_role_repair) runs after the (possibly reranked) schema is chosen; assignment code does not call acceptance scoring.
4. **Exact method names / baselines that expose it:** `tfidf_acceptance_rerank`, `tfidf_hierarchical_acceptance_rerank`, `bm25_acceptance_rerank`, `bm25_hierarchical_acceptance_rerank` (see `main()` lines 2542–2567).
5. **Query-side features:** type_counts, role_evidence, operator_evidence, structural, n_numeric, cue_words. **Schema-side features:** slot_names, type_expectations, role_expectations, structural, family, subgroup (see tables above).

---

## 3. Verdict on Idea 2

**Idea 2 (hierarchical grouping) is partially implemented and active.** The repo has **two levels only:** family and subgroup (no sub-subgroup). Families and subgroups are **rule-based labels** (keywords and slot-name patterns), not learned. The hierarchy is used as **soft reranking**: family/subgroup consistency bonuses and family-mismatch penalty in the acceptance score, and when `use_hierarchy=True` an extra −0.2 penalty if the schema family is not in the query’s family hints. There is **no** pipeline that “first infers main group, then subgroup, then sub-subgroup, then selects”; query hints are computed once and used inside the same acceptance score. So: family + subgroup exist and are active; sub-subgroup and hierarchical inference (coarse-to-fine selection) are not present.

---

## 4. Evidence for Idea 2 in code

**Same file:** `tools/nlp4lp_downstream_utility.py`.

| Component | Location | What it does |
|-----------|----------|---------------|
| **Family keywords** | `FAMILY_KEYWORDS` (1765–1776) | allocation, production, transportation, scheduling, packing, covering, network, inventory, ratio, **generic_lp** (keyword lists per family). |
| **Schema family** | `_schema_family(schema_id, schema_text, slot_names)` (1780–1791) | Combines id + description + slot names; assigns one family by keyword count (default generic_lp). |
| **Schema subgroup** | `_schema_subgroup(schema_id, slot_names)` (1795–1812) | From slot name patterns only: total_budget_per_unit_profit, capacity_demand, min_max_bounds, ratio_fraction, fixed_cost_penalty, time_resource, item_count, or "". |
| **Query family hints** | `_query_family_hints(query)` (1815–1825) | Returns set of family names present in query (by FAMILY_KEYWORDS); if none, returns {generic_lp}. |
| **Query subgroup hints** | `_query_subgroup_hints(query)` (1829–1846) | Returns set of subgroup tags present in query (phrase/keyword rules). |
| **Use in scoring** | `_acceptance_score` (1988–1994) | family_consistency_bonus if schema family in query family hints; family_mismatch_penalty if query has specific family and schema family does not match; subgroup bonus if schema subgroup in query subgroup hints. |
| **Hierarchy flag** | `make_rerank_rank_fn(..., use_hierarchy=False)` (2009–2094) | When `use_hierarchy=True`: extra penalty `final -= 0.2` if schema family not in query family hints and not generic_lp (line 2091–2092). |

**Answers to the six Idea-2 questions:**

1. **Main group / family level?** Yes. `_schema_family` and `_query_family_hints`; FAMILY_KEYWORDS define allocation, production, transportation, scheduling, packing, covering, network, inventory, ratio, generic_lp.
2. **Subgroup level?** Yes. `_schema_subgroup` and `_query_subgroup_hints`; subgroups are total_budget_per_unit_profit, capacity_demand, min_max_bounds, ratio_fraction, fixed_cost_penalty, time_resource, item_count.
3. **Sub-subgroup level?** No. Only family and subgroup exist in code and docs.
4. **Query first classified into family then subgroup before final schema choice?** No. Query hints (family + subgroup) are computed once and used inside the same acceptance score; there is no staged “infer family → then subgroup → then sub-subgroup → then pick schema.”
5. **Hierarchy used as hard filtering, soft reranking, penalties/bonuses, or just documentation?** **Soft reranking:** bonuses/penalties in `_acceptance_score`; when use_hierarchy=True, an extra penalty for family mismatch.
6. **Levels explicit labels, inferred heuristically, or partial?** **Explicit rule-based labels:** family from keyword counts over schema id/description/slots; subgroup from slot-name substring rules; query hints from keyword/phrase presence in query text.

---

## 5. Method names / CLI exposure

- **CLI:** `--baseline` accepts (among others): `tfidf_acceptance_rerank`, `tfidf_hierarchical_acceptance_rerank`, `bm25_acceptance_rerank`, `bm25_hierarchical_acceptance_rerank` (see help at line 2511 and branches at 2542–2567).
- **`--acceptance-k`:** type=int, default=10; top-k from base retriever before acceptance reranking (line 2514).
- **Effective baseline name:** The string passed to `run_setting` is exactly the baseline name (e.g. `tfidf_acceptance_rerank`); no suffix is added for assignment mode unless you also change `--assignment-mode` (assignment mode is independent).
- **retrieval/:** No acceptance or hierarchy code; everything is in `tools/nlp4lp_downstream_utility.py`.

---

## 6. Docs found

| Doc | Path | Content (summary) |
|-----|------|-------------------|
| **Acceptance rerank implementation** | `docs/NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md` | Where reranking plugs in (main(), make_rerank_rank_fn); files/functions changed; family/subgroup design; query acceptance features; schema acceptance profile; scoring formula; method names (tfidf_acceptance_rerank, tfidf_hierarchical_acceptance_rerank, bm25_*). Matches code. |
| **Acceptance rerank examples** | `docs/NLP4LP_ACCEPTANCE_RERANK_EXAMPLES.md` | How to compare flat vs reranked per-query; acceptance breakdown; when rerank helps/hurts. |
| **Acceptance rerank results** | `docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md` | Tables for tfidf vs tfidf_acceptance_rerank vs tfidf_hierarchical_acceptance_rerank; Schema R@1 lower with rerank; instantiation_ready higher with hierarchical. |
| **Other mentions** | JOURNAL_READINESS_AUDIT.md, RESULTS_VS_CODE_VERIFICATION.md, OPTIMIZATION_ROLE_METHOD_AUDIT.md, OPTIMIZATION_ROLE_METRICS_COMPARISON.md, CURRENT_STATE_AUDIT.md, REPO_CLEANUP_PLAN.md | Acceptance_rerank and hierarchical_acceptance_rerank mentioned as baselines; doc-only, no new implementation detail. |
| **GAMSPy** | GAMSPY_LOCAL_EXAMPLES_COLLECTION.md, GAMSPY_LOCAL_EXAMPLES_NEXT_STEPS.md | “Schema acceptance”, “family classification” in the context of GAMSPy examples and future use; not the NLP4LP acceptance reranker. |

The three NLP4LP acceptance-rerank docs describe the same design as the code: problem-accepts-statement style scoring (type/role/operator/fillability/family/subgroup) and two-level hierarchy used in reranking.

---

## 7. Results/artifacts found

**Downstream summary:** `results/paper/nlp4lp_downstream_summary.csv` contains rows (orig):

- `orig,tfidf_acceptance_rerank` — schema_R1 0.8761, param_coverage 0.7974, type_match 0.2275, key_overlap 0.8856, exact5_on_hits 0.1822, exact20_on_hits 0.2058, instantiation_ready 0.0816
- `orig,tfidf_hierarchical_acceptance_rerank` — schema_R1 0.8459, param_coverage 0.7771, type_match 0.2303, key_overlap 0.8592, exact5 0.1705, exact20 0.1965, instantiation_ready 0.0846

**Per-query / JSON (orig):**

- `results/paper/nlp4lp_downstream_orig_tfidf_acceptance_rerank.json`
- `results/paper/nlp4lp_downstream_per_query_orig_tfidf_acceptance_rerank.csv`
- `results/paper/nlp4lp_downstream_orig_tfidf_hierarchical_acceptance_rerank.json`
- `results/paper/nlp4lp_downstream_per_query_orig_tfidf_hierarchical_acceptance_rerank.csv`

**BM25 acceptance variants:** No rows in the downstream summary for `bm25_acceptance_rerank` or `bm25_hierarchical_acceptance_rerank`; CLI supports them but current result files only show TF-IDF acceptance runs.

**Paper-facing tables:** The artifact script `make_nlp4lp_paper_artifacts.py` uses a fixed baseline list for the **final** downstream table: `order = ["random", "lsa", "bm25", "tfidf", "oracle", "tfidf_untyped", "oracle_untyped"]` (line 295). So `tfidf_acceptance_rerank` and `tfidf_hierarchical_acceptance_rerank` do **not** appear in `nlp4lp_downstream_final_table_orig.csv` or the corresponding LaTeX; they are in the full summary and per-query outputs only.

---

## 8. What is fully implemented vs partial

| Idea | Fully implemented | Partial / missing |
|------|--------------------|--------------------|
| **Idea 1** | Candidate-side acceptance scoring (type, role, operator, fillability); schema profile + query features; reranking by combined retrieval + acceptance; fillability_bonus, type_coverage, role_coverage; exposed as *_acceptance_rerank and *_hierarchical_acceptance_rerank; CLI and results for TF-IDF. | BM25 acceptance runs not in current summary; paper final/section tables omit acceptance baselines. |
| **Idea 2** | Family level (FAMILY_KEYWORDS, _schema_family, _query_family_hints); subgroup level (_schema_subgroup, _query_subgroup_hints); family_consistency_bonus, family_mismatch_penalty, subgroup bonus; optional extra hierarchy penalty when use_hierarchy=True. | No sub-subgroup level. No “infer main group → subgroup → sub-subgroup → select” pipeline; hierarchy is used only as soft bonuses/penalties inside acceptance scoring. |

---

## 9. Whether these ideas are still active in the current repo

**Yes.** Both ideas are in the active code path:

- **Idea 1:** Choosing baseline `tfidf_acceptance_rerank` or `tfidf_hierarchical_acceptance_rerank` (or bm25 variants) builds `rank_fn` via `make_rerank_rank_fn`, which uses `_extract_query_acceptance_features`, `_build_schema_acceptance_profile`, and `_acceptance_score`. No dead code or feature flags that disable this.
- **Idea 2:** Family and subgroup are computed for every candidate in the reranker; acceptance score includes family/subgroup terms; `use_hierarchy=True` for `*_hierarchical_acceptance_rerank` adds the extra family-mismatch penalty. No removal or bypass found.

---

## 10. Appendix: file-by-file evidence list

| File | What was inspected |
|------|--------------------|
| **tools/nlp4lp_downstream_utility.py** | Lines 1751–2094: ACCEPTANCE_RERANK_WEIGHTS, FAMILY_KEYWORDS, _schema_family, _schema_subgroup, _query_family_hints, _query_subgroup_hints, _extract_query_acceptance_features, _get_expected_scalar_for_schema, _build_schema_acceptance_profile, _acceptance_score, make_rerank_rank_fn. Lines 2511, 2514, 2542–2567: CLI --baseline and --acceptance-k, construction of rank_fn for acceptance_rerank and hierarchical_acceptance_rerank. |
| **docs/NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md** | Full file: entry point, functions, family/subgroup design, query and schema features, scoring, method names. |
| **docs/NLP4LP_ACCEPTANCE_RERANK_EXAMPLES.md** | Comparison of flat vs reranked; acceptance breakdown. |
| **docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md** | Tables and interpretation. |
| **results/paper/nlp4lp_downstream_summary.csv** | Rows for orig tfidf_acceptance_rerank, orig tfidf_hierarchical_acceptance_rerank. |
| **results/paper/** | Presence of nlp4lp_downstream_orig_tfidf_acceptance_rerank.json, nlp4lp_downstream_per_query_orig_tfidf_acceptance_rerank.csv, and same for hierarchical_acceptance_rerank. |
| **tools/make_nlp4lp_paper_artifacts.py** | Lines 295, 308–317, 523: fixed baseline list for final_table_orig and section table; acceptance baselines not in list. |
| **retrieval/** | Grep: no matches for acceptance, rerank, hierarchical, family, subgroup — confirmation that acceptance/hierarchy live only in tools/nlp4lp_downstream_utility.py. |
| **tools/collect_gams_examples_manifest.py** | likely_family for GAMSPy examples (different pipeline); schema_acceptance/family_classification mentioned as usage strings for manifest, not the NLP4LP reranker. |

---

## Summary answers (Phase 5 style)

1. **Does current evidence show that Idea 1 was implemented?** Yes. Schema acceptance profile, query acceptance features, fillability/type/role/operator scoring, and reranking by “how well the problem accepts the statement” are all in `tools/nlp4lp_downstream_utility.py` and used by `make_rerank_rank_fn`.
2. **Does current evidence show that Idea 2 was implemented?** Partially. Family and subgroup are implemented and used in scoring; sub-subgroup and coarse-to-fine hierarchical selection are not.
3. **If implemented, complete or partial?** Idea 1: complete for retrieval reranking (no downstream assignment use). Idea 2: partial — two levels, soft scoring only.
4. **Still active now?** Yes. Both are on the active path for *_acceptance_rerank and *_hierarchical_acceptance_rerank.
5. **Present in paper-facing tables or only full summaries?** Only in full summary and per-query files; paper final/section tables use a fixed baseline list that omits acceptance_rerank and hierarchical_acceptance_rerank.
6. **Main evidence files/functions:** `tools/nlp4lp_downstream_utility.py`: `_extract_query_acceptance_features`, `_build_schema_acceptance_profile`, `_acceptance_score`, `make_rerank_rank_fn`, `_schema_family`, `_schema_subgroup`, `_query_family_hints`, `_query_subgroup_hints`; docs: `NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md`; results: `nlp4lp_downstream_summary.csv` and per-query JSON/CSV for tfidf_acceptance_rerank and tfidf_hierarchical_acceptance_rerank.
