# NLP4LP Acceptance Rerank Examples

Examples are obtained by comparing `nlp4lp_downstream_per_query_orig_tfidf.csv` (flat) vs `nlp4lp_downstream_per_query_orig_tfidf_acceptance_rerank.csv` (reranked): same query_id, different predicted_doc_id and metrics.

**What to show per example (when debug logging is added):**
- **Query snippet**
- **Top-k retrieved schemas before reranking** (e.g. top-5 doc_ids and retrieval scores)
- **Acceptance score breakdown** (type/role/operator/fillability/family/subgroup) for top candidates
- **Family/subgroup hints** for the query
- **Final reranked schema** (top-1)
- **Where the method helped or still failed**

**Example 1 — Rerank improved downstream (same or better schema choice):**  
Queries where `tfidf` had schema_hit=0 but `tfidf_acceptance_rerank` had schema_hit=1 are rare (rerank lowered overall Schema R@1). Queries where rerank improved instantiation (e.g. higher param_coverage or type_match for the chosen schema) can be found by comparing per-query rows: filter where baseline=tfidf_acceptance_rerank and (param_coverage or type_match) > same query’s tfidf row.

**Example 2 — Rerank hurt retrieval:**  
Queries where tfidf had schema_hit=1 and tfidf_acceptance_rerank had schema_hit=0: the acceptance scorer demoted the gold schema in favor of another candidate that matched query evidence better.

**Example 3 — Rerank helped instantiation despite wrong schema:**  
Query where both chose the wrong schema (schema_hit=0) but rerank chose a schema with higher param_coverage/type_match for that query (more fillable wrong schema).

To generate 3–5 full examples with acceptance breakdown, add optional debug output in `make_rerank_rank_fn` (e.g. write a JSONL of query_id, top-5 candidates, acceptance debug, final choice) and run on a small subset of eval queries.
