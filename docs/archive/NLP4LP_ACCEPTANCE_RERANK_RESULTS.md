# NLP4LP Acceptance Rerank Results

## Retrieval comparison vs flat retrieval

- **Schema R@1 (orig):** tfidf = 0.9063, tfidf_acceptance_rerank = 0.8761. Reranking slightly **lowered** Schema R@1 (−0.030).
- **Source:** `results/paper/nlp4lp_downstream_summary.csv` (schema_R1 column per baseline).

## Downstream comparison vs current pipeline

**Orig, typed assignment:**

| Baseline                    | schema_R1 | param_coverage | type_match | key_overlap | instantiation_ready |
|----------------------------|-----------|----------------|------------|-------------|---------------------|
| tfidf                      | 0.9063    | 0.8222         | 0.2260     | 0.9188      | 0.0755              |
| tfidf_acceptance_rerank    | 0.8761    | 0.7974         | 0.2275     | 0.8856      | **0.0816**          |
| tfidf_hierarchical_acceptance_rerank | 0.8459 | 0.7771    | 0.2303     | 0.8592      | **0.0846**          |

- **param_coverage:** slightly lower with rerank (0.822 → 0.797).
- **type_match:** slightly higher with rerank (0.226 → 0.228).
- **instantiation_ready:** **higher** with rerank (0.0755 → 0.0816).

## Did reranking improve Schema R@1?

**No.** Schema R@1 decreased from 0.906 to 0.876 on orig. The acceptance scorer sometimes promotes a schema that fits the query’s “evidence” better but is not the gold schema.

## Did reranking improve InstantiationReady?

**Yes.** Instantiation_ready increased from 0.0755 to 0.0816 on orig. Among cases where the chosen schema differs from flat TF-IDF, the reranker occasionally picks a schema that is more fillable (better type/coverage on the chosen schema), even when it is not the gold schema.

## Short interpretation

- Acceptance reranking trades off a small amount of **retrieval accuracy** (Schema R@1) for a small gain in **downstream utility** (instantiation_ready) on orig. The method is interpretable and deterministic; tuning weights or adding more query/schema features could improve both.
- Family/subgroup and type/role coverage are the main levers; on this dataset flat TF-IDF is already strong, so gains are modest.
- **Hierarchical** variant (family-mismatch penalty) further lowers Schema R@1 (0.846) but gives the highest instantiation_ready (0.0846), suggesting the hierarchy pushes toward more fillable schemas at the cost of retrieval accuracy.

## Commands to reproduce

```bash
# Flat TF-IDF (existing)
python -m tools.nlp4lp_downstream_utility --variant orig --baseline tfidf --assignment-mode typed

# Acceptance rerank (TF-IDF top-10, rerank by acceptance)
python -m tools.nlp4lp_downstream_utility --variant orig --baseline tfidf_acceptance_rerank --assignment-mode typed --acceptance-k 10

# Hierarchical + acceptance rerank
python -m tools.nlp4lp_downstream_utility --variant orig --baseline tfidf_hierarchical_acceptance_rerank --assignment-mode typed --acceptance-k 10
```

Then inspect `results/paper/nlp4lp_downstream_summary.csv` for rows with baseline `tfidf`, `tfidf_acceptance_rerank`, `tfidf_hierarchical_acceptance_rerank`.
