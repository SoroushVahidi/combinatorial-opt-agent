# Learning Experiment: Deterministic Baseline Artifacts

Evidence-based audit of where the current 7-method (safe-mode) evaluation outputs live and which metrics are available for comparison with learned downstream runs.

## 1. Exact result files found

**Location:** `results/paper/`

| Method | JSON (aggregate metrics) | Per-query CSV |
|--------|---------------------------|----------------|
| tfidf_acceptance_rerank | `nlp4lp_downstream_orig_tfidf_acceptance_rerank.json` | `nlp4lp_downstream_per_query_orig_tfidf_acceptance_rerank.csv` |
| tfidf_hierarchical_acceptance_rerank | `nlp4lp_downstream_orig_tfidf_hierarchical_acceptance_rerank.json` | `nlp4lp_downstream_per_query_orig_tfidf_hierarchical_acceptance_rerank.csv` |
| tfidf_optimization_role_repair | `nlp4lp_downstream_orig_tfidf_optimization_role_repair.json` | `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_repair.csv` |
| tfidf_optimization_role_relation_repair | `nlp4lp_downstream_orig_tfidf_optimization_role_relation_repair.json` | `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_relation_repair.csv` |
| tfidf_optimization_role_anchor_linking | `nlp4lp_downstream_orig_tfidf_optimization_role_anchor_linking.json` | `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_anchor_linking.csv` |
| tfidf_optimization_role_bottomup_beam_repair | `nlp4lp_downstream_orig_tfidf_optimization_role_bottomup_beam_repair.json` | `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_bottomup_beam_repair.csv` |
| tfidf_optimization_role_entity_semantic_beam_repair | `nlp4lp_downstream_orig_tfidf_optimization_role_entity_semantic_beam_repair.json` | `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_entity_semantic_beam_repair.csv` |

**Focused summary (all 7 in one CSV):** `nlp4lp_focused_eval_summary.csv` — one row per (variant, baseline) with columns: variant, baseline, schema_R1, param_coverage, type_match, key_overlap, exact5_on_hits, exact20_on_hits, instantiation_ready, n.

**Full downstream summary (all runs ever):** `nlp4lp_downstream_summary.csv` — upserted by each run of `nlp4lp_downstream_utility.py`.

## 2. Metrics available in deterministic JSON files

Each `nlp4lp_downstream_orig_<baseline>.json` has:

- **config:** variant, baseline, k, random_control
- **aggregate:** variant, baseline, schema_R1, param_coverage, type_match, exact5_on_hits, exact20_on_hits, param_coverage_hits, param_coverage_miss, type_match_hits, type_match_miss, key_overlap, key_overlap_hits, key_overlap_miss, **instantiation_ready**, n

**Interpretation:**

- **schema_R1:** Retrieval recall@1 (fraction of queries where predicted doc = gold doc).
- **param_coverage:** Among expected scalar params (on predicted schema), fraction that got a filled value.
- **type_match:** Fraction of filled params where value type matches expected (percent/int/currency/float).
- **exact5_on_hits:** Among schema-hit queries, fraction of params within 5% relative error (or exact).
- **exact20_on_hits:** Among schema-hit queries, fraction of params within 20% relative error (or exact).
- **instantiation_ready:** Fraction of queries with param_coverage ≥ 0.8 and type_match ≥ 0.8 (readiness for instantiation).

These metrics are from the **full pipeline**: retrieval (e.g. TF-IDF) → schema hit → downstream assignment (e.g. optimization_role_repair) over the **query-level** eval set (e.g. 331 queries, variant orig).

## 3. Which baseline numbers are trusted for comparison

- **Primary deterministic baseline for downstream comparison:** **tfidf_optimization_role_repair**.  
  Numbers in `results/paper/nlp4lp_downstream_orig_tfidf_optimization_role_repair.json` and in `nlp4lp_focused_eval_summary.csv` are trusted when the file exists and was produced by `tools/run_nlp4lp_focused_eval.py` (or equivalent `nlp4lp_downstream_utility.py` run).

- **Secondary:** **tfidf_optimization_role_relation_repair**.  
  Same provenance; file: `nlp4lp_downstream_orig_tfidf_optimization_role_relation_repair.json`.

- **Other 5 methods (acceptance_rerank, hierarchical_acceptance_rerank, anchor_linking, bottomup_beam_repair, entity_semantic_beam_repair):** Artifacts exist; we use them for context but do **not** treat them as the main comparison target. Best deterministic downstream (by exact20_on_hits and instantiation_ready) is optimization_role_repair per existing situation reports.

## 4. What is missing or ambiguous

- **Evaluation setup difference:** Deterministic baselines are evaluated on the **full pipeline** (retrieval + downstream) over **query-level** NLP4LP eval (e.g. orig, 331 queries). Learned runs are evaluated on **ranker data** (slot-level pairwise, then slot/exact-instance metrics). So:
  - **Learned metrics:** pairwise_accuracy, slot_selection_accuracy, exact_slot_fill_accuracy, type_match_after_decoding (on ranker test set).
  - **Deterministic metrics:** schema_R1, param_coverage, exact20_on_hits, instantiation_ready (on full pipeline, query-level).
- Direct numerical comparison (e.g. learned “exact_slot_fill_accuracy” vs deterministic “exact20_on_hits”) is **not** apples-to-apples; we compare in separate tables and state the difference in the report.
- If `results/paper/` JSONs are missing (e.g. fresh clone), run:
  `python tools/run_nlp4lp_focused_eval.py --variant orig --safe`
  or with `--experimental` for all 7 methods.

## 5. Which deterministic baseline(s) we compare against in Stage 3

- **Primary:** **tfidf_optimization_role_repair** — best current deterministic downstream (exact20_on_hits, instantiation_ready).
- **Secondary:** **tfidf_optimization_role_relation_repair** — relation-aware variant; we include it in the comparison table when the artifact exists.
- We **do not** replace these with guessed numbers; we read from the actual JSON/CSV files and report “missing” if not present.
