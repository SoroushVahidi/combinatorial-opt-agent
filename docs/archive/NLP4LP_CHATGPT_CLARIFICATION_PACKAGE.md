# NLP4LP Clarification Package for ChatGPT (Experiments Section)

Concise, file-cited answers for manuscript-facing clarity. Use this when assessing and rewriting the experiments section. Do not rewrite the whole manuscript; focus on experiment logic and supporting file paths.

---

## 1. Random baseline clarification

**Why retrieval uses 1/331 = 0.0030 and downstream uses 2/331 = 0.0060**

- **Retrieval:** No random retrieval run exists. The retrieval pipeline (`training/run_baselines.py`) only runs BM25, TF-IDF, LSA (and optionally others) on the eval file. The paper-artifact script injects a **theoretical** lower bound: Recall@1 = 1/N with N = 331.
- **Downstream:** The downstream pipeline **does** run a random baseline: for each query it picks a schema via `random.Random(_md5_seed(query_id)).randrange(len(doc_ids))`. That run is deterministic (fixed seed per query). In the current run, exactly 2 queries get the correct schema by chance, so Schema R@1 = 2/331 ≈ 0.0060.

**Intentional design choice?**

Yes. Retrieval uses a **theoretical** random (1/N) so the table does not depend on running a random retriever. Downstream uses an **empirical** random run to show real end-to-end performance with random schema choice. Both are intentional and documented in the verification report.

**Exact files/scripts/notes that prove this**

- **Retrieval random = 1/331:**  
  `tools/make_nlp4lp_paper_artifacts.py` lines 905–930: comment “random is implicit (schema_R1=1/331 ≈ 0.006)”, then `n_queries = 331.0`, `r1_random = 1.0 / n_queries`, and row `("random", r1_random, r1_random, r1_random)` written to the retrieval main table.  
  Output: `results/paper/nlp4lp_retrieval_main_table.csv` (random row has 0.0030).

- **Downstream random = 2/331:**  
  `tools/nlp4lp_downstream_utility.py` lines 352–354: `elif mode == "random":` then `rng = random.Random(_md5_seed(qid))`, `pred_id = doc_ids[rng.randrange(len(doc_ids))]`.  
  `results/paper/nlp4lp_downstream_summary.csv`: row `orig,random,...,0.006042296072507553,...` (2/331 ≈ 0.00604).

- **Summary document:**  
  `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` §1.3 and §1.1: states retrieval random is theoretical 1/N, downstream random is deterministic single run with empirical 2/331.

**Safest manuscript wording to avoid reviewer confusion**

- **Retrieval table (or caption):** “Random is the theoretical lower bound 1/N (N = 331); no random retrieval run is performed.”
- **Downstream (methods or caption):** “Random baseline: one deterministic run in which the schema is chosen uniformly at random per query (fixed seed per query ID); reported Schema R@1 is the empirical fraction (2/331 on the original variant).”

---

## 2. Exact5 / Exact20 clarification

**Subset and denominator**

- **Subset:** Only **schema-hit** queries (queries where the retrieved/chosen schema equals the gold schema).
- **Further restriction:** Only schema-hit queries that have **at least one comparable error** (i.e. `comparable_errs` non-empty) contribute. For each such query, the code computes one value: the fraction of that query’s comparable errors that are ≤ 5% (Exact5) or ≤ 20% (Exact20).
- **Aggregate:** `exact5_on_hits` = mean of those per-query fractions; `exact20_on_hits` = same for 20%. So the **denominator is the number of schema-hit queries with non-empty comparable_errs**, not 331 and not the total number of schema-hit queries.

**What “comparable_errs” means in practice**

- For a given query, after filling scalar parameters from the query text, the code compares filled **numeric** values to gold **numeric** values only when: (1) the query is a schema hit, (2) the filled value is numeric, and (3) the gold value for that parameter is scalar. For each such parameter it appends the **relative error** `|pred - gold| / max(1, |gold|)` to `comparable_errs`.
- So “comparable” = we have both a filled numeric value and a gold scalar value for that slot on a schema-hit query. No numeric comparison (hence no comparable_err) for non-scalar gold or when the extractor doesn’t produce a numeric value.

**Exact file(s) / code location(s)**

- **Definition of comparable_errs and Exact5/Exact20:**  
  `tools/nlp4lp_downstream_utility.py`:  
  - Lines 409, 440–451: `comparable_errs = []`; only when `schema_hit and tok.value is not None and _is_scalar(gold_params.get(p))` we compute `err = _rel_err(float(tok.value), gold_val)` and append to `comparable_errs`; `_rel_err` at 227–228 is `abs(pred - gold) / max(1.0, abs(gold))`.  
  - Lines 457–465: if `schema_hit` and `comparable_errs` non-empty, per-query `exact5` = fraction of `comparable_errs` ≤ 0.05, `exact20` = fraction ≤ 0.20.  
  - Lines 496–499, 511–513: only when `exact5`/`exact20` are float do we append to `exact5_vals`/`exact20_vals`; aggregate is `sum(exact5_vals)/len(exact5_vals)` and same for exact20.

---

## 3. Oracle clarification

**What oracle means in this project**

- Oracle = **gold schema always**: for every query, the “retrieved” schema is set to the gold schema. There is **no retrieval step**; it is an upper bound on downstream utility when retrieval is perfect.

**Does oracle set pred_id = gold_id for every query and skip retrieval?**

- Yes.  
  `tools/nlp4lp_downstream_utility.py` lines 349–351: `if mode == "oracle": pred_id = gold_id`. No call to `rank_fn`; retrieval is skipped.

**Why oracle InstantiationReady can still be low**

- Oracle only fixes **schema choice**. The rest of the pipeline (number extraction from query text, type assignment, and filling slots) is unchanged. Many queries do not contain enough numeric information to fill 80% of expected scalar slots, or the extractor/heuristics do not achieve ≥ 0.8 type_match. So the fraction of queries with coverage ≥ 0.8 and type_match ≥ 0.8 (InstantiationReady) remains low (e.g. 0.0816 on orig).  
  Supporting value: `results/paper/nlp4lp_downstream_summary.csv` row `orig,oracle,...,0.08157099697885196` for `instantiation_ready`.

**Supporting files**

- Logic: `tools/nlp4lp_downstream_utility.py` lines 349–351.  
- Numbers: `results/paper/nlp4lp_downstream_summary.csv` (oracle rows).  
- Explanation: `docs/NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md` §1.2; `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` §1.4.

---

## 4. Hit/miss clarification

**Definitions**

- **Schema-hit:** A query for which the **retrieved (or chosen) schema equals the gold schema**: `predicted_doc_id == gold_doc_id`.
- **Schema-miss:** Any query for which the retrieved schema ≠ gold schema.

**Which metrics are averaged over subsets and what are the support counts (orig)**

- All six columns in the hit/miss table are **means over a subset of queries**:
  - **param_coverage_hits, type_match_hits, key_overlap_hits:** mean over **schema-hit** queries only. Denominator = number of hit queries for that baseline.
  - **param_coverage_miss, type_match_miss, key_overlap_miss:** mean over **schema-miss** queries only. Denominator = number of miss queries for that baseline.
- **Support counts (orig):** n_hits = schema_R1 × 331, n_miss = 331 − n_hits. From `results/paper/nlp4lp_downstream_summary.csv` (orig):  
  - LSA: 0.85498… → 283 hits, 48 miss.  
  - BM25: 0.88519… → 293 hits, 38 miss.  
  - TF-IDF: 0.90634… → 300 hits, 31 miss.

**Exact files**

- Definition and aggregation: `tools/nlp4lp_downstream_utility.py` lines 357–358 (schema_hit), 486–495 (cov_hits/cov_miss etc.), 513–519 (param_coverage_hits/miss, type_match_hits/miss, key_overlap_hits/miss).
- Table and values: `results/paper/nlp4lp_downstream_hitmiss_table_orig.csv`, `results/paper/nlp4lp_error_hitmiss_table.csv` (same numbers).  
- Source of aggregates: `results/paper/nlp4lp_downstream_summary.csv` (variant=orig, baselines lsa, bm25, tfidf).  
- Verification: `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` §1.5.

---

## 5. Per-type clarification

**Micro- vs macro-averaged**

- **Micro-averaged.** Per-type metrics are **not** “average of per-query type-level rates.” They are: total count (over all queries) for that type in the numerator, total count for that type in the denominator.

**Exact denominators**

- **Coverage (per type):** `type_filled_total[t] / type_expected_total[t]` — denominator = total number of **expected** scalar slots of that type across all 331 queries.
- **Type match (per type):** `type_correct_total[t] / type_filled_total[t]` — denominator = total number of **filled** scalar slots of that type across all 331 queries.

**Support counts per type (from outputs)**

- From `results/paper/nlp4lp_downstream_types_summary.csv` (variant=orig): each row has `n_expected`, `n_filled`; `param_coverage` = n_filled/n_expected, `type_match` = (implied by correct/filled). Example totals (same for all baselines for n_expected): currency 382, float 1209, integer 147, percent 109. Example n_filled for TF-IDF orig: currency 304, float 1009, integer 109, percent 95.

**Exact files**

- Computation: `tools/nlp4lp_downstream_utility.py` lines 526–531 (cov_t = n_fill_t / max(1, n_exp_t), tm_t = type_correct_total[t] / n_fill_t), with type_expected_total, type_filled_total, type_correct_total accumulated over all queries (lines 334–338, 384–438).
- Output: `results/paper/nlp4lp_downstream_types_summary.csv`.  
- Paper tables: `results/paper/nlp4lp_downstream_types_table_orig.csv`, `results/paper/nlp4lp_error_types_table.csv`.  
- Verification: `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` §1.6; `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` §4.

---

## 6. Reviewer-confusing values (reproducible but worth explaining)

| Value | Why it happens | Supporting evidence |
|-------|----------------|---------------------|
| **0.0030 vs 0.0060 (random)** | Retrieval table uses theoretical 1/331; downstream uses one deterministic random run (2 hits). Two different definitions by design. | §1 above; `make_nlp4lp_paper_artifacts.py` 927–930; `nlp4lp_downstream_utility.py` 352–354; `nlp4lp_downstream_summary.csv` orig random row. |
| **Oracle InstantiationReady &lt; 0.1 (e.g. 0.0816)** | Oracle only fixes schema; extraction and typing still limit how many queries reach coverage ≥ 0.8 and type_match ≥ 0.8. | `nlp4lp_downstream_summary.csv` orig,oracle instantiation_ready; consistency plan §1.2. |
| **Noisy type_match = 0** | Noisy variant uses `<num>` placeholders; the evaluator does not resolve these to numeric values, so type_match is 0. | `nlp4lp_downstream_summary.csv` noisy rows (type_match 0); reporting package / consistency plan. |
| **Float type_match ≈ 0.03** | Float parameters are hardest for the type heuristic; many filled slots are not correct type. Micro-averaged over many float slots. | `results/paper/nlp4lp_downstream_types_summary.csv` orig tfidf/oracle float type_match; reporting package §4. |
| **Random exact20_on_hits = 0.1250** | Only 2 schema-hit queries for random on orig; only some have non-empty comparable_errs. The reported value is the mean of the per-query “fraction of comparable errors ≤ 20%” over those few queries (e.g. 1/8). | `nlp4lp_downstream_summary.csv` orig,random exact20_on_hits; `nlp4lp_downstream_utility.py` 457–465, 511–513. |
| **Random integer type_match = 1.0000** | Random has only 1 filled integer slot on orig (n_filled=1) and that one slot has correct type, so type_correct/type_filled = 1/1. | `results/paper/nlp4lp_downstream_types_summary.csv` row orig,random,integer (n_filled=1, type_match=1.0). |

---

## 7. Manuscript-facing note (8–12 bullets for ChatGPT)

Use these when rewriting the experiments section. Focus: what to clarify in prose, what in captions, key numbers, and what not to over-claim.

- **State test set size once:** “We evaluate on 331 queries per variant (orig, noisy, short); the same query IDs are used across variants, with different query text.”

- **Clarify random in two places:** In the retrieval table caption (or methods): “Random is the theoretical lower bound 1/N (N = 331); no random retrieval run is performed.” In the downstream section: “The random baseline is one deterministic run (schema chosen uniformly at random per query, fixed seed per query ID); on the original variant, Schema R@1 is 2/331.”

- **Define oracle once in the downstream section:** “Oracle uses the gold schema for every query (no retrieval); all other pipeline steps (extraction, typing, evaluation) are unchanged.”

- **Define InstantiationReady in methods or caption:** “InstantiationReady is the fraction of queries with param_coverage ≥ 0.8 and type_match ≥ 0.8 (denominator 331).”

- **In the downstream main table caption (or footnote):** “Exact5 and Exact20 are computed only over schema-hit queries that have at least one comparable error (filled numeric value vs gold scalar); the reported value is the mean of per-query fractions, so the denominator is not 331.”

- **In the hit/miss table caption:** “Schema-hit = queries where the retrieved schema equals the gold schema; schema-miss = all other queries. Each cell is the mean over that subset only (hit count and miss count vary by baseline; e.g. TF-IDF 300 hits, 31 miss on orig).”

- **In the per-type table caption:** “Per-type metrics are micro-averaged (pooled counts over all queries): coverage = total filled slots of that type / total expected slots of that type; type match = total correct-type filled / total filled for that type.”

- **Key numbers to keep consistent:** Schema R@1 orig: TF-IDF 0.906, BM25 0.885, LSA 0.855; random retrieval 0.003, downstream random 0.006; oracle instantiation_ready 0.082; N = 331 everywhere unless stated otherwise (e.g. Exact5/Exact20).

- **Do not over-claim:** Do not say oracle gives “perfect” downstream performance; say it gives perfect retrieval only, and instantiation_ready remains low. Do not say Exact5/Exact20 are “over all queries”; say they are over schema-hit queries with comparable errors.

- **Rounding:** Either state “Reported metrics are rounded to 4 decimal places” in methods or one caption, or keep table (4 dp) and prose (e.g. 3 dp) consistent and note if needed.

- **Existing docs to quote:** `docs/NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md` has ready-to-adapt captions and denominator table. `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` has the full verification table and special checks (random, oracle, hit/miss, per-type). `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` has the exact numbers and denominator definitions for all metric families.

---

**Summary:** This package answers each clarification point with a short explanation and **exact file paths** (code and result files). The manuscript-facing bullets at the end are written so ChatGPT can use them directly when rewriting the experiments section and captions, without rewriting the rest of the manuscript or changing code.
