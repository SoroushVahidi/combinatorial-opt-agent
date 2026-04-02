# NLP4LP Manuscript-Ready Reporting Package

**Test set:** 331 queries per variant. Same query IDs across variants; query text differs (orig, noisy, short).  
**Eval files:** `data/processed/nlp4lp_eval_orig.jsonl`, `nlp4lp_eval_noisy.jsonl`, `nlp4lp_eval_short.jsonl` (331 lines each).  
**Export rounding:** 4 decimal places (`f"{float(v):.4f}"`) for all downstream and section tables; retrieval table same.  
**Source CSVs:** `results/nlp4lp_retrieval_summary.csv`, `results/paper/nlp4lp_downstream_summary.csv`, `results/paper/nlp4lp_downstream_types_summary.csv`.  
**Table-generation script:** `tools/make_nlp4lp_paper_artifacts.py`. **Metric computation:** `tools/nlp4lp_downstream_utility.run_one()`, `training/run_baselines.py`, `training/metrics.compute_metrics()`.

---

## 1. Main retrieval results (Schema R@1 = Recall@1)

**Denominator:** Number of eval queries = **331** (all metrics).  
**Definition:** Recall@1 = P@1 = (number of queries where the gold schema is in rank 1) / 331.  
**Random:** Not run in retrieval pipeline; theoretical = 1/331 ≈ **0.0030**.  
**Oracle:** Not a retrieval baseline; only in downstream.

| Variant | BM25    | TF-IDF  | LSA     | random (theoretical) |
|---------|--------|---------|---------|-----------------------|
| **orig**  | 0.8852 | 0.9063  | 0.8550  | 0.0030                |
| **noisy** | 0.8943 | 0.9033  | 0.8912  | 0.0030                |
| **short** | 0.7734 | 0.7855  | 0.7704  | 0.0030                |

**Source:** `results/nlp4lp_retrieval_summary.csv` (Recall@1); random from 1/331 in `make_nlp4lp_paper_artifacts.py` §4i. Paper table: `results/paper/nlp4lp_retrieval_main_table.csv`.

---

## 2. Main downstream results (by variant)

**Denominators:**
- **Schema_R@1:** 331 (all queries).
- **InstantiationReady:** 331 (fraction of queries with param_coverage ≥ 0.8 and type_match ≥ 0.8).
- **Coverage (param_coverage):** Per query = n_filled / n_expected_scalar; aggregate = mean over 331.
- **TypeMatch:** Per query = type_matches / n_filled; aggregate = mean over 331.
- **KeyOverlap:** Per query = |pred_scalar ∩ gold_scalar| / |gold_scalar|; aggregate = mean over 331.
- **Exact5 / Exact20:** Mean over **schema-hit queries that have comparable_errs** (denominator = count of those queries, not 331).

**Rounding:** 4 decimal places.

### 2.1 Orig (331 queries)

| Baseline | Schema_R@1 | InstantiationReady | Coverage | TypeMatch | KeyOverlap | Exact5   | Exact20  |
|----------|------------|--------------------|----------|----------|------------|----------|----------|
| random   | 0.0060     | 0.0060             | 0.0101   | 0.0060   | 0.0082     | 0.0000   | 0.1250   |
| lsa      | 0.8550     | 0.0604             | 0.7976   | 0.2063   | 0.8657     | 0.1727   | 0.1965   |
| bm25     | 0.8852     | 0.0755             | 0.8133   | 0.2251   | 0.8936     | 0.1936   | 0.2175   |
| tfidf    | 0.9063     | 0.0725             | 0.8222   | 0.2267   | 0.9188     | 0.1876   | 0.2140   |
| oracle   | 1.0000     | 0.0816             | 0.8695   | 0.2475   | 0.9953     | 0.1636   | 0.1871   |

### 2.2 Noisy (331 queries)

| Baseline | Schema_R@1 | InstantiationReady | Coverage | TypeMatch | KeyOverlap | Exact5 | Exact20 |
|----------|------------|--------------------|----------|----------|------------|--------|--------|
| random   | 0.0060     | 0.0000             | 0.0101   | 0.0      | 0.0082     | —      | —      |
| lsa      | 0.8912     | 0.0000             | 0.7080   | 0.0      | 0.9087     | —      | —      |
| bm25     | 0.8943     | 0.0000             | 0.7073   | 0.0      | 0.9036     | —      | —      |
| tfidf    | 0.9033     | 0.0000             | 0.7100   | 0.0      | 0.9201     | —      | —      |
| oracle   | 1.0000     | 0.0000             | 0.7536   | 0.0      | 0.9953     | —      | —      |

(Exact5/Exact20 on hits not populated for noisy in summary; type_match 0 due to `<num>` placeholders.)

### 2.3 Short (331 queries)

| Baseline | Schema_R@1 | InstantiationReady | Coverage | TypeMatch | KeyOverlap | Exact5   | Exact20  |
|----------|------------|--------------------|----------|----------|------------|----------|----------|
| random   | 0.0060     | 0.0000             | 0.0000   | 0.0      | 0.0082     | —        | —        |
| lsa      | 0.7704     | 0.0030             | 0.0297   | 0.0196   | 0.7753     | 0.2121   | 0.2121   |
| bm25     | 0.7734     | 0.0091             | 0.0354   | 0.0317   | 0.7878     | 0.2059   | 0.2353   |
| tfidf    | 0.7855     | 0.0060             | 0.0333   | 0.0272   | 0.7884     | 0.0588   | 0.0588   |
| oracle   | 1.0000     | 0.0030             | 0.0373   | 0.0287   | 0.9953     | 0.1190   | 0.1190   |

**Source:** `results/paper/nlp4lp_downstream_summary.csv`. Section table: `results/paper/nlp4lp_downstream_section_table.csv` (orig only). Variant table: `results/paper/nlp4lp_downstream_variant_table.csv`.

---

## 3. Hit/miss breakdown (orig; LSA, BM25, TF-IDF only)

**Subset definitions:** Schema-hit = queries where predicted_doc_id == gold_doc_id. Schema-miss = all other queries.  
**Denominators:** For each baseline, metrics on hits = mean over **hit queries**; on misses = mean over **miss queries**.  
**Support counts (derived):** n_hits = schema_R1 × 331, n_miss = 331 − n_hits.

| Baseline | n_hits | n_miss | ParamCoverage (hit) | ParamCoverage (miss) | TypeMatch (hit) | TypeMatch (miss) | KeyOverlap (hit) | KeyOverlap (miss) |
|----------|--------|--------|--------------------|----------------------|-----------------|-----------------|------------------|-------------------|
| lsa      | 283    | 48     | 0.8993             | 0.1979               | 0.2334          | 0.0465          | 0.9980           | 0.0860            |
| bm25     | 293    | 38     | 0.8898             | 0.2237               | 0.2378          | 0.1272          | 0.9946           | 0.1148            |
| tfidf    | 300    | 31     | 0.8821             | 0.2419               | 0.2377          | 0.1204          | 0.9948           | 0.1833            |

**Exact denominators:** Each cell is the mean over the corresponding subset; denominator for hit metrics = n_hits, for miss metrics = n_miss.  
**Source:** `results/paper/nlp4lp_downstream_summary.csv` (variant=orig); hit/miss table: `results/paper/nlp4lp_downstream_hitmiss_table_orig.csv`, `results/paper/nlp4lp_error_hitmiss_table.csv`. **Rounding:** 4 decimal places.

---

## 4. Per-type breakdown (orig; percent, integer, currency, float)

**Averaging:** **Micro-averaged.**  
- **param_coverage (per type):** (sum over all queries of filled slots of that type) / (sum over all queries of expected slots of that type) = n_filled / n_expected.  
- **type_match (per type):** (sum over all queries of correct-type filled slots) / (sum over all queries of filled slots of that type) = n_correct / n_filled.  
- **exact5_on_hits / exact20_on_hits (per type):** Among schema-hit queries only; numerator = count of queries (of that type) meeting the exact threshold; denominator = schema-hit queries that have that type and comparable_errs.

**Support counts (n_expected, n_filled)** from types summary (orig, 331 queries):

| param_type | n_expected (total) | tfidf n_filled | oracle n_filled | random n_filled |
|------------|--------------------|---------------|-----------------|-----------------|
| currency   | 382                | 304           | 322             | 2               |
| float      | 1209               | 1009          | 1069            | 5               |
| integer    | 147                | 109           | 121             | 1               |
| percent    | 109                | 95            | 101             | 0               |

**Reported metrics per type (orig) — TF-IDF, Oracle, Random (4 dp):**

| param_type | TF-IDF coverage | TF-IDF type_match | Oracle coverage | Oracle type_match | Random coverage | Random type_match |
|------------|-----------------|-------------------|-----------------|-------------------|-----------------|-------------------|
| currency   | 0.7958          | 0.3586            | 0.8429           | 0.3696           | 0.0052           | 0.5000           |
| float      | 0.8346          | 0.0287           | 0.8842           | 0.0281           | 0.0041           | —                |
| integer    | 0.7415          | 0.9908           | 0.8231           | 0.9917           | 0.0068           | 1.0000           |
| percent    | 0.8716          | 0.4842           | 0.9266           | 0.4950           | 0.0000           | —                |

**Source:** `results/paper/nlp4lp_downstream_types_summary.csv`; pivoted tables: `results/paper/nlp4lp_downstream_types_table_orig.csv`, `results/paper/nlp4lp_error_types_table.csv`. **Rounding:** 4 decimal places. **n_queries:** 331 for all rows.

---

## 5. Typed vs untyped ablation (orig; 331 queries)

**Denominators:** Same as main downstream (331 for schema_R1, coverage, type_match, instantiation_ready; exact5/exact20 on hits use schema-hit-with-comparable_errs count).

| Baseline        | Coverage | TypeMatch | Exact20 (on hits) | InstantiationReady |
|-----------------|----------|-----------|--------------------|--------------------|
| tfidf           | 0.8222   | 0.2267    | 0.2140             | 0.0725             |
| tfidf_untyped   | 0.8222   | 0.1677    | 0.1539             | 0.0453             |
| oracle          | 0.8695   | 0.2475    | 0.1871             | 0.0816             |
| oracle_untyped  | 0.8695   | 0.1887    | 0.1754             | 0.0453             |

**Support:** 331 queries for all four rows. No separate support counts per cell; denominators as in §2.  
**Source:** `results/paper/nlp4lp_downstream_summary.csv`; table: `results/paper/nlp4lp_error_ablation_table.csv`. **Rounding:** 4 decimal places.

---

## 6. Compact manuscript-facing summary (verified numbers)

- On **331 test queries per variant**, retrieval Recall@1 (Schema R@1) for **TF-IDF** is **0.906** (orig), **0.903** (noisy), and **0.785** (short); **BM25** and **LSA** are slightly lower; **random** (theoretical 1/331) is **0.003**.

- **Downstream Schema R@1** on orig matches retrieval: TF-IDF **0.906**, BM25 **0.885**, LSA **0.855**, random **0.006** (2/331), oracle **1.0**.

- **InstantiationReady** (fraction of queries with coverage ≥ 0.8 and type_match ≥ 0.8) on orig is **0.073** (TF-IDF), **0.076** (BM25), **0.060** (LSA), **0.006** (random), and **0.082** (oracle); denominator is **331** for all.

- **Oracle** on orig reaches **0.082** instantiation_ready and **0.869** coverage, only modestly above TF-IDF (**0.073**, **0.822**), so retrieval is not the only bottleneck.

- On **noisy** and **short**, instantiation_ready for lexical baselines is **0** or near zero (e.g. TF-IDF short **0.006**); on noisy, type_match is **0** because `<num>` placeholders are not recovered.

- **Schema-hit vs schema-miss (orig):** For LSA/BM25/TF-IDF, **param_coverage on hits** is **0.88–0.90** and **on misses 0.20–0.24**; **type_match on hits** is **0.23–0.24** and on misses **0.05–0.13**; **key_overlap on hits** is **≈0.99** and on misses **0.09–0.18**. Hit counts are **283–300**, miss **31–48**, depending on baseline.

- **Per-type metrics (orig)** are **micro-averaged**: coverage = total filled / total expected per type, type_match = total correct / total filled per type. **Integer** has the highest type_match (**≈0.99** for TF-IDF/oracle), **float** the lowest (**≈0.03**); currency and percent are in between.

- **Typed vs untyped (orig, 331 queries):** **Type_match** for TF-IDF increases from **0.168** (untyped) to **0.227** (typed) and **instantiation_ready** from **0.045** to **0.073**; coverage stays **0.822**. Oracle shows the same pattern (type_match **0.189** → **0.247**, instantiation_ready **0.045** → **0.082**).

- All reported downstream and section numbers use **4 decimal places** in the export scripts; retrieval table uses the same rounding.

- **Exact5/Exact20** are computed only over **schema-hit queries with non-empty comparable_errs**; denominator is that query count, not 331.

- **Test set size** is **331** for orig, noisy, and short; eval files are `nlp4lp_eval_orig.jsonl`, `nlp4lp_eval_noisy.jsonl`, and `nlp4lp_eval_short.jsonl`.
