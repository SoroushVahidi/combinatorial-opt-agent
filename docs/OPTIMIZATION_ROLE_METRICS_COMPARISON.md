# Optimization-Role vs Alternatives: Metrics Comparison (orig, existing results only)

**Source:** `results/paper/nlp4lp_downstream_summary.csv` (orig variant only).  
**No new runs;** all values from current result files.

---

## Comparison table (orig)

| Baseline | schema_R1 | param_coverage | type_match | key_overlap | exact5_on_hits | exact20_on_hits | instantiation_ready |
|----------|-----------|----------------|------------|-------------|----------------|-----------------|---------------------|
| **tfidf** | 0.9063 | 0.8222 | 0.2260 | 0.9188 | 0.2053 | 0.2330 | **0.0755** |
| **tfidf_constrained** | 0.9063 | 0.7720 | 0.1950 | 0.9188 | 0.2921 | **0.3250** | 0.0272 |
| **tfidf_semantic_ir_repair** | 0.9063 | 0.7783 | **0.2539** | 0.9188 | 0.2345 | 0.2614 | 0.0634 |
| **tfidf_optimization_role_repair** | 0.9063 | 0.8218 | 0.2427 | 0.9188 | **0.2514** | 0.2772 | 0.0604 |
| **tfidf_acceptance_rerank** | 0.8761 | 0.7974 | 0.2275 | 0.8856 | 0.1822 | 0.2058 | **0.0816** |
| **tfidf_hierarchical_acceptance_rerank** | 0.8459 | 0.7771 | 0.2303 | 0.8592 | 0.1705 | 0.1965 | **0.0846** |
| **oracle** | 1.0 | 0.8695 | 0.2401 | 0.9953 | 0.1824 | 0.2044 | **0.0816** |
| **oracle_constrained** | 1.0 | 0.8195 | 0.2092 | 0.9953 | 0.2938 | 0.3206 | 0.0211 |
| **oracle_semantic_ir_repair** | 1.0 | 0.8249 | **0.2798** | 0.9953 | 0.2360 | 0.2581 | 0.0695 |
| **oracle_optimization_role_repair** | 1.0 | **0.8691** | **0.2688** | 0.9953 | **0.2465** | **0.2702** | 0.0695 |

**Bold** in table = best in column among the listed methods (for instantiation_ready, bold marks the best TF-IDF and best Oracle separately where relevant).

---

## Raw values (copy-paste from summary CSV)

| Baseline | schema_R1 | param_coverage | type_match | key_overlap | exact5_on_hits | exact20_on_hits | instantiation_ready |
|----------|-----------|----------------|------------|-------------|----------------|-----------------|---------------------|
| tfidf | 0.9063444108761329 | 0.8221793563938585 | 0.22596988442909896 | 0.918759890663214 | 0.20531573278052154 | 0.23302949007174353 | 0.0755287009063444 |
| tfidf_constrained | 0.9063444108761329 | 0.7720175091776307 | 0.19498437111730169 | 0.918759890663214 | 0.2921479635388087 | 0.3249923785135052 | 0.027190332326283987 |
| tfidf_semantic_ir_repair | 0.9063444108761329 | 0.7782540219096117 | 0.25393762833339883 | 0.918759890663214 | 0.234536257951751 | 0.261436675609211 | 0.0634441087613293 |
| tfidf_optimization_role_repair | 0.9063444108761329 | 0.8218017128893268 | 0.24268025529354859 | 0.918759890663214 | 0.25142026400829226 | 0.27716805885819995 | 0.06042296072507553 |
| tfidf_acceptance_rerank | 0.8761329305135952 | 0.797361475382624 | 0.2275380041241069 | 0.8856111830432072 | 0.182153521660821 | 0.20579083325433692 | 0.08157099697885196 |
| tfidf_hierarchical_acceptance_rerank | 0.8459214501510574 | 0.7770766242820628 | 0.23033997724330055 | 0.8591689445163766 | 0.1704928068564432 | 0.19647776465958283 | 0.08459214501510574 |
| oracle | 1.0 | 0.8694613104645093 | 0.24009266161834156 | 0.9952524816573155 | 0.1823649729899731 | 0.20442451067451078 | 0.08157099697885196 |
| oracle_constrained | 1.0 | 0.8194721002789253 | 0.20920294877998813 | 0.9952524816573155 | 0.29379127816627815 | 0.3206498362748364 | 0.021148036253776436 |
| oracle_semantic_ir_repair | 1.0 | 0.8249435940123704 | 0.27977323951644223 | 0.9952524816573155 | 0.23602947977947977 | 0.2580775705775707 | 0.06948640483383686 |
| oracle_optimization_role_repair | 1.0 | 0.8690836669599775 | 0.26884675411865766 | 0.9952524816573155 | 0.24645851833351856 | 0.27021855459355487 | 0.06948640483383686 |

---

## Answers to your questions

### 1. Which current method is best on InstantiationReady?

**Among the listed methods (orig):**

- **tfidf_hierarchical_acceptance_rerank** has the highest **instantiation_ready**: **0.0846**.
- Next: **tfidf_acceptance_rerank** and **oracle** (both **0.0816**), then **tfidf** (**0.0755**), then **oracle_semantic_ir_repair** and **oracle_optimization_role_repair** (both **0.0695**), then **tfidf_semantic_ir_repair** (0.0634), **tfidf_optimization_role_repair** (0.0604), **tfidf_constrained** (0.0272), **oracle_constrained** (0.0211).

So for the paper’s “instantiation ready” metric, **the best current method is tfidf_hierarchical_acceptance_rerank**, then the two flat acceptance-rerank / oracle baselines.

---

### 2. Which current method is best overall for the paper’s likely main story?

**Depends how you weight the story:**

- **If the main story is “best end-to-end correctness” (exact match on hits):**  
  **tfidf_constrained** has the highest exact20_on_hits (0.325) and exact5_on_hits (0.292) among TF-IDF methods; **oracle_constrained** leads for oracle (0.321 / 0.294). So **constrained** is strongest on exact-match metrics.

- **If the main story is “best instantiation readiness” (model-ready parameters):**  
  **tfidf_hierarchical_acceptance_rerank** (0.0846) or **tfidf_acceptance_rerank** (0.0816) are best; **optimization_role_repair** is not the leader here.

- **If the main story is “strong retrieval + strong assignment without hurting coverage”:**  
  **tfidf_optimization_role_repair** is a good candidate: same schema_R1 and almost same param_coverage as base tfidf (0.906, 0.822), better type_match (0.243 vs 0.226), and **best exact5/exact20 among non-constrained TF-IDF** (0.251, 0.277). So among **non-constrained** methods, **optimization_role_repair** is the best on exact-match-on-hits while keeping coverage high.

**Practical recommendation for “main” story:**  
- For **exact-match emphasis**: center **constrained** (tfidf_constrained / oracle_constrained).  
- For **instantiation-ready emphasis**: center **tfidf_hierarchical_acceptance_rerank** (or tfidf_acceptance_rerank).  
- For **“optimization-aware assignment improves exact match without losing coverage”**: center **tfidf_optimization_role_repair** (and oracle_optimization_role_repair) and contrast with typed/constrained/semantic_ir_repair.

---

### 3. Is `optimization_role_repair` at least as good as the older acceptance-rerank results?

**No.** On the metrics in the current summary:

- **instantiation_ready:**  
  - tfidf_acceptance_rerank **0.0816** > tfidf_optimization_role_repair **0.0604**  
  - tfidf_hierarchical_acceptance_rerank **0.0846** > tfidf_optimization_role_repair **0.0604**

- **exact5_on_hits / exact20_on_hits:**  
  - tfidf_optimization_role_repair (**0.251**, **0.277**) is **better** than tfidf_acceptance_rerank (0.182, 0.206) and tfidf_hierarchical_acceptance_rerank (0.170, 0.196).

- **param_coverage / schema_R1:**  
  - tfidf_optimization_role_repair (0.822, 0.906) is **better** than both acceptance-rerank variants (lower schema_R1 and param_coverage for rerank).

So: **optimization_role_repair is better than acceptance-rerank on exact-match and coverage, but worse on instantiation_ready.** It is not “at least as good” on every metric; it trades off better exact-match and coverage for lower instantiation_ready.

---

### 4. Are the paper-facing tables currently missing this stronger method?

**Yes.** The paper-facing tables (e.g. `nlp4lp_downstream_final_table_orig.csv`, section/main tables) are produced by `make_nlp4lp_paper_artifacts.py` from a **fixed list of baselines** that does not include `*_optimization_role_repair` (see `docs/RESULTS_VS_CODE_VERIFICATION.md` and `docs/OPTIMIZATION_ROLE_METHOD_AUDIT.md`). So:

- **optimization_role_repair** is in the **full** downstream summary and in per-query JSON/CSV.
- It is **not** in the current paper final/section/main table CSVs.

If you want the “stronger” method (for exact-match-on-hits and coverage) in the manuscript tables, the artifact script’s baseline list needs to be updated to include `tfidf_optimization_role_repair` and optionally `oracle_optimization_role_repair`, and the paper tables regenerated.

---

### 5. Which method should we center in the manuscript right now?

**Depends on the narrative:**

1. **“Best instantiation-ready performance”**  
   → Center **tfidf_hierarchical_acceptance_rerank** (or tfidf_acceptance_rerank).  
   optimization_role_repair is weaker on instantiation_ready.

2. **“Best exact match on retrieved hits”**  
   → Center **constrained** (tfidf_constrained / oracle_constrained).  
   optimization_role_repair is strong but not the top on exact5/exact20.

3. **“Optimization-aware assignment: better exact match without sacrificing coverage”**  
   → Center **optimization_role_repair** (tfidf + oracle).  
   Compare to typed, constrained, semantic_ir_repair; show that it beats base tfidf and semantic_ir_repair on exact5/exact20 while keeping param_coverage and schema_R1 high.

4. **“One number to rule them all”**  
   There is no single method that wins every metric; the table shows a clear trade-off (constrained → best exact, acceptance_rerank → best instantiation_ready, optimization_role_repair → best exact among non-constrained with high coverage).

**Recommendation:**  
- If the paper’s **main** metric is **instantiation_ready**, center **tfidf_hierarchical_acceptance_rerank** and report optimization_role_repair as a strong alternative on exact-match and coverage.  
- If the paper’s **main** metric is **exact match on hits** and you want to highlight **optimization-aware assignment**, center **tfidf_optimization_role_repair** (and oracle_optimization_role_repair) and add them to the paper-facing tables; then you can honestly say optimization-role repair gives the best exact-match performance among non-constrained methods while preserving coverage.

---

*All values from `results/paper/nlp4lp_downstream_summary.csv`; no new experiments run.*
