# NLP4LP Manuscript Consistency Plan

Final consistency plan for experiments section, table captions, and footnotes. Use this document to rewrite captions and to ensure reviewer-facing clarity without rewriting the manuscript prose.

---

## 1. Places that require explicit clarification in the manuscript (tables / captions / notes)

### 1.1 Random baseline definition

- **Retrieval table:** Random is **not run**. The value shown is **theoretical** Recall@1 = 1/N with N = 331 (≈ 0.0030). No random retrieval baseline is executed; the number is injected in the table-generation script.
- **Downstream tables:** Random **is run**: one deterministic run with schema chosen per query via `random.Random(seed(query_id)).randrange(len(catalog))`. Empirical Schema R@1 = 2/331 (≈ 0.0060) in the current run.
- **Clarification needed:** The manuscript must state explicitly which definition applies in each table (retrieval vs downstream), or standardize and explain the chosen convention (see Section 3).

### 1.2 Oracle definition

- **Retrieval:** Oracle is **not** a retrieval baseline; it does not appear in the retrieval table.
- **Downstream:** Oracle = “gold schema always”: for every query, predicted schema = gold schema (no retrieval step). All other downstream steps (extraction, typing, evaluation) are unchanged. Subset = all 331 queries with correct schema by construction.
- **Clarification needed:** In the downstream section, state once that “Oracle uses the gold schema for every query (no retrieval)” so reviewers do not assume an upper-bound retrieval system.

### 1.3 Schema-hit / schema-miss subset definition

- **Hit:** Queries where the **retrieved (or chosen) schema equals the gold schema** (predicted_doc_id == gold_doc_id).
- **Miss:** All other queries (predicted_doc_id ≠ gold_doc_id).
- **Clarification needed:** In the hit/miss table caption or a footnote, state: “Schema-hit (resp. schema-miss) = queries where the retrieved schema is correct (resp. incorrect). Metrics in the hit/miss columns are means over those subsets only.”

### 1.4 Per-type micro vs macro averaging

- **Current implementation:** All per-type metrics are **micro-averaged**:
  - **Coverage (per type):** (total filled slots of that type over all queries) / (total expected slots of that type over all queries).
  - **Type match (per type):** (total correct-type filled slots of that type) / (total filled slots of that type).
- This is **not** macro-averaging (i.e. not “average of per-query type-level rates”).
- **Clarification needed:** In the per-type table caption or a footnote, state: “Per-type metrics are micro-averaged (pooled counts over all queries), not macro-averaged.”

### 1.5 Denominator definitions for every metric family

| Metric family | Denominator / aggregation | Where to state |
|---------------|---------------------------|----------------|
| **Schema R@1 / Recall@1** | 331 (all test queries) | Retrieval and downstream table captions or methods. |
| **Coverage (param_coverage)** | Per query: n_filled / n_expected_scalar; aggregate: mean over 331. | Downstream caption or footnote. |
| **TypeMatch** | Per query: type_correct / n_filled; aggregate: mean over 331. | Same. |
| **KeyOverlap** | Per query: \|pred ∩ gold\| / \|gold\|; aggregate: mean over 331. | Same. |
| **Exact5 / Exact20** | **Not** 331. Mean over **schema-hit queries that have comparable_errs** (variable count per baseline). | Downstream caption or footnote; important to avoid misinterpretation. |
| **InstantiationReady** | 331 (fraction of queries with coverage ≥ 0.8 and type_match ≥ 0.8). | Same. |
| **Hit/miss columns** | Mean over hit queries only (resp. miss queries only); counts vary by baseline (e.g. 283–300 hits for LSA–TF-IDF on orig). | Hit/miss table caption. |
| **Per-type** | Micro: denominator for coverage = total expected for that type; for type_match = total filled for that type. | Per-type table caption. |

### 1.6 Rounding policy

- **Tables:** All main experiment tables use **4 decimal places** in the export pipeline (`f"{float(v):.4f}"`).
- **In-text:** Some generated notes use 3 decimal places (e.g. “schema_R1=0.906”); if the manuscript copies such numbers, state “values rounded to 3 (or 4) decimal places” where relevant, or keep table and text consistent (e.g. 4 dp in tables, 3 dp in prose if desired).
- **Clarification needed:** One short sentence in methods or caption: “Reported metrics are rounded to 4 decimal places unless otherwise noted.”

---

## 2. Polished technical captions / footnotes (directly adaptable)

### 2.1 Retrieval main table

**Caption (technical):**  
“Schema retrieval accuracy (Recall@1) by query variant and baseline. Each cell is the fraction of test queries for which the gold schema appears at rank 1 (denominator N = 331 per variant). BM25, TF-IDF, and LSA are retrieval baselines; random is the theoretical lower bound 1/N (no random retrieval run). Values rounded to 4 decimal places.”

**Footnote (if needed):**  
“Recall@1 = (number of queries with gold schema at rank 1) / 331. Random = 1/331; same value for all variants.”

---

### 2.2 Downstream main table

**Caption (technical):**  
“Downstream utility on NLP4LP (original queries, N = 331). Schema R@1 = fraction of queries where the retrieved schema equals the gold schema. Coverage = mean over queries of (filled scalar slots / expected scalar slots). TypeMatch = mean over queries of (correct-type filled / filled). KeyOverlap = mean over queries of |pred ∩ gold| / |gold|. Exact5 (Exact20) = mean over **schema-hit queries with comparable errors** of the fraction of errors within 5% (20%). InstantiationReady = fraction of queries with coverage ≥ 0.8 and type_match ≥ 0.8. Oracle uses the gold schema for every query (no retrieval). Random is one deterministic run (schema per query from fixed seed). Values rounded to 4 decimal places.”

**Footnote (denominators):**  
“All metrics except Exact5/Exact20 use denominator 331. Exact5 and Exact20 use only schema-hit queries that have comparable_errs; denominator therefore varies by baseline.”

---

### 2.3 Hit/miss table

**Caption (technical):**  
“Schema-hit vs schema-miss downstream behavior (original queries, retrieval baselines LSA, BM25, TF-IDF). Schema-hit = queries where the retrieved schema equals the gold schema; schema-miss = all other queries. Each cell is the mean of the corresponding metric over that subset only (denominator = number of hit queries or miss queries, which varies by baseline). Coverage = param_coverage; Type = type_match; Key = key_overlap. Values rounded to 4 decimal places.”

**Footnote (support):**  
“Hit counts (approx.): LSA 283, BM25 293, TF-IDF 300; miss = 331 − hit.”

---

### 2.4 Per-type table

**Caption (technical):**  
“Per-parameter-type downstream behavior (original queries, N = 331). Types: percent, integer, currency, float. Coverage (per type) = total filled slots of that type / total expected slots of that type over all queries. Type match (per type) = total correct-type filled slots of that type / total filled slots of that type. **Micro-averaged** (pooled counts over queries), not macro-averaged. Baselines: TF-IDF, Oracle (gold schema always), Random (one deterministic run). Values rounded to 4 decimal places.”

**Footnote (optional):**  
“Support: n_expected (total) per type is fixed (e.g. currency 382, float 1209, integer 147, percent 109); n_filled varies by baseline.”

---

### 2.5 Typed vs untyped ablation table

**Caption (technical):**  
“Effect of type-aware parameter assignment on original queries (N = 331). Typed = assignment respects inferred parameter types; untyped = type-agnostic assignment. Same denominators as the main downstream table: 331 for Coverage, TypeMatch, InstantiationReady; Exact20 is over schema-hit queries with comparable_errs. Oracle uses gold schema for every query. Values rounded to 4 decimal places.”

---

## 3. Presentation choice: two random definitions vs one

### Option A: Keep both definitions and explain them clearly

- **Description:** Retrieval table keeps theoretical random 1/331; downstream tables keep empirical random (deterministic run, 2/331). Manuscript states explicitly in each context which definition applies.
- **Pros:** Matches the code and artifacts exactly; no code or table changes. Theoretic retrieval baseline is clean (1/N); downstream shows a concrete “random run” that is reproducible.
- **Cons:** Two different numbers (0.003 vs 0.006) for “random”; reviewers may be confused unless the distinction is clearly stated in captions or methods.

### Option B: Standardize to a single notion of random

- **Description:** Either (B1) use theoretical 1/N everywhere (change downstream tables/artifact to report 1/331 for random in downstream) or (B2) use empirical random everywhere (run random retrieval and report 2/331 in retrieval table, or average over multiple runs).
- **Pros:** One number for “random” in the whole paper; simpler narrative.
- **Cons:** (B1) Downstream random would no longer reflect the actual run (weaker story). (B2) Requires changing retrieval pipeline or reporting (e.g. multi-run average) and possibly new experiments.

### Final recommendation

**Recommend Option A (keep both, explain clearly).** The code and results are already consistent and reproducible; the only cost is one or two explicit sentences. Suggested wording:

- In the **retrieval** table caption or methods: “Random is the theoretical lower bound 1/N (N = 331); no random retrieval run is performed.”
- In the **downstream** section or table caption: “Random baseline: one deterministic run (schema chosen per query by seeded RNG); reported Schema R@1 is the empirical fraction (e.g. 2/331 on orig).”

No code or table changes are required; only caption/methods text.

---

## 4. Values that may confuse reviewers unless explained

- **Two different “random” numbers (0.003 vs 0.006):** Explain which table uses which definition (see Section 3).
- **Exact5 / Exact20 denominator ≠ 331:** State that these metrics are over schema-hit queries with comparable_errs only, so the denominator is smaller and baseline-dependent.
- **Oracle instantiation_ready &lt; 0.1 (e.g. 0.082):** Reviewers may expect “oracle = perfect.” Clarify that oracle only fixes retrieval; extraction and typing remain, so full instantiation is still rare.
- **Noisy type_match = 0:** Explain that noisy queries use `<num>` placeholders that are not deterministically recovered, so type_match is 0 by design in the current evaluation.
- **Per-type “type match” for float ≈ 0.03:** Low rate is expected (hard type); stating “micro-averaged” and “float is hardest” avoids the impression of a bug.
- **Hit/miss support counts (283–300 vs 31–48):** Stating approximate hit/miss counts in the hit/miss table footnote avoids confusion about whether denominators are equal across baselines.

---

## 5. Safe-reporting checklist for the paper

### 5.1 Must be stated in the main text (methods or experiments)

- **Test set size:** “We evaluate on 331 queries per variant (orig, noisy, short); same query IDs across variants, with different query text.”
- **Random baseline (if Option A):** One sentence that retrieval uses theoretical 1/N and downstream uses one deterministic random run (and, if desired, that Schema R@1 for that run is 2/331 on orig).
- **Oracle (downstream):** “Oracle uses the gold schema for every query (no retrieval); all other pipeline steps are unchanged.”
- **InstantiationReady definition:** “InstantiationReady is the fraction of queries with param_coverage ≥ 0.8 and type_match ≥ 0.8.”

### 5.2 Must be stated in table captions or footnotes

- **Retrieval table:** Denominator 331; random = theoretical 1/N; rounding (4 dp) if you want to be explicit.
- **Downstream main table:** Denominator 331 for all metrics except Exact5/Exact20; Exact5/Exact20 over schema-hit queries with comparable_errs; oracle and random definitions; rounding if desired.
- **Hit/miss table:** Definition of schema-hit and schema-miss; denominators = hit count and miss count respectively; optional approximate hit/miss counts.
- **Per-type table:** Micro-averaging (pooled counts); denominator for coverage = total expected per type, for type_match = total filled per type; rounding if desired.
- **Ablation table:** N = 331; same denominator rules as downstream main; typed vs untyped defined in one phrase.

### 5.3 Can remain implicit (if methods are clear)

- Exact formula for key_overlap (|pred ∩ gold| / |gold|) can stay in supplement or code; caption can say “key overlap (pred–gold scalar set overlap)”.
- Exact rounding implementation (e.g. Python `.4f`) need not appear in the manuscript; “4 decimal places” in one place is enough.
- Names of eval files (`nlp4lp_eval_orig.jsonl`, etc.) can stay in reproducibility notes or code; “331 queries per variant” in text is sufficient.
- Per-query definitions of coverage and type_match (n_filled / n_expected_scalar, etc.) can be summarized as “mean over queries” in the caption once the denominator 331 is stated.

---

**Summary:** Use Section 1 as the list of required clarifications; Section 2 as drop-in or near drop-in captions/footnotes; Section 3’s recommendation (Option A) and Section 4’s list to preempt reviewer questions; Section 5 as the minimal checklist when rewriting the experiments section and table captions.
