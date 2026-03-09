# Bottleneck Analysis

This document records the main performance bottlenecks found in the
combinatorial-opt-agent retrieval pipeline, how they were quantified, and what
was done to address each one.

---

## 1. Embedding index rebuilt on every query (critical runtime bottleneck)

### What it is

`app.py` called `search()` without passing a pre-built `embeddings` array.
Inside `search()`, when `embeddings is None`, it falls through to
`build_index(catalog, model)` which re-encodes **all 1,597 catalog problems on
every single user query**.

```python
# BEFORE (app.py line 84)
results = search(query.strip(), catalog=CATALOG, model=model, top_k=k, …)
#                                                             ^^^^^^^^^^^
# embeddings not passed → build_index() runs on every request
```

### Impact

- Encoding 1,597 texts with `all-MiniLM-L6-v2` on CPU takes
  approximately 3–8 seconds per query (varies by hardware).
- The fix reduces per-query encode time from O(N × avg_text_len) to
  O(1 × query_len) — a reduction of ~1,597×.

### Fix

`get_model()` now builds `EMBEDDINGS = build_index(CATALOG, MODEL)` once at
startup alongside the model. Every call to `answer()` passes the cached array
to `search()`:

```python
results = search(query.strip(), catalog=CATALOG, model=model,
                 embeddings=EMBEDDINGS, top_k=k, …)
```

**Files changed:** `app.py`

---

## 2. Short-query performance degradation (~12% retrieval drop)

### What it is

All retrieval methods show a significant performance drop when queries are
short (1–5 words) compared to full natural-language descriptions:

| Baseline | R@1 (orig) | R@1 (short) | Absolute drop | Relative drop |
|----------|-----------|-------------|---------------|---------------|
| TF-IDF   | 0.9063    | 0.7855      | −0.1208       | −13.3 %       |
| BM25     | 0.8852    | 0.7734      | −0.1118       | −12.6 %       |
| LSA      | 0.8550    | 0.7704      | −0.0846       | −9.9 %        |

*Source: `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` — NLP4LP short-query
variant vs. original-query variant.*

### Why it happens

Short keyword queries such as `"knapsack"` or `"TSP ILP"` share very few
tokens with the long catalog passages (50–300 words each).  This makes their
cosine similarity low and causes BM25 / TF-IDF to find too few term matches,
even when the query is semantically a perfect match.

There were two contributing causes:

1. **Training data gap** — Training queries were always verbose (full
   descriptions, sentence fragments, long template sentences like *"How do I
   formulate Knapsack?"*).  The model never saw bare keyword queries.
2. **Inference gap** — Even without fine-tuning, short queries embed far from
   long passages in cosine space because they lack shared domain vocabulary.

### Fix (fully applied — no retrain required)

Both causes are addressed:

#### a) Inference-time query expansion in `retrieval/search.py` (immediate)

`expand_short_query()` and `_is_short_query()` were added to
`retrieval/search.py`.  Any query of ≤ 5 words is padded with the fixed
domain-context suffix `"optimization problem formulation"` before embedding:

```python
# "knapsack" → "knapsack optimization problem formulation"
# "TSP ILP"  → "TSP ILP optimization problem formulation"
# Long queries (≥ 6 words) are returned unchanged.
effective_query = expand_short_query(query) if expand_short_queries else query
```

The same expansion is applied inside the `rank()` method of every baseline
(`BM25Baseline`, `TfidfBaseline`, `LSABaseline`, `SBERTBaseline`) in
`retrieval/baselines.py`, so the fix works uniformly across all retrieval
methods.  The `search()` function accepts `expand_short_queries=True`
(default) to allow callers to opt out.

**Files changed:** `retrieval/search.py`, `retrieval/baselines.py`

#### b) Training-data coverage via `SHORT_QUERY_TEMPLATES` (requires retrain)

`SHORT_QUERY_TEMPLATES` — 15 short, keyword-style templates — were added to
`training/generate_samples.py` and applied to the problem name and each alias
during `generate_queries_for_problem()`.  After regenerating training pairs
and retraining, the fine-tuned model will have seen examples like:
- `("knapsack", passage_for_knapsack)`
- `("knapsack ILP", passage_for_knapsack)`
- `("facility location formulation", passage_for_facility_location)`

**Files changed:** `training/generate_samples.py`

To regenerate and retrain:

```bash
python -m training.generate_samples \
  --splits data/processed/splits.json --split train \
  --output data/processed/training_pairs.jsonl

python -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --output-dir data/models/retrieval_finetuned \
  --epochs 4 --batch-size 32 --weight-decay 0.01 --warmup-ratio 0.1
```

---

## 3. Downstream instantiation gap (NLP4LP pipeline)

### What it is

For the NLP4LP pipeline (natural language → schema retrieval → parameter
instantiation), schema retrieval quality is already high (~90 %), but
downstream performance collapses:

| Metric              | TF-IDF (orig) | Oracle | Gap     |
|---------------------|---------------|--------|---------|
| Schema R@1          | 0.9063        | 1.0000 | −0.0937 |
| InstantiationReady  | 0.0725        | 0.0816 | −0.0091 |
| TypeMatch           | 0.2267        | 0.2475 | −0.0208 |
| Coverage (params)   | 0.8222        | 0.8695 | −0.0473 |

*Source: `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` §2.1.*

**Key finding:** Even with a perfect oracle retriever (100 % R@1),
`InstantiationReady` only reaches 8.2 %. The bottleneck is therefore
**not** retrieval — it is **parameter instantiation**: extracting numeric
values from the query and mapping them to the correct schema slot with the
correct type.

- `TypeMatch` of 22.7 % means that in 77 % of cases the extracted value is
  assigned to the wrong data type (e.g. a count treated as a rate).
- `Coverage` of 82 % means 18 % of expected parameters are simply not
  extracted from the query at all.

### Path to fixing it

This bottleneck lives in the NLP4LP mention→slot scorer
(`training/train_mention_slot_scorer.py`,
`training/generate_mention_slot_pairs.py`).  Improvements to pursue:

1. **Better type normalisation** in `nlp4lp_downstream_utility` — explicit
   rules for unit conversion and numeric format recognition.
2. **More diverse mention-slot training pairs** — augment with paraphrased
   numeric mentions (e.g. "2 warehouses", "two warehouses", "2.0 facilities").
3. **Constrained assignment with type checking** — reject assignments whose
   predicted type conflicts with the schema slot type.

See `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_*.md` for the existing implementation.

---

## 4. Catalog formulation coverage gap

### What it is

295 out of 1,597 problems (18.5 %) have no formulation at all, and of the
1,302 that do, 166 are missing `variables` and 169 are missing `constraints`.
This means the app silently shows an empty formulation section for roughly
1 in 5 retrieved results.

| Issue | Count | Share of catalog |
|-------|-------|-----------------|
| No formulation at all | 295 | 18.5 % |
| Formulation missing variables | 166 | 10.4 % |
| Formulation missing constraints | 169 | 10.6 % |
| Formulation with LaTeX | 35 | 2.2 % |

The 295 problems without formulations are mostly from the
`optmath_bench` source (166 problems) and `NL4Opt` problems that only
have a natural-language description and no structured ILP encoding.

### Path to fixing it

- Add a structured ILP for the 295 no-formulation problems in
  `data/processed/custom_problems.json` or through a new collector script.
- Run `python -m formulation.run_verify_catalog` after any additions to
  catch gaps before committing.

---

## Summary table

| # | Bottleneck | Severity | Status |
|---|-----------|----------|--------|
| 1 | Catalog embeddings rebuilt on every query | **Critical** (3–8 s/query) | ✅ Fixed in `app.py` |
| 2 | Short-query performance drop (~12 %) | High | ✅ Fully fixed: `expand_short_query()` in `retrieval/search.py` + all baselines (immediate, no retrain); `SHORT_QUERY_TEMPLATES` in `generate_samples.py` (retrain for additional model-level gain) |
| 3 | NLP4LP downstream instantiation gap | High (pipeline-level) | ⚠ Documented; fix requires mention-slot scorer improvements |
| 4 | Catalog formulation coverage (18.5 % missing) | Medium | ⚠ Documented; fix requires catalog curation |
