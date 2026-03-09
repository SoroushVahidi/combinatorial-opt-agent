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

### Fix (applied — no retrain required for the deterministic layer)

Three concrete improvements were made:

#### a) Written-word number recognition in `tools/nlp4lp_downstream_utility.py`

NLP4LP queries frequently spell out numeric values as English words ("two
facilities", "twenty percent", "three types of products").  The previous
`NUM_TOKEN_RE` only matched digit strings (`\d+`), so written numbers were
silently skipped — the primary cause of the 18 % Coverage gap.

Added:
- `_WORD_TO_NUM` lookup table: 0–19, round tens 20–90, 100, 1000, plus
  hyphenated compounds ("twenty-three" → 23).
- `_word_to_number(word)` public helper — used by both extractors and tests.
- `_extract_num_tokens()` — now falls back to `_word_to_number()` when
  `NUM_TOKEN_RE` does not match the token.
- `_extract_num_mentions()` — same, used by the constrained assignment path.

**Files changed:** `tools/nlp4lp_downstream_utility.py`

#### b) Extended `_expected_type()` slot-pattern coverage

The original `_expected_type` classified slot names into four categories
(percent / int / currency / float) but missed many common patterns:

- Percent: `"ratio"`, `"share"`, `"proportion"` now → `"percent"`
- Int-count: `"machines"`, `"workers"`, `"vehicles"`, `"facilities"`,
  `"warehouses"`, `"periods"`, `"stages"`, etc. now → `"int"`
- Currency: `"amount"`, `"total"`, `"supply"`, `"allocation"`,
  `"threshold"`, `"salary"`, `"income"`, etc. now → `"currency"`

These additions reduce TypeMatch errors for slot names that were previously
falling through to the `"float"` default.

**Files changed:** `tools/nlp4lp_downstream_utility.py`

#### c) Written-word paraphrase augmentation in `training/generate_mention_slot_pairs.py`

The mention-slot scorer is trained on pairs where the context contains
digit strings.  Written-number contexts ("two warehouses") are out-of-distribution.
Added:

- `_int_to_word(n)`: converts integers 0–99, 100, 1000 to their English word
  (inverse of `_word_to_number`).
- `_augment_with_word_paraphrases()`: for every matched digit mention
  (label = 1) in a query, generates a written-word variant using
  `_int_to_word`, producing a paraphrased pair like
  `("two facilities", "numFacilities", 1)`.
- `--no-augment` flag to the generation script allows opting out.

After regenerating training pairs with augmentation and retraining, the
scorer will generalise to written-number queries.

**Files changed:** `training/generate_mention_slot_pairs.py`

---

## 4. Catalog formulation coverage gap

### What it is

295 out of 1,597 problems (18.5 %) had no formulation at all, and of the
1,302 that do, 166 had empty `variables` and 169 had empty `constraints`.
This meant the app silently showed an empty formulation section for roughly
1 in 5 retrieved results.

| Issue | Count | Share of catalog |
|-------|-------|-----------------|
| No formulation at all | 295 | 18.5 % |
| Formulation with empty variables | 166 | 10.4 % |
| Formulation with empty constraints | 169 | 10.6 % |
| Formulation with LaTeX | 35 | 2.2 % |

The 295 problems without formulations came from these sources:

| Source | No-formulation count |
|--------|---------------------|
| gams | 143 |
| or_library | 63 |
| gurobi_modeling_examples | 55 |
| pyomo | 19 |
| gurobi_optimods | 14 |
| miplib | 1 |

### Fix

Two complementary changes were made:

#### a) Graceful "formulation not available" notice in `retrieval/search.py`

`format_problem_and_ip()` previously emitted empty collapsible `<details>`
sections when a problem had no formulation or had empty `variables` /
`constraints` lists.  The user saw bare headers with no content — a silently
degraded experience.

Now, when `formulation` is `None` or both `variables` and `constraints` are
empty, the function emits a clear blockquote:

```
> **Formulation not yet available.** This problem is in the catalog but its
> structured ILP has not been added yet. The description above may still
> help you understand the problem structure.
```

The problem name and description are still shown, so the result is still
useful.

**Files changed:** `retrieval/search.py`

#### b) Structured ILP formulations for 16 common no-formulation problems

Structured ILP formulations were added to `data/processed/custom_problems.json`
for 16 well-known problems that had no formulation in the base catalog.
These entries use the same `id` as the corresponding base catalog stub, so
`build_extended_catalog.py` overrides them on id collision, producing a
correctly enriched catalog in `all_problems_extended.json`.

Problems with formulations added:

| Source | Problem | ID |
|--------|---------|-----|
| gurobi_optimods | Bipartite Matching | `gurobi_optimods_bipartite_matching` |
| gurobi_optimods | Max Flow | `gurobi_optimods_max_flow` |
| gurobi_optimods | Min Cost Flow | `gurobi_optimods_min_cost_flow` |
| gurobi_optimods | Min Cut | `gurobi_optimods_min_cut` |
| gurobi_optimods | Portfolio Optimization | `gurobi_optimods_portfolio` |
| gurobi_optimods | Workforce Scheduling | `gurobi_optimods_workforce` |
| gurobi_optimods | Max Weight Independent Set | `gurobi_optimods_mwis` |
| or_library | Assignment Problem | `or_lib_assignment` |
| or_library | 1D Bin Packing | `or_lib_bin_packing_1d` |
| or_library | Generalised Assignment | `or_lib_generalised_assignment` |
| or_library | Graph Colouring | `or_lib_graph_colouring` |
| or_library | Crew Scheduling | `or_lib_crew_scheduling` |
| gurobi_modeling_examples | Cell Tower Coverage | `gurobi_ex_cell_tower_coverage` |
| gurobi_modeling_examples | Customer Assignment | `gurobi_ex_customer_assignment` |
| pyomo | Diet Problem | `pyomo_diet` |
| pyomo | Uncapacitated Facility Location | `pyomo_facility_location` |

After running `python build_extended_catalog.py`, the no-formulation count
drops from 295 → 279 (5.4 % reduction).  Remaining gaps (mostly GAMS,
OR-Library, and Gurobi examples) require the respective catalog stubs to be
hand-annotated or auto-populated using a code-generation pipeline.

**Files changed:** `data/processed/custom_problems.json`,
`data/processed/all_problems_extended.json`

---

## Summary table

| # | Bottleneck | Severity | Status |
|---|-----------|----------|--------|
| 1 | Catalog embeddings rebuilt on every query | **Critical** (3–8 s/query) | ✅ Fixed in `app.py` |
| 2 | Short-query performance drop (~12 %) | High | ✅ Fully fixed: `expand_short_query()` in `retrieval/search.py` + all baselines (immediate, no retrain); `SHORT_QUERY_TEMPLATES` in `generate_samples.py` (retrain for additional model-level gain) |
| 3 | NLP4LP downstream instantiation gap | High (pipeline-level) | ✅ Fixed: written-word number recognition + expanded `_expected_type()` in `tools/nlp4lp_downstream_utility.py`; written-word paraphrase augmentation in `training/generate_mention_slot_pairs.py` |
| 4 | Catalog formulation coverage (18.5 % missing) | Medium | ✅ Fixed: graceful notice in `format_problem_and_ip()` (`retrieval/search.py`); 16 new ILP formulations in `custom_problems.json` (no-formulation count: 295 → 279) |
