# Evaluation leakage analysis

## 1) Functions that generate queries and how seeds are used

### training/generate_samples.py

| Function | Role | Seed usage | Overlap risk |
|----------|------|------------|--------------|
| `generate_queries_for_problem(problem, rng, target_per_problem)` | Produces a list of query strings for one problem. Uses templates (QUERY_TEMPLATES) with `{text}` filled by name, aliases, description snippets; adds full/short description and sentence chunks. | Uses `rng` only for `rng.shuffle(queries)` before capping. **Order varies by seed; exact set of queries is deterministic** (same inputs → same set, order differs). | Same problem + same target_per_problem → **same query set**. Different seed only changes order; after `[:target_per_problem]` the **same queries can be selected** if the set is identical. |
| `generate_all_samples(catalog_path, seed=42, instances_per_problem=100, ...)` | Loads catalog from disk, creates one `random.Random(seed)` per run, iterates **every problem** in catalog, calls `generate_queries_for_problem(problem, rng, target_per_problem)`, collects (query, passage) pairs. | Single global seed 42. Same catalog + same seed → **identical sequence of queries** for every problem. | **All problems** are used for training; no holdout. |
| `main()` | CLI: `--seed 42`, `--instances-per-problem 100`. Writes pairs to JSONL. | Passes args to `generate_all_samples`. | Training uses **full catalog** and default seed 42. |

**Key point:** The set of query strings for a given problem is **deterministic** given the problem dict and `target_per_problem`. The only randomness is shuffle order; with 100 requested and typically 50+ templates × (name + aliases + desc snippets), many queries are **identical** across runs. Changing seed (e.g. 42 vs 999) only changes which 100 (or 50) are kept after shuffle, but **overlap is guaranteed** because both draws come from the same pool (same templates + same name/aliases/description).

### training/evaluate_retrieval.py

| Function | Role | Seed usage | Overlap risk |
|----------|------|------------|--------------|
| `_generate_eval_instances(seed=999, num_instances=500)` | Loads **same catalog** via `_load_catalog()`, creates `random.Random(seed)`, iterates **every problem** in catalog, calls `generate_queries_for_problem(problem, rng, target_per_problem=50)`, collects (query, problem_index), shuffles, takes first 500. | Seed 999. Same catalog + same seed → **same 500 instances**. Each instance is (query, problem_index). | **Same problems** as training. **Same query generator** and **same templates**. So the **query strings** for a given problem are from the same deterministic set; only the count (50 vs 100) and the shuffle differ. So many queries that appear in training (seed 42, 100 per problem) **also appear in eval** (seed 999, 50 per problem). |
| `main()` | Loads or regenerates eval instances, loads catalog, runs retrieval, computes P@1/P@5. | `--seed 999`, `--num 500`. | Eval uses **same catalog** as training; no problem holdout. |

**Conclusion:** Training and eval both iterate the **same catalog** and use the **same** `generate_queries_for_problem()`. The only differences are: (1) training seed 42 vs eval seed 999, (2) 100 vs 50 queries per problem, (3) eval shuffles and caps at 500 instances. So **every problem appears in both train and eval**; and for each problem, the **query pools are the same** (templates + name/aliases/description), so **exact query string overlap is highly likely**.

---

## 2) Three concrete query strings likely in BOTH train and eval (same problem)

For a problem with **name** `"0-1 Knapsack Problem"` and **description** starting with "Given items with weights and values...", the following are produced by the **same** templates and text in both training (seed 42, 100 per problem) and eval (seed 999, 50 per problem). They are **deterministic** given the problem:

1. **`"0-1 Knapsack Problem"`**  
   From `add(name)` in `generate_queries_for_problem` (line 106). Always included.

2. **`"What is 0-1 Knapsack Problem?"`**  
   From `QUERY_TEMPLATES`: `"What is {text}?"` with `text=name` (lines 107–109). Always in the pool; whether it appears in the final 50 or 100 depends only on shuffle order, but both train and eval draw from the same pool, so it can appear in both.

3. **`"ILP for 0-1 Knapsack Problem."`**  
   From `QUERY_TEMPLATES`: `"ILP for {text}."` with `text=name`. Same as above.

4. **`"Given items with weights and values, select a subset of items to put in a knapsack of limited capacity so that total value is maximized and total weight does not exceed capacity. Each item can be taken at most once."`**  
   From `add(desc)` (line 95). Full description is always the first “query” added. So this **exact string** appears in both train and eval for this problem.

So at least **four** queries (name, “What is {name}?”, “ILP for {name}.”, full description) are **guaranteed** to be in the query set for that problem in both scripts; and after shuffling, many more template-based queries can coincide. Hence **same problem + same query string in both train and eval** is not rare—it’s systematic.

---

## 3) Every place where training and eval read the same catalog entries

| Location | What is read | Same as other? |
|----------|----------------|-----------------|
| **training/generate_samples.py** `generate_all_samples()` | `catalog = json.load(catalog_path)` with default `data/processed/all_problems.json` (or extended). Iterates **all** problems. | Same file as below. |
| **training/evaluate_retrieval.py** `_generate_eval_instances()` | `catalog = _load_catalog()` which uses `retrieval.search._load_catalog()` → `data/processed/all_problems_extended.json` or `all_problems.json`. Iterates **all** problems. | **Same catalog content** (extended if present, else base). |
| **training/evaluate_retrieval.py** `main()` | `catalog = _load_catalog()` again for building the index and resolving expected problem names. | Same as above. |
| **retrieval/search.py** `_load_catalog()` | Used by eval; same path logic. | Same. |

So: **training** and **eval** both read the **same catalog file(s)** and both iterate **every** problem in that catalog. There is **no** split by problem: train and eval use the same set of problems and the same query generator, so **problem-level and query-level leakage** are both present.

---

## 4) Fix: leak-free evaluation

- **Split by problem:** Produce disjoint train / dev / test **problem IDs** (e.g. in `training/splits.py`), stratifying by `source` when possible.
- **Training:** Generate pairs **only** for problems in the **train** split (`generate_samples --splits ... --split train`).
- **Eval:** Generate eval instances **only** for problems in **dev** or **test** split (`evaluate_retrieval --splits ... --split test`). Evaluation is then on problems the model **never saw** during training.
- **Catalog:** At eval time, retrieval still runs over the **full** catalog (so the model can retrieve any problem); metrics are computed **only** on instances whose **expected** problem is in the chosen split (dev/test). This matches standard retrieval evaluation: train on train-set problems only, evaluate on held-out problems.
