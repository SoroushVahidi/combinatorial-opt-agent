# Q1 Journal Submission — Code-Level Audit

Senior researcher code audit for the combinatorial-optimization-agent repo. References exact file paths, functions, and line ranges where applicable.

---

## A) Missing vs Claimed — Gap List

Features **promised** in README (lines 1–21), vision, and Phase checklists that are **not implemented**, with **exact paths** where they should live.

| # | Claimed feature | Where claimed | Implemented? | Where it should live (exact path) |
|---|------------------|---------------|--------------|-----------------------------------|
| 1 | **LP relaxation** (output) | README L7, L12, L20 | **No** — only ILP/formulation displayed; no LP relaxation derivation or display | `retrieval/search.py`: extend `format_problem_and_ip()` to accept/show `lp_relaxation`; add `formulation/lp_relaxation.py` (or `formulation/relaxation.py`) to derive LP from ILP (e.g. relax binary → [0,1]) and attach to problem dict; schema in `data/` problem records: `lp_relaxation` / `formulation_latex_lp` |
| 2 | **Solver-ready code** (Pyomo, Gurobi, PuLP) | README L8, L20, L57; How to run L57 | **No** — no code generation or templates | `formulation/solver_code.py` (new): e.g. `generate_pyomo(problem)`, `generate_pulp(problem)`, `generate_gurobi(problem)`; optionally `formulation/templates/`; call from `app.py` and/or `retrieval/search.py` (e.g. after `format_problem_and_ip`) or new endpoint |
| 3 | **Entity and constraint extraction** (NL → structure) | README L16 (Architecture) | **No** | `nlp/extract_entities.py` or `nlp/constraint_extraction.py` (new): functions to extract entities/constraints from raw NL; optional integration in `app.py` before/after retrieval |
| 4 | **Problem classifier (embeddings + LLM)** | README L17 | **Partial** — embeddings only in `retrieval/search.py`; no LLM in classification path | LLM component: e.g. `retrieval/classifier_llm.py` (new) or extend `retrieval/search.py` with optional LLM rerank/confirm step using top‑k from embedding search |
| 5 | **For unknown problems: generate formulation via LLM and verify** | README L19 | **No** | `formulation/llm_generate.py` (new): generate formulation from NL when no match; `formulation/verify.py` (new): validate formulation (syntax, consistency); orchestration in `app.py` when `search()` returns low confidence or “unknown” |
| 6 | **Verification** (of formulations) | README L19; Phase 4 “Validation against benchmark instances” | **No** | `formulation/verify.py`: validate structure (variables, objective, constraints), optional instance check; hook in `app.py` and in retrieval pipeline (see Section D) |
| 7 | **Disambiguation** (clarifying questions) | README Phase 3 (L163) | **No** | `retrieval/disambiguation.py` (new): when top‑k are close in score, return clarifying questions or multiple candidates; call from `app.py` after `search()` or inside `answer()` |
| 8 | **Complexity class when applicable** | README L20 | **Partial** — `complexity` field exists in catalog and is displayed in `format_problem_and_ip()` (search.py L156–157); not computed or guaranteed per problem | Either: extend collectors/merge to set `complexity` for all problems, or add `formulation/complexity.py` to infer from formulation; no new file strictly required if schema is filled in data |
| 9 | **Use generated formulation and solver code** (user workflow) | README L57 | **No** — no generation, so nothing to “use” | Resolved by implementing (2) and (6); then document in README/Quick start |
| 10 | **FAISS / ChromaDB** (vector search) | README L167, L190 | **No** — search uses in-memory NumPy in `retrieval/search.py` (`build_index`, `search`) | `retrieval/faiss_index.py` or `retrieval/chroma_index.py` (new): build and query FAISS/ChromaDB index; `retrieval/search.py`: optional path to use FAISS/Chroma when available (e.g. for large catalogs) |
| 11 | **Backend API (FastAPI)** | README Phase 5 (L175) | **Partial** — Gradio is mounted on FastAPI in `app.py` (L199–201); no dedicated REST API for search/formulation | `api/` or routes in `app.py`: e.g. `POST /search`, `GET /problem/<id>`; keep existing Gradio at `/` |
| 12 | **LaTeX + solver code output formatting** | README Phase 4 (L171) | **Partial** — LaTeX shown in `format_problem_and_ip()` (search.py L143–155); solver code not generated | Add solver code generation (see (2)); optionally `format_solver_code()` in `formulation/solver_code.py` or in `retrieval/search.py` |

**Summary:** The only fully delivered “vision” items are: (a) NL → retrieval over a DB of known problems, (b) display of ILP (variables, objective, constraints, optional LaTeX), and (c) optional complexity display when present in data. LP relaxation, solver code, verification, disambiguation, LLM for unknown problems, and FAISS/ChromaDB are **missing** and should live in the paths above.

---

## B) Minimal Publishable Roadmap (4–6 Milestones)

Concrete milestones that can be implemented inside this repo, with **exact files**, **key functions/classes**, **unit tests**, and **quantitative evaluation**.

---

### Milestone 1: Leak-free evaluation protocol and extra metrics

**Goal:** Train/dev/test split by problem and by query type; add MRR, nDCG, coverage/abstention; no eval set overlap with training.

**Modules/files to add or modify:**

- **Add:** `training/splits.py`  
  - `split_problems_by_source(catalog, seed)` → train_names, dev_names, test_names (e.g. by `source` or by name hash).  
  - `split_queries_by_type(queries_meta)` → train/dev/test by query template or problem family.  
  - `build_splits(catalog, seed, eval_ratio_dev, eval_ratio_test)` → return dict with `train`, `dev`, `test` problem name sets and/or instance lists.

- **Modify:** `training/evaluate_retrieval.py`  
  - Replace `_generate_eval_instances(seed, num_instances)` with: load split from `training/splits.py` so **test** uses only problems (and optionally query types) not seen in train.  
  - Add metrics: **MRR** (mean reciprocal rank of first correct), **nDCG@k** (e.g. k=5), **Coverage** (% of test problems with correct in top‑k), **Abstention** (optional: skip if max score < threshold; report accuracy on non-abstained).  
  - Write results to `data/processed/eval_results.json` (per-split metrics).

- **Add:** `tests/test_splits.py`  
  - `test_split_no_overlap()`: train ∩ test = ∅.  
  - `test_split_reproducibility(seed)`.

- **Quantitative evaluation:** Run `python -m training.evaluate_retrieval --split test` and report P@1, P@5, MRR, nDCG@5, Coverage@5, and optional abstention rate/accuracy.

---

### Milestone 2: Baseline suite (BM25, TF-IDF, MiniLM no-tune, fine-tuned, LLM reranker)

**Goal:** Reproducible baselines for retrieval on the same test set.

**Modules/files to add or modify:**

- **Add:** `retrieval/baselines.py`  
  - `bm25_search(query, catalog, top_k)`: BM25 on `_searchable_text(problem)` (use `rank_bm25` or sklearn `TfidfVectorizer` + cosine).  
  - `tfidf_search(query, catalog, top_k)`: TF-IDF + cosine similarity.  
  - `minilm_search(query, catalog, model, top_k)`: current `search()` with `_default_model_path()` base model (no fine-tune).  
  - `finetuned_search(query, catalog, model, top_k)`: current `search()` with fine-tuned model when present.  
  - `llm_rerank(query, top_k_problems, model_name)`: optional LLM reranker: given query + list of problem names/descriptions, return reordered list (e.g. via API or local LLM).

- **Modify:** `training/evaluate_retrieval.py` (or **add** `training/run_baselines.py`):  
  - CLI: `--baseline bm25|tfidf|minilm|finetuned|llm_rerank`.  
  - Run same test instances through selected baseline; output same metrics (P@1, P@5, MRR, nDCG@5).

- **Add:** `tests/test_baselines.py`  
  - `test_bm25_returns_top_k()`, `test_tfidf_returns_top_k()`, `test_minilm_vs_finetuned_different_or_same()` (smoke test).

- **Quantitative evaluation:** Table: P@1, P@5, MRR, nDCG@5 for each baseline on the **test** split defined in Milestone 1.

---

### Milestone 3: Safe use of `user_queries.jsonl` in evaluation

**Goal:** Use real user queries for eval without leaking into training; optional human labels for a small held-out set.

**Modules/files to add or modify:**

- **Add:** `data/processed/user_queries_eval_schema.json` (or doc in `training/README.md`): schema for eval records: `query`, `expected_problem_name` or `expected_problem_id`, optional `annotator`, `split` (dev/test).  
  - **Rule:** Any row in `user_queries.jsonl` that is used for training (e.g. via `collected_queries_to_pairs`) must **not** appear in the eval set.  
  - **Add:** `training/user_queries_to_eval.py`: script that copies a subset of `user_queries.jsonl` to `data/processed/user_queries_eval.jsonl` and optionally adds `expected_problem_name` (manual or from top-1 at collection time with a “gold” flag).  
  - **Modify:** `training/evaluate_retrieval.py`: if `--eval-file data/processed/user_queries_eval.jsonl` and file has `expected_problem_name`, run evaluation on that file; ensure this file is **excluded** from training data (document in README).

- **Add:** `tests/test_user_queries_eval.py`  
  - `test_eval_schema_has_expected_name()`, `test_no_overlap_with_training_pairs()` (mock: ensure script that builds eval set doesn’t include training pair IDs).

- **Quantitative evaluation:** Report P@1, P@5, MRR on `user_queries_eval.jsonl` (when available) in addition to synthetic test set; note in paper “real user queries, held-out from training.”

---

### Milestone 4: Solver-ready code generation (Pyomo first)

**Goal:** For each retrieved problem, output runnable Pyomo code (skeleton or template-based).

**Modules/files to add or modify:**

- **Add:** `formulation/solver_code.py`  
  - `generate_pyomo(problem: dict) -> str`: from `problem["formulation"]` (variables, objective, constraints) and optional `formulation_latex`, produce Pyomo code string (concrete model with placeholder data).  
  - `get_solver_code(problem: dict, backend="pyomo")`: dispatcher; later add `pulp`, `gurobi`.

- **Modify:** `retrieval/search.py`  
  - Optional: add `solver_code_pyomo(problem)` in `format_problem_and_ip()` or a new `format_problem_ip_and_code()`.  
  Or **modify:** `app.py`: after displaying formulation, show “Solver code (Pyomo)” in a collapsible block (call `formulation.solver_code.generate_pyomo(problem)`).

- **Add:** `tests/test_solver_code.py`  
  - `test_generate_pyomo_returns_string()`, `test_pyomo_contains_model_and_solve()` (regex or AST check); optional: run Pyomo with tiny data and assert no exception.

- **Quantitative evaluation:** (1) For 20 hand-picked problems, run generated code and check it builds/solves. (2) Report “syntax valid %” and “runs without error %” in a table.

---

### Milestone 5: LP relaxation (display + optional derivation)

**Goal:** Show LP relaxation in UI and optionally derive it from ILP when not in catalog.

**Modules/files to add or modify:**

- **Add:** `formulation/lp_relaxation.py`  
  - `get_lp_relaxation(problem: dict) -> dict`: if `problem.get("lp_relaxation")` exists, return it; else derive from `formulation` (e.g. replace binary/integer domains with continuous bounds) and return structured dict + optional `formulation_latex_lp`.  
  - `derive_lp_from_ilp(formulation: dict) -> dict`: simple rule-based (e.g. all binary → [0,1], integer → continuous with same bounds).

- **Modify:** `retrieval/search.py`: `format_problem_and_ip()` to accept optional `include_lp=True` and append LP block (from `get_lp_relaxation(problem)`).  
  Or **modify:** `app.py`: second collapsible “LP relaxation” with output of `get_lp_relaxation(problem)`.

- **Add:** `tests/test_lp_relaxation.py`  
  - `test_derived_lp_has_continuous_domains()`, `test_knapsack_lp_relaxation_bounds()` (known case).

- **Quantitative evaluation:** (1) For classic problems with known LP (e.g. Knapsack), compare derived relaxation to reference. (2) Report “LP shown %” (catalog has or derived) in UI.

---

### Milestone 6: Lightweight validation hooks (formulation + code)

**Goal:** Automatic checks on formulation structure and generated code; integrate into app and retrieval.

**Modules/files to add or modify:**

- **Add:** `formulation/verify.py`  
  - `verify_formulation(problem: dict) -> tuple[bool, list[str]]`: check required keys (`variables`, `objective`, `constraints`), non-empty, variable refs in objective/constraints; return (ok, list of error messages).  
  - `verify_pyomo_syntax(code: str) -> tuple[bool, str]`: run Pyomo parser or `ast.parse` on extracted Python block; return (ok, error message).

- **Modify:** `app.py`: after retrieval, call `verify_formulation(problem)` for each top result; if not ok, append “⚠ Validation: …” to the displayed block. After generating solver code (Milestone 4), call `verify_pyomo_syntax(code)` and show “✓ Syntax OK” or “⚠ Syntax error: …”.  
- **Modify:** `retrieval/search.py`: optional `validate=True` in `search()` or in a wrapper: before returning, run `verify_formulation(problem)` and attach `_validation_errors` to problem dict (or filter out invalid if desired).

- **Add:** `tests/test_verify.py`  
  - `test_verify_formulation_valid()`, `test_verify_formulation_missing_objective()`, `test_verify_pyomo_syntax_valid()`, `test_verify_pyomo_syntax_invalid()`.

- **Quantitative evaluation:** Run verification on full catalog; report “% valid formulation” and “% generated code with valid syntax.”

---

## C) Evaluation Redesign — Leak-free Protocol

### C.1 Train / dev / test split strategy

- **By problem (primary):**  
  - Split **problems** into train / dev / test by **source** or **name** (e.g. 70/15/15) so that no problem appears in more than one split.  
  - **Training pairs:** only (query, passage) where the passage’s problem is in **train**.  
  - **Dev:** evaluate retrieval on (query, expected_problem) where expected_problem ∈ dev set; use for hyperparameters and model selection.  
  - **Test:** same as dev but problems ∈ test set; report final metrics only on test.  
  - **Implementation:** `training/splits.py` (see Milestone 1); `training/train_retrieval.py` to filter `load_pairs()` by train problem names; `training/evaluate_retrieval.py` to load dev/test splits and evaluate only on the corresponding instances.

- **By query type (optional):**  
  - Within each problem split, tag queries by “template” (e.g. from `generate_queries_for_problem` template list in `training/generate_samples.py`).  
  - Ensure dev/test include all query types; optionally stratify so that train/dev/test have similar template distribution.  
  - **Implementation:** in `training/generate_samples.py`, optionally add a `query_type` or `template_id` to each generated query; in `_generate_eval_instances` (or replacement in `evaluate_retrieval.py`), attach this and use in split construction in `training/splits.py`.

### C.2 Incorporating `user_queries.jsonl` safely

- **Rule:** Rows in `user_queries.jsonl` used for **training** (e.g. converted to pairs via `training/collected_queries_to_pairs.py`) must be **excluded** from evaluation.  
- **Process:**  
  1. Reserve a random subset (e.g. 10–20%) of collected queries **before** adding to training data; write these to `data/processed/user_queries_eval.jsonl` with schema `query`, `expected_problem_name`, `split` (dev/test).  
  2. Manually label `expected_problem_name` for eval rows (or use “top-1 at collection time” only for ablations, clearly stated).  
  3. In `evaluate_retrieval.py`, support `--eval-file data/processed/user_queries_eval.jsonl` and compute the same metrics (P@1, P@5, MRR, nDCG@5) on this set.  
  4. Document in `training/README.md`: “Eval set is held out from training; do not use user_queries_eval.jsonl for training.”

### C.3 Metrics beyond P@1 / P@5

- **MRR (Mean Reciprocal Rank):** 1/rank of first correct answer; average over instances. Implement in `training/evaluate_retrieval.py` (or `training/metrics.py`).  
- **nDCG@k (e.g. k=5):** relevance = 1 for correct problem, 0 else; DCG/IDCG. Implement in `training/metrics.py` and call from `evaluate_retrieval.py`.  
- **Coverage@k:** fraction of test instances where the correct problem appears in top‑k.  
- **Calibration / abstention:** If max similarity score < threshold τ, abstain; report **abstention rate** and **accuracy on non-abstained** only. Implement as optional in `evaluate_retrieval.py` and `retrieval/search.py` (return confidence/score with each result).

### C.4 Baseline suite

- **BM25:** `retrieval/baselines.py` — `bm25_search()`; evaluate with `--baseline bm25`.  
- **TF-IDF:** `retrieval/baselines.py` — `tfidf_search()`; `--baseline tfidf`.  
- **MiniLM no-tune:** current embedding model (no fine-tuning); `--baseline minilm`.  
- **Fine-tuned:** current pipeline with `data/models/retrieval_finetuned/final`; `--baseline finetuned`.  
- **LLM reranker:** top‑k from MiniLM/fine-tuned, then rerank with LLM; `--baseline llm_rerank`.  
- **Implementation:** `retrieval/baselines.py` (see Milestone 2); `training/run_baselines.py` or flags in `evaluate_retrieval.py` to run and log metrics for each baseline on the same test split.

### C.5 Where to implement (exact paths)

| Component | File path |
|-----------|-----------|
| Split logic (train/dev/test by problem and optional query type) | `training/splits.py` (new) |
| Metrics (MRR, nDCG@k, Coverage@k, abstention) | `training/metrics.py` (new) or inside `training/evaluate_retrieval.py` |
| Eval driver (load split, run retrieval, compute metrics) | `training/evaluate_retrieval.py` (modify) |
| User-queries eval schema and exclusion from training | `training/user_queries_to_eval.py` (new); `training/README.md` (doc) |
| Baselines (BM25, TF-IDF, MiniLM, fine-tuned, LLM reranker) | `retrieval/baselines.py` (new) |
| Entrypoint for baseline runs | `training/run_baselines.py` (new) or `evaluate_retrieval.py --baseline ...` |

---

## D) Validation Hooks — Lightweight Verifier and Integration

### D.1 Verifier design

- **Formulation verifier:**  
  - Input: one `problem` dict (from catalog or LLM).  
  - Checks: `formulation` exists; `variables` non-empty; `objective` has `sense` and `expression`; `constraints` is a list; variable symbols referenced in objective/constraints exist in `variables`.  
  - Output: `(is_valid: bool, errors: list[str])`.  
  - **Location:** `formulation/verify.py` — function `verify_formulation(problem)`.

- **Generated code verifier (partial):**  
  - Input: solver code string (e.g. Pyomo).  
  - Checks: (1) Python syntax (`ast.parse`); (2) optional: Pyomo-specific (e.g. `pyomo.environ` import, `ConcreteModel()`, `solve()`).  
  - Output: `(syntax_ok: bool, message: str)`.  
  - **Location:** `formulation/verify.py` — function `verify_pyomo_syntax(code)` (or `verify_solver_code(code, backend="pyomo")`).

### D.2 Integration points

- **In `app.py`:**  
  - After `search()` returns (around L84–93), for each problem in `results`, call `verify_formulation(problem)`. If not valid, append to the displayed block a short line: “⚠ Formulation validation: …” with the first error.  
  - When solver code is implemented (Milestone 4), after generating code call `verify_pyomo_syntax(code)` and show “✓ Solver code syntax OK” or “⚠ Solver code: …” under the code block.  
  - **Exact location:** inside `answer()`, after building `out` list (e.g. after `format_problem_and_ip`), add a loop that calls `verify_formulation(problem)` and appends validation note to `out`.

- **In `retrieval/search.py`:**  
  - Option A (lightweight): add an optional parameter `validate: bool = False` to `search()`. When True, after building the result list, for each `(problem, score)` run `verify_formulation(problem)` and attach to `problem` a key `_validation` = `{"ok": bool, "errors": list}`; do not filter out, let the caller (e.g. app) display.  
  - Option B: add a wrapper function `search_with_validation(query, catalog, model, top_k, validate=True)` in `search.py` that calls `search()` then runs `verify_formulation` on each result and returns results plus validation info.  
  - **Exact location:** `retrieval/search.py`: extend `search()` signature (e.g. after `top_k`) with `validate=False`; after the line that builds `results` (list of (problem, score)), if validate: for each problem call `verify_formulation(problem)` and store in `problem["_validation"]`.  
  - Ensure `format_problem_and_ip` does not break if `problem` has extra keys like `_validation`; optionally in `format_problem_and_ip`, if `problem.get("_validation", {}).get("ok") is False`, append validation errors to the formatted string.

### D.3 Summary

- **New file:** `formulation/verify.py` with `verify_formulation(problem)` and `verify_pyomo_syntax(code)`.  
- **Modify:** `app.py` — in `answer()`, after formatting each result, call verifier and append validation message.  
- **Modify:** `retrieval/search.py` — optional `validate` in `search()` and attach `_validation` to each problem; or add `search_with_validation()` that calls `search()` and then verifies.

This gives a minimal, automatic check that can be extended later (e.g. instance-based verification or LP feasibility).

---

*End of audit. All references are to the repository state at the time of the audit; line numbers may shift slightly after edits.*
