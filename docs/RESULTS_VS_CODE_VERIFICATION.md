# Verification: Result Files vs Latest Code State

**Purpose:** Check whether current summary/result files reflect the latest code and whether paper tables are stale relative to the full summaries.

**Method:** Content comparison (exact values in paper tables vs current summary), script logic (what reads what, what writes what). Timestamps could not be reliably obtained in this environment (shell `stat`/`ls` failed); recommend re-running timestamp checks locally.

---

## 1. Summary of findings

| Finding | Severity |
|--------|----------|
| **Paper tables (orig) are stale subsets of the current summary** | **High** |
| **Paper tables omit constrained/repair baselines by design** | Medium (documented) |
| **Artifact script reads current summary when run; tables are only updated when script is run** | Design |
| **Retrieval summary vs retrieval main table** | Needs value check if retrieval was re-run |

---

## 2. Downstream: paper tables vs current summary

**Source of truth:** `results/paper/nlp4lp_downstream_summary.csv` is updated by `tools/nlp4lp_downstream_utility.py` (`run_one()` → `_upsert_summary_row()`) every time a downstream run completes (e.g. `--variant orig --baseline tfidf`).

**Paper tables:** These are **written only when** `tools/make_nlp4lp_paper_artifacts.py` is run. They read `results/paper/nlp4lp_downstream_summary.csv` at that time and write:

- `nlp4lp_downstream_final_table_orig.csv` / `.tex` — baselines `["random", "lsa", "bm25", "tfidf", "oracle", "tfidf_untyped", "oracle_untyped"]` (line 299).
- `nlp4lp_downstream_main_table_orig.csv` / `.tex` — same 7 baselines (line 1000).
- `nlp4lp_downstream_section_table.csv` / `.tex` — baselines `["random", "lsa", "bm25", "tfidf", "oracle"]` (line 523).

**Value comparison (orig, tfidf):**

| Metric | Current summary (`nlp4lp_downstream_summary.csv`) | Paper tables (final_table_orig, main_table_orig, section_table) |
|--------|--------------------------------------------------|------------------------------------------------------------------|
| exact5_on_hits | 0.20531573278052154 | **0.1876** |
| exact20_on_hits | 0.23302949007174353 | **0.2140** |
| type_match | 0.22596988442909896 | 0.2267 (close; 0.2267 could be from earlier run) |
| param_coverage | 0.8221793563938585 | 0.8222 (rounded match) |

**Conclusion:** The numbers in the paper tables **do not** match the current summary for at least `exact5_on_hits` and `exact20_on_hits` for orig/tfidf. So either:

1. The summary was updated **after** the last run of `make_nlp4lp_paper_artifacts.py` (e.g. by re-running downstream for orig/tfidf), or  
2. The summary was modified by some other process.

In both cases, **the paper tables are stale**: they reflect an older state of the summary. Re-running `python tools/make_nlp4lp_paper_artifacts.py` will overwrite them from the current summary and bring them in line.

---

## 3. Paper tables as intentional subsets

The artifact script **hardcodes** the list of baselines for the “final” and “main” orig tables (lines 299 and 1000):

```python
order = ["random", "lsa", "bm25", "tfidf", "oracle", "tfidf_untyped", "oracle_untyped"]
```

So **tfidf_constrained**, **oracle_constrained**, **tfidf_acceptance_rerank**, **tfidf_hierarchical_acceptance_rerank**, **tfidf_semantic_ir_repair**, **tfidf_optimization_role_repair**, **oracle_semantic_ir_repair**, **oracle_optimization_role_repair**, and **random_untyped** are **never** written to the final/main table CSVs even though they exist in the full summary. This is a design choice (subset for the main paper table), but it means:

- The “main” or “final” table is not a complete view of the current results.
- If you want constrained/repair in the main table, the script must be changed (add those baseline names to `order` or to a separate table).

---

## 4. Data flow (for re-verification)

1. **Downstream runs**  
   `python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf [--assignment-mode ...]`  
   → Updates **only** the corresponding row(s) in `results/paper/nlp4lp_downstream_summary.csv` (and types summary, per-query files).  
   → Does **not** run `make_nlp4lp_paper_artifacts.py`.

2. **Paper artifact generation**  
   `python tools/make_nlp4lp_paper_artifacts.py`  
   → Reads **current** `results/nlp4lp_retrieval_summary.csv`, `results/nlp4lp_stratified_metrics.csv`, and `results/paper/nlp4lp_downstream_summary.csv` (and types summary).  
   → Overwrites all paper CSVs/TeX/plots in `results/paper/` (e.g. final_table_orig, main_table_orig, section_table, retrieval_main_table, etc.).

So: **any change in the downstream summary is reflected in paper tables only after re-running `make_nlp4lp_paper_artifacts.py`.**

---

## 5. Retrieval summary vs paper retrieval table

- **Retrieval metrics:** Produced by `training/run_baselines.py` with `--out results/nlp4lp_retrieval_metrics_<variant>.json`; then `tools/summarize_nlp4lp_results.py` aggregates to `results/nlp4lp_retrieval_summary.csv`.
- **Paper retrieval table:** `make_nlp4lp_paper_artifacts.py` reads `results/nlp4lp_retrieval_summary.csv` and writes `results/paper/nlp4lp_main_table.csv` and `results/paper/nlp4lp_retrieval_main_table.csv` (and TeX).

If retrieval baselines or summarization were re-run after the last artifact run, the retrieval paper tables could also be stale. A quick check: compare a value from `results/nlp4lp_retrieval_summary.csv` (e.g. orig tfidf Recall@1) with the same value in `results/paper/nlp4lp_retrieval_main_table.csv` (after rounding). If they match, retrieval tables are likely in sync; if not, re-run the artifact script.

---

## 6. Recommended timestamp checks (run locally)

From project root:

```bash
# Scripts that produce or consume results
ls -la tools/nlp4lp_downstream_utility.py tools/make_nlp4lp_paper_artifacts.py tools/summarize_nlp4lp_results.py training/run_baselines.py training/external/build_nlp4lp_benchmark.py

# Main result files
ls -la results/nlp4lp_retrieval_summary.csv results/paper/nlp4lp_downstream_summary.csv
ls -la results/paper/nlp4lp_downstream_final_table_orig.csv results/paper/nlp4lp_downstream_section_table.csv results/paper/nlp4lp_retrieval_main_table.csv
```

**Interpretation:**

- If **downstream_summary.csv** is **newer** than **final_table_orig.csv** (or section_table, main_table_orig), then paper tables are stale and should be regenerated.
- If any **producer script** (e.g. `nlp4lp_downstream_utility.py`) is **newer** than the **summary** it writes to, then the summary may be stale relative to code (re-run the downstream runs that update that summary).

---

## 7. Concrete actions

1. **Regenerate paper tables from current summary**  
   Run: `python tools/make_nlp4lp_paper_artifacts.py` (from project root).  
   This will refresh `nlp4lp_downstream_final_table_orig.csv`, `nlp4lp_downstream_main_table_orig.csv`, `nlp4lp_downstream_section_table.csv`, and all other artifact CSVs/TeX/plots from the current retrieval and downstream summaries.  
   *Note: The agent runs on the login node and cannot fork; to regenerate on HPC, use a compute node. A SLURM job is provided: `sbatch jobs/run_paper_artifacts.slurm` (submit from project root, or from an `srun --pty bash` session if the login node blocks sbatch). Output: `logs/paper_artifacts_<jobid>.out`.*

2. **Optional: include constrained/repair in main orig table**  
   In `tools/make_nlp4lp_paper_artifacts.py`, extend the `order` list (e.g. around lines 299 and 1000) to include `tfidf_constrained`, `oracle_constrained`, and optionally one repair variant, or add a separate “extended” table that includes them.

3. **After any change to downstream or retrieval code**  
   Re-run the relevant pipeline (downstream runs and/or retrieval baselines + summarization), then re-run `make_nlp4lp_paper_artifacts.py` so paper tables and the summary stay in sync.

---

## 8. Files inspected

| File | Role |
|------|------|
| `results/paper/nlp4lp_downstream_summary.csv` | Current full downstream summary (source of truth for downstream) |
| `results/paper/nlp4lp_downstream_final_table_orig.csv` | Subset table (7 baselines); values stale vs summary |
| `results/paper/nlp4lp_downstream_main_table_orig.csv` | Same subset, same stale values |
| `results/paper/nlp4lp_downstream_section_table.csv` | 5-baseline subset; same stale values for shared baselines |
| `tools/make_nlp4lp_paper_artifacts.py` | Reads summaries, writes all paper tables/TeX/plots (lines 289–340, 514–569, 998–1044) |
