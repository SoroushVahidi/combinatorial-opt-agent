# NLP4LP Focused Evaluation — Downstream Methods

Reproducible bundle for evaluating the **evidence-supported** downstream methods. The default pipeline runs **4 methods**; experimental/archived methods can be re-enabled with `--experimental`.

## Methods (same split, variant=orig by default)

**Default (4 methods):**

| Method | Retrieval | Assignment |
|--------|-----------|------------|
| tfidf_acceptance_rerank | TF-IDF + acceptance rerank | typed |
| tfidf_hierarchical_acceptance_rerank | TF-IDF + hierarchical acceptance rerank | typed |
| tfidf_optimization_role_repair | TF-IDF | optimization_role_repair |
| tfidf_optimization_role_relation_repair | TF-IDF | optimization_role_relation_repair |

**Experimental/archived (not in default pipeline; use `--experimental` to include):**  
`tfidf_optimization_role_anchor_linking`, `tfidf_optimization_role_bottomup_beam_repair`, `tfidf_optimization_role_entity_semantic_beam_repair` — available in git and via CLI for reproducibility; aggregate metrics did not improve over opt_repair/relation_repair.

## Entrypoints

1. **Run focused methods and produce side-by-side summary**  
   `tools/run_nlp4lp_focused_eval.py` [--variant orig] [--experimental] [--safe]  
   - Default: runs the 4 evidence-supported methods. With `--experimental`, runs all 7 (adds anchor_linking, bottomup_beam, entity_semantic_beam).  
   - Invokes `nlp4lp_downstream_utility.py` once per method.  
   - Reads `results/paper/nlp4lp_downstream_summary.csv`, filters to run baselines, writes `results/paper/nlp4lp_focused_eval_summary.csv`.

2. **Per-instance comparison**  
   `tools/build_nlp4lp_per_instance_comparison.py` [--variant orig] [--experimental] [--safe]  
   - Default: runs only **opt_repair** and **relation_repair** per instance; output CSV has columns for those two only.  
   - With `--experimental`: also runs anchor_linking, bottomup_beam, entity_semantic_beam and adds those columns.  
   - Uses TF-IDF retrieval (same for all). Writes `results/paper/nlp4lp_focused_per_instance_comparison.csv`.

3. **Disagreement labels**  
   `tools/analyze_nlp4lp_downstream_disagreements.py`  
   - Reads the per-instance comparison, finds rows where pred_opt_repair ≠ pred_relation_repair.  
   - Assigns heuristic labels: objective_vs_bound, lower_vs_upper_bound, total_vs_per_unit, percent_ratio_confusion, wrong_variable_association, other.  
   - Writes `results/paper/nlp4lp_focused_disagreement_labels.csv`.

4. **Failure audit**  
   `tools/build_nlp4lp_failure_audit.py`  
   - Reads the per-instance comparison (and optionally disagreement labels).  
   - Produces: `nlp4lp_downstream_failure_patterns.csv` (failure families, counts, opt_wrong_relation_right vs both_wrong, representative query_ids), `nlp4lp_downstream_hard_cases.csv` (schema_hit=1, both exact=0), `nlp4lp_downstream_failure_audit.md` (ranked families, missing signals, recommended next change).

## Metrics in outputs

- **Schema R@1** (retrieval): in summary as `schema_R1`.  
- **InstantiationReady**: in summary as `instantiation_ready`; in per-instance as `inst_ready_opt_repair`, `inst_ready_relation_repair` (and, with `--experimental`, `inst_ready_anchor`, `inst_ready_beam`, `inst_ready_entity_semantic_beam`).  
- **Exactness**: in summary as `exact5_on_hits`, `exact20_on_hits`; in per-instance as `exact_opt_repair`, `exact_relation_repair` (and experimental columns when built with `--experimental`).  
- **param_coverage, type_match, key_overlap**: in summary CSV.

## Commands (Wulver) — use compute nodes

Run from project root. Prefer **compute nodes** (sbatch or interactive srun) to avoid login-node fork limits.

```bash
# Submit to compute nodes (recommended)
bash jobs/submit_nlp4lp_focused_eval.sh
# or directly:
sbatch jobs/run_nlp4lp_focused_eval.slurm
```

Or run step by step (e.g. in an interactive compute session):

```bash
cd /path/to/combinatorial-opt-agent
source venv/bin/activate   # or: module load python/3.10

# 1. Run 4 methods + build focused summary CSV (default)
python tools/run_nlp4lp_focused_eval.py --variant orig
# Or with --safe for low-resource; or --experimental for all 7 methods

# 2. Per-instance comparison (default: opt_repair + relation_repair only)
python tools/build_nlp4lp_per_instance_comparison.py --variant orig
# Add --experimental to include anchor_linking, bottomup_beam, entity_semantic_beam columns

# 3. Disagreement analysis
python tools/analyze_nlp4lp_downstream_disagreements.py

# 4. Failure audit (patterns, hard cases, report)
python tools/build_nlp4lp_failure_audit.py
```

## Expected output paths

| Path | Description |
|------|-------------|
| `results/paper/nlp4lp_downstream_summary.csv` | Updated with rows for the run baselines (4 by default, 7 with --experimental). |
| `results/paper/nlp4lp_focused_eval_summary.csv` | Side-by-side summary: variant, baseline, schema_R1, param_coverage, type_match, key_overlap, exact5_on_hits, exact20_on_hits, instantiation_ready, n. |
| `results/paper/nlp4lp_downstream_per_query_orig_<baseline>.csv` | Per-query metrics per baseline. |
| `results/paper/nlp4lp_downstream_orig_<baseline>.json` | Aggregate JSON per baseline. |
| `results/paper/nlp4lp_focused_per_instance_comparison.csv` | One row per instance. Default columns: query_id, gold_doc_id, pred_doc_id, schema_hit, mentions_summary, gold_assignments, pred_opt_repair, pred_relation_repair, n_expected_scalar, n_filled_opt/relation, exact_opt_repair, exact_relation_repair, inst_ready_*. With --experimental build: also pred_anchor_linking, pred_bottomup_beam, pred_entity_semantic_beam and their exact/inst_ready columns. |
| `results/paper/nlp4lp_focused_disagreement_labels.csv` | Rows where the two repair methods disagree: query_id, gold_doc_id, pred_doc_id, schema_hit, disagreement_labels, n_filled_opt, n_filled_relation. |
| `results/paper/nlp4lp_downstream_failure_patterns.csv` | Failure families: failure_family, count, count_schema_hit, count_opt_wrong_relation_right, count_both_wrong, representative_query_ids, short_explanation. |
| `results/paper/nlp4lp_downstream_hard_cases.csv` | Hard cases (schema_hit=1, opt and relation exact=0). Columns include pred_opt_repair, pred_relation_repair, likely_failure_family, brief_reason; experimental pred/exact columns only if present in per-instance CSV. |
| `results/paper/nlp4lp_downstream_failure_audit.md` | Report: top failure families, missing signals, single highest-value next deterministic change. |
| `logs/nlp4lp_focused_eval_<jobid>.out` | SLURM stdout when using the job script. |

## Assumptions and dependencies

- **Data:** `data/processed/nlp4lp_eval_orig.jsonl` and `data/catalogs/nlp4lp_catalog.jsonl` exist.  
- **Hugging Face:** Gold schema/parameters are loaded from `udell-lab/NLP4LP` (split=test). Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` if the dataset is gated.  
- **Python:** Same as main project (`requirements.txt`); `retrieval.baselines` and `tools.nlp4lp_downstream_utility` must be importable.  
- **No modeling changes:** Only evaluation, comparison, and artifact scripts; downstream utility is run as-is.  
- **Wulver:** Run on a compute node (e.g. `sbatch` or `srun`) to avoid login-node resource limits.
