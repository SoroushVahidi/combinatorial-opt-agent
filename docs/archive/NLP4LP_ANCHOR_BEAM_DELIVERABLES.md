# NLP4LP Anchor-Linking and Bottom-Up Beam — Deliverables

## 1. Exact files changed

| File | Changes |
|------|--------|
| `tools/nlp4lp_downstream_utility.py` | Added `ANCHOR_WEIGHTS`, `_slot_profile_tokens`, `_score_mention_slot_anchor` (context-aware + entity alignment + edge pruning). Extended `_opt_role_global_assignment` with optional `score_matrix`. Added `_run_optimization_role_anchor_linking`, `_run_optimization_role_bottomup_beam_repair`. Registered `optimization_role_anchor_linking` and `optimization_role_bottomup_beam_repair` in `--assignment-mode` and `effective_baseline`. Added two `elif assignment_mode == ...` blocks in `run_one()` to fill metrics for the new methods. |
| `tools/run_nlp4lp_focused_eval.py` | Added `tfidf_optimization_role_anchor_linking` and `tfidf_optimization_role_bottomup_beam_repair` to `FOCUSED_BASELINES` and `BASELINE_ASSIGNMENT`. Updated docstring to list 6 methods. |
| `tools/build_nlp4lp_per_instance_comparison.py` | Imported `_run_optimization_role_anchor_linking` and `_run_optimization_role_bottomup_beam_repair`. Added columns: `pred_anchor_linking`, `pred_bottomup_beam`, `n_filled_anchor`, `n_filled_beam`, `exact_anchor`, `exact_beam`, `inst_ready_anchor`, `inst_ready_beam`. Per instance: call both new runners and compute exact/inst_ready. |
| `tools/build_nlp4lp_failure_audit.py` | Extended hard_cases CSV columns to include `pred_anchor_linking`, `pred_bottomup_beam`, `exact_anchor`, `exact_beam` when present in comparison CSV. |
| `docs/NLP4LP_FOCUSED_EVAL.md` | Updated methods table and entrypoints to include anchor_linking and bottomup_beam; updated output path descriptions and metric names. |
| `docs/NLP4LP_ANCHOR_AND_BEAM_IMPLEMENTATION_PLAN.md` | New file: audit summary and implementation plan. |
| `docs/NLP4LP_ANCHOR_BEAM_DELIVERABLES.md` | This file. |

## 2. Exact commands to run on Wulver

From project root (prefer compute nodes, e.g. `sbatch` or interactive `srun`):

```bash
# 1. Run all 6 focused methods (writes summary + per-query CSVs/JSONs)
python tools/run_nlp4lp_focused_eval.py --variant orig

# 2. Build per-instance comparison (opt, relation, anchor, beam)
python tools/build_nlp4lp_per_instance_comparison.py --variant orig

# 3. Label disagreements (opt vs relation)
python tools/analyze_nlp4lp_downstream_disagreements.py

# 4. Failure audit (patterns, hard cases, report)
python tools/build_nlp4lp_failure_audit.py
```

Or submit the single Slurm job that runs steps 1–4:

```bash
bash jobs/submit_nlp4lp_focused_eval.sh
# or: sbatch jobs/run_nlp4lp_focused_eval.slurm
```

Run a single baseline (e.g. anchor_linking or beam) only:

```bash
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_anchor_linking
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_bottomup_beam_repair
```

## 3. Expected output paths

| Path | Description |
|------|-------------|
| `results/paper/nlp4lp_downstream_summary.csv` | Updated with rows for all baselines (including anchor_linking, bottomup_beam). |
| `results/paper/nlp4lp_focused_eval_summary.csv` | Side-by-side: variant, baseline, schema_R1, param_coverage, type_match, key_overlap, exact5_on_hits, exact20_on_hits, instantiation_ready, n (6 rows for focused baselines). |
| `results/paper/nlp4lp_downstream_per_query_orig_tfidf_optimization_role_anchor_linking.csv` | Per-query metrics for anchor_linking. |
| `results/paper/nlp4lp_downstream_per_query_orig_tfidf_optimization_role_bottomup_beam_repair.csv` | Per-query metrics for bottomup_beam. |
| `results/paper/nlp4lp_focused_per_instance_comparison.csv` | Per-instance: gold_assignments, pred_opt_repair, pred_relation_repair, pred_anchor_linking, pred_bottomup_beam, n_filled_*, exact_*, inst_ready_* for all four downstream methods. |
| `results/paper/nlp4lp_focused_disagreement_labels.csv` | Disagreements between opt_repair and relation_repair with heuristic labels. |
| `results/paper/nlp4lp_downstream_failure_patterns.csv` | Failure families (counts, opt_wrong_relation_right, both_wrong, representative query_ids). |
| `results/paper/nlp4lp_downstream_hard_cases.csv` | schema_hit=1, exact_opt=0, exact_relation=0; includes pred_anchor_linking, pred_bottomup_beam, exact_anchor, exact_beam. |
| `results/paper/nlp4lp_downstream_failure_audit.md` | Report: top failure families, missing signals, recommended next change. |

## 4. Runtime tradeoffs

- **optimization_role_repair**: Bipartite matching (scipy or DP). Same as before.
- **optimization_role_relation_repair**: Greedy incremental with admissibility. Same as before.
- **optimization_role_anchor_linking**: Same pipeline as repair but with anchor scoring (one extra pass over mention×slot for `_score_mention_slot_anchor`). Slightly more work per pair than `_score_mention_slot_opt` (entity alignment, profile overlap, edge pruning). Same assignment step (bipartite). **Modest increase** in CPU per query.
- **optimization_role_bottomup_beam_repair**: Builds score matrix (anchor or opt), then beam over partial bundles. Beam width default 5; at each step O(beam × atoms) extensions, with admissibility check per extension. **Larger increase** than anchor_linking when many mentions/slots (beam × |atoms| × steps). Still deterministic and no external APIs.

## 5. Which failure families each new method is most likely to improve

| Failure family | Anchor-linking | Bottom-up beam |
|----------------|----------------|----------------|
| **lower_vs_upper_bound** | Yes (operator compatibility + edge pruning: min↔min, max↔max; penalty for min→upper-only slot). | Yes (relation bonus and admissibility reduce wrong min/max pairing). |
| **objective_vs_bound** | Yes (fragment/slot objective vs bound alignment; edge pruning and operator compat). | Yes (bundles that mix objective and bound incorrectly can be outscored). |
| **total_vs_per_unit** | Yes (total/total-like and per-unit/per-unit alignment; edge pruning total↔per-unit). | Yes (admissibility and relation consistency favor coherent total vs per-unit sets). |
| **percent_ratio_confusion** | Yes (percent mention → non-percent slot pruning; percent/ratio slot alignment). | Yes (type and role in scoring; admissibility). |
| **wrong_variable_association** | Yes (entity alignment: mention context + slot/alias overlap). | Yes (beam keeps multiple partial assignments; can prefer the one with better entity alignment). |
| **multiple_float_like_values** | Partially (better local grounding reduces some swaps). | Yes (beam explores alternative full assignments; can recover from one wrong commitment). |
| **currency_vs_scalar_confusion** | Partially (type and unit compatibility in base score). | Partially (same scoring; beam can still swap). |
| **missing_operator_grounding** | Yes (operator compatibility and edge pruning tie operators to slots). | Yes (admissibility and relation bonus reinforce operator–slot consistency). |

## 6. Current codebase: assignment strategy classification

- **optimization_role_repair**: **Bipartite matching** (maximum-weight assignment via scipy `linear_sum_assignment` or, if scipy missing, DP over slot subset mask). One-to-one; then validate-and-repair for unfilled slots.
- **optimization_role_relation_repair**: **Greedy incremental** (slot order by role/alias size; for each slot pick best mention such that partial stays admissible; add relation bonus). No-reuse via `used_mention_ids`.
- **optimization_role_anchor_linking**: **Bipartite matching** with anchor-based scores (same max-weight assignment, different score matrix).
- **optimization_role_bottomup_beam_repair**: **Constrained search** (beam over partial assignment bundles; compatibility and admissibility constrain combinations; deterministic top-K).

So the repo is a **hybrid**: bipartite (repair, anchor_linking), greedy incremental (relation_repair), and beam search (bottomup_beam), all sharing the same validate-and-repair post-pass and no-reuse semantics where applicable.

## Ablations (optional, programmatic)

- **Anchor-linking without edge pruning**: Call `_run_optimization_role_anchor_linking(..., use_edge_pruning=False)`. Without entity alignment: `use_entity_alignment=False`.
- **Bottom-up beam using old scores only**: Call `_run_optimization_role_bottomup_beam_repair(..., use_anchor_scores=False)` (uses `_score_mention_slot_opt` instead of anchor scores).
- **Bottom-up beam using anchor_linking scores**: Default; `use_anchor_scores=True`.

These are not exposed as separate `--assignment-mode` values or baseline names to keep the CLI simple. To run them in the eval pipeline, add corresponding entries to `BASELINE_ASSIGNMENT` and call the runner with the desired kwargs (or add a thin wrapper that calls the same function with different params and a distinct effective_baseline name).

## 7. Richer constrained maximum-weight assignment as a future extension

Yes. A **constrained maximum-weight assignment** formulation would be a clean next step:

- Keep the same cost/score matrix (from opt, anchor, or a combination).
- Add **linear ordering constraints** (e.g. value(min_slot) ≤ value(max_slot) when both are numeric) or **slot-slot relation constraints** (e.g. total slot and per-unit slot must receive values from the same “group” of mentions).
- Implement via a small **integer/linear program** (e.g. one binary variable per (mention, slot), objective = sum of scores, constraints = one mention per slot, one slot per mention, plus min/max ordering and optional grouping). Solvers: `scipy.optimize.linprog` with binary variables (e.g. via a MIP library) or a dedicated assignment + constraints API.

The current **admissibility** in `_is_partial_admissible` (type, total vs per-unit, min/max tags) could be re-expressed as **hard constraints** in such a formulation; **relation bonuses** could stay as additive terms in the objective or as soft constraints. This would keep the pipeline deterministic and avoid greedy/beam heuristics while still encoding the same signals.
