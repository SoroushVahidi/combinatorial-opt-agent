# Global Consistency Grounding

## Method Intuition

Existing downstream methods either assign mentions greedily (one slot at a time, locally) or apply structured bipartite matching (one-to-one, no global penalty). Both approaches miss an important class of errors: cross-slot inconsistencies, such as assigning a total-budget number to a per-unit coefficient slot when a much better match exists globally.

**Global Consistency Grounding (GCG)** reformulates number-to-slot assignment as a global optimisation problem:

1. **Generate candidates** — for every slot, every numeric mention is a candidate.
2. **Prune** implausible pairs below a local score threshold.
3. **Search** over full assignments using beam search.
4. **Score** each candidate assignment jointly with both local compatibility and global consistency rewards/penalties.
5. **Return** the assignment with the maximum combined score.

The key insight is that the right assignment for *slot A* depends on what is assigned to *slot B*: a percent value should go to the percent slot, not steal it from the total-budget slot even if the local overlap score happens to be slightly higher.

---

## Scoring Components

### Local Score (`_gcg_local_score`, weights in `GCG_LOCAL_WEIGHTS`)

Per-pair score between one numeric mention and one slot.  Reuses the same feature signals as `optimization_role_repair`:

| Feature | Description |
|---|---|
| `type_exact_bonus` | Strong reward when mention type exactly matches slot expected type (e.g. percent→percent) |
| `type_loose_bonus` | Partial reward for loosely compatible types (int↔float↔currency) |
| `type_incompatible_penalty` | Hard blocker for impossible pairs (−∞) |
| `opt_role_overlap` | Per-tag reward for shared optimization-role tags (budget, demand, objective, etc.) |
| `fragment_compat_bonus` | Reward when mention fragment (objective/constraint/resource) aligns with slot role |
| `operator_match_bonus` | Reward for matching min/max operator context |
| `lex_context_overlap` | Lexical overlap between mention context and slot name/aliases |
| `lex_sentence_overlap` | Sentence-level overlap |
| `unit_match_bonus` | Percent, currency, integer marker match |
| `entity_resource_overlap` | Entity / resource token alignment |
| `coefficient_vs_total_bonus` | Reward when total/per-unit status aligns between mention and slot |
| `schema_prior_bonus` | Small constant reward for any feasible pair |
| `weak_match_penalty` | Penalty if all other scores are zero |

### Global Score (`_gcg_global_penalty`, weights in `GCG_GLOBAL_WEIGHTS`)

Applied once to a full (or partial) assignment as a whole:

| Term | Description |
|---|---|
| `coverage_reward_per_slot` | Reward for each filled slot |
| `type_consistency_reward` | Reward per slot where mention type is consistent with expected type |
| `percent_misuse_penalty` | Penalty when a percent mention is assigned to a non-percent slot, *given that a percent mention exists in the query* |
| `non_percent_to_pct_slot_penalty` | Penalty when a non-percent mention is assigned to a percent slot, *given that a percent mention exists* |
| `total_to_coeff_penalty` | Penalty when a total-looking mention goes to a per-unit (coefficient) slot |
| `coeff_to_total_penalty` | Penalty when a per-unit mention goes to a total/budget slot |
| `bound_flip_penalty` | Penalty when a max-tagged mention is assigned to a min slot (or vice versa) |
| `duplicate_mention_penalty` | Hard penalty for using the same mention twice |
| `plausibility_coverage_bonus` | Bonus when ≥ 80 % of slots are filled |

---

## Search Strategy

**Beam search** over partial slot assignments.

- Slots are processed in index order.
- Each beam state is a frozen set of committed `(slot_index, mention_index)` pairs plus the running local score sum.
- At each step, every beam state is extended by: (a) skipping the current slot, or (b) assigning any admissible (above prune threshold, not yet used) mention.
- After each slot, the beam is pruned to the top `GCG_BEAM_WIDTH` states by local score sum.
- After all slots are processed, every surviving state is scored with the global consistency delta.
- The state with the **highest total score** (local sum + global delta) is the output.

This gives an exact solution when the beam covers all relevant states (true for small slot counts ≤ 8–10), and a near-exact solution for larger instances.

Configurable parameters:

| Constant | Default | Description |
|---|---|---|
| `GCG_BEAM_WIDTH` | 8 | Number of states kept in the beam at each step |
| `GCG_PRUNE_THRESHOLD` | −0.5 | Local score below which a mention-slot pair is not considered |
| `GCG_LOCAL_WEIGHTS` | see file | All local scoring weights |
| `GCG_GLOBAL_WEIGHTS` | see file | All global penalty/reward weights |

---

## Limitations

1. **Context window is sentence-local.** The mention extractor uses a ±14 token context window, so mentions far from relevant context tokens may receive weak role tags.
2. **Written numbers.** The extractor handles common English number words but may miss complex multi-word numbers (e.g. "one hundred and twenty-five").
3. **Sentence segmentation.** The method does not use true sentence boundaries; token windows may bleed across sentence boundaries in long queries.
4. **Beam may miss global optimum** for instances with very many slots (> ~15) and many mentions, since the beam only keeps the top-`GCG_BEAM_WIDTH` states.
5. **No external model.** The method is fully rule-based and deterministic; it does not use any trained model or embeddings, which limits its ability to resolve subtle semantic confusions.

---

## How to Run

### Single method run (CLI)

```bash
python tools/nlp4lp_downstream_utility.py \
    --variant orig \
    --baseline tfidf \
    --assignment-mode global_consistency_grounding
```

This writes results to `results/paper/`.

### Via the focused evaluation script

```bash
# Run all 5 default methods (including global_consistency_grounding)
python tools/run_nlp4lp_focused_eval.py --variant orig

# Low-resource / no-subprocess mode (resume-safe)
python tools/run_nlp4lp_focused_eval.py --variant orig --safe
```

### Run the unit tests

```bash
python -m pytest tests/test_global_consistency_grounding.py -v
```

### Run in-process via Python

```python
from tools.nlp4lp_downstream_utility import _run_global_consistency_grounding

filled_values, filled_mentions, diagnostics = _run_global_consistency_grounding(
    query="Each unit earns a profit of 5 dollars. The total budget is 1000 dollars.",
    variant="orig",
    expected_scalar=["profit_per_unit", "total_budget"],
)
print(filled_values)
# Inspect top candidate assignments and active penalties:
for entry in diagnostics["top_assignments"][:3]:
    print(entry["rank"], entry["total_score"], entry["active_reasons"])
```

---

## Files Modified / Created

| File | Change |
|---|---|
| `tools/nlp4lp_downstream_utility.py` | Added `GCG_LOCAL_WEIGHTS`, `GCG_GLOBAL_WEIGHTS`, `GCG_PRUNE_THRESHOLD`, `GCG_BEAM_WIDTH`, `_gcg_local_score`, `_gcg_global_penalty`, `_gcg_beam_search`, `_run_global_consistency_grounding`; wired into `run_setting`, `run_single_setting`, `main` |
| `tools/run_nlp4lp_focused_eval.py` | Added `tfidf_global_consistency_grounding` to default focused baselines list |
| `tests/test_global_consistency_grounding.py` | 29 focused tests (new file) |
| `docs/global_consistency_grounding.md` | This document |
| `docs/learning_runs/global_consistency_grounding_results.md` | Benchmark note |
