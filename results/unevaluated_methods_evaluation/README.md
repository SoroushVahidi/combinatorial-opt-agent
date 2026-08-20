# Evaluation of previously-unevaluated grounding methods (Phase 3, 2026-08-12)

**Finding: all three grounding-method families that `docs/METHOD_INVENTORY.md`
(Phase 2) flagged as "implemented + unit-tested but never evaluated on real
NLP4LP data" turn out to substantially and significantly outperform the
canonical `tfidf_typed_greedy` baseline (InstantiationReady 0.5287) when
actually run on the full 331-query `orig` benchmark.**

| Method | InstantiationReady | vs. typed greedy (0.5287) | p (paired bootstrap) |
|---|---|---|---|
| `max_weight_matching` | **0.7432** | +0.2145 | <0.001 |
| `search_structured_grounding` | 0.7039 | +0.1752 | <0.001 |
| `hierarchical_structured_grounding` | 0.7039 | +0.1752 | <0.001 |

All three even exceed the manuscript's own `Oracle-TG` upper-bound-style
control (0.5680), which was previously the highest InstantiationReady value
recorded anywhere in this repository's evaluated-method history.

## What changed vs. typed greedy

All three methods reuse the same richer optimization-role pairwise scoring
function (`_score_mention_slot_opt` in
`tools/nlp4lp_downstream_utility.py`) that `optimization_role_repair`
already used -- but `optimization_role_repair`'s repair-based decode only
reaches 0.4411 (a documented negative result, `docs/NEGATIVE_RESULTS.md`
NR3). `max_weight_matching` decodes the SAME family of local scores with an
exact global optimum (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`,
already implemented in `_run_max_weight_matching_grounding`) instead of a
greedy/repair heuristic. This is the most likely explanation for the gap:
greedy and repair-based decoding can get trapped in locally-good-but-
globally-suboptimal assignments that an exact bipartite matching avoids by
construction.

## How this was produced

```bash
export NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json
python3 -m tools.nlp4lp_downstream_utility --variant orig --baseline tfidf \
    --assignment-mode max_weight_matching \
    --output-dir <out_dir>
# repeat for --assignment-mode search_structured_grounding
# repeat for --assignment-mode hierarchical_structured_grounding
```

No code in `tools/nlp4lp_downstream_utility.py` was modified to produce
this result -- these assignment modes and their scoring/decoding logic
already existed; they had simply never been invoked against the real
benchmark before this pass. Total compute cost: under 2 seconds per method
for all 331 queries (CPU only, no GPU, no external API).

## Files

- `nlp4lp_downstream_orig_tfidf_{method}.json` -- aggregate metrics per method
- `nlp4lp_downstream_per_query_orig_tfidf_{method}.csv` -- per-query detail
- `significance.json` -- paired bootstrap significance vs. typed greedy (B=1000, seed=42)

## Status and next step

This is a **newly discovered, statistically significant positive result**,
not yet integrated into the manuscript or into
`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
(that table remains the camera-ready, already-submitted headline and was
NOT modified by this discovery -- see `results/CANONICAL_RESULTS.md` for
the full provenance record and the explicit decision not to fold this into
Table 1 without a dedicated manuscript-integration pass). See
`PROJECT_STATUS.md` and `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` for how
this reprioritizes future work: `max_weight_matching` should now be treated
as the strongest known baseline for any future learned-scorer comparison,
superseding typed greedy.
