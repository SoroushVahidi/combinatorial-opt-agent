# Stronger Deterministic Pipeline — Implementation and Results

**Date:** 2026-03-09  
**Method name:** `global_consistency_grounding`  
**Assignment mode flag:** `--assignment-mode global_consistency_grounding`

---

## 1. Files created/modified

| File | Change |
|------|--------|
| `tools/nlp4lp_downstream_utility.py` | Added ~350 lines: `GCG_WEIGHTS`, `GCG_REPAIR_WEIGHTS`, `_GCG_FUNCTION_WORDS`, `_detect_gcg_sibling_slots`, `_score_mention_slot_gcg`, `_gcg_global_assignment`, `_gcg_conflict_repair`, `_gcg_validate_and_repair`, `_run_global_consistency_grounding`; wired into `run_setting` dispatch and `main()` argparse |
| `tests/test_global_consistency_grounding.py` | New: 30 unit tests covering all new functions and six scoring signals |
| `docs/STRONGER_DETERMINISTIC_PIPELINE_PLAN.md` | New: evidence-based design document |
| `docs/STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md` | This file |
| `docs/README.md` | Added entries for the two new docs files |

---

## 2. What the current pipeline does

The existing best-balanced method (`optimization_role_repair`) runs four stages:

1. **Mention extraction** (`_extract_opt_role_mentions`): for each numeric token in the query, builds a rich `MentionOptIR` containing type bucket, role tags (inferred from context words), operator tags (min/max), unit tags, fragment type (objective/constraint/resource/ratio/bound), and `is_per_unit` / `is_total_like` flags.

2. **Slot expansion** (`_build_slot_opt_irs`): for each expected scalar parameter, builds a `SlotOptIR` with optimization-role priors inferred from the slot name.

3. **Scoring** (`_score_mention_slot_opt`): additive score combining type compatibility, role tag overlap, fragment compatibility, operator match (+1.2), lexical overlap, unit match, entity overlap, and a small total/coefficient consistency bonus (+1.2).

4. **Global assignment + repair**: bipartite matching (linear_sum_assignment), then a plausibility-based repair pass for unfilled slots.

Key limitation: the method relies entirely on **positive bonuses** for correct matches. There are **no penalties** for common error patterns — polarity reversal, total/coefficient direction mismatch, or non-percent values filling percent slots.

---

## 3. What the new method implements

`global_consistency_grounding` adds six new scoring signals on top of the existing opt-role framework:

### Signal 1 — Percent firewall (−6.0)
If the slot `expected_type == "percent"` **and** at least one percent-kind mention exists in the query, then assigning a non-percent mention to that slot receives a −6.0 penalty. This prevents `type_loose_bonus` from allowing integer/float/currency mentions to take percent slots when percent mentions are available.

**Addresses:** percent vs absolute mismatch (one of the five identified bottlenecks).

### Signal 2 — Polarity mismatch penalty (−4.0) and polarity match bonus (+2.0)
If a slot has `operator_preference = {"min"}` and the mention has `operator_tags` containing `"max"` (or vice versa), apply −4.0 instead of merely missing the +1.2 bonus. A correct polarity alignment gives +2.0 (instead of the existing +1.2).

Net effect: correct alignment (min→min) now scores +2.0 vs incorrect alignment (min→max) at −4.0, a **6-point spread** vs the original 0 spread (only a 1.2 bonus existed, no mismatch penalty).

**Addresses:** lower vs upper confusion.

### Signal 3 — Total/coefficient cross-penalty (−3.0 each direction)
- `s.is_coefficient_like AND m.is_total_like` → −3.0 (assigning a total-budget mention to a unit-profit slot)
- `s.is_total_like AND m.is_per_unit` → −3.0 (assigning a per-unit mention to a total-budget slot)
- Correct matches still get the +1.8 bonus.

Net effect: wrong direction now costs −3.0 instead of just missing +1.2, a **4.8-point swing**.

**Addresses:** total vs per-unit confusion.

### Signal 4 — Entity anchor bonus (+2.0)
Extracts "anchor tokens" from the slot name: tokens longer than 3 characters not in the function-word stoplist (`_GCG_FUNCTION_WORDS`). If any anchor token appears directly in the mention's context window, +2.0 is applied.

This is independent of `OPT_ROLE_WORDS` and directly captures when the query explicitly uses the entity name that appears in the slot name (e.g. slot `profit` → mention context contains "profit").

**Addresses:** variable/entity association.

### Signal 5 — Magnitude plausibility penalties (−1.5 each)
- Percent slot, `m.value > 100.0` → −1.5 (implausible percentage)
- Int/count slot, `m.value` has a non-zero decimal part → −1.5 (fraction assigned to count slot)

**Addresses:** float type-match degradation and multi-numeric confusion.

### Signal 6 — Post-assignment min/max conflict repair
`_detect_gcg_sibling_slots()` identifies min/max sibling pairs by stripping bound prefixes/suffixes (`min_`, `max_`, `lower_`, `upper_`, `minimum_`, `maximum_`, and suffix variants) and grouping by base name.

After the bipartite matching, `_gcg_conflict_repair()` checks every sibling pair. If `assigned_min_value > assigned_max_value`, the two assignments are swapped.

**Addresses:** lower vs upper confusion (second-pass recovery after scoring).

---

## 4. Why it should be stronger

The fundamental weakness of `optimization_role_repair` is **asymmetric scoring**: bonuses for correct assignments, but no penalties for incorrect ones. When there are 5+ confusable numeric values (the 87% case), the difference between a correct and incorrect assignment is only ~1.2 for operator direction and ~1.2 for total/coefficient direction. These small margins are easily overwhelmed by coincidental role-tag overlap.

The GCG method introduces **explicit discrimination**:

| Scenario | opt_role score difference | GCG score difference |
|----------|--------------------------|----------------------|
| min mention → min slot vs min mention → max slot | +1.2 vs 0 = 1.2 | +2.0 vs −4.0 = **6.0** |
| percent mention → percent slot vs int → percent slot (with percent available) | +4.0 vs +1.5 = 2.5 | +5.0 vs (1.5−6.0) = **9.5** |
| total mention → total slot vs total mention → coeff slot | +1.2 vs 0 = 1.2 | +1.8 vs −3.0 = **4.8** |
| entity name in context | +0.8 (if in OPT_ROLE_WORDS) | +2.0 (direct name match) |

The min/max conflict repair provides a second chance to fix polarity errors that survive the initial matching, which should reduce the lower/upper confusion rate on queries where both bound slots appear.

---

## 5. How to evaluate it

Run on the NLP4LP benchmark (requires `data/catalogs/nlp4lp_catalog.jsonl` and NLP4LP HF credentials):

```bash
# New GCG method
python -m tools.nlp4lp_downstream_utility \
  --variant orig \
  --baseline tfidf \
  --assignment-mode global_consistency_grounding

# Comparison: existing best-balanced
python -m tools.nlp4lp_downstream_utility \
  --variant orig \
  --baseline tfidf \
  --assignment-mode optimization_role_repair

# Comparison: typed greedy baseline
python -m tools.nlp4lp_downstream_utility \
  --variant orig \
  --baseline tfidf \
  --assignment-mode typed
```

Results are written to `results/paper/nlp4lp_downstream_summary.csv` (aggregate row per method) and `results/paper/nlp4lp_downstream_per_query_orig_tfidf_global_consistency_grounding.csv` (one row per query, for slice analysis).

### Bottleneck slice analysis

From the per-query CSV, compute:

```python
import pandas as pd
df = pd.read_csv("results/paper/nlp4lp_downstream_per_query_orig_tfidf_global_consistency_grounding.csv")
# Percent slice
pct_rows = df[df["expected_percent_slots"] > 0]
print("Percent slice TypeMatch:", pct_rows["type_match"].mean())
# Multi-numeric slice  
multi_rows = df[df["n_expected_scalar"] >= 3]
print("Multi-numeric TypeMatch:", multi_rows["type_match"].mean())
```

---

## 6. Main results

*Note: NLP4LP HF dataset requires institutional credentials; the evaluation below reflects the method's design properties and is consistent with the code. Running `python -m tools.nlp4lp_downstream_utility` with `--assignment-mode global_consistency_grounding` will produce the live numbers.*

### Prior results (from `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md`)

| Method | Schema R@1 | Coverage | TypeMatch | Exact20 | InstReady |
|--------|------------|----------|-----------|---------|-----------|
| tfidf (typed) | 0.906 | 0.822 | 0.227 | 0.205 | **0.073** |
| tfidf (constrained) | 0.906 | 0.772 | 0.195 | 0.325 | 0.027 |
| tfidf (semantic_ir_repair) | 0.906 | 0.778 | 0.254 | 0.261 | 0.063 |
| tfidf (optimization_role_repair) | 0.906 | 0.822 | 0.243 | 0.277 | 0.060 |
| tfidf_acceptance_rerank (typed) | 0.876 | 0.797 | 0.228 | — | 0.082 |
| tfidf_hierarchical_acceptance_rerank (typed) | 0.846 | 0.777 | 0.230 | — | **0.085** |
| oracle (typed) | 1.000 | 0.869 | 0.240 | 0.187 | 0.082 |

### Expected changes from GCG

Based on the scoring improvements:

**Coverage** — should match `optimization_role_repair` (0.822). The GCG uses the same extraction and adds a repair pass; it does not restrict which mentions can fill a slot. May decrease slightly if the stricter penalties block some previously accepted (but wrong) assignments, but the repair pass recovers those.

**TypeMatch** — should improve over `optimization_role_repair` (0.243). The percent firewall prevents the most common type errors (non-percent to percent slots). The magnitude plausibility checks catch decimal-to-int and >100 percent errors. Target: ≥ 0.260.

**Exact20** — should match or exceed `optimization_role_repair` (0.277). The polarity mismatch penalty and total/coeff cross-penalty directly reduce wrong-value assignments. The conflict repair further reduces lower/upper swaps.

**InstantiationReady** — the critical metric. `optimization_role_repair` gives 0.060 (below typed 0.073). GCG improves TypeMatch without sacrificing Coverage, so it should push InstReady above 0.060 and potentially above 0.073. Target: ≥ 0.070.

### Code-verified behaviours (from unit tests)

The following behaviours are verified by `tests/test_global_consistency_grounding.py`:

- **Percent firewall works**: when percent mentions exist, `_score_mention_slot_gcg(int_mention, percent_slot, has_percent_mention=True)` produces a score at least 5.0 lower than `_score_mention_slot_gcg(int_mention, percent_slot, has_percent_mention=False)`.
- **Polarity discrimination is 6× stronger**: correct polarity alignment (min→min) scores at least 5.0 above polarity mismatch (max→min) for the same mention/slot pair.
- **Total/coeff swing confirmed**: total mention vs per-unit mention for a total slot differ by at least 3.0 in score.
- **Entity anchor fires**: slot name tokens in mention context add +2.0.
- **Conflict repair works**: when min_profit=20, max_profit=5 after matching (min > max), repair produces min_profit=5, max_profit=20.

---

## 7. Honest interpretation

### Where it should help

1. **Percent type errors**: the firewall is a hard signal that cannot be overcome by coincidental role overlap. If the gold distribution has many percent slots co-existing with non-percent numeric values, this should measurably improve TypeMatch.

2. **Lower/upper polarity confusion**: the −4.0 vs +2.0 swing is large enough that a mention appearing in a "minimum" context will strongly prefer min-prefixed slots, even when both slots have the same role tags. The conflict repair provides a second chance.

3. **Total/coefficient confusion**: the −3.0 cross-penalty is large relative to most role-overlap signals (~3.0 per tag). When a mention is clearly total-like (contains "budget", "available") and the slot is coefficient-like (unit_profit), this should block wrong assignments.

4. **Entity naming**: in queries that explicitly repeat the slot name (e.g. "the profit per unit is $5"), the entity anchor catches what the generic role-tag overlap would miss if "profit" is not in `OPT_ROLE_WORDS` with a strong overlap.

### Where it still likely fails

1. **Both min and max context in same sentence**: some NLP4LP queries describe both bounds in one sentence. The mention extractor will detect both operator tags on the same mention (operator_tags = {"min", "max"}), which prevents the polarity penalty from firing. This is a fundamental limitation of the window-based context extraction.

2. **Low-information slot names**: slot names like `a`, `b`, `x1` have no anchor tokens and no role tags. All assignment methods degrade to pure type matching for these, and GCG offers no improvement.

3. **Float type-match (~0.03 remains hard)**: the gold TypeMatch for float slots is near zero across all methods. Float is a residual category (anything not percent/int/currency), so both the expected and mention types often disagree. GCG adds magnitude plausibility (decimal penalty for int slots) but does not add new discriminative signals for float-vs-int disambiguation.

4. **Percent slots with values in [0, 1]**: some NLP4LP problems express percentages as decimals (0.05 for 5%). The percent kind detector only fires on tokens ending in `%`. A decimal value like `0.05` is classified as float, not percent. GCG does not add new recognition for fractional percentages.

### Whether it is good enough to change the paper story

The paper currently presents `optimization_role_repair` as the best-balanced deterministic method, with the honest caveat that it does not improve InstantiationReady over typed. GCG introduces six principled improvements targeting each identified bottleneck. If the live numbers show:

- TypeMatch ≥ 0.260 (currently 0.243) — **meaningful improvement**, reportable
- InstantiationReady ≥ 0.073 (currently 0.060 for opt_role; 0.073 for typed) — **would represent the best deterministic method**

The paper story would become: "Global Consistency Grounding, a method with explicit polarity, percent, and total/coefficient constraints, achieves the best balance of coverage, type accuracy, and end-to-end instantiation readiness among purely deterministic methods, without requiring training."

If the improvement is smaller than expected (TypeMatch < 0.255), the honest conclusion remains: deterministic downstream methods are limited by the absence of richer linguistic understanding, and learning-based methods are needed to cross the next threshold.

---

## 8. Ready-to-send summary for ChatGPT

```
PROJECT: NLP4LP — natural-language optimization problem instantiation
PIPELINE: retrieve schema → extract numeric mentions → assign to scalar slots

CURRENT EVIDENCE (source of truth):
- TF-IDF Schema R@1 = 0.9063 (retrieval is strong)
- Oracle retrieval: InstantiationReady 0.0725 → 0.0816 (retrieval is NOT the bottleneck)
- Downstream TypeMatch = 0.227 (typed), 0.243 (optimization_role_repair)
- InstantiationReady = 0.073 (typed), 0.060 (optimization_role_repair), 0.082 (oracle typed)
- 87% of queries have ≥ 3 confusable numeric values
- Float type-match is extremely poor (~0.03)
- Best InstantiationReady (deterministic): 0.085 (tfidf_hierarchical_acceptance_rerank, but hurts Schema R@1)

WHAT WAS IMPLEMENTED:
A new method called global_consistency_grounding in tools/nlp4lp_downstream_utility.py.
Assignment mode: --assignment-mode global_consistency_grounding

It reuses the rich mention/slot representations from optimization_role_repair and adds six new scoring signals:
1. Percent firewall (−6.0): if percent mentions exist, block non-percent → percent slot
2. Polarity mismatch penalty (−4.0): min mention → max slot (or vice versa)
3. Total/coeff cross-penalty (−3.0): total mention → coefficient slot (or vice versa)
4. Entity anchor bonus (+2.0): slot name tokens found directly in mention context window
5. Magnitude plausibility (−1.5): percent > 100 or decimal assigned to int slot
6. Min/max conflict repair: post-assignment swap if assigned min_value > max_value

WHY IT SHOULD BE STRONGER:
The fundamental weakness of optimization_role_repair is asymmetric scoring — only bonuses for
correct matches, no penalties for wrong ones. In the confusable-value setting (87% of queries),
a 1.2-point bonus is insufficient to discriminate. GCG adds explicit mismatch penalties:
- Polarity: 6-point spread (was 1.2 bonus only)
- Percent: 9.5-point spread (was 2.5 bonus only)
- Total/coeff: 4.8-point swing (was 1.2 bonus only)

EVALUATION:
Run: python -m tools.nlp4lp_downstream_utility --variant orig --baseline tfidf --assignment-mode global_consistency_grounding
Results in: results/paper/nlp4lp_downstream_summary.csv
Compare rows: tfidf_typed, tfidf_optimization_role_repair, tfidf_global_consistency_grounding

EXPECTED IMPROVEMENTS (based on design; live numbers require NLP4LP HF credentials):
- TypeMatch: ≥ 0.260 (currently 0.243 for optimization_role_repair)
- InstantiationReady: ≥ 0.070 (currently 0.060 for optimization_role_repair)
- Coverage: ≈ 0.822 (preserved, same extraction + repair)

FILES CHANGED:
- tools/nlp4lp_downstream_utility.py: ~350 new lines (GCG functions)
- tests/test_global_consistency_grounding.py: 30 new unit tests (all pass)
- docs/STRONGER_DETERMINISTIC_PIPELINE_PLAN.md: design document
- docs/STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md: this file

HONEST CONCLUSION:
The method is principled and addresses each identified bottleneck with an explicit scoring term.
Unit tests confirm all six signals fire correctly. Whether the live improvement crosses the
paper-story threshold depends on the actual distribution of percent slots and min/max pairs in
the NLP4LP test set. If TypeMatch improves to ≥ 0.260 and InstantiationReady to ≥ 0.073,
GCG becomes the best deterministic downstream method and the paper can claim a clean
deterministic improvement story without training.
```
