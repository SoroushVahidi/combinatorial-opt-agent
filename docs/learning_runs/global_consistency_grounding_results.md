# Global Consistency Grounding — Benchmark Note

## Status

**Canonical benchmark is blocked** by unavailability of the gated NLP4LP gold test split in this environment (the HuggingFace dataset requires authentication and is not accessible in the sandbox).  No benchmark numbers are fabricated here.

The method is fully integrated into the evaluation pipeline and can be benchmarked by any user with access to the gold data by running:

```bash
python tools/nlp4lp_downstream_utility.py \
    --variant orig \
    --baseline tfidf \
    --assignment-mode global_consistency_grounding

# or, via the focused eval runner (all 5 default methods side-by-side):
python tools/run_nlp4lp_focused_eval.py --variant orig
```

---

## What Would Be Compared

When gold data is available, the canonical comparison set is:

| Method | Assignment mode | Notes |
|---|---|---|
| `tfidf_acceptance_rerank` | `typed` (greedy) | Greedy typed baseline |
| `tfidf_optimization_role_repair` | `optimization_role_repair` | Current best structured deterministic method |
| `tfidf_optimization_role_relation_repair` | `optimization_role_relation_repair` | Relation-aware + incremental admissible |
| **`tfidf_global_consistency_grounding`** | `global_consistency_grounding` | **New method** |

Primary metrics reported per method:

- `param_coverage` — fraction of expected scalar slots that were filled
- `type_match` — fraction of filled slots where the mention type matches expected
- `exact20_on_hits` — fraction of schema-hit queries where assigned value is within 20 % relative error
- `instantiation_ready` — fraction of queries where coverage ≥ 0.8 and type_match ≥ 0.8
- Per-type breakdowns: `percent`, `integer`, `currency`, `float`

---

## Qualitative Behavior (From Unit Tests)

The following behavior is validated by unit tests in `tests/test_global_consistency_grounding.py`:

| Scenario | GCG behaviour |
|---|---|
| Percent slot vs scalar slot | Assigns percent mention to percent slot and scalar to scalar slot |
| Per-unit coefficient vs total budget | Assigns 3 to `profit_per_unit`, 600 to `total_budget` |
| Lower bound vs upper bound | Assigns smaller value to min slot, larger to max slot |
| Three distinct float coefficients | Fills all three slots without duplication |
| Global duplicate-mention penalty | Penalises same mention used for two slots |
| Percent-misuse global penalty | Penalises percent mention in non-percent slot |
| Bound-flip global penalty | Penalises max-tagged mention in min slot |

---

## Design Rationale for Not Inventing Numbers

The problem statement explicitly states: *"Do not invent benchmark numbers."*

Running on local synthetic examples is possible but would not constitute a meaningful comparison against the existing methods' published results.  The unit tests in `tests/test_global_consistency_grounding.py` provide ground-truth verification of the method's correctness on targeted confusion cases without requiring the gated dataset.

---

## Reproducing Full Benchmark (When Gold Data Is Available)

1. Set the HuggingFace token:
   ```bash
   export HF_TOKEN=<your_token>
   ```
2. Run:
   ```bash
   python tools/run_nlp4lp_focused_eval.py --variant orig --safe
   ```
3. Results will appear in `results/paper/nlp4lp_focused_eval_summary.csv` and
   `results/paper/nlp4lp_downstream_summary.csv`.
