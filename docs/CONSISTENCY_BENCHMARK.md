# Consistency Benchmark

Evaluates the old vs repaired number-role checkers on 15 synthetic cases (7 correct, 8 wrong).

## Running

```bash
python scripts/run_consistency_benchmark.py
```

Outputs are written to `outputs/consistency_benchmark/`.

## Metrics

| Metric | Description |
|---|---|
| `fpr_correct` | Fraction of correct cases falsely flagged as wrong |
| `recall_wrong` | Fraction of wrong cases correctly detected |

## Old Checker

Flags an answer if its literal string is not present in the question text. This produces high false-positive rates on computed answers (e.g., `60` not in `"12 widgets per hour … 5 hours"`).

## Repaired Checker

Uses the full pipeline:
1. `extract_number_mentions` + `annotate_relevance`
2. `repair_number_roles` + `calibrate_required_flags`
3. `detect_suspicious_missing_roles` — flags only if `suspicious_missing=True` and `confidence != "low"`

## Output Files

| File | Contents |
|---|---|
| `summary.json` | Aggregate metrics for both checkers |
| `per_candidate_results.csv` | Per-case flags and correctness |
| `failure_type_summary.csv` | Accuracy by failure type |
| `role_signal_summary.csv` | FPR and recall comparison |

## Synthetic Case Types

- **correct** (7 cases): answer computed from inputs; old checker incorrectly flags most of these.
- **intermediate_as_final** (4 cases): model stops at an intermediate value.
- **wrong_target_quantity** (4 cases): model reports a constraint bound instead of the optimized quantity.
