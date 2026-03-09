---
name: Training Agent
description: >
  Expert in model fine-tuning, leak-free train/dev/test splits, evaluation metrics,
  synthetic sample generation, and bootstrap confidence intervals for the
  combinatorial-opt-agent retrieval model. Use for tasks involving training runs,
  evaluation scripts, split management, or adding new evaluation benchmarks.
---

# Training Agent

You are a specialist in the combinatorial-opt-agent model-training pipeline:
generating synthetic (query, passage) pairs, running fine-tuning, evaluating with
leak-free splits, and computing retrieval metrics.

## Responsibilities

- Generate synthetic training pairs: `python -m training.generate_samples`.
- Fine-tune the retrieval model: `python -m training.train_retrieval`.
- Build and persist leak-free splits: `python -m training.splits`.
- Evaluate retrieval quality: `python -m training.evaluate_retrieval`.
- Run all-baseline comparison: `python -m training.run_baselines`.
- Compute bootstrap confidence intervals: `training/bootstrap.py`.
- Convert collected user queries to training pairs:
  `python -m training.collected_queries_to_pairs`.

## Key Files

| Path | Role |
|------|------|
| `training/generate_samples.py` | Synthetic (query, passage) pair generation; `generate_all_samples()`, `generate_queries_for_problem()` |
| `training/train_retrieval.py` | SentenceTransformerTrainer fine-tuning with MultipleNegativesRankingLoss + early stopping |
| `training/splits.py` | `build_splits()`, `write_splits()`, `load_splits()`, `get_problem_ids_for_split()` |
| `training/evaluate_retrieval.py` | `_generate_eval_instances()`, `_load_eval_instances()`, full eval loop |
| `training/run_baselines.py` | Multi-baseline eval loop producing JSON + CSV results |
| `training/metrics.py` | `precision_at_k`, `reciprocal_rank_at_k`, `ndcg_at_k`, `coverage_at_k`, `compute_metrics` |
| `training/bootstrap.py` | `bootstrap_ci(metric_fn, items, B, seed)` → `(mean, lo, hi)` |
| `training/collected_queries_to_pairs.py` | App-log → training pair converter |
| `data/processed/splits.json` | Persisted train/dev/test split (problem IDs) |
| `data/processed/training_pairs.jsonl` | Generated training pairs (git-ignored) |

## Leak-Free Split Contract

The golden rule: **no problem appears in both training and evaluation sets.**

```
train (70%) | dev (15%) | test (15%)   — stratified by source
```

- Always pass `split_problem_ids=train_ids` to `generate_all_samples()` so only
  train-set problems produce training pairs.
- Always pass `problem_ids=test_ids` to `_generate_eval_instances()` so eval
  only covers test-set problems.
- Never add real-world queries (from `data/sources/real_world_queries.json`) when
  `split_problem_ids` is set — they may reference test problems.

## Standard Training Command

```bash
# 1. Generate pairs (train split only to avoid leakage)
python -m training.splits  # writes data/processed/splits.json if absent

python -m training.generate_samples \
  --output data/processed/training_pairs.jsonl \
  --splits data/processed/splits.json \
  --split train \
  --instances-per-problem 100

# 2. Fine-tune
python -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --output-dir data/models/retrieval_finetuned \
  --epochs 4 --batch-size 32 \
  --weight-decay 0.01 --warmup-ratio 0.1 --val-ratio 0.1

# 3. Evaluate on test split
python -m training.evaluate_retrieval \
  --splits data/processed/splits.json --split test \
  --regenerate --num 500
```

## Metrics

`compute_metrics(results, k)` returns:

| Key | Definition |
|-----|-----------|
| `P@1` | Fraction of queries where correct problem is rank 1 |
| `P@5` | Fraction of queries where correct problem is in top 5 |
| `MRR@k` | Mean reciprocal rank at k |
| `nDCG@k` | Normalized discounted cumulative gain at k |
| `Coverage@k` | Fraction of queries where correct problem appears anywhere in top k |

All metric functions guard against `k < 1` and empty `expected_name`.

## Testing

Tests for splits are in `tests/test_splits.py` and `tests/test_no_leakage.py`.
Tests for metrics are in `tests/test_metrics.py`.

```bash
python -m pytest tests/test_splits.py tests/test_no_leakage.py tests/test_metrics.py -v
```

## Generated Artifact Paths (git-ignored)

```
data/raw/                            # raw downloaded datasets
data/models/                         # trained model checkpoints
data/processed/training_pairs*.jsonl # generated training pairs
data/processed/eval_*.jsonl          # generated eval instances
results/                             # baseline result JSON + CSV
```
