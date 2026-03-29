# Patch summary: leak-free split-by-problem evaluation

## 1) Exact new/modified file paths

### New files
- **`training/splits.py`** — Build and write train/dev/test problem ID splits (stratified by `source`); `load_splits()`, `get_problem_ids_for_split()`.
- **`training/metrics.py`** — P@1, P@5, MRR@k, nDCG@k, Coverage@k; `compute_metrics(results, k=10)`.
- **`docs/EVALUATION_LEAKAGE_ANALYSIS.md`** — Leakage analysis (query generation, seeds, overlap, three example queries, catalog overlap).
- **`tests/test_splits.py`** — Unit tests: disjoint splits, reproducibility, write/load, `get_problem_ids_for_split`.
- **`tests/test_no_leakage.py`** — Unit tests: train∩test=∅, generate_samples respects split, eval instances only for split, full pipeline no overlap.

### Modified files
- **`training/generate_samples.py`**
  - `generate_all_samples(..., split_problem_ids=None)`: when set, only problems whose `id` is in `split_problem_ids` are used; when using splits, `include_real_world` is False to avoid leakage.
  - CLI: `--splits PATH` and `--split {train,dev,test}`; both required together.
- **`training/evaluate_retrieval.py`**
  - Eval instances are `(query, problem_id)`; generation can be restricted to `problem_ids` (split).
  - `_generate_eval_instances(catalog, seed, num_instances, problem_ids=None)`.
  - `_load_eval_instances(eval_path, catalog)` supports both `problem_id` and legacy `problem_index` in JSONL.
  - CLI: `--splits PATH`, `--split {train,dev,test}`, `--results-dir`, `--top-k` (default 10).
  - Metrics: P@1, P@5, MRR@10, nDCG@10, Coverage@10 via `training.metrics.compute_metrics`.
  - Writes metrics JSON to `results/eval_{split}.json` (or `results/eval.json` when no split).
- **`.gitignore`** — Added `data/processed/eval_*.jsonl`, `data/processed/splits.json`, `results/`.

---

## 2) New CLI commands: leak-free pipeline end-to-end

All commands from **project root** unless noted.

```bash
# 1) Build catalog (existing)
pip install -r requirements.txt
python pipeline/run_collection.py
# Optional: scripts/merge_catalog.py if you use merge only

# 2) Build train/dev/test splits (stratified by source)
python -m training.splits --splits-out data/processed/splits.json --seed 42

# 3) Generate training pairs ONLY for train split (no eval problems in training)
python -m training.generate_samples \
  --splits data/processed/splits.json \
  --split train \
  --output data/processed/training_pairs.jsonl \
  --seed 42 \
  --instances-per-problem 100

# 4) Train retrieval model on train pairs only
python -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --output-dir data/models/retrieval_finetuned \
  --epochs 2 --batch-size 32

# 5) Evaluate on TEST split only (leak-free: test problems never seen in training)
python -m training.evaluate_retrieval \
  --splits data/processed/splits.json \
  --split test \
  --regenerate \
  --num 500 \
  --seed 999 \
  --results-dir results \
  --top-k 10
```

Optional: evaluate on dev for tuning, then run once on test for final numbers:

```bash
python -m training.evaluate_retrieval --splits data/processed/splits.json --split dev --regenerate --num 500 --results-dir results
```

---

## 3) Expected outputs written to `results/`

| Filename | When produced | Contents |
|----------|----------------|----------|
| **`results/eval_test.json`** | `evaluate_retrieval --split test ...` | `{"split": "test", "num_instances": N, "metrics": {"P@1": ..., "P@5": ..., "MRR@10": ..., "nDCG@10": ..., "Coverage@10": ...}}` |
| **`results/eval_dev.json`** | `evaluate_retrieval --split dev ...` | Same structure, `"split": "dev"`. |
| **`results/eval.json`** | `evaluate_retrieval` without `--splits`/`--split` | Same structure, `"split": null`. |

Additional data files (intermediate):

| Path | Producer |
|------|----------|
| **`data/processed/splits.json`** | `python -m training.splits` |
| **`data/processed/eval_test.jsonl`** | `evaluate_retrieval --split test --regenerate` (or first run without existing file) |
| **`data/processed/eval_dev.jsonl`** | Same for `--split dev`. |

---

## 4) Function signatures (implementation-ready)

**training/splits.py**
```python
def load_catalog(catalog_path: Path | None = None) -> list[dict]: ...
def build_splits(
    catalog: list[dict],
    seed: int = 42,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]: ...
def write_splits(splits: dict[str, list[str]], out_path: Path) -> None: ...
def load_splits(splits_path: Path) -> dict[str, list[str]]: ...
def get_problem_ids_for_split(splits: dict[str, list[str]], split_name: str) -> list[str]: ...
```

**training/metrics.py**
```python
def precision_at_k(ranked_names: list[str], expected_name: str, k: int) -> float: ...
def reciprocal_rank_at_k(ranked_names: list[str], expected_name: str, k: int) -> float: ...
def ndcg_at_k(ranked_names: list[str], expected_name: str, k: int) -> float: ...
def coverage_at_k(ranked_names: list[str], expected_name: str, k: int) -> float: ...
def compute_metrics(
    results: list[tuple[list[str], str]],
    k: int = 10,
) -> dict[str, float]: ...
```

**training/generate_samples.py**
```python
def generate_all_samples(
    catalog_path: Path | None = None,
    seed: int = 42,
    instances_per_problem: int = 100,
    include_real_world: bool = True,
    split_problem_ids: list[str] | None = None,
) -> list[tuple[str, str]]: ...
```

**training/evaluate_retrieval.py**
```python
def _generate_eval_instances(
    catalog: list[dict],
    seed: int = 999,
    num_instances: int = 500,
    problem_ids: list[str] | None = None,
) -> list[tuple[str, str]]: ...  # (query, problem_id)
def _load_eval_instances(eval_path: Path, catalog: list[dict]) -> list[tuple[str, str]]: ...
```

---

## 5) Run unit tests

```bash
python -m pytest tests/test_splits.py tests/test_no_leakage.py -v
```

Expected: 9 passed (5 in test_splits, 4 in test_no_leakage).
