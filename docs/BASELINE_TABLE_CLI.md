# Baseline table: exact CLI to reproduce paper-ready results

## File paths

| Path | Description |
|------|--------------|
| **`retrieval/baselines.py`** | BM25, TF-IDF, SBERT baselines; `fit(catalog)`, `rank(query, top_k) -> [(problem_id, score)]`; `get_baseline(name)`. |
| **`training/run_baselines.py`** | Runner: loads eval, runs all baselines, writes JSON + CSV. |
| **`results/baselines_<split>.json`** | Full config + metrics per baseline. |
| **`results/baselines_<split>.csv`** | One row per baseline (baseline, P@1, P@5, MRR@10, nDCG@10, Coverage@10). |

## Function / class signatures (retrieval/baselines.py)

```python
def _searchable_text(problem: dict) -> str: ...

class RetrievalBaseline(ABC):
    def fit(self, catalog: list[dict]) -> RetrievalBaseline: ...
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]: ...  # [(problem_id, score)]

class BM25Baseline(RetrievalBaseline): ...
class TfidfBaseline(RetrievalBaseline): ...
class SBERTBaseline(RetrievalBaseline):  # __init__(self, model_path: str | None = None)

def bm25() -> BM25Baseline: ...
def tfidf() -> TfidfBaseline: ...
def sbert(model_path: str | None = None) -> SBERTBaseline: ...

def get_baseline(name: str, model_path: str | None = None) -> RetrievalBaseline:  # name in bm25, tfidf, sbert
```

## Exact commands: build catalog → splits → eval → run baselines → CSV for paper

Run from **project root**:

```bash
# 1) Build catalog
python pipeline/run_collection.py

# 2) Build leak-free splits (stratified by source)
python -m training.splits --splits-out data/processed/splits.json --seed 42

# 3) Generate test-split eval instances (deterministic with seed)
python -m training.run_baselines \
  --splits data/processed/splits.json \
  --split test \
  --regenerate \
  --num 500 \
  --seed 999 \
  --k 10 \
  --baselines bm25 tfidf sbert \
  --results-dir results
```

If the eval file `data/processed/eval_test.jsonl` already exists and you do not want to regenerate:

```bash
python -m training.run_baselines \
  --splits data/processed/splits.json \
  --split test \
  --eval-file data/processed/eval_test.jsonl \
  --k 10 \
  --baselines bm25 tfidf sbert \
  --results-dir results
```

## Outputs (ready to paste into paper)

- **`results/baselines_test.json`** — Config (catalog, split, eval_file, num_instances, seed, k, baselines) and per-baseline metrics (P@1, P@5, MRR@10, nDCG@10, Coverage@10).
- **`results/baselines_test.csv`** — Table:

  | baseline | P@1 | P@5 | MRR@10 | nDCG@10 | Coverage@10 |
  |----------|-----|-----|--------|---------|-------------|
  | bm25     | ... | ... | ...    | ...     | ...         |
  | tfidf    | ... | ... | ...    | ...     | ...         |
  | sbert    | ... | ... | ...    | ...     | ...         |

Copy the CSV (or open in a spreadsheet) and paste into the paper table. Metrics come from `training/metrics.py` (same as `evaluate_retrieval`).

## Optional: custom catalog or eval file

```bash
python -m training.run_baselines \
  --catalog data/processed/all_problems.json \
  --eval-file data/processed/eval_test.jsonl \
  --k 10 \
  --baselines bm25 tfidf sbert \
  --results-dir results
```

Without `--splits`/`--split`, output files are **`results/baselines.json`** and **`results/baselines.csv`** (no split suffix).

## Determinism

- Eval set: fixed by `--seed 999` when using `--regenerate`.
- Baselines: BM25 and TF-IDF are deterministic; SBERT (SentenceTransformer) is deterministic for a given model and catalog order.
