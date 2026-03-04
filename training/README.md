# Training the retrieval model

Fine-tune the sentence-transformers retrieval model on synthetic (query, passage) pairs so it better matches natural-language queries to problems in the catalog. The catalog includes **NL4Opt**, **OptMATH benchmark**, and **classic + classic_extra** problems (see `docs/open_datasets.md`).

## 1. Generate samples

From the project root:

```bash
python -m training.generate_samples --output data/processed/training_pairs.jsonl
# Or 100 instances per problem (for stronger recognition):
python -m training.generate_samples --output data/processed/training_pairs.jsonl --instances-per-problem 100
```

This creates multiple query phrasings per problem (description, name, aliases, ILP/formulation templates) paired with the same passage (name + aliases + description). With `--instances-per-problem 100` the model gets more examples to recognize each problem; the app then shows the matched problem's description, ILP, variables, constraints, etc.

## 2. Install training deps

```bash
pip install datasets 'accelerate>=1.1.0'
```

(Sentence-transformers and torch are already in the stack; `accelerate` is required by the Trainer. Use a GPU environment for speed.)

## 3. Train (local or interactive)

```bash
python -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --output-dir data/models/retrieval_finetuned \
  --epochs 2 \
  --batch-size 32
```

On a machine with GPU, training will use it automatically.

## 4. Run training on Wulver (batch, GPU)

```bash
sbatch scripts/train_retrieval_gpu.slurm
```

The script will:

- Request 1 GPU, 8 CPUs, 32 GB RAM, 2 hours (partition `gpu`, QOS `standard`)
- Generate training pairs
- Fine-tune `sentence-transformers/all-MiniLM-L6-v2` with `MultipleNegativesRankingLoss`
- Save the model to `data/models/retrieval_finetuned/final`

If your account uses a different partition or QOS, edit the `#SBATCH` lines in `scripts/train_retrieval_gpu.slurm` (e.g. `--partition=general`, `--qos=low`, and remove `--gres=gpu:1` if you run on CPU).

To check job status and logs once in a while:

```bash
bash scripts/check_training_status.sh
```

**Automatic checks (run once, then it keeps checking):**

```bash
# Check every 60 seconds until you Ctrl+C
bash scripts/watch_training.sh

# Check every 2 minutes
bash scripts/watch_training.sh 120

# When training stops: write status to training_latest_status.txt and tell you to inform the assistant (so the assistant can do next steps)
bash scripts/watch_training.sh 60 --notify-when-stopped

# Same, run in background
nohup bash scripts/watch_training.sh 60 --notify-when-stopped > watch_training.log 2>&1 &
tail -f watch_training.log
```

When you use `--notify-when-stopped`, the script will print a line like: **"Training stopped - check training_latest_status.txt and do the next steps."** — copy that into Cursor chat so the assistant can read the status file and resubmit or confirm.

## 5. Use the fine-tuned model

Point the app or search to the saved model by setting the model path in `retrieval/search.py` or `app.py` to `data/models/retrieval_finetuned/final` when that path exists (or add a small wrapper that loads the fine-tuned model if present).

---

## 6. Collect real user prompts (for training)

When the app runs (e.g. on Wulver), every search is logged to **`data/collected_queries/user_queries.jsonl`** on the server. Each line is a JSON object:

- `ts`: ISO timestamp (UTC)
- `query`: the user’s natural-language prompt
- `top_k`: number of results requested
- `results`: list of `{ "name": "<problem name>", "score": <relevance> }` for the top results

So prompts from different people using the app are saved under the project directory on Wulver.

### Convert collected queries to training pairs

To turn these into (query, passage) pairs (using the top result’s problem as the passage):

```bash
python -m training.collected_queries_to_pairs
```

Output: **`data/processed/collected_training_pairs.jsonl`**. Optional args:

- `--input path` — input JSONL (default: `data/collected_queries/user_queries.jsonl`)
- `--output path` — output JSONL (default: `data/processed/collected_training_pairs.jsonl`)
- `--min-score 0.5` — only use queries whose top result has relevance ≥ this (default 0)

You can then merge `collected_training_pairs.jsonl` with synthetic pairs (e.g. concatenate files) and run `training.train_retrieval` on the combined file for training.
