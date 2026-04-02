# Wulver first training run: mention–slot scorer

## What was trained

- **Target:** A small **mention–slot compatibility model** (binary classifier / scorer) for the downstream slot-assignment stage of NLP4LP.
- **Architecture:** `nreimers/MiniLM-L6-H384-uncased` (Hugging Face; L6 is under nreimers, not microsoft) via `AutoModelForSequenceClassification` (binary).
- **Input:** Pairs `(sentence1=mention context, sentence2=slot name)`.
- **Output:** Binary label (match / no match); training uses cross-entropy.

## Data used

- **Source:** Eval items from `data/processed/nlp4lp_eval_orig.jsonl` plus gold parameters from Hugging Face `udell-lab/NLP4LP` test set.
- **Generated files:**
  - `data/processed/mention_slot_pairs.jsonl` — training pairs (11,271 examples).
  - `data/processed/mention_slot_pairs_dev.jsonl` — dev pairs (1,111 examples).
- **Generation script:** `training/generate_mention_slot_pairs.py`.

## SLURM resources requested

- **Partition:** `gpu`
- **QOS:** `low`
- **Account:** `ikoutis`
- **GPU:** 1× `gpu:a100_40g` (nodes observed with A100-SXM4-80GB)
- **Memory:** 32G
- **CPUs:** 8
- **Time:** 2 hours

## Job IDs and outcomes

| Job ID  | Outcome | Note |
|---------|--------|------|
| 853350  | Failed | Wrong `PROJECT_ROOT` (script ran under SLURM spool dir); no repo/data. |
| 853351  | Failed | `PROJECT_ROOT` fixed; PyTorch not found on compute node (Anaconda base has no `torch`). Model load failed; training did not run. |
| 853352  | Smoke only | Conda env with torch not found; smoke fallback ran but smoke test also failed (no torch for from_config). Script now uses `nreimers/MiniLM-L6-H384-uncased` and smoke writes a placeholder when PyTorch is missing. |

## Log and output paths

- **Logs:** `logs/mention_slot_%j.out`, `logs/mention_slot_%j.err` (on this cluster the `%j` in the filename may appear literally; check both).
- **Intended outputs (when training runs):**
  - Checkpoint: `results/mention_slot_scorer/final/`
  - Metrics: `results/mention_slot_scorer/metrics.json`
  - Training log: Trainer logging to stdout (in the `.out` log).

## Current status

- **Pipeline:** Batch script and training script are in place; data generation works; PROJECT_ROOT is set to repo on GPFS.
- **Blocker:** On GPU nodes, the default Anaconda Python does not have `torch`. Training therefore fails when loading the Hugging Face model.
- **Script change:** The batch script now tries to activate a conda env that has `torch` (tries `CONDA_ENV` if set, else `pt`, `pytorch`, `base`). If PyTorch is still missing, it runs the smoke test and exits successfully so you get a clear log and optional smoke artifact.

## Next commands

1. **Use a conda env that has torch (recommended)**  
   Create or use an env with `torch` and `transformers`, then submit with that env:
   ```bash
   cd /mmfs1/home/sv96/combinatorial-opt-agent   # or your repo path
   # Example: conda activate your_env
   export CONDA_ENV=your_env_with_torch
   sbatch jobs/wulver_train_mention_slot_2h.slurm
   ```
   Or set in the script the exact env name you use (e.g. edit `CONDA_ENV` in the batch file).

2. **Monitor the job**
   ```bash
   squeue -u $USER
   ```

3. **Inspect logs (replace JOBID if your cluster writes numeric job ID into the filename)**
   ```bash
   cat logs/mention_slot_%j.out
   cat logs/mention_slot_%j.err
   # If your cluster expands %j:
   # cat logs/mention_slot_<JOBID>.out
   ```

4. **After a successful run, check artifacts**
   ```bash
   ls -la results/mention_slot_scorer/
   cat results/mention_slot_scorer/metrics.json
   ```

## Files involved

- **Batch script:** `jobs/wulver_train_mention_slot_2h.slurm`
- **Training script:** `training/train_mention_slot_scorer.py`
- **Data generation:** `training/generate_mention_slot_pairs.py`
- **Data:** `data/processed/mention_slot_pairs.jsonl`, `data/processed/mention_slot_pairs_dev.jsonl`
