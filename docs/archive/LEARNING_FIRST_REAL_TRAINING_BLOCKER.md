# Learning Stage 3: First Real Training Blocker

**Status:** ⛔ Blocked — training not yet run  
**Date:** 2026-03-09

---

## What was attempted

1. Audited all existing training infrastructure and Slurm scripts.
2. Created environment check utility (`src/learning/check_training_env.py`) and ran it locally.
3. Created all required batch scripts and shell scripts for Stage-3 experiments.
4. Created the experiment matrix configuration (`configs/learning/experiment_matrix_stage3.json`).
5. Fixed `scripts/train_retrieval_gpu.slurm` with robust environment activation.

## Exact blocker

**`torch` and `transformers` are not installed in the current Python environment.**

The environment check reports:

```
=== Required packages ===
  [✗] torch: No module named 'torch'
  [✗] transformers: No module named 'transformers'
  [✗] sentence_transformers: No module named 'sentence_transformers'
  [✗] datasets: No module named 'datasets'
  [✗] accelerate: No module named 'accelerate'

Overall required-package check: FAILED ✗
```

Additionally, the two NLP4LP learning data directories are missing:

```
  [✗] artifacts/learning_ranker_data/nlp4lp: MISSING
  [✗] artifacts/learning_aux_data/nl4opt: MISSING
```

And the training Python scripts themselves are not yet implemented:

- `src/learning/train_nlp4lp_pairwise_ranker.py` — **not yet created**
- `src/learning/train_multitask_grounder.py` — **not yet created**
- `src/learning/eval_nlp4lp_pairwise_ranker.py` — **not yet created**
- `src/learning/eval_bottleneck_slices.py` — **not yet created**
- `src/learning/run_stage3_experiments.py` — **not yet created**

## Whether check_training_env passed

**No.** `scripts/learning/run_check_training_env.sh` exits with code 1 (packages missing).

On Wulver with the correct conda env activated (one that has `torch`, `transformers`, etc.),
the check would pass for packages. The data artifact directories would still be missing.

## Files changed / created

### New files
| File | Purpose |
|------|---------|
| `src/learning/check_training_env.py` | Lightweight env diagnostic (Python, packages, CUDA, artifacts) |
| `batch/learning/check_training_env.sbatch` | Slurm job to run env check on a GPU node |
| `batch/learning/train_nlp4lp_ranker.sbatch` | Slurm job for NLP4LP pairwise ranker training |
| `batch/learning/train_multitask_grounder.sbatch` | Slurm job for multitask grounder training |
| `batch/learning/run_stage3_experiments.sbatch` | Slurm orchestration for full Stage-3 matrix |
| `scripts/learning/run_check_training_env.sh` | Local launcher for env check |
| `scripts/learning/run_stage3_experiments.sh` | Local/sequential Stage-3 launcher |
| `configs/learning/experiment_matrix_stage3.json` | Stage-3 experiment matrix |

### Modified files
| File | Change |
|------|--------|
| `scripts/train_retrieval_gpu.slurm` | Robust env activation, `set -euo pipefail`, pre-flight checks, `OUTPUT_BASE` support, `%x_%j` log names |

## Minimum next actions needed

### Step A — activate the correct conda env on Wulver

```bash
# Check what envs exist on the cluster
conda env list

# Activate the env that has torch (e.g. 'pt' or 'pytorch')
export CONDA_ENV=pt

# Verify
sbatch batch/learning/check_training_env.sbatch
```

### Step B — create the learning data artifacts

Data directories for NLP4LP and NL4Opt learning must be prepared. The existing data generation
scripts are for retrieval training pairs; new data-prep scripts for the pairwise ranker and
multitask grounder still need to be written.

Interim: the existing `data/processed/mention_slot_pairs.jsonl` can serve as a starting point
for the pairwise ranker data.

### Step C — implement the training scripts

The actual training and evaluation scripts in `src/learning/` must be implemented:

1. `src/learning/train_nlp4lp_pairwise_ranker.py`
2. `src/learning/train_multitask_grounder.py`
3. `src/learning/eval_nlp4lp_pairwise_ranker.py`
4. `src/learning/eval_bottleneck_slices.py`
5. `src/learning/run_stage3_experiments.py`

### Step D — run the priority experiment

Once A–C are done, submit the priority run:

```bash
cd /mmfs1/home/sv96/combinatorial-opt-agent
export CONDA_ENV=pt
RUN_NAME=nlp4lp_pairwise_text_only sbatch batch/learning/train_nlp4lp_ranker.sbatch
```

Monitor: `squeue -u $USER`

## Environment activation pattern (established)

The existing repo uses this cascade (also used in all new batch scripts):

```bash
# a) try CONDA_ENV if set
# b) try conda env named combinatorial-train, pt, or pytorch
# c) fall back to project venv (venv/bin/activate)
# d) fall back to module load python/3.10
```

See `docs/WULVER_RESOURCE_INVENTORY.md` — the inventory shows `torch 2.7.0` is available at
user site-packages on the Wulver login node. The GPU compute nodes may need a conda env.

## Infrastructure ready

All batch scripts, shell scripts, and the environment check utility are in place.
Once the Python env and training scripts are ready, Stage-3 experiments can be launched
immediately with:

```bash
sbatch batch/learning/check_training_env.sbatch          # verify env
sbatch batch/learning/train_nlp4lp_ranker.sbatch         # priority run
sbatch batch/learning/train_multitask_grounder.sbatch    # optional second run
sbatch batch/learning/run_stage3_experiments.sbatch      # full matrix
```
