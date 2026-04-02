# Learning / Stage 3 Training Environment Audit

**Date:** 2025-03-08  
**Purpose:** Document actual Python/torch/transformers/GPU assumptions for batch training on Wulver so one real learned run can complete.

---

## 1. Current batch Python path

| Context | Python used | Source |
|--------|-------------|--------|
| **run_stage3_experiments.sbatch** | Bare `python` (no activation) | Line 49: `python -m src.learning.run_stage3_experiments` |
| **train_nlp4lp_ranker.sbatch** | Bare `python` | Line 31: `python -m src.learning.train_nlp4lp_pairwise_ranker` |
| **train_multitask_grounder.sbatch** | Bare `python` | Line 41: `python -m src.learning.train_multitask_grounder` |

So **all three** rely on whatever `python` is first in `PATH` when the job starts. On Wulver GPU nodes that is typically the **default Anaconda base** (`/apps/easybuild/software/Anaconda3/2023.09-0/bin/python`), which **does not** have `torch` in its environment (per WULVER_FIRST_TRAINING_RUN.md and inspection).

---

## 2. Torch / transformers on the cluster

- **Login node (inspection):**
  - `which python` → `/apps/easybuild/software/Anaconda3/2023.09-0/bin/python`
  - Anaconda base: **torch MISSING** (No module named 'torch'); **transformers 5.2.0** present (but limited without torch).
  - User site-packages: `pip show torch` → **torch 2.7.0** in `~/.local/lib/python3.11/site-packages` (Anaconda’s `python3` may not prefer this).
- **Repo venv:**
  - `venv/bin/python` → symlink to Anaconda; venv’s site-packages used when activated.
  - With `source venv/bin/activate`: **torch 2.10.0+cu128** and **transformers** available.
- **Conclusion:** For batch jobs we must **activate the repo venv** (or another env that has torch + transformers) so that `python` sees torch. Relying on default `python` without activation leads to “torch/transformers required” and config-only fallback.

---

## 3. CUDA / GPU visibility

- **Login node:** No GPU; `nvidia-smi` not in PATH.
- **GPU partition:** A100 (40G/20G/10G) and L40 nodes; CUDA drivers present on compute nodes.
- **WULVER_RESOURCE_INVENTORY.md:** PyTorch in user site-packages built with CUDA 12; on GPU nodes `torch.cuda.is_available()` should be True when the correct Python (with torch) is used.
- **Batch scripts:** All request `#SBATCH --gres=gpu:1` and `--partition=gpu` (stage3 and train_*). So GPU is allocated; visibility in the job depends on using a Python that has torch with CUDA support (e.g. venv with torch 2.10+cu128).

---

## 4. Existing environment conventions in the repo

- **README.md:** `python -m venv venv && source venv/bin/activate`.
- **docs/NLP4LP_FOCUSED_EVAL.md:** `source venv/bin/activate` or `module load python/3.10`.
- **docs/ENVIRONMENT_RESOURCES_AUDIT.md:** “venv and venv_sbert present at repo root”.
- **jobs/wulver_train_mention_slot_2h.slurm:** Uses **repo venv first**: if `$PROJECT_ROOT/venv/bin/python -c 'import torch'` works, prepends `venv/bin` to PATH; else tries conda envs `CONDA_ENV`, `pt`, `pytorch`, `base`. Prints diagnostics (python, torch version, CUDA available, nvidia-smi). Fails to a smoke path if torch still missing.
- **requirements.txt:** Does **not** list `torch` or `transformers`; they are installed in venv (or user/conda) for GPU training.

**Conclusion:** Reuse the pattern from `jobs/wulver_train_mention_slot_2h.slurm`: prefer **repo `venv`** (activate or set PATH so `python` is `venv/bin/python`), then optionally try conda envs; print diagnostics; fail early with a clear message if torch is missing.

---

## 5. What activation is needed for real training

1. **Set REPO_ROOT** (or PROJECT_ROOT) to the repo directory (e.g. `SLURM_SUBMIT_DIR` or `/mmfs1/home/sv96/combinatorial-opt-agent`).
2. **Use a Python that has torch and transformers:**
   - **Preferred:** If `$REPO_ROOT/venv/bin/python` exists and `venv/bin/python -c 'import torch'` succeeds, set `PATH="$REPO_ROOT/venv/bin:$PATH"` (or `source venv/bin/activate`) so that `python` in the script is this interpreter.
   - **Fallback:** Conda: `source $(conda info --base)/etc/profile.d/conda.sh` then `conda activate <env_with_torch>` (e.g. set `CONDA_ENV`).
3. **Before running training:** Print `which python`, `python --version`, `python -c 'import torch; print(torch.__version__)'`, `torch.cuda.is_available()`, and optionally `nvidia-smi`. Exit with a clear error if `import torch` or `import transformers` fails.
4. **Stage 3 launcher:** Runs as `python -m src.learning.run_stage3_experiments`; it then spawns subprocesses with `sys.executable` for training. So the **same** activated Python (with torch) must be the one used to start the launcher; then all child training calls will use that executable.

---

## 6. Summary table (actual inspection)

| Item | Finding |
|------|---------|
| **Batch Python path** | Unset; default is Anaconda base (no torch). |
| **torch in default python** | No. |
| **transformers in default python** | Yes (5.2.0) but limited without torch. |
| **torch in repo venv** | Yes (2.10.0+cu128 when venv activated). |
| **GPU visible** | Only on compute nodes; requires `--partition=gpu` and `--gres=gpu:1`. |
| **Env convention** | Repo venv; wulver job uses venv-first then conda. |

**Required change:** In `run_stage3_experiments.sbatch`, `train_nlp4lp_ranker.sbatch`, and `train_multitask_grounder.sbatch`: activate repo venv (or conda env with torch) and print diagnostics before running any Python training. Use the same pattern as `jobs/wulver_train_mention_slot_2h.slurm` so one real learned run can train and write a checkpoint + metrics.
