# First Learning Run — Scientific Validity Audit (Copy-Paste for ChatGPT)

---

## SECTION 1 — RUN SUMMARY

| Field | Value |
|-------|-------|
| **Training path** | NLP4LP pairwise mention-slot ranker (binary classification) |
| **Training entrypoint** | `src.learning.train_nlp4lp_pairwise_ranker` |
| **Batch script** | `batch/learning/train_nlp4lp_first_run.sbatch` |
| **Submit helper** | `scripts/learning/submit_train_first_run.sh` |
| **SLURM job id** | 854608 |
| **Job final status** | COMPLETED |
| **Start time** | Mon Mar 9 18:35:21 EDT 2026 (from log) |
| **End time** | Mon Mar 9 18:36 (approx; ~1 min runtime) |
| **Log path** | `logs/learning/train_nlp4lp_first_run_854608.out` |
| **Stderr path** | `logs/learning/train_nlp4lp_first_run_854608.err` |
| **Checkpoint path** | `artifacts/learning_runs/first_learning_run/checkpoint.pt` |
| **Config path** | `artifacts/learning_runs/first_learning_run/config.json` |
| **Output directory** | `artifacts/learning_runs/first_learning_run/` |

---

## SECTION 2 — EXACT COMMANDS AND CONFIG

**Exact sbatch command used:**
```bash
sbatch batch/learning/train_nlp4lp_first_run.sbatch
```

**Exact training command executed inside the batch job:**
```bash
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name first_learning_run \
  --data_dir /mmfs1/home/sv96/combinatorial-opt-agent/artifacts/learning_ranker_data/nlp4lp \
  --save_dir /mmfs1/home/sv96/combinatorial-opt-agent/artifacts/learning_runs \
  --encoder distilroberta-base \
  --epochs 1 \
  --max_steps 200 \
  --lr 2e-5 \
  --batch_size 8
```

**Effective hyperparameters:**
| Parameter | Value |
|-----------|-------|
| epochs | 1 |
| max_steps | 200 |
| batch_size | 8 |
| lr | 2e-5 |
| max_length | 256 (hardcoded in train script) |
| model | distilroberta-base |
| optimizer | AdamW (hardcoded) |
| scheduler | None |
| random seed | Not set (no seed in training loop) |
| use_structured_features | false |

**GPU usage:** Yes. Log shows:
- `CUDA available: True`
- `Device count: 1`
- `NVIDIA A100-SXM4-80GB` (80GB VRAM)
- `ranker.model = ranker.model.to(device)` in train script

---

## SECTION 3 — DATA SOURCES AND SPLITS

**Paths:**
| Split | Path | Lines |
|-------|------|-------|
| train | `artifacts/learning_ranker_data/nlp4lp/train.jsonl` | 14,298 |
| dev | (none) | — |
| test | `artifacts/learning_ranker_data/nlp4lp/test.jsonl` | 14,298 |

**Are train and test distinct?** **NO.** They are byte-identical:
```
md5sum: 159d4e17f63239d6b62debff09290ab8 (both files)
```

**Fallback logic:** Yes. In `src/learning/run_stage3_experiments.py` (lines 101–111):
```python
# Fallback: if train.jsonl missing (e.g. no HF train/dev), use test as train for minimal learned run
if not (data_dir / "train.jsonl").exists() and (data_dir / "test.jsonl").exists():
    import shutil
    shutil.copy(data_dir / "test.jsonl", data_dir / "train.jsonl")
    print("Warning: train.jsonl missing; used test.jsonl as train (test-as-train fallback)", file=sys.stderr)
```

**Data source chain:**
1. `build_common_grounding_corpus` writes NLP4LP corpus by split. For `nlp4lp` with `split=all`, it builds test from `NLP4LP_GOLD_CACHE` or HF `udell-lab/NLP4LP` test; train/dev require HF `load_dataset` for train/dev splits.
2. `build_nlp4lp_pairwise_ranker_data` reads corpus from `artifacts/learning_corpus/` and writes `{split}.jsonl` per split. It only writes splits that exist in the corpus.
3. Corpus was built with only `test` split (train/dev missing because HF train/dev were not loaded or not available).
4. So only `test.jsonl` was produced. The Stage 3 launcher (or a prior run) then copied `test.jsonl` → `train.jsonl` via the fallback.

**Plain-English conclusion:**
> **Potential leakage: train uses test fallback.** Train and test are identical. The model was trained on the same data it would be evaluated on. Any reported metrics are invalid for scientific comparison; they are exploratory/smoke-test only.

---

## SECTION 4 — LOG HEALTH CHECK

**Training completed normally:** Yes. Log ends with:
```
Saved to .../first_learning_run (steps=200)
=== Training complete ===
SUCCESS: checkpoint.pt written to ...
```

**Warnings:** None in stdout. Stderr shows only:
- HuggingFace weight-loading progress (UNEXPECTED keys for lm_head — normal when loading encoder-only for classification)
- No NaNs, no errors

**Loss trend:** Not logged. The trainer does not print loss per step.

**Final train metrics:** None reported. Training script does not compute or log validation/train accuracy.

**Checkpoint save:** Confirmed. `checkpoint.pt` 328,520,634 bytes.

**Systems perspective:** Healthy. Venv activated, GPU used, no crashes.

---

## SECTION 5 — CHECKPOINT / ARTIFACT CHECK

| Check | Result |
|-------|--------|
| **Checkpoint exists** | Yes |
| **Checkpoint size** | 328,520,634 bytes (~313 MB) |
| **Loadable** | Yes. `torch.load(..., map_location='cpu', weights_only=True)` succeeds. |
| **Keys** | `model_state`, `encoder_name` |
| **Model class** | `PairwiseRanker` / `_PairwiseRankerModule` (DistilRoBERTa + linear head) |
| **Config exists** | Yes |
| **Config contents** | `{"encoder": "distilroberta-base", "use_features": false, "steps": 200}` |

---

## SECTION 6 — EVALUATION READINESS

**Evaluation script:** `src.learning.eval_nlp4lp_pairwise_ranker`

**Exact command to evaluate this checkpoint:**
```bash
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir artifacts/learning_ranker_data/nlp4lp \
  --run_dir artifacts/learning_runs/first_learning_run \
  --split test \
  --decoder argmax
```

**Scientifically valid?** **No.** Because train and test are identical:
- Test metrics are inflated (model has seen test data during training).
- Comparison to deterministic baselines is invalid until a proper train/test split is used.

**Useful for smoke testing?** Yes. Running the eval confirms the checkpoint loads and the pipeline runs end-to-end.

**Before valid benchmark comparison:**
1. Build proper train/dev from HF `udell-lab/NLP4LP` (train/dev splits) or another held-out source.
2. Retrain with distinct train and test.
3. Re-run evaluation on held-out test.

---

## SECTION 7 — WHAT CHATGPT SHOULD SEE NEXT

### 3 most important issues
1. **Data leakage:** Train and test are identical (test-as-train fallback). Results are exploratory only.
2. **No proper train split:** NLP4LP train/dev from HF were not used; corpus had only test.
3. **No loss/metric logging:** Training did not log loss or validation metrics; no way to monitor convergence.

### 3 most important next actions
1. **Build proper train/dev:** Run `build_common_grounding_corpus` with HF token and `--split all` so train and dev exist, then rebuild ranker data; ensure no test-as-train fallback.
2. **Retrain with distinct splits:** Run training again with real train.jsonl ≠ test.jsonl.
3. **Add loss logging:** Add loss (and optionally validation accuracy) logging to the training loop for future runs.

### File contents to paste next into ChatGPT
- `artifacts/learning_runs/first_learning_run/config.json`
- The fallback block from `src/learning/run_stage3_experiments.py` (lines 101–111)
- Output of `md5sum artifacts/learning_ranker_data/nlp4lp/train.jsonl artifacts/learning_ranker_data/nlp4lp/test.jsonl`
