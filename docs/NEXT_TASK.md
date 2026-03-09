# Next Task

## Objective

Run the NLP4LP real-benchmark evaluation for `global_consistency_grounding` (GCG)
on a machine with network access to `huggingface.co`, then submit the Stage-3
pairwise ranker training job on the Wulver GPU cluster.

## Why It Matters

- GCG is fully implemented and tested but its real `Exact20` number is **unknown**.
  All paper comparisons for GCG are currently placeholders (`_TBD_`).
- The pairwise ranker (`nlp4lp_pairwise_text_only`) is the simplest, fastest
  Stage-3 run. If it works, it gives the first learning-based Exact20 number —
  the primary proof-of-concept for the learned downstream approach.
- Both tasks unblock the paper's results section.

## Files Involved

| File | Role |
|------|------|
| `tools/nlp4lp_downstream_utility.py` | Run GCG real eval |
| `src/learning/build_nlp4lp_pairwise_ranker_data.py` | Build pairwise training data |
| `src/learning/train_nlp4lp_pairwise_ranker.py` | Train ranker |
| `src/learning/eval_nlp4lp_pairwise_ranker.py` | Evaluate ranker |
| `batch/learning/run_stage3_experiments.sbatch` | Full Stage-3 SLURM job |
| `configs/learning/experiment_matrix_stage3.json` | Experiment definitions |
| `docs/LEARNING_STAGE3_FIRST_RESULTS.md` | Update with actual results |

## Step-by-Step

### Step 1 — GCG real eval (requires HF network access)

```bash
# From repo root, in an environment with: pip install datasets sentence-transformers
python -m tools.nlp4lp_downstream_utility \
  --variant orig \
  --baseline tfidf \
  --assignment-mode global_consistency_grounding \
  --split test
# Repeat for: semantic_ir_repair, optimization_role_repair (baseline comparison)
```

Record Coverage, TypeMatch, Exact20, InstReady in `docs/GCG_FINAL_EVAL_REPORT.md`
(replace all `_TBD_` placeholders).

### Step 2 — Stage-3 pairwise ranker (requires Wulver GPU node)

```bash
# From repo root on Wulver login node:
sbatch batch/learning/run_stage3_experiments.sbatch
# OR run the simplest single experiment first:
sbatch batch/learning/train_nlp4lp_ranker.sbatch
```

After completion:
```bash
python src/learning/eval_nlp4lp_pairwise_ranker.py \
  --model-dir artifacts/learning_runs/nlp4lp_pairwise_text_only \
  --data-dir artifacts/learning_ranker_data/nlp4lp
```

## Success Criteria

- GCG: `Exact20` ≥ 0.277 (beats `optimization_role_repair`) on orig/tfidf.
- Pairwise ranker: training completes, checkpoint saved, eval Exact20 > rule baseline.
- `docs/LEARNING_STAGE3_FIRST_RESULTS.md` updated with real numbers.

## Constraints

- Do **not** retrain the retrieval model — retrieval is already strong.
- Do **not** change `tools/nlp4lp_downstream_utility.py` scoring logic without adding tests.
- Do **not** commit `artifacts/learning_runs/` or `logs/` — they are gitignored.
- GCG eval may take 10–30 min for 331 queries on CPU; use `--workers 4` if available.
