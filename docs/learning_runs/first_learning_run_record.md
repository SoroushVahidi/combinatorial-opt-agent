# First Learning Run Record

**Date:** 2025-03-09  
**Purpose:** First successful learned model for NLP4LP number-to-slot grounding.

---

## 1. Run summary

| Field | Value |
|-------|-------|
| **Job ID** | 854608 |
| **Status** | Completed successfully |
| **Task** | Pairwise mention-slot ranker (binary classification) |
| **Dataset** | NLP4LP (`artifacts/learning_ranker_data/nlp4lp`) |
| **Train size** | ~14,298 pairs (train.jsonl) |
| **Encoder** | distilroberta-base |
| **Max steps** | 200 |
| **Batch size** | 8 |
| **LR** | 2e-5 |

---

## 2. Datasets used

- **NLP4LP** (primary): `artifacts/learning_ranker_data/nlp4lp/train.jsonl`, `test.jsonl`
- Source: Common grounding corpus built from `data/processed/nlp4lp_eval_orig.jsonl` + gold from `NLP4LP_GOLD_CACHE` or HF `udell-lab/NLP4LP`
- No NL4OPT, Text2Zinc, or ORQA in this run.

---

## 3. Task formulation

- **Input:** (slot_name, slot_role, mention_surface, context) formatted as text
- **Output:** Binary label (1 = correct slot–mention pair, 0 = incorrect)
- **Model:** PairwiseRanker (DistilRoBERTa + optional structured features; this run uses text-only)
- **Training:** Binary cross-entropy, AdamW, 200 steps

---

## 4. Entrypoints and paths

| Item | Path |
|------|------|
| **Batch script** | `batch/learning/train_nlp4lp_first_run.sbatch` |
| **Submit helper** | `scripts/learning/submit_train_first_run.sh` |
| **Training module** | `src.learning.train_nlp4lp_pairwise_ranker` |
| **Config** | Inline in batch script (no separate config file) |

---

## 5. Cluster environment

| Item | Value |
|------|-------|
| **Cluster** | Wulver |
| **Partition** | gpu |
| **QoS** | standard |
| **Account** | ikoutis |
| **Resources** | 1 GPU, 24G RAM, 4 CPUs, 2h walltime |
| **Python** | Repo venv (torch 2.10+cu128, transformers) |

---

## 6. Output locations

| Artifact | Path |
|----------|------|
| **Checkpoint** | `artifacts/learning_runs/first_learning_run/checkpoint.pt` |
| **Config** | `artifacts/learning_runs/first_learning_run/config.json` |
| **Stdout** | `logs/learning/train_nlp4lp_first_run_854608.out` |
| **Stderr** | `logs/learning/train_nlp4lp_first_run_854608.err` |

---

## 7. Status

- **Submitted:** 2025-03-09
- **Model trained:** Yes (200 steps completed)
- **Checkpoint exists:** Yes — `artifacts/learning_runs/first_learning_run/checkpoint.pt` (~313 MB)

---

## 8. Next steps after this run

1. **Verify checkpoint:** `ls -la artifacts/learning_runs/first_learning_run/checkpoint.pt`
2. **Run evaluation:** Use `src.learning.eval_nlp4lp_pairwise_ranker` with `--checkpoint` pointing to the checkpoint
3. **Compare to rule baseline:** Run bottleneck slice eval with learned model vs rule
4. **Optional:** Run full Stage 3 launcher (`sbatch batch/learning/run_stage3_experiments.sbatch`) for multi-run comparison

---

## 9. Commands used

```bash
# Submit
sbatch batch/learning/train_nlp4lp_first_run.sbatch

# Monitor
squeue -u $USER

# Check logs (after job starts)
tail -f logs/learning/train_nlp4lp_first_run_854608.out
```
