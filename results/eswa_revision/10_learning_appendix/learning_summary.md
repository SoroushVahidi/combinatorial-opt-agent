# Learning Appendix

**Date:** 2026-03-10  
**Status:** Definitive negative result. Learning is future work.

## Experiment summary

**Source:** `docs/learning_runs/real_data_only_learning_check.md` (job 854626 on GPU cluster)

### Data split
| Set | Instances | Pairwise pairs |
|-----|-----------|----------------|
| Train | 230 | 9,729 |
| Dev | 50 | 2,230 |
| Test | 50 | 2,339 |

Split by `instance_id`, seed 42. No overlap. No test-as-train fallback.
Source: `data/processed/nlp4lp_eval_orig.jsonl` + gold cache `results/paper/nlp4lp_gold_cache.json`.

### Model
`distilroberta-base`, 500 steps, batch_size 8, lr 2e-5, seed 42.

### Results (held-out test split, 50 instances)

| Metric | Learned model | Rule baseline (same split) |
|--------|--------------|---------------------------|
| pairwise_accuracy | 0.197 | 0.247 |
| slot_selection_accuracy | 0.182 | 0.229 |
| exact_slot_fill_accuracy | 0.000 | 0.022 |
| type_match_after_decoding | 0.068 | 0.125 |

**Conclusion:** Learned model is below rule baseline on all metrics.

## Previous invalid runs

| Run | Issue | Validity |
|-----|-------|---------|
| Stage 3 full run (local) | No torch; all fell back to rule baseline | INVALID |
| First learning run (job 854608) | train = test (fallback); models trained on test set | INVALID |

## Recommendation for ESWA

Include a compact learning appendix (1 table) showing:
1. Valid experiment design (distinct splits, no synthetic aux)
2. Honest result: learned < rule on all metrics
3. Brief explanation: limited training data (230 instances), pairwise formulation limitations
4. Frame as future work: larger datasets, better formulation, GPT-4 distillation

## CSV
`results/eswa_revision/13_tables/learning_summary.csv`
