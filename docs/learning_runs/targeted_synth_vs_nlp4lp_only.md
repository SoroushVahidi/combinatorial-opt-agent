# Targeted synthetic aux vs NLP4LP-only

**Job ID:** 854618 (COMPLETED 2026-03-09, exit 0).  
**Logs:** `logs/learning/targeted_synth_comparison_854618.out` / `.err`

**Purpose:** One controlled experiment to test whether **targeted high-precision synthetic** auxiliary data improves the pairwise ranker on held-out NLP4LP test.

**Benchmark validity:** Evaluation is on the same held-out NLP4LP test only. Synthetic data is auxiliary only; labels are template-defined, not gold. This run used a **very small** NLP4LP split on the cluster (1 train instance, 1 test instance); results are therefore **not generalizable** and must be interpreted with caution.

---

## Actual metrics (job 854618)

| Metric | NLP4LP-only baseline | Targeted-synth then NLP4LP |
|--------|----------------------|-----------------------------|
| **pairwise_accuracy** | 0.0 (0/8) | 0.125 (1/8) |
| **slot_selection_accuracy** | 0.0 | 0.125 |
| **exact_slot_fill_accuracy** | 0.0 (0/1 instances) | 0.0 (0/1 instances) |
| **type_match_after_decoding** | **0.875 (7/8)** | **0.0 (0/8)** |
| pair_correct / pair_total | 0 / 8 | 1 / 8 |
| slot_correct / slot_total | 0 / 8 | 1 / 8 |
| total_instances (test) | 1 | 1 |

**Raw counts:** 1 test instance, 8 slot decisions per run.

---

## Experiment description

- **Baseline:** Train pairwise ranker on NLP4LP train only (45 pairs from 1 instance), seed 42, max 200 steps (actual steps: 6 due to data size).
- **Aux run:** Train 50 steps on targeted synth (155 rows), then train on same NLP4LP train (45 pairs) from that checkpoint, max 200 steps (actual: 6).
- **Evaluation:** Same held-out NLP4LP test (1 instance, 8 slots) for both.

---

## Conservative interpretation

1. **Pairwise / slot:** Targeted-synth run got 1 slot correct vs baseline 0. With **8 slot decisions and 1 test instance**, this difference is **not reliable**; it could easily be noise.
2. **Type match:** Baseline achieved 87.5% type match (7/8); targeted-synth run achieved **0%** (0/8). That is a **large drop**. The model after synth pretrain + finetune selected mentions that did not match the expected slot types on this test instance.
3. **Exact slot-fill:** Neither run solved the single test instance exactly (0% exact).
4. **Training:** Both runs saw very little real data (1 train instance). Synth pretrain ran 20 steps; NLP4LP phase ran 6 steps in both cases. Training was stable (loss decreased) but under such small data no strong conclusion can be drawn.

**Likely contributors to the mixed result:**

- **Tiny test set:** One instance (8 decisions) makes any metric difference uninterpretable; a single different prediction flips type_match sharply.
- **Type_match collapse:** Suggests synth pretraining may have shifted the model toward slot–mention associations that do not preserve type semantics on this NLP4LP instance, or the single test instance is type-wise unrepresentative.
- **Formulation / data size:** With only 1 train instance, neither run could learn robust NL→slot behavior; the comparison is underpowered.

---

## Decision: **STOP**

**Do not scale** the targeted-synthetic auxiliary training direction on this evidence.

**Rationale:**

- The **type_match_after_decoding** drop (87.5% → 0%) is a strong negative. It outweighs the tiny pairwise gain (1 vs 0 correct), which is within noise for n=8.
- The test set is **a single instance**; we do not have enough signal to conclude that targeted synth helps. We do have enough to see that it can **hurt** type alignment on this instance.
- Therefore: treat the result as **neutral-to-harmful**. Do not invest in scaling this aux approach (more templates, more data) until there is either (a) a larger, fixed NLP4LP test set and a clear positive effect, or (b) a precise refinement that addresses the type_match drop.

---

## Best next move

1. **Prefer more real data:** Run the same valid benchmark (distinct train/dev/test on NLP4LP) in an environment where the **full** NLP4LP corpus is available (e.g. full split from `split_nlp4lp_corpus_for_benchmark` with 330 instances), so that training and evaluation use many more instances and metrics are interpretable.
2. **Alternative:** Treat learning as secondary for now; keep **deterministic (rule) methods** as the main story and use learning only when a larger, reproducible NLP4LP (or other) benchmark is in place.
3. **Do not:** Revive the broad GAMS weak-label path (already a documented negative). Do not add more targeted-synth templates or scale synth aux until the type_match issue is understood or the evaluation set is larger.

---

## Artifacts

- **Generator:** `tools/build_targeted_synth_ranker_data.py`
- **Synth data:** `artifacts/learning_ranker_data/targeted_synth/train.jsonl`, `stats.json`, `README.md`
- **Batch script:** `batch/learning/train_nlp4lp_targeted_synth_comparison.sbatch`
- **Run dirs:** `artifacts/learning_runs/nlp4lp_only_baseline/`, `targeted_synth_pretrain/`, `targeted_synth_then_nlp4lp/`
- **Metrics:** `artifacts/learning_runs/nlp4lp_only_baseline/metrics.json`, `artifacts/learning_runs/targeted_synth_then_nlp4lp/metrics.json`

## Caveats

- Single seed; no variance. Test set was 1 instance (8 slot decisions) on this run.
- GAMS weak-label experiment (854616) remains a separate negative result; do not revive unless there is a strong new reason.
