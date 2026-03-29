# Real-Data-Only Learning Check — Decision Document

**Purpose:** Answer whether the current learning formulation has any chance when trained and evaluated on enough real NLP4LP data. No synthetic aux, no GAMS weak labels. This is the key benchmark before deciding if learning stays in scope or moves to future work.

**Context (do not revive):**
- **GAMS weak-label aux:** Negative result; do not revive.
- **Targeted synthetic aux:** Stopped; type_match collapsed; do not scale.
- Previous cluster runs used tiny effective splits (e.g. 1 test instance) due to missing `NLP4LP_GOLD_CACHE` or stale corpus; results were not informative.
- Deterministic methods remain the main reliable story.

---

## 1. Data source and split used

- **Source:** `data/processed/nlp4lp_eval_orig.jsonl` (331 lines).
- **Gold:** `results/paper/nlp4lp_gold_cache.json` (split `test`, 331 `gold_by_id` entries). Set via **`NLP4LP_GOLD_CACHE`** when building the corpus; without it the corpus builder produces few or no records.
- **Pipeline:** Build test corpus → split by `instance_id` 70/15/15 (seed 42) → build pairwise ranker data.
- **Chosen split (largest valid in this environment):**
  - **Corpus (instance) counts:** train 230, dev 50, test 50.
  - **Pairwise (ranker) counts:** train 9,729, dev 2,230, test 2,339.
- **Why this split:** It is the full available NLP4LP eval set (330 records with gold), properly split with no train=test fallback and no leakage. Smaller numbers (e.g. 1/0/1 or 45/0/112) came from runs where the gold cache was unset or an old tiny corpus was reused.

**Why cluster runs had tiny splits:** The corpus builder needs `NLP4LP_GOLD_CACHE` pointing to a JSON file with `split` and `gold_by_id`. If unset or wrong path, the test corpus gets few or no records; then split yields 1 train / 0 dev / 1 test (or similar). So for any run (cluster or local), set `NLP4LP_GOLD_CACHE` to e.g. `results/paper/nlp4lp_gold_cache.json` before building the corpus. The batch script sets it by default to that path.

---

## 2. Split integrity proof

| File | Path | Instance (corpus) | Pairs (ranker) |
|------|------|-------------------|----------------|
| train | `artifacts/learning_corpus/nlp4lp_train.jsonl` | 230 | 9,729 |
| dev   | `artifacts/learning_corpus/nlp4lp_dev.jsonl`   | 50  | 2,230 |
| test  | `artifacts/learning_corpus/nlp4lp_test.jsonl`  | 50  | 2,339 |

- **Content/hash:** `verify_split_integrity --data_dir artifacts/learning_ranker_data/nlp4lp` reports: train, dev, test are **distinct** (SHA-256 hashes differ).
- **No leakage:** No overlap between splits; split is by `instance_id` so each instance appears in exactly one of train/dev/test.
- **No train=test fallback:** Benchmark mode only; corpus was built with gold cache then split.

To reproduce integrity check:
```bash
python -m src.learning.verify_split_integrity --data_dir artifacts/learning_ranker_data/nlp4lp
```

---

## 3. Training configuration

- **Model:** NLP4LP pairwise mention–slot ranker (no synthetic aux, no GAMS).
- **Encoder:** `distilroberta-base`.
- **Data:** `artifacts/learning_ranker_data/nlp4lp` (train/dev/test as above).
- **Hyperparameters:** seed 42, max_steps 500, batch_size 8, lr 2e-5, epochs 1.
- **Output:** `artifacts/learning_runs/real_data_only_learning_check/` (checkpoint, config.json, logs).
- **Script:** `batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch` (or `scripts/learning/run_real_data_only_learning_check.sh` locally with a torch-enabled env).

---

## 4. Final metrics (to be filled after run)

After running the training job:

- **Learned model (held-out test):** See `artifacts/learning_runs/real_data_only_learning_check/metrics.json`:
  - `pairwise_accuracy`, `slot_selection_accuracy`, `exact_slot_fill_accuracy`, `type_match_after_decoding`.

- **Rule baseline (same test split):** See `artifacts/learning_runs/rule_baseline_same_test/metrics.json`. Obtained by running the same evaluator **without** `--run_dir` (so it uses the built-in rule scorer).

**Results (job 854626 completed):**

| Metric | Learned model | Rule baseline |
|--------|----------------|----------------|
| pairwise_accuracy | **0.197** | **0.247** |
| slot_selection_accuracy | **0.182** | **0.229** |
| exact_slot_fill_accuracy | **0.000** | **0.022** |
| type_match_after_decoding | **0.068** | **0.125** |

Learned model is **below** the rule baseline on all metrics on the same held-out test split.

---

## 5. Deterministic comparison on same split

- **Baseline:** The evaluator’s built-in rule (handcrafted features: type match, operator cue, slot–mention overlap). When `eval_nlp4lp_pairwise_ranker` is run **without** `--run_dir`, it uses this rule on the same test split.
- **Same split:** Both learned and rule runs use `artifacts/learning_ranker_data/nlp4lp/test.jsonl` (2,339 pairs, 50 instances).
- **Rule-only eval without GPU:** `python -m src.learning.eval_rule_baseline_only --data_dir artifacts/learning_ranker_data/nlp4lp --split test --out_dir artifacts/learning_runs/rule_baseline_same_test` (no transformer load; run on login node to get rule metrics). Rule metrics above were produced this way.

---

## 6. Conclusion

**Not promising; move learning to future work.** On the largest valid real-data-only NLP4LP split (230 train / 50 dev / 50 test instances, 9,729 train pairs), the learned pairwise ranker (500 steps, distilroberta-base) achieved **lower** pairwise, slot, exact-fill, and type-match accuracy than the deterministic rule baseline on the same held-out test. The formulation does not show an advantage over the rule with this data scale and setup; deterministic methods remain the reliable story. Learning is documented as future work.

---

## 7. How to run the check

**On a GPU cluster (recommended):**
```bash
# Ensure gold cache is available (e.g. in repo)
export NLP4LP_GOLD_CACHE="$(pwd)/results/paper/nlp4lp_gold_cache.json"
sbatch batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch
```
Then from the job output or artifact dirs, copy learned and rule metrics into Section 4 and set Section 6 conclusion.

**Locally (with venv/conda that has torch + transformers):**
```bash
export NLP4LP_GOLD_CACHE="$(pwd)/results/paper/nlp4lp_gold_cache.json"
./scripts/learning/run_real_data_only_learning_check.sh
```

---

## 8. Final recommendation (Section F)

- [ ] **Continue learning:** Real-data-only run is clearly competitive or better than rule on the same test.
- [ ] **One last refinement:** Result is inconclusive; one more bounded change (e.g. decoder or data prep) then re-run this same check.
- [x] **Stop and keep learning as future work:** Real-data-only run is not competitive; keep deterministic methods as main story and document learning as future work.

---

## Summary (for reporting back)

| Section | Content |
|---------|---------|
| **A — Largest valid NLP4LP split** | 330 source records → 230 train / 50 dev / 50 test instances; 9,729 / 2,230 / 2,339 pairwise rows. Source: `nlp4lp_eval_orig.jsonl` + `nlp4lp_gold_cache.json` (test). |
| **B — Split integrity** | `verify_split_integrity` OK; train/dev/test hashes distinct; no leakage. |
| **C — Training run** | One run: pairwise ranker, distilroberta-base, seed 42, 500 steps, real data only. Scripts: `batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch`, `scripts/learning/run_real_data_only_learning_check.sh`. |
| **D — Final metrics** | Learned: pairwise 0.197, slot 0.182, exact 0.0, type_match 0.068. Rule: 0.247, 0.229, 0.022, 0.125. |
| **E — Deterministic comparison** | Rule baseline on same test: 0.247 / 0.229 / 0.022 / 0.125. Learned is below rule on all metrics. |
| **F — Final recommendation** | **Stop and keep learning as future work.** (See Section 6.) |
