# Stage 3 first results memo

Evidence-based summary of the first experiment round. **No learned runs completed** in the environment where the launcher was run (torch/transformers not available); all reported “learned” metrics are rule-baseline scores. This memo records what ran, what failed, the numbers we have, and the next step.

---

## 1. What actually ran

**Launcher:** `python -m src.learning.run_stage3_experiments --config configs/learning/experiment_matrix_stage3.json --mode local`

- **Completed**
  - **Corpus:** Built NLP4LP corpus (330 records; only test split had local data; train/dev require HF).
  - **Ranker data:** Built from corpus; only `test.jsonl` existed initially. **Test-as-train fallback** used: `train.jsonl` was created by copying `test.jsonl` (see manifest warnings).
  - **NL4Opt aux data:** Built (entity/bound/role JSONL + summary).
  - **Eval:** Ran for all five run names (rule_baseline, nlp4lp_pairwise_text_only, nlp4lp_pairwise_text_plus_features, nl4opt_pretrain_then_finetune, nl4opt_joint_multitask). In every case the **rule baseline** was used (no checkpoint loaded).
  - **Bottleneck slices:** Ran; slice_metrics.json and bottleneck_slice_report.md written.
  - **Collect + report:** stage3_results_summary.json/.md, stage3_comparison_table.csv, stage3_paper_comparison.md written.

- **Failed / blocked**
  - **nlp4lp_pairwise_text_only / nlp4lp_pairwise_text_plus_features:** Trainer reported “Training requires torch. Saving config only.” No checkpoint written; eval fell back to rule.
  - **nl4opt_pretrain_then_finetune / nl4opt_joint_multitask:** `train_multitask_grounder` exited with “torch/transformers required”. No checkpoint; eval fell back to rule.

**Split:** test (ranker data: `artifacts/learning_ranker_data/nlp4lp/test.jsonl`).

**Manifest:** `artifacts/learning_runs/stage3_manifest.json` — `trained: true` only for the two NLP4LP pairwise runs (config-only save); `errors` list includes the two NL4Opt train failures; `warnings` includes `test_as_train_fallback`.

---

## 2. Main result snapshot

All learned-run metrics in this round are **rule baseline** (no model loaded).

| Run | pairwise_acc | slot_acc | exact_slot_fill_acc | type_match | model_source |
|-----|--------------|----------|---------------------|------------|---------------|
| rule_baseline | 0.272 | 0.223 | 0.030 | 0.106 | rule |
| nlp4lp_pairwise_text_only | 0.272 | 0.223 | 0.030 | 0.106 | rule |
| nlp4lp_pairwise_text_plus_features | 0.272 | 0.223 | 0.030 | 0.106 | rule |
| nl4opt_pretrain_then_finetune | 0.272 | 0.223 | 0.030 | 0.106 | rule |
| nl4opt_joint_multitask | 0.272 | 0.223 | 0.030 | 0.106 | rule |

- **Counts:** 270 instances, 1843 slot-level pairs, 1509 with gold; 8 exact-instance correct, 411 slot-level correct.
- **Deterministic baselines** (from `results/paper/`, full pipeline, not directly comparable):
  - **tfidf_optimization_role_repair:** param_coverage 0.822, exact20_on_hits 0.274, instantiation_ready 0.060.
  - **tfidf_optimization_role_relation_repair:** param_coverage 0.821, exact20_on_hits 0.250, instantiation_ready 0.054.

---

## 3. Direct interpretation

- **Did learning beat the rule baseline?** **Unknown.** No learned checkpoint was produced; all metrics are rule scoring.
- **Did any learned run beat optimization_role_repair?** **N/A.** Learned runs did not train; comparison to deterministic baselines is not possible in this round.
- **Did NL4Opt help over NLP4LP-only?** **N/A.** No NL4Opt-augmented (or any) learned model was trained.
- **Which bottleneck slices improved most?** **N/A.** Only rule baseline was evaluated on slices; no learned comparison.
- **Exactness vs readiness?** **Unknown.** Only rule baseline numbers available on ranker eval.

---

## 4. Honest caveats

- **No torch in run environment:** Training scripts exited without saving a real checkpoint; pairwise trainers wrote config only; multitask trainer exited on “torch/transformers required”.
- **Test-as-train fallback:** Because only `test.jsonl` existed for NLP4LP ranker data, the launcher copied it to `train.jsonl`. Any future learned run that uses this setup is effectively training on test data (overfit/minimal-validity setup) unless proper train data (e.g. from HF) is used.
- **Evaluation mismatch:** Learned metrics are on **ranker** (slot-level, test.jsonl); deterministic metrics are on **full pipeline** (query-level, retrieval + downstream). They are not directly comparable.
- **Partial round:** This round validates the pipeline and produces a consistent rule baseline and comparison report, but does not provide evidence on learning or NL4Opt.

---

## 5. Immediate next recommendation

**Run the full experiment round on a GPU node with torch/transformers installed** using the existing batch flow:

```bash
sbatch batch/learning/run_stage3_experiments.sbatch
```

Then re-run collect and report (or rely on the launcher’s built-in collect/report step). That will produce at least:

1. **rule_baseline** (unchanged).
2. **One NLP4LP-only learned run** (nlp4lp_pairwise_text_only or text_plus_features) with a real checkpoint and metrics.
3. **One NL4Opt-augmented run** (nl4opt_pretrain_then_finetune or nl4opt_joint_multitask) if multitask training succeeds.

For a proper train split, ensure NLP4LP train (and optionally dev) are built (e.g. HF token and `build_common_grounding_corpus` for nlp4lp train/dev) so that the test-as-train fallback is not used, or explicitly document “test-as-train” in the memo for that run.

---

## Artifact locations

| Artifact | Path |
|----------|------|
| Manifest | `artifacts/learning_runs/stage3_manifest.json` |
| Per-run metrics | `artifacts/learning_runs/<run_name>/metrics.json`, `metrics.md` |
| Bottleneck slices | `artifacts/learning_runs/bottleneck_slices/slice_metrics.json`, `bottleneck_slice_report.md` |
| Summary | `artifacts/learning_runs/stage3_results_summary.json`, `stage3_results_summary.md` |
| Comparison table | `artifacts/learning_runs/stage3_comparison_table.csv` |
| Paper comparison | `artifacts/learning_runs/stage3_paper_comparison.md` |
