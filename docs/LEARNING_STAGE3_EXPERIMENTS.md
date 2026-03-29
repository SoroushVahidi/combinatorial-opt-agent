# Stage 3: First experiment round

First evidence-producing experiment suite for the learned downstream model: compare rule baseline, NLP4LP-only learned runs, NL4Opt-augmented runs, and deterministic downstream baselines.

## Baseline artifacts (deterministic)

See **docs/LEARNING_EXPERIMENT_BASELINES.md** for exact paths and metrics. Primary comparison baseline: **tfidf_optimization_role_repair** (results in `results/paper/nlp4lp_downstream_orig_tfidf_optimization_role_repair.json`). Secondary: **tfidf_optimization_role_relation_repair**.

## Experiment matrix

**configs/learning/experiment_matrix_stage3.json** defines:

1. **rule_baseline** — no checkpoint; handcrafted rule scoring
2. **nlp4lp_pairwise_text_only** — Stage 1 pairwise, text-only, 200 steps
3. **nlp4lp_pairwise_text_plus_features** — Stage 1 pairwise, text + structured features, 200 steps
4. **nl4opt_pretrain_then_finetune** — Stage 2: pretrain on NL4Opt aux (100 steps), finetune on NLP4LP (200 steps)
5. **nl4opt_joint_multitask** — Stage 2: joint training 300 steps, all aux heads

Defaults: encoder distilroberta-base, eval_split test, decoder argmax.

## How to run the full round

**Print what would run (no execution):**
```bash
./scripts/learning/run_stage3_experiments.sh print_only
```

**Run locally (corpus/ranker/aux if needed, train all runs, eval, bottleneck slices, then collect + report):**
```bash
./scripts/learning/run_stage3_experiments.sh local
```

**Force rerun of completed steps:**
```bash
./scripts/learning/run_stage3_experiments.sh local --force
```

**Submit as a single batch job:**
```bash
sbatch batch/learning/run_stage3_experiments.sbatch
# Or: FORCE=1 sbatch batch/learning/run_stage3_experiments.sbatch
```

**After a partial run (e.g. only eval), gather results and build report:**
```bash
./scripts/learning/run_collect_stage3_results.sh
./scripts/learning/run_build_stage3_comparison_report.sh
# Or via batch: sbatch batch/learning/collect_stage3_results.sbatch
#              sbatch batch/learning/build_stage3_comparison_report.sbatch
```

## Outputs produced

| Location | Description |
|----------|-------------|
| **artifacts/learning_runs/stage3_manifest.json** | Run manifest: which runs were trained, metrics written, bottleneck done, errors |
| **artifacts/learning_runs/<run_name>/metrics.json** | Per-run metrics (run_name, split, pairwise_accuracy, slot_selection_accuracy, exact_slot_fill_accuracy, type_match_after_decoding, model_source, training_mode, use_structured_features, use_nl4opt_*) |
| **artifacts/learning_runs/<run_name>/metrics.md** | Human-readable summary for that run |
| **artifacts/learning_runs/bottleneck_slices/slice_metrics.json** | Slice metrics per run (overall, multiple_float_like, lower_upper_cues, multi_entity) |
| **artifacts/learning_runs/bottleneck_slices/bottleneck_slice_report.md** | Slice report markdown |
| **artifacts/learning_runs/stage3_results_summary.json** | Collected learned + deterministic + slices (from collect script) |
| **artifacts/learning_runs/stage3_results_summary.md** | Summary markdown |
| **artifacts/learning_runs/stage3_comparison_table.csv** | Learned runs comparison table |
| **artifacts/learning_runs/stage3_paper_comparison.md** | Paper-ready comparison: learned table, deterministic tables, bottleneck table, evidence-based interpretation |

## Prerequisites

- **Corpus and ranker data:** If `artifacts/learning_ranker_data/nlp4lp/train.jsonl` or `test.jsonl` are missing, the launcher runs `build_common_grounding_corpus` and `build_nlp4lp_pairwise_ranker_data` (requires NLP4LP corpus / HF or NLP4LP_GOLD_CACHE).
- **NL4Opt aux data:** Required for nl4opt_pretrain_then_finetune and nl4opt_joint_multitask; launcher runs `build_nl4opt_aux_data` if needed.
- **Deterministic baselines:** For the comparison report, `results/paper/nlp4lp_downstream_orig_tfidf_optimization_role_repair.json` (and optionally relation_repair) must exist. Produce with: `python tools/run_nlp4lp_focused_eval.py --variant orig --safe`.

## Boundaries

- TAT-QA/FinQA are not used in this round.
- Conclusions are limited to the first experiment round; no overstated claims.
- Deterministic numbers are read from actual repo artifacts only.
