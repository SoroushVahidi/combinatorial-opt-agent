# Experiments

This document provides a consolidated overview of all experiments conducted in this project. Detailed results, implementation notes, and reproduction commands are linked in each section.

> **EAAI manuscript note:** For the current EAAI submission, the authoritative paper-facing results are in `results/paper/eaai_camera_ready_tables/` and `analysis/eaai_*_report.md`. The sections below cover the full experiment history including pre-EAAI exploratory work. Dense retrieval baselines (E5, BGE, SBERT) documented here are supplementary and not in the EAAI main results table. See [`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md) for the canonical paper framing.

> **Audit note:** For a strict evidence-driven breakdown of which experiments are truly measured vs. placeholder/scaffolding, see [docs/EXPERIMENT_AUDIT.md](docs/EXPERIMENT_AUDIT.md).

---

## 1. Problem Retrieval Experiments

**Goal:** Evaluate how well different retrieval baselines match a natural-language query to the correct optimization problem schema.

**Eval set:** 331 NLP4LP queries in three variants — `orig` (original), `noisy` (numeric values replaced with `<num>`), and `short` (abbreviated queries).

**Baselines evaluated:**

| Baseline | Description |
|----------|-------------|
| Random | Theoretical 1/331 ≈ 0.003 for retrieval; empirical 2/331 for downstream |
| LSA | Latent semantic analysis |
| BM25 | BM25 sparse retrieval |
| TF-IDF | TF-IDF cosine similarity (primary production baseline) |
| SBERT (fine-tuned) | Sentence-transformers retrieval, fine-tuned on synthetic pairs |
| E5 (`intfloat/e5-base-v2`) | Dense retrieval with asymmetric prefixes: `"passage: "` on corpus, `"query: "` on queries |
| BGE (`BAAI/bge-large-en-v1.5`) | Dense retrieval with task-specific instruction prefix on queries; passages encoded without prefix |

**Key metric — Schema R@1 (orig):**

| Baseline | Schema R@1 (orig) |
|----------|-------------------|
| Random | 0.003 |
| TF-IDF | 0.906 |
| BM25 | ~0.88 |

**Variant summary:**

| Variant | TF-IDF Schema R@1 |
|---------|--------------------|
| orig | 0.9063 |
| noisy | 0.9033 |
| short | 0.7855 |

**How to reproduce:**

```bash
python -m training.run_baselines \
  --splits data/processed/splits.json --split test \
  --regenerate --num 500 --seed 999 --k 10 \
  --baselines bm25 tfidf lsa sbert e5 bge --results-dir results
```

> **Note:** E5 and BGE require downloading model weights from HuggingFace at runtime. Omit them if offline or if model downloads are unavailable.

**Result files:** `results/baselines_test.csv`, `results/nlp4lp_retrieval_summary.csv`

**Detailed docs:** [docs/BASELINE_TABLE_CLI.md](docs/BASELINE_TABLE_CLI.md), [docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md](docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md)

---

## 2. Downstream Grounding / Assignment Method Experiments

After schema retrieval, the pipeline assigns numeric mentions from the NL query to schema parameter slots. Several assignment methods have been implemented and compared.

**Eval set:** 331 NLP4LP test queries (`orig` variant, plus `noisy` and `short` robustness variants).

**Metrics:**

| Metric | Description |
|--------|-------------|
| `schema_R1` | Fraction of queries where top-1 retrieved schema is correct |
| `param_coverage` | Fraction of expected scalar slots that were filled |
| `type_match` | Fraction of filled slots where the mention type matches expected |
| `exact20_on_hits` | Fraction of schema-hit queries where assigned value is within 20% relative error |
| `instantiation_ready` | Fraction of queries with coverage ≥ 0.8 and type_match ≥ 0.8 |

### 2.1 Typed Greedy Assignment (Baseline)

The primary production baseline. Assigns numeric mentions to schema slots greedily, respecting parameter types.

| Schema | Coverage | TypeMatch | Exact20 | InstantiationReady |
|--------|----------|-----------|---------|-------------------|
| TF-IDF | 0.822 | 0.226 | 0.233 | 0.076 |
| Oracle | 0.870 | 0.240 | 0.204 | 0.082 |

### 2.2 Untyped Assignment (Ablation)

Type-agnostic assignment; used to quantify the contribution of type-awareness.

| Schema | Coverage | TypeMatch | Exact20 | InstantiationReady |
|--------|----------|-----------|---------|-------------------|
| TF-IDF | 0.822 | 0.168 | — | 0.045 |
| Oracle | 0.870 | 0.189 | — | 0.045 |

**Conclusion:** Removing type awareness substantially lowers TypeMatch and InstantiationReady.

### 2.3 Constrained Assignment

Global one-to-one bipartite assignment with strict type constraints. Improves numeric accuracy (Exact20) at the cost of lower coverage.

| Schema | Coverage | TypeMatch | Exact20 | InstantiationReady |
|--------|----------|-----------|---------|-------------------|
| TF-IDF | 0.772 | 0.195 | **0.325** | 0.027 |
| Oracle | 0.820 | 0.205 | **0.315** | 0.021 |

**Conclusion:** More precise on filled slots but leaves more slots empty; InstantiationReady drops significantly.

**Detailed docs:** [docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md](docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md)

### 2.4 Acceptance Reranking

Re-ranks the top-K retrieved schemas by an acceptance scorer before assignment, rather than always taking rank-1. Also tested with a hierarchical family-mismatch penalty variant.

| Method | Schema R@1 | Coverage | TypeMatch | InstantiationReady |
|--------|------------|----------|-----------|-------------------|
| tfidf (baseline) | 0.906 | 0.822 | 0.226 | 0.076 |
| tfidf_acceptance_rerank | 0.876 | 0.797 | 0.228 | **0.082** |
| tfidf_hierarchical_acceptance_rerank | 0.846 | 0.777 | 0.230 | **0.085** |

**Conclusion:** Reranking trades a small amount of Schema R@1 for a small gain in InstantiationReady. The hierarchical variant gives the highest InstantiationReady (0.085) at the lowest Schema R@1.

**Detailed docs:** [docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md](docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md)

### 2.5 Semantic IR + Repair

Uses semantic roles (operator/unit tags) plus a repair pass to fill unfilled slots with type-compatible fallbacks. Improves TypeMatch over constrained assignment and the repair pass partially restores coverage.

| Schema | Coverage | TypeMatch | Exact20 | InstantiationReady |
|--------|----------|-----------|---------|-------------------|
| TF-IDF | 0.778 | **0.254** | 0.261 | 0.063 |
| Oracle | 0.825 | **0.280** | 0.258 | 0.069 |

**Conclusion:** Best TypeMatch among deterministic methods. Coverage and InstantiationReady are between typed and constrained.

**Detailed docs:** [docs/NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md](docs/NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md)

### 2.6 Optimization-Role Repair

Adds optimization-role tags (objective coefficient, bound, RHS) as an extra signal during assignment, then applies a repair pass. Preserves coverage at the level of typed greedy while improving TypeMatch and Exact20.

| Schema | Coverage | TypeMatch | Exact20 | InstantiationReady |
|--------|----------|-----------|---------|-------------------|
| TF-IDF | **0.822** | 0.243 | 0.277 | 0.060 |
| Oracle | **0.869** | 0.269 | 0.270 | 0.069 |

**Conclusion:** Current strongest structured deterministic method for balancing coverage, TypeMatch, and Exact20. InstantiationReady is slightly below typed greedy but much above constrained.

**Detailed docs:** [docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md](docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md)

### 2.7 Optimization-Role + Relation-Aware Repair

Extends the optimization-role repair with relation-aware features (incremental admissible decoding). Results pending full run.

**How to run:**

```bash
python tools/nlp4lp_downstream_utility.py --variant orig \
    --baseline tfidf --assignment-mode optimization_role_relation_repair
```

**Detailed docs:** [docs/NLP4LP_RELATION_AWARE_METHOD_RESULTS.md](docs/NLP4LP_RELATION_AWARE_METHOD_RESULTS.md)

### 2.8 Global Consistency Grounding (GCG)

A new method that enforces global consistency constraints during assignment (e.g. duplicate-mention penalty, percent-misuse penalty, bound-flip penalty). Results pending availability of gated gold data.

**Validated behaviors (from unit tests in `tests/test_global_consistency_grounding.py`):**

| Scenario | GCG behavior |
|----------|-------------|
| Percent slot vs scalar slot | Assigns percent mention to percent slot |
| Per-unit coefficient vs total budget | Assigns correctly sized values to each role |
| Lower bound vs upper bound | Assigns smaller to min, larger to max |
| Global duplicate-mention penalty | Penalizes same mention for two slots |

**How to run (requires HF gold data):**

```bash
python tools/run_nlp4lp_focused_eval.py --variant orig --safe
```

**Detailed docs:** [docs/global_consistency_grounding.md](docs/global_consistency_grounding.md), [docs/learning_runs/global_consistency_grounding_results.md](docs/learning_runs/global_consistency_grounding_results.md)

---

## 3. Robustness Experiments

**Variants tested:** `orig` (standard queries), `noisy` (numeric values replaced with `<num>`), `short` (abbreviated queries).

| Method | orig InstReady | noisy InstReady | short InstReady |
|--------|---------------|-----------------|-----------------|
| tfidf typed | 0.076 | 0.0 | ~0.006 |
| tfidf constrained | 0.027 | 0.0 | ~0.006 |

**Observations:**
- `noisy`: TypeMatch and InstantiationReady collapse to 0 because `<num>` placeholders are not resolvable to numeric values.
- `short`: All metrics are very low due to insufficient query content for reliable mention extraction.

---

## 4. Retrieval Model Fine-Tuning

**Goal:** Fine-tune the sentence-transformers retrieval model on synthetic (query, passage) pairs to improve problem matching.

**Setup:**
- **Base model:** `all-MiniLM-L6-v2` (or similar sentence-transformer)
- **Training data:** Synthetic pairs generated from the problem catalog
- **Optional augmentation:** Real user queries collected from the app (`data/collected_queries/user_queries.jsonl`)

**Evaluation metric:** Precision@1 / Precision@5 on 500 held-out instances.

**How to run:**

```bash
# Generate synthetic training pairs
python -m training.generate_samples \
  --output data/processed/training_pairs.jsonl --instances-per-problem 100

# Train
python -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --output-dir data/models/retrieval_finetuned \
  --epochs 4 --batch-size 32

# Evaluate
python -m training.evaluate_retrieval --regenerate --num 500
```

**Detailed docs:** [training/README.md](training/README.md)

---

## 5. Learning Experiments (NLP4LP Grounding)

Experiments on training a learned pairwise mention–slot ranker for the grounding pipeline.

### 5.1 GAMS Weak-Label Auxiliary Training

**Outcome:** Negative result. TypeMatch collapsed. Do not revive.

### 5.2 Targeted Synthetic Auxiliary Training

**Outcome:** Stopped. TypeMatch collapsed when scaled. Do not revive.

### 5.3 Real-Data-Only Learning Check

The key benchmark for learned grounding: train and evaluate entirely on real NLP4LP data with no synthetic or GAMS auxiliary data. Uses benchmark-safe train/dev/test splits.

**Split:** 330 NLP4LP source records → 230 train / 50 dev / 50 test instances; 9,729 / 2,230 / 2,339 pairwise pairs.

**Training config:**
- Encoder: `distilroberta-base`
- Steps: 500, batch size 8, lr 2e-5, seed 42

**Results (job 854626):**

| Metric | Learned model | Rule baseline |
|--------|---------------|---------------|
| pairwise_accuracy | 0.197 | **0.247** |
| slot_selection_accuracy | 0.182 | **0.229** |
| exact_slot_fill_accuracy | 0.000 | **0.022** |
| type_match_after_decoding | 0.068 | **0.125** |

**Conclusion:** Learned model is below the rule baseline on all metrics. **Learning is documented as future work.**

**Detailed docs:** [docs/learning_runs/real_data_only_learning_check.md](docs/learning_runs/real_data_only_learning_check.md)

### 5.4 Stage 3 Experiment Round

A structured experiment round comparing the rule baseline against five learned configurations:

| Run name | Description |
|----------|-------------|
| `rule_baseline` | No checkpoint; handcrafted rule scoring |
| `nlp4lp_pairwise_text_only` | Pairwise ranker, text only, 200 steps |
| `nlp4lp_pairwise_text_plus_features` | Pairwise ranker, text + structured features, 200 steps |
| `nl4opt_pretrain_then_finetune` | Pretrain on NL4Opt (100 steps) → finetune on NLP4LP (200 steps) |
| `nl4opt_joint_multitask` | Joint multitask training, 300 steps |

**Status of first run:** No learned runs completed (torch/transformers not available in run environment). All reported metrics are rule baseline scores (pairwise_acc 0.272, slot_acc 0.223).

**How to run (GPU cluster required):**

```bash
sbatch batch/learning/run_stage3_experiments.sbatch
```

**Detailed docs:** [docs/LEARNING_STAGE3_EXPERIMENTS.md](docs/LEARNING_STAGE3_EXPERIMENTS.md), [docs/LEARNING_STAGE3_FIRST_RESULTS.md](docs/LEARNING_STAGE3_FIRST_RESULTS.md), [docs/learning_runs/README.md](docs/learning_runs/README.md)

---

## 6. Copilot vs. Our Model Comparison

A benchmark comparing our retrieval-based agent against GitHub Copilot (and GPT-4 acting as Copilot) on a set of handcrafted and NL4Opt test cases.

**Benchmark cases:** `artifacts/copilot_vs_model/benchmark_cases.jsonl`

**Evaluation rubric:** `artifacts/copilot_vs_model/evaluation_rubric.md`

| Component | File |
|-----------|------|
| Benchmark cases | `artifacts/copilot_vs_model/benchmark_cases.jsonl` |
| Copilot prompts (30+) | `artifacts/copilot_vs_model/copilot_prompts/` |
| Prompt template | `artifacts/copilot_vs_model/copilot_prompt_template.md` |
| Run GPT-4 as Copilot | `artifacts/copilot_vs_model/run_gpt4_as_copilot.py` |
| Run our model | `artifacts/copilot_vs_model/run_our_model.py` |
| Score comparison | `artifacts/copilot_vs_model/score_comparison.py` |

**Detailed docs:** [docs/copilot_vs_our_model_comparison.md](docs/copilot_vs_our_model_comparison.md)

---

## 7. ESWA Revision Experiments

Structured experiment suite for the journal paper revision. Results are stored under `results/eswa_revision/` in 14 subdirectories:

| Dir | Description |
|-----|-------------|
| `00_env/` | Environment configuration baseline |
| `01_retrieval/` | Problem retrieval results |
| `03_prefix_vs_postfix/` | Prefix vs postfix schema representation |
| `04_method_comparison/` | Full method comparison (all assignment methods) |
| `05_retrieval_vs_grounding/` | Bottleneck analysis: retrieval vs grounding |
| `06_robustness/` | Robustness across query variants |
| `07_sae/` | Semantic acceptability evaluation |
| `08_error_taxonomy/` | Error type breakdown |
| `09_case_studies/` | Qualitative case studies |
| `10_learning_appendix/` | Learning details (appendix) |
| `11_runtime/` | Runtime performance profiling |
| `12_figures/` | Paper figures |
| `13_tables/` | Aggregate metrics tables |
| `14_reports/` | Human-readable reports |

**Manifest:** `results/eswa_revision/manifests/experiment_manifest.json`

---

## 8. Summary of Key Findings

| Experiment | Key finding |
|------------|-------------|
| Retrieval | TF-IDF achieves Schema R@1 = 0.906 on orig; strong across variants |
| Typed greedy assignment | Primary production baseline; best balance of coverage and InstReady |
| Constrained assignment | Best Exact20 (0.325) but lowest InstantiationReady (0.027) |
| Acceptance reranking | Highest InstantiationReady (0.085 hierarchical) at cost of Schema R@1 |
| Semantic IR repair | Best TypeMatch (0.254) among deterministic methods |
| Optimization-role repair | Best balance of coverage + TypeMatch + Exact20; current recommended method |
| Robustness | Noisy queries collapse downstream metrics; short queries are very limited |
| Real-data learning | Learned model below rule baseline; learning is future work |
| Stage 3 experiments | Infra validated; full GPU run still needed |

---

## 9. Reproducing the Full Benchmark

```bash
# Run all downstream baselines (requires HF gold data)
python tools/run_nlp4lp_focused_eval.py --variant orig --safe

# Aggregate results
python tools/summarize_nlp4lp_results.py

# Build paper artifacts (tables, LaTeX)
python tools/make_nlp4lp_paper_artifacts.py
```

**CI / GitHub Actions:** The `NLP4LP downstream benchmark (authenticated)` workflow (`.github/workflows/downstream_benchmark.yml`) runs the full benchmark matrix. Measured wall-clock time: ~32 seconds for all 30 settings. Trigger from **Actions → NLP4LP downstream benchmark (authenticated) → Run workflow**.

See [HOW_TO_RUN_BENCHMARK.md](HOW_TO_RUN_BENCHMARK.md) for the complete guide.
