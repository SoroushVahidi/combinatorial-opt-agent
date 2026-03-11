# Documentation index

Overview of documentation in this repository.

## Data and setup

- **[data_sources.md](data_sources.md)** — Canonical list of optimization data sources (OR-Library, Gurobi, NL4Opt, etc.).
- **[open_datasets.md](open_datasets.md)** — Open datasets used for training and evaluation.
- **[PUBLIC_OPTIMIZATION_DATA_COLLECTION.md](PUBLIC_OPTIMIZATION_DATA_COLLECTION.md)** — Public optimization data collection notes.

## GAMSPy

- **[GAMSPY_SETUP_AND_LICENSE.md](GAMSPY_SETUP_AND_LICENSE.md)** — Installing GAMSPy and configuring the license.
- **[GAMSPY_MANUAL_LICENSE_STEP.md](GAMSPY_MANUAL_LICENSE_STEP.md)** — Manual license activation steps if needed.
- **[GAMSPY_DATA_DISCOVERY.md](GAMSPY_DATA_DISCOVERY.md)** — Where GAMSPy and example data were discovered.
- **[GAMSPY_LOCAL_EXAMPLES_COLLECTION.md](GAMSPY_LOCAL_EXAMPLES_COLLECTION.md)** — Local GAMSPy example collection: what was copied/cataloged, counts by family, usefulness.
- **[GAMSPY_LOCAL_EXAMPLES_NEXT_STEPS.md](GAMSPY_LOCAL_EXAMPLES_NEXT_STEPS.md)** — Next steps: which examples to process first, support for reranking/family/roles/train-dev-test.

## NLP4LP (NL-to-optimization)

- **Acceptance rerank:** [NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md](NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md), [NLP4LP_ACCEPTANCE_RERANK_EXAMPLES.md](NLP4LP_ACCEPTANCE_RERANK_EXAMPLES.md), [NLP4LP_ACCEPTANCE_RERANK_RESULTS.md](NLP4LP_ACCEPTANCE_RERANK_RESULTS.md)
- **Constrained assignment:** [NLP4LP_CONSTRAINED_ASSIGNMENT_*.md](NLP4LP_CONSTRAINED_ASSIGNMENT_EXAMPLES.md)
- **Optimization role method:** [NLP4LP_OPTIMIZATION_ROLE_METHOD_*.md](NLP4LP_OPTIMIZATION_ROLE_METHOD_IMPLEMENTATION.md)
- **Semantic IR / repair:** [NLP4LP_SEMANTIC_IR_REPAIR_*.md](NLP4LP_SEMANTIC_IR_REPAIR_IMPLEMENTATION.md)
- **Manuscript and reporting:** [NLP4LP_MANUSCRIPT_*.md](NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md), [NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md](NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md), [NLP4LP_CHATGPT_CLARIFICATION_PACKAGE.md](NLP4LP_CHATGPT_CLARIFICATION_PACKAGE.md)

## HPC and runs

- **[wulver.md](wulver.md)** — NJIT Wulver cluster setup and usage.
- **[wulver_webapp.md](wulver_webapp.md)** — Running the web app on Wulver.
- **[WULVER_RESOURCE_INVENTORY.md](WULVER_RESOURCE_INVENTORY.md)**, **[WULVER_FIRST_TRAINING_RUN.md](WULVER_FIRST_TRAINING_RUN.md)** — Resource inventory and first training run notes.

## Evaluation and tooling

- **[BASELINE_TABLE_CLI.md](BASELINE_TABLE_CLI.md)** — Baseline table and CLI usage.
- **[PATCH_LEAK_FREE_EVAL.md](PATCH_LEAK_FREE_EVAL.md)** — Leak-free evaluation patch.
- **[VALIDATION_AND_ANALYSIS_CLI.md](VALIDATION_AND_ANALYSIS_CLI.md)** — Validation and analysis CLI.

## Experiments

- **[../EXPERIMENTS.md](../EXPERIMENTS.md)** — Consolidated overview of all experiments: retrieval baselines, grounding/assignment methods, learning runs, copilot comparison, and ESWA revision experiments.
- **[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)** — Strict evidence-based audit: which experiments are truly measured vs placeholder/scaffolding only. Updated to reflect benchmark run `22922351003`.
- **[CI_ROOT_CAUSE_AUDIT.md](CI_ROOT_CAUSE_AUDIT.md)** — Root-cause analysis of why prior workflow runs appeared fast (push-triggered stubs) and confirmation that the real benchmark ran in 32 s.

## Repo maintenance

- **[REPO_CLEANUP_PLAN.md](REPO_CLEANUP_PLAN.md)** — Cleanup plan and deleted-files manifest.
