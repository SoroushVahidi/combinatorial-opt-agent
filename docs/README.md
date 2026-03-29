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
- **Constrained assignment:** [NLP4LP_CONSTRAINED_ASSIGNMENT_EXAMPLES.md](NLP4LP_CONSTRAINED_ASSIGNMENT_EXAMPLES.md), [NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md](NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md)
- **Optimization role method:** [NLP4LP_OPTIMIZATION_ROLE_METHOD_IMPLEMENTATION.md](NLP4LP_OPTIMIZATION_ROLE_METHOD_IMPLEMENTATION.md), [NLP4LP_OPTIMIZATION_ROLE_METHOD_EXAMPLES.md](NLP4LP_OPTIMIZATION_ROLE_METHOD_EXAMPLES.md), [NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md](NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md)
- **Semantic IR / repair:** [NLP4LP_SEMANTIC_IR_REPAIR_IMPLEMENTATION.md](NLP4LP_SEMANTIC_IR_REPAIR_IMPLEMENTATION.md), [NLP4LP_SEMANTIC_IR_REPAIR_EXAMPLES.md](NLP4LP_SEMANTIC_IR_REPAIR_EXAMPLES.md), [NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md](NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md)
- **Manuscript and reporting:** [NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md](NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md), [NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md](NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md), [NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md](NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md), [NLP4LP_CHATGPT_CLARIFICATION_PACKAGE.md](NLP4LP_CHATGPT_CLARIFICATION_PACKAGE.md)

## HPC and runs

- **[wulver.md](wulver.md)** — NJIT Wulver cluster setup and usage.
- **[wulver_webapp.md](wulver_webapp.md)** — Running the web app on Wulver.
- **[WULVER_RESOURCE_INVENTORY.md](WULVER_RESOURCE_INVENTORY.md)**, **[WULVER_FIRST_TRAINING_RUN.md](WULVER_FIRST_TRAINING_RUN.md)** — Resource inventory and first training run notes.

## Evaluation and tooling

- **[BOTTLENECK_ANALYSIS.md](BOTTLENECK_ANALYSIS.md)** — Identified performance bottlenecks (embedding cache, short-query degradation, downstream instantiation gap, catalog coverage), with quantitative evidence and remediation status.
- **[EVALUATION_LEAKAGE_ANALYSIS.md](EVALUATION_LEAKAGE_ANALYSIS.md)** — Analysis of evaluation data leakage: which functions generate queries, how seeds are used, and leakage remediation.
- **[BASELINE_TABLE_CLI.md](BASELINE_TABLE_CLI.md)** — Baseline table and CLI usage.
- **[PATCH_LEAK_FREE_EVAL.md](PATCH_LEAK_FREE_EVAL.md)** — Leak-free evaluation patch.
- **[VALIDATION_AND_ANALYSIS_CLI.md](VALIDATION_AND_ANALYSIS_CLI.md)** — Validation and analysis CLI.

## Learning and training

- **[LEARNING_AUDIT_ANALYSIS.md](LEARNING_AUDIT_ANALYSIS.md)** — CPU-only bottleneck audit and data-quality analysis of the NLP4LP learning pipeline.
- **[LEARNING_FIRST_REAL_TRAINING_BLOCKER.md](LEARNING_FIRST_REAL_TRAINING_BLOCKER.md)** — Learning Stage 3: first real training blocker, status, and remediation steps.

## Stronger deterministic pipeline

- **[STRONGER_DETERMINISTIC_PIPELINE_PLAN.md](STRONGER_DETERMINISTIC_PIPELINE_PLAN.md)** — Evidence-based design plan for the `global_consistency_grounding` downstream method: what the current pipeline does, why it fails, and the exact stronger design.
- **[STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md](STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md)** — Implementation summary, metrics comparison against all existing methods, honest interpretation, and ChatGPT-ready copy-paste summary.

## Repo maintenance

- **[COPY_PASTE_INFO.md](COPY_PASTE_INFO.md)** — Copy-paste reference card: key numbers, pipeline overview, and pointers to all major docs.
- **[Q1_JOURNAL_AUDIT.md](Q1_JOURNAL_AUDIT.md)** — Senior researcher code audit for Q1 journal submission; references exact file paths, functions, and line ranges.
- **[FULL_REPO_SUMMARY.md](FULL_REPO_SUMMARY.md)** — Comprehensive audit of the current GitHub repo state (generated 2026-03-09).
- **[REPO_CLEANUP_PLAN.md](REPO_CLEANUP_PLAN.md)** — Cleanup plan and deleted-files manifest.
- **[REPO_CLEANUP_DELETED_FILES_MANIFEST.txt](REPO_CLEANUP_DELETED_FILES_MANIFEST.txt)** — Paths deleted by `scripts/cleanup_old_artifacts.sh`; used for reproducible cleanup.

## Experiments

- **[../EXPERIMENTS.md](../EXPERIMENTS.md)** — Consolidated overview of all experiments: retrieval baselines, grounding/assignment methods, learning runs, copilot comparison, and ESWA revision experiments.
- **[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)** — Strict evidence-based audit: which experiments are truly measured vs placeholder/scaffolding only. Updated to reflect benchmark run `22922351003`.
- **[CI_ROOT_CAUSE_AUDIT.md](CI_ROOT_CAUSE_AUDIT.md)** — Root-cause analysis of why prior workflow runs appeared fast (push-triggered stubs) and confirmation that the real benchmark ran in 32 s.
