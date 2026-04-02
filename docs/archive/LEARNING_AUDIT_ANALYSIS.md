# NLP4LP Learning Audit and Analysis

## Overview

This document describes the CPU-only bottleneck audit and data-quality analysis pass
for the NLP4LP pairwise ranker. These tools require no GPU, no Wulver access, and no
external model inference. They run locally from the repo root in seconds to minutes.

The audit targets the three main downstream grounding failure modes identified previously:
1. Wrong variable/entity association
2. Multiple confusable float-like values
3. Lower vs upper bound confusion

Plus two additional corpus-observed risks:
4. Total vs per-unit confusion
5. Percentage vs absolute value confusion

---

## Scripts and What They Do

### 1. `src/learning/audit_nlp4lp_bottlenecks.py`

**Purpose**: Apply transparent heuristics to the NLP4LP eval corpus to identify
instances likely to be hard for downstream slot grounding.

**Inputs**:
- `data/processed/nlp4lp_eval_orig.jsonl` (331 instances; always available)
- `artifacts/learning_ranker_data/nlp4lp/` (optional; if present, rows are cross-referenced)

**Heuristics** (one per slice):

| Slice | Heuristic |
|-------|-----------|
| `entity_association_risk` | Query contains a title+name (Mr./Mrs./Dr./...) or multi-word proper noun alongside numeric mentions |
| `lower_upper_risk` | Query contains both lower-bound cues ("at least", "minimum", "≥") AND upper-bound cues ("at most", "maximum", "≤") |
| `multi_numeric_confusion` | Query contains ≥3 distinct numeric values (integers, floats, percents, currency) |
| `total_vs_per_unit_risk` | Query uses both aggregate language ("total", "combined", "sum") and per-unit language ("per", "each", "unit") with numbers |
| `percent_vs_absolute_risk` | Query mixes percentage mentions (e.g. "20%") and large absolute dollar/integer values |

**Outputs**:
- `artifacts/learning_audit/bottleneck_audit_summary.json` — counts and fractions per slice
- `artifacts/learning_audit/bottleneck_audit_summary.md` — human-readable summary
- `artifacts/learning_audit/entity_association_risk_examples.jsonl`
- `artifacts/learning_audit/lower_upper_risk_examples.jsonl`
- `artifacts/learning_audit/multi_numeric_confusion_examples.jsonl`
- `artifacts/learning_audit/total_vs_per_unit_risk_examples.jsonl`
- `artifacts/learning_audit/percent_vs_absolute_risk_examples.jsonl`

Each flagged example includes: `instance_id`, `query`, `numeric_mentions`, `heuristic_reason`, `slice`.

---

### 2. `src/learning/check_nlp4lp_pairwise_data_quality.py`

**Purpose**: Inspect pairwise ranker data files for structural quality issues.

**Inputs**:
- `artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl` (optional; reports gracefully if absent)

**Checks performed**:
- Row counts per split
- Positive / negative counts and label balance
- Groups with no positive candidate (training signal leakage risk)
- Groups with multiple positive candidates (ambiguity risk)
- Duplicate rows
- Missing required fields (`slot_name`, `mention_surface`, `group_id`)
- Feature coverage and constant-feature detection for known numeric features
- Mean feature value on positives vs negatives (separation signal)
- Suspicious rows (invalid labels, slot_name == mention_surface, etc.)

**Outputs**:
- `artifacts/learning_audit/pairwise_data_quality.json`
- `artifacts/learning_audit/pairwise_data_quality.md`

---

### 3. `src/learning/analyze_pairwise_features.py`

**Purpose**: Compute simple descriptive statistics for structured features to assess
their expected usefulness as training signals.

**Inputs**:
- `artifacts/learning_ranker_data/nlp4lp/` (preferred; full per-row stats if present)
- `data/processed/nlp4lp_eval_orig.jsonl` (fallback corpus-proxy mode)

**Features analyzed**:

| Feature | Description |
|---------|-------------|
| `type_match` | Inferred numeric type vs slot expected type |
| `operator_cue_match` | Operator cues (≤/≥/=) alignment with slot role |
| `lower_cue_present` | Lower-bound language near the mention |
| `upper_cue_present` | Upper-bound language near the mention |
| `slot_mention_overlap` | Lexical Jaccard overlap between slot name and mention context |
| `entity_match` | Entity/name co-occurrence match |
| `sentence_proximity` | Mention distance to slot-description sentence |

**Per-feature statistics**:
- Coverage (% of rows with non-null value)
- Mean on positive pairs vs mean on negative pairs
- Separation signal (mean_pos − mean_neg)
- Whether the feature is constant (no variance)

**Corpus-proxy mode** (when no pairwise data): measures feature prevalence on eval queries.

**Outputs**:
- `artifacts/learning_audit/pairwise_feature_analysis.json`
- `artifacts/learning_audit/pairwise_feature_analysis.md`

---

### 4. `src/learning/export_manual_inspection_cases.py`

**Purpose**: Export compact, readable sets of hard cases for manual review.

**Categories exported** (up to 25 per category):
- Entity-association-heavy examples
- Lower/upper-bound-heavy examples
- Multi-number confusion examples
- Mixed hard cases (flagged in ≥2 slices)

**Per case includes**:
- `instance_id`, `category`, `heuristic_reason`
- `query_snippet` (first 400 characters)
- `first_sentence`
- `numeric_mentions` (deduplicated)
- `slot_names` and `mention_surfaces` from ranker data (if available)

**Outputs**:
- `artifacts/learning_audit/manual_inspection_cases.md` — human-readable with problem text snippets
- `artifacts/learning_audit/manual_inspection_cases.jsonl` — machine-readable

---

## Findings from the NLP4LP Eval Corpus (331 instances)

| Metric | Value |
|--------|-------|
| Instances flagged in any slice | 320 / 331 (96.7%) |
| Flagged in ≥2 slices | 213 / 331 (64.4%) |
| Entity association risk | 14 (4.2%) |
| Lower/upper bound risk | 116 (35.1%) |
| Multi-numeric confusion | 288 (87.0%) |
| Total vs per-unit risk | 191 (57.7%) |
| Percent vs absolute risk | 43 (13.0%) |
| Avg numeric mentions/query | 7.6 |
| Fraction with lower-bound cue | 66.2% |
| Fraction with upper-bound cue | 46.2% |
| Fraction with both bound cues | 30.2% |

**Main findings**:
1. **Multi-numeric confusion dominates**: 87% of queries have ≥3 distinct numeric values. This is the most pervasive grounding risk.
2. **Total vs per-unit**: 58% of queries mix aggregate and per-unit language — a major confounding factor for slot assignment.
3. **Lower/upper risk**: 35% of queries have both bound types in the same problem statement.
4. **Entity risk is targeted**: Only 14 instances have strong named-entity risk, but these are precisely the hardest for variable-entity association errors.
5. **Pairwise data not yet available**: Feature analysis runs in corpus-proxy mode. Once pairwise data is generated, per-feature separation signals can be computed.

---

## What These Audits Can and Cannot Prove

**Can prove / establish**:
- Which proportion of the corpus falls into each bottleneck category
- Which individual instances are candidate hard cases for downstream grounding
- Whether pairwise training data has structural quality issues (once available)
- Rough feature prevalence and co-occurrence patterns

**Cannot prove**:
- Ground-truth difficulty labels (no gold annotations for grounding errors)
- Actual model accuracy on each slice (requires trained model and eval pipeline)
- Which features actually improve model accuracy (requires ablation experiments)
- Whether heuristic-flagged examples are actually harder (correlation, not causation)

---

## How to Run

### Locally (no Slurm)

```bash
# All four scripts (run from repo root):
bash scripts/learning/run_audit_nlp4lp_bottlenecks.sh
bash scripts/learning/run_check_nlp4lp_pairwise_data_quality.sh
bash scripts/learning/run_analyze_pairwise_features.sh
bash scripts/learning/run_export_manual_inspection_cases.sh

# Or run Python scripts directly:
python src/learning/audit_nlp4lp_bottlenecks.py
python src/learning/check_nlp4lp_pairwise_data_quality.py
python src/learning/analyze_pairwise_features.py
python src/learning/export_manual_inspection_cases.py
```

### On Wulver (Slurm, CPU partition)

```bash
sbatch batch/learning/audit_nlp4lp_bottlenecks.sbatch
sbatch batch/learning/check_nlp4lp_pairwise_data_quality.sbatch
sbatch batch/learning/analyze_pairwise_features.sbatch
sbatch batch/learning/export_manual_inspection_cases.sbatch
```

All sbatch scripts use `--partition=cpu`, `--mem=8G`, `--time=0:30:00`. They do not
require torch, transformers, or GPU resources.

### Override output directory

```bash
OUTPUT_BASE=/my/output/dir bash scripts/learning/run_audit_nlp4lp_bottlenecks.sh
# or
OUTPUT_BASE=/my/output/dir sbatch batch/learning/audit_nlp4lp_bottlenecks.sbatch
```

---

## Output Artifacts Summary

```
artifacts/learning_audit/
  bottleneck_audit_summary.json       ← slice counts and fractions
  bottleneck_audit_summary.md         ← human-readable summary + examples
  entity_association_risk_examples.jsonl
  lower_upper_risk_examples.jsonl
  multi_numeric_confusion_examples.jsonl
  total_vs_per_unit_risk_examples.jsonl
  percent_vs_absolute_risk_examples.jsonl
  pairwise_data_quality.json          ← data quality report (empty if no ranker data)
  pairwise_data_quality.md
  pairwise_feature_analysis.json      ← feature stats (corpus-proxy if no ranker data)
  pairwise_feature_analysis.md
  manual_inspection_cases.jsonl       ← 89 hard cases (25 per category max)
  manual_inspection_cases.md          ← human-readable inspection doc

logs/learning/
  audit_nlp4lp_bottlenecks_<timestamp>.log
  check_pairwise_quality_<timestamp>.log
  analyze_pairwise_features_<timestamp>.log
  export_inspection_cases_<timestamp>.log
```
