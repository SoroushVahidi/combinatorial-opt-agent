# Evaluation Rubric — Copilot vs Our Model

## Overview

This rubric is used to score both systems on the 30-case benchmark
(`artifacts/copilot_vs_model/benchmark_cases.jsonl`).

All scoring is automated where possible.  Where exact gold is unavailable (NLP4LP cases
from HuggingFace, which require authentication), proxy metrics based on the available
catalog schema text are used.  The 4 hand-crafted cases (`handcrafted_*`) have full gold
and are scored with exact comparisons.

---

## Dimension 1 — Schema / Problem-Type Correctness

**What it measures:** Did the system identify the correct optimization schema / problem type?

### For our model (tfidf retrieval)

| Condition | Score |
|---|---|
| `predicted_schema_id == gold_schema_id` | 1.0 |
| predicted schema is close (top-3 retrieval hit) | 0.5 |
| wrong schema | 0.0 |

Automated: compare `predicted_schema_id` to `gold_schema_id` in benchmark manifest.

### For Copilot

| Condition | Score |
|---|---|
| `predicted_problem_type` matches expected family (manually judged or heuristic) | 1.0 |
| partially correct (e.g. says "lp" when it is "investment_lp") | 0.5 |
| wrong or missing | 0.0 |

Heuristic: check if the predicted_problem_type string is a substring / overlap of the gold schema family tag.

---

## Dimension 2 — Numeric Grounding Quality

**What it measures:** Were numeric values assigned to the correct parameter slots?

### Sub-metric 2a — Coverage (slot recall)

$$\text{coverage} = \frac{|\text{filled slots} \cap \text{expected slots}|}{|\text{expected slots}|}$$

Where "expected slots" = `gold_scalar_slots` from the benchmark manifest.

Scored 0.0 – 1.0.

### Sub-metric 2b — Type correctness

For each filled slot:
- percent slot (name contains "percent"/"rate"/"fraction") filled with value in [0,1]: +1
- non-percent slot filled with value > 1: +1
- wrong type mapping: +0

Scored as fraction of filled slots that pass the type check.

### Sub-metric 2c — Value correctness (hand-crafted cases only)

For cases with `gold_param_values`:

$$\text{exact\_match} = \mathbf{1}\left[|v_{\text{pred}} - v_{\text{gold}}| / \max(1, |v_{\text{gold}}|) \leq 0.01\right]$$

Exact match rate across all gold-parameterized slots.

### Sub-metric 2d — No hallucinated values

Count slots assigned values that are NOT present as numbers anywhere in the input text.
Penalty: subtract 0.1 per hallucinated value, floored at 0.

---

## Dimension 3 — End-to-End Instantiation Usefulness

**What it measures:** Is the output usable / close to usable as a complete optimization problem instance?

Defined as:

$$\text{instantiation\_ready} = \mathbf{1}\left[\text{schema\_correct} = 1 \text{ AND } \text{coverage} \geq 0.8\right]$$

Binary: 1 if both schema was found correctly AND at least 80 % of slots were filled.

---

## Dimension 4 — Objective Direction Correctness

**What it measures:** Did the system correctly identify maximize vs minimize?

| Condition | Score |
|---|---|
| Correct direction | 1.0 |
| Unknown (when direction is clear from text) | 0.5 |
| Wrong direction | 0.0 |

Heuristic gold: if query or schema contains "maximize" → gold = "maximize"; "minimize" → "minimize".

---

## Aggregate Score

$$\text{overall} = 0.30 \cdot \text{schema} + 0.35 \cdot \text{grounding\_coverage} + 0.20 \cdot \text{type\_correct} + 0.10 \cdot \text{objective\_dir} + 0.05 \cdot \text{no\_hallucination}$$

The weights reflect the priority: getting the right schema structure matters most,
followed by filling the right slots with the right values.

---

## Winner Determination Per Case

| Condition | Winner |
|---|---|
| our_model_overall > copilot_overall + 0.05 | our_model |
| copilot_overall > our_model_overall + 0.05 | copilot |
| else | tie |

The 0.05 tolerance avoids declaring a winner when scores are essentially equal.

---

## Limitations of This Rubric

1. **No HF gold for 26/30 NLP4LP cases:** Value-correctness (2c) is only computable on the 4 hand-crafted cases. For the remaining 26, coverage is used as a proxy.
2. **Copilot schema scoring is heuristic:** Without retrieving from the same 331-schema catalog, we use string-overlap on problem type names.
3. **Potential contamination:** Copilot may have seen the NLP4LP test problems during pre-training. The 4 hand-crafted cases are the cleanest comparison point.
4. **Manual Copilot execution:** The comparison is partially manual — Copilot outputs must be collected by a human pasting prompts into Copilot Chat and recording responses.
5. **Single retriever baseline:** Our system uses TF-IDF. Using BM25 or learning-based retrieval would likely improve our schema recall.

---

## Reproducibility

All scores are computed by `artifacts/copilot_vs_model/score_comparison.py`.
The script reads:
- `benchmark_cases.jsonl` — test set + gold
- `our_model_outputs.jsonl` — our system's outputs
- `copilot_outputs.jsonl` — Copilot's outputs

and writes:
- `comparison_summary.csv` — one row per case
- stdout summary table
