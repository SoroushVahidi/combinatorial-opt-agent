# GitHub Copilot Prompt Template — Optimization Problem Instantiation

## Purpose

This template is used to evaluate GitHub Copilot as a baseline for natural-language optimization problem instantiation.
The **same prompt structure** must be used for every benchmark case; only the `{{PROBLEM_TEXT}}` placeholder changes.

---

## Prompt Template

```
You are an expert in mathematical optimization (linear programming, integer programming, etc.).

Given the following natural-language optimization problem description, produce a structured instantiation.
Do NOT solve the problem. Only identify the structure and extract parameter values from the text.

---
PROBLEM:
{{PROBLEM_TEXT}}
---

Return your answer as a valid JSON object with exactly these fields:

{
  "predicted_problem_type": "<string: e.g. 'linear_programming', 'integer_programming', 'transportation', 'diet', 'investment', 'product_mix', etc.>",
  "objective_direction": "<string: 'maximize' or 'minimize' or 'unknown'>",
  "decision_variables": [
    {"name": "<variable name>", "description": "<brief description>"}
  ],
  "constraints_summary": ["<one constraint per item, plain English>"],
  "extracted_numeric_mentions": [
    {"value": <number>, "unit": "<string or null>", "context": "<surrounding words>"}
  ],
  "slot_value_assignments": {
    "<slot_name>": <numeric_value>
  },
  "modeling_notes": "<optional: any caveats or assumptions you made>"
}

Rules:
- slot_value_assignments must contain every numeric parameter mentioned in the problem.
- Use the exact numeric value from the text (do NOT compute or infer values).
- Slot names should be CamelCase descriptive names (e.g. TotalBudget, ProfitPerUnit, MinDemand).
- If a value is a percentage (e.g. 20%), store it as a decimal (0.20) AND note the percentage context in the slot name.
- Do not add, remove, or change any numbers.
- Output only valid JSON. No markdown, no commentary outside the JSON.
```

---

## Usage Instructions

### Automated (preferred)

If you have access to a Copilot API endpoint, call it with this prompt template substituting `{{PROBLEM_TEXT}}` for each benchmark case.

### Manual (fallback)

1. Open GitHub Copilot Chat (VS Code, github.com/copilot, or Copilot CLI).
2. Paste the full prompt above, replacing `{{PROBLEM_TEXT}}` with the `input_text` from each case in `benchmark_cases.jsonl`.
3. Copy the JSON response exactly as returned by Copilot.
4. Paste it into the corresponding entry in `copilot_outputs.jsonl`.

### Per-case prompts

Pre-filled per-case prompts are saved to `copilot_prompts/` (one `.txt` file per case).
These are ready to paste directly into Copilot Chat without any modification.

---

## Output Schema

Each Copilot response must be stored as one JSON line in `copilot_outputs.jsonl`:

```json
{
  "case_id": "<same as benchmark_cases.jsonl>",
  "model": "github-copilot",
  "raw_response": "<exact string returned by Copilot>",
  "parsed": {
    "predicted_problem_type": "...",
    "objective_direction": "...",
    "decision_variables": [...],
    "constraints_summary": [...],
    "extracted_numeric_mentions": [...],
    "slot_value_assignments": {...},
    "modeling_notes": "..."
  },
  "parse_error": null
}
```

If Copilot returns invalid JSON, set `"parse_error": "<error message>"` and `"parsed": null`.

---

## Fairness Constraints

- **Same prompt structure** for every case — no case-specific hints.
- **No gold labels** given to Copilot — it must infer everything from the problem text alone.
- **No human-assisted correction** of Copilot outputs before scoring.
- **Same test cases** used for both our system and Copilot.

---

## Anti-contamination Notes

The 20 NLP4LP test cases in the benchmark are from the public `nlp4lp` HuggingFace dataset test split.
There is a possibility that GPT-4 / Copilot has seen these during training.
The 4 handcrafted cases (`handcrafted_01` to `handcrafted_04`) were written specifically for this benchmark
and are not part of any public dataset, reducing contamination risk for those cases.
