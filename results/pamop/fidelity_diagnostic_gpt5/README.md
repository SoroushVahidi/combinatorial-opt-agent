# PaMOP fidelity diagnostic — C3 (stronger deployment), 2026-08-12

**Purpose:** resolve the `FIDELITY_DIAGNOSTIC_REQUIRED` gate from
`PROJECT_STATUS.md` §10 — is PaMOP's low semantic correctness (1/6 on the
6-problem pilot, despite 6/6 execution success) primarily model-limited or
prompt-limited?

**Method:** same 6 problem ids as the existing pilot/forensics run (14,
23, 34, 72, 84, 88), same reconstructed prompts (unmodified), same
temperature (0.2), only the Azure deployment changed:

- **C1** (baseline, not rerun — reused from `results/pamop/forensics_targeted/`):
  `gpt-4.1-mini-2025-04-14`.
- **C3** (this directory): `gpt-5.4`, the strongest Azure deployment
  available on this workstation (verified via
  `AZURE_OPENAI_STRONG_DEPLOYMENT` before running — no other stronger
  deployment was found).

C2/C4 (prompt-strengthening variants) were **not** run in this pass —
time-bounded scope reduction. This diagnostic answers the model-vs-prompt
question along the model axis only; a full C1-C4 matrix is future work if
still needed after reading the result below.

**Generator:** `scripts/pamop_fidelity_diagnostic.py` (new; reuses
`tools/pamop_pilot_benchmark.py`'s own `run_problem`/`write_summary`/etc.
directly, no changes to that file or to any tracked PaMOP config —
constructs an in-memory `dataclasses.replace()`'d config instead of
touching the CLI script's intentional gpt-4.1-mini safety pin).

## Result

| | C1 (gpt-4.1-mini) | C3 (gpt-5.4) |
|---|---|---|
| Initial execution success | 2/6 (0.333) | 5/6 (0.833) |
| Final execution success (after correction loop) | 6/6 (1.0) | 5/6 (0.833) |
| Semantic correctness (exact objective match vs. gold) | **1/6** | **4/5 evaluable (0.8)** |
| Mean tokens/problem | 4032 | 3177 |

**Per-problem objective values (gold vs. produced):**

| Problem | Gold objective | C1 objective | C1 match | C3 objective | C3 match |
|---|---|---|---|---|---|
| 14 | 486.0 | 475.0 | ✗ | 480.0 | ✗ |
| 23 | 40.0 | 52.0 | ✗ | 40.0 | ✓ |
| 34 | 400000.0 | 75000.0 | ✗ | 400000.0 | ✓ |
| 72 | 8.0 | 5.0 | ✗ | 8.0 | ✓ |
| 84 | 1555.56 | 800.0 | ✗ | (AMPL parse failure) | n/a |
| 88 | 142.0 | 142.0 | ✓ | 142.0 | ✓ |

**Interpretation: MODEL_LIMITED.** Swapping only the deployment (no prompt
changes) took semantic correctness from 1/6 to 4/5 evaluable. C3's one
execution failure (problem 84, `AMPL_PARSE_FAILURE`) is a different,
narrower failure mode than C1's dominant issue (executes fine but produces
the wrong objective). This strongly suggests the reconstructed
prompts/pipeline are not the primary limitation — a stronger underlying
model resolves most of the semantic-correctness gap without any prompt
engineering.

## Final PaMOP status (supersedes `FIDELITY_DIAGNOSTIC_REQUIRED`)

**B. MODEL_LIMITED.**

- Do NOT conclude the reconstructed prompts need major rework before
  further investment — the model swap alone recovered most of the gap.
- **Recommendation for any future scale-up:** if an 18- or 269-case run is
  attempted, use `gpt-5.4` (or the strongest available deployment), not
  `gpt-4.1-mini`. This diagnostic does not itself authorize that
  scale-up — per the task's explicit instruction, no 18- or 269-case rerun
  was launched in this pass.
- A full C2/C4 (prompt-strengthening) comparison was not run; if a future
  agent wants to isolate the model-vs-prompt contributions more precisely
  (e.g. confirm prompt changes add nothing on top of the stronger model,
  or add something), that is a well-scoped, cheap follow-up (6 problems,
  same cost class as this run).

## Known limitation of this diagnostic

n=5 evaluable problems is a very small sample; 4/5 vs 1/6 is suggestive,
not a rigorously powered statistical result. Treat "MODEL_LIMITED" as the
best-supported reading of the available evidence, not a certainty.

## Files

- `per_problem.csv`, `summary.json`, `comparison_with_ours.csv`,
  `failure_analysis.csv`, `correction_analysis.csv` — same schema as
  `results/pamop/forensics_targeted/` (C1), for direct comparison.
- `incremental/problem_*.json` — per-problem trace/metadata (no gated
  NLP4LP text; same structure as the already-committed C1 equivalent).

## Reproduction

```bash
export PAMOP_AMPLPY_PYTHON=/home/soroush/.venvs/gurobi/bin/python
python3 scripts/pamop_fidelity_diagnostic.py --deployment gpt-5.4 \
    --output-dir results/pamop/fidelity_diagnostic_gpt5
```
