# Known Issues

This document records the main problems with the application as of the current state of the codebase.
Each issue describes the symptom, root cause, and current status.

---

## 1. Code / Test Issues

### 1.1 Optional runtime dependencies must be installed for full functionality

**Symptom:** Several tests fail with `ModuleNotFoundError` for `starlette`, `gradio`, or `pypdf`
when those packages are not installed in the environment.

**Root cause (now fixed):** `app.py` previously imported `starlette.concurrency` and `gradio`
unconditionally at the top level.  Any test that imported `app` would fail immediately, even
though the test only exercised functions (`_log_user_query`, `answer`) that do not use those
libraries at all.

**Fix applied:**
- `app.py` — the `starlette` HPC thread-pool patch and the `import gradio as gr` statement are
  now wrapped in `try/except ImportError` blocks so that `app` can be imported in lightweight
  test environments where those packages are not installed.
- `tests/test_pdf_upload.py` — `test_whitespace_collapsing` calls `patch("pypdf.PdfReader")`,
  which requires `pypdf` to be importable.  A `pytest.importorskip("pypdf")` guard is added
  so the test is **skipped** (not failed) when `pypdf` is absent.

**How to install all runtime dependencies:**
```bash
pip install -r requirements.txt
```

**Status:** ✅ Fixed.

---

## 2. Application-Level Limitations

### 2.1 Float type matching is nearly zero

**Symptom:** Overall `type_match ≈ 0.226`, but broken down by numeric type:

| Numeric type | type_match (before fixes) |
|---|---|
| Float | ≈ 0.029 |
| Integer | ≈ 0.991 |

The vast majority of numeric parameters in NLP4LP are floats.

**Root causes identified and fixed:**

1. `_is_type_match("float", "int")` previously returned `False` — integer tokens
   (the most common representation in NL for coefficients like `RequiredEggs = 2`)
   were never counted as matches for float slots.  **Fixed** in a prior session.

2. `_expected_type` misclassified `demand`, `capacity`, `minimum`, `maximum`, and
   `limit` as `"currency"` — these are quantity constraints, not monetary values.
   For a `MinimumDemand = 100` slot, the token `"100"` (kind=`int`) received no type
   bonus and a weak-match penalty (-1.0) instead of the full `type_match_bonus` (+3.0).
   **Fixed:** those five keywords removed from the currency list; they now correctly
   fall through to `"float"`.

3. `_is_type_match("currency", "int/float")` returned `False` — monetary slots
   (budget, cost, price, …) whose values appear without an explicit `$` prefix in NL
   were never credited as type matches.  **Fixed:** both `int` and `float` are now
   treated as full matches for `currency` slots.

4. All four scoring functions (`_score_mention_slot`, `_score_mention_slot_ir`,
   `_score_mention_slot_opt`, `_gcg_local_score`) were missing the
   `expected == "currency" and kind in {"int","float"}` branch, so plain-integer
   monetary values were silently ignored during assignment scoring.  **Fixed.**

**Status:** ✅ Fixed — the three coding-level root causes above are resolved.
The end-to-end TypeMatch improvement requires a downstream re-evaluation run
(needs HF dataset access), but the algorithmic fixes are in place and verified
by 25 new targeted tests in `tests/test_float_type_match.py` (classes
`TestQuantityConstraintSlotTypes` and `TestCurrencySlotPlainNumericTokens`).

---

### 2.2 InstantiationReady rate is low (≤ 8.5 %)

**Symptom:** The end-to-end metric `InstantiationReady` (coverage ≥ 0.8 **and**
type_match ≥ 0.8) is ≤ 0.082 for all evaluated assignment methods.

| Method | Schema R@1 | Coverage | TypeMatch | InstantiationReady |
|--------|------------|----------|-----------|-------------------|
| Typed Greedy (TF-IDF) | 0.906 | 0.822 | 0.226 | 0.076 |
| Typed Greedy (Oracle) | 1.000 | 0.870 | 0.240 | 0.082 |
| Constrained (TF-IDF) | 0.906 | 0.772 | 0.195 | 0.027 |

**Root cause:** High coverage and high type accuracy are in tension.  Typed greedy fills
more slots (high coverage) but misidentifies float types; constrained assignment improves
type accuracy but leaves more slots empty.  Both fail the joint threshold.

**Status:** ⚠️ Open research problem.  Primary bottleneck for downstream usability.

---

### 2.3 Short-query retrieval and grounding performance is degraded

**Symptom:**
- Retrieval Schema R@1 drops from **0.906** (full queries) to **0.786** (short/first-sentence
  queries) for TF-IDF.
- Short-query downstream coverage ≈ 0.03 — effectively zero numeric grounding because short
  queries contain almost no numeric information.

**Status:** ⚠️ Partially mitigated.  Short-query expansion (`retrieval/utils.py:expand_short_query`)
improves the retrieval gap; downstream coverage remains an open problem.

---

### 2.4 Learned model does not outperform the deterministic rule baseline

**Symptom:** No learned model checkpoints have been produced.  A real-data-only benchmark run
confirmed that the learned formulation did not improve over the typed greedy rule baseline.

**Root cause:** GPU / `torch` required for training.  Training is blocked in CPU-only
environments (e.g. the Wulver login node) and no usable checkpoint was generated.

**Status:** ⚠️ Future work.  Infrastructure (benchmark-safe splits, pairwise ranker code,
training scripts) exists in `src/learning/` and `training/`.  See
[`docs/learning_runs/README.md`](docs/learning_runs/README.md).

---

### 2.5 Downstream evaluation requires gated HuggingFace dataset access

**Symptom:** Full downstream rerun (parameter gold values, exact-match scoring) cannot be
performed offline.  The dataset `udell-lab/NLP4LP` is gated and requires a HuggingFace
account with approved access and a personal access token.

**Impact:** The downstream metrics in EXPERIMENTS.md (TypeMatch, Exact20,
InstantiationReady) cannot be re-verified in a fresh offline environment.

**Status:** ⚠️ Environment dependency.  See README "HuggingFace dataset access" section for
setup instructions.

---

### 2.6 No LP/ILP solver for output validation

**Symptom:** The `validate` flag in `answer()` is wired into the UI but no solver
(Pyomo, Gurobi, PuLP, OR-Tools) is installed or invoked in the current codebase.
Validation therefore has no effect at runtime.

**Impact:** Solver feasibility/optimality of generated formulations cannot be confirmed
automatically.

**Status:** ⚠️ Planned feature.  Skeleton present; no solver integration implemented.

---

## 3. One Known Warning

### 3.1 `RuntimeWarning: invalid value encountered in divide` in LSA decomposition

**Symptom:** `sklearn`'s `TruncatedSVD` occasionally emits a `RuntimeWarning` about division
by zero during `explained_variance_ratio_` calculation when the corpus is very small.

**Impact:** Cosmetic only — the LSA baseline still produces correct rankings.

**Status:** ℹ️ Benign warning; suppressed in the test suite where applicable.
