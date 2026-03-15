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

5. `_parse_num_token` classified **any value ≥ 1000** as `kind="currency"` via a
   size-based heuristic (`abs(val) >= 1000`), regardless of monetary context.
   This was a direct regression with fix 2 above: once `TimeLimit`, `MaxCapacity`,
   `MaximumDemand`, etc. were correctly typed as `"float"` by `_expected_type`,
   large non-monetary values like `5000` (hours) or `2000` (units) still got
   `kind="currency"` from the heuristic — and `_is_type_match("float", "currency")`
   is `False`, causing a systematic type-match failure for all large non-monetary
   parameters.  **Fixed:** the `abs(val) >= 1000` branch was removed.  Currency
   classification now requires an explicit `$` prefix or a MONEY_CONTEXT word
   (budget, cost, price, profit, revenue, dollar, dollars, $, €, usd, eur).
   18 new tests in `tests/test_float_type_match.py::TestLargeNumberNotCurrency`
   cover the regression and all boundary cases.  One existing test in
   `tests/test_global_vs_local_grounding.py` that relied on the size heuristic
   to make a type mismatch was corrected to use a properly-classified
   coefficient-like slot (`HoursPerProduct`).

**Status:** ✅ Fixed — all five coding-level root causes above are resolved.
The end-to-end TypeMatch improvement requires a downstream re-evaluation run
(needs HF dataset access), but the algorithmic fixes are in place and verified
by 43 targeted tests in `tests/test_float_type_match.py`.

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

**Code-level fixes applied (contributing improvements):**

1. `_choose_token` in `tools/nlp4lp_downstream_utility.py` had a scoring inconsistency
   for `currency` slots.  The four scoring functions
   (`_score_mention_slot`, `_score_mention_slot_ir`, `_score_mention_slot_opt`,
   `_gcg_local_score`) all give `int`/`float` tokens a full `type_exact_bonus` for
   `currency` slots, but `_choose_token` gave them `pref=0` — identical to `unknown`
   tokens.  This meant a plain-integer budget value ("100") could lose the value-selection
   tiebreaker to an unclassified token with a larger absolute value ("9999_unknown").
   **Fixed:** `_choose_token` now assigns `pref=1` to `int`/`float` tokens for currency
   slots (below `pref=2` for explicit currency tokens, above `pref=0` for others).
   9 new tests in `tests/test_float_type_match.py::TestChooseTokenCurrencySlot` verify
   all ranking cases including the regression.

2. `_is_partial_admissible` (the PICARD-style incremental admissibility gate) did not
   check the **numeric ordering** of assigned min/max slot values — it only checked
   operator-tag direction.  Schemas with paired bound slots (e.g. `MinDemand`/`MaxDemand`,
   `LowerBound`/`UpperBound`) could therefore receive inverted assignments (min value >
   max value) without being rejected, directly causing `lower_vs_upper_bound` failures.
   **Fixed:** a new `_slot_stem` helper strips min/max/lower/upper affixes to identify
   paired bound slots sharing the same quantity stem.  `_is_partial_admissible` now
   enforces `value(min_slot) ≤ value(max_slot)` for every such paired slot in the
   partial assignment, rejecting inverted configurations before they propagate.
   22 new tests in `tests/test_bound_role_layer.py` (`TestSlotStem` and
   `TestAdmissibleMinMaxOrdering`) cover all cases: correct ordering, inverted ordering,
   equal values, different-stem pairs (not enforced), and incomplete partials.

**Status:** ⚠️ Open research problem.  The primary bottleneck (joint coverage+type
threshold) remains unresolved, but the `_choose_token` scoring inconsistency and the
`lower_vs_upper_bound` admissibility gap are now fixed.

---

### 2.3 Short-query retrieval and grounding performance is degraded

**Symptom:**
- Retrieval Schema R@1 drops from **0.906** (full queries) to **0.786** (short/first-sentence
  queries) for TF-IDF.
- Short-query downstream coverage ≈ 0.03 — effectively zero numeric grounding because short
  queries contain almost no numeric information.

**Fixes applied:**
- `retrieval/utils.py` — `_DOMAIN_EXPANSION_MAP` extended with six new problem families
  that were not previously handled: LP/MIP/ILP formulations, quadratic/convex programs (QP),
  portfolio/finance optimisation, bipartite matching, resource/inventory capacity planning,
  and cutting/layout/strip packing.  Short queries like `"lp"`, `"portfolio"`, `"matching"`,
  `"inventory"`, `"qp"`, `"ilp"` now receive rich domain-specific expansion phrases instead
  of the generic fallback.  This improves retrieval R@1 for these families.

**Status:** ✅ Domain-expansion coverage improved.  Downstream grounding for short queries
(which carry almost no numeric information) remains an open problem.

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
Full feasibility/optimality checking therefore cannot be performed.

**Fixes applied:**
- `formulation/verify.py` — new `verify_lp_consistency` function that catches
  LP/ILP structural inconsistencies that do not require a solver:
  - Invalid objective sense (anything other than `minimize`, `maximize`, `min`, `max`).
  - Variables with missing or empty `symbol` fields.
  - Duplicate variable symbols within the same formulation.
  - Constraints with missing or empty `expression` fields.
- `run_all_problem_checks` now calls `verify_lp_consistency` and returns a third key
  `lp_consistency_errors` alongside the existing `schema_errors` and `formulation_errors`.
- `app.py` — the validation output block surfaces `lp_consistency_errors` to the user
  and updates the all-clear message to
  "✓ Schema, formulation, and LP consistency OK".
- 16 new tests in `tests/test_verify.py` (`TestVerifyLpConsistency`) exercise all
  new checks, including the integration path through `run_all_problem_checks`.

**Impact:** The validate checkbox now catches a meaningful class of formulation authoring
errors at structure-check time, without requiring a solver installation.  Full
feasibility/optimality verification still requires an external solver.

**Status:** ✅ LP structural consistency checks implemented.  Full solver-based
feasibility/optimality validation remains future work.

---

## 3. One Known Warning

### 3.1 `RuntimeWarning: invalid value encountered in divide` in LSA decomposition

**Symptom:** `sklearn`'s `TruncatedSVD` emits a `RuntimeWarning` about division
by zero during `explained_variance_ratio_` calculation when the corpus is very small.

**Fix applied:** Both call sites that invoke `TruncatedSVD.fit_transform` on
potentially small corpora now wrap the call with a
`warnings.catch_warnings() / warnings.simplefilter("ignore", RuntimeWarning)` context:

* `retrieval/baselines.py` — `LSABaseline.fit()`
* `tools/run_overlap_analysis.py` — LSA section of the overlap-analysis runner

The full test suite now passes with `-W error::RuntimeWarning` (zero warnings).

**Impact:** None — the decomposition result is correct; only the explained-variance
ratio statistic (not used at run-time) is undefined on tiny corpora.

**Status:** ✅ Fixed.
