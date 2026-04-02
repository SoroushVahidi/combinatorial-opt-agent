# Known Issues

Structured record of active limitations and resolved historical issues for the
EAAI companion codebase.

> **See also:** `docs/CURRENT_STATUS.md` for a concise reviewer-facing summary of
> validated vs non-validated components.

---

## 1. Scientific Limitations

### 1.1 Downstream grounding leaves ~47% of queries not fully instantiation-ready

**Symptom:** The joint metric `InstantiationReady` (coverage ≥ 0.8 **and**
type_match ≥ 0.8) is 0.5257 for TF-IDF typed greedy — meaning roughly half of
queries reach the joint threshold, but the remaining ~47% still fall short.

**Canonical values** (source: `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`):

| Method | Schema R@1 | Coverage | TypeMatch | InstantiationReady |
|--------|------------|----------|-----------|-------------------|
| Typed Greedy (TF-IDF) | 0.9094 | 0.8639 | 0.7513 | 0.5257 |
| Typed Greedy (Oracle) | 1.0000 | 0.9151 | 0.8030 | 0.5650 |
| Constrained (TF-IDF) | 0.9094 | 0.8112 | 0.7113 | 0.4230 |

> **Historical note:** Earlier exploratory runs (pre float-type-match fix) showed
> InstantiationReady ≤ 0.082 and TypeMatch ≈ 0.226. Those values reflected five
> code-level bugs since resolved (see §4.2). The table above is the current
> canonical state.

**Root cause (remaining):** The joint threshold requires both coverage ≥ 0.8 and
TypeMatch ≥ 0.8 simultaneously. Typed greedy fills most slots (Coverage 0.8639)
and matches types well (TypeMatch 0.7513), but the intersection still leaves ~47%
of queries below the joint bar.

**Status:** ⚠️ Open research problem. Float type-match bugs are resolved; the
remaining gap is an inherent coverage × type-accuracy trade-off. Further
improvement requires additional grounding signal or relaxed threshold definitions.

---

### 1.2 Short-query downstream grounding is near zero

**Symptom:** Short-query downstream coverage ≈ 0.03 — effectively zero numeric
grounding because short queries contain almost no numeric information.

**Context:** Short-query retrieval Schema R@1 = 0.786 (degraded vs 0.906 for full
queries). Domain-expansion coverage has been improved (see §4.3), but downstream
grounding for short queries is an inherent data limitation: first-sentence-only
queries simply do not contain numeric values to ground.

**Status:** ⚠️ Open limitation — not fixable without additional numeric context in
the query.

---

### 1.3 No full LP/ILP solver for benchmark-wide validation

**Symptom:** Full feasibility/optimality checking is not available across the
NLP4LP benchmark.

**Detail:** The solver-backed study is restricted to 20 instances, using a SciPy
HiGHS compatibility shim. GAMSPy/Pyomo/PuLP/Gurobi appear in demo code only and
are outside the paper scope.

**Status:** ⚠️ By design — paper scope is explicitly restricted to the 20-instance
solver-backed subset. See `docs/paper_vs_demo_scope.md`.

---

## 2. Reproducibility / Access Limitations

### 2.1 Downstream evaluation requires gated HuggingFace dataset access

**Symptom:** Full downstream rerun (parameter gold values, exact-match scoring)
cannot be performed in a fresh offline environment.

**Detail:** The dataset `udell-lab/NLP4LP` is gated and requires a HuggingFace
account with approved access and a personal access token set as `HF_TOKEN`.

**Impact:** The downstream metrics (TypeMatch, Exact20, InstantiationReady) cannot
be re-verified without `HF_TOKEN`. Retrieval metrics (Schema R@1) can be reproduced
without HF access using the local catalog files in `data/processed/`.

**Workaround:** See `HOW_TO_REPRODUCE.md` and the HuggingFace access section in
`README.md` for token setup instructions.

**Status:** ⚠️ Environment dependency — unresolved by design (gated dataset).

---

### 2.2 Optional API baselines (OpenAI / Gemini) are outside camera-ready tables

**Symptom:** Confusion between **Tables 1–5** (deterministic / EAAI scripts) and **optional** two-stage LLM baselines (`tools/llm_baselines.py`).

**Detail:** **OpenAI** downstream CSV/JSON for some variants exists under `results/paper/` from a completed historical batch. **Gemini** has Slurm + preflight infrastructure (**[`docs/GEMINI_RERUN_REPORT.md`](docs/GEMINI_RERUN_REPORT.md)**); a **full** Gemini NLP4LP rerun is **not** claimed in docs unless matching artifacts exist under `results/rerun/gemini/…`. **Mistral** is not wired in this repository.

**Workaround:** Treat **`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`** as headline authority; read **`docs/RESULTS_PROVENANCE.md`** before mixing columns from `nlp4lp_downstream_summary.csv`.

**Status:** ⚠️ By design — optional tooling, not paper-core.

---

## 3. Engineering / Repository Limitations

### 3.1 Learned model does not outperform the deterministic rule baseline

**Symptom:** No learned model checkpoints have been produced that improve over the
typed-greedy rule baseline on the held-out NLP4LP evaluation set.

**Root cause:** GPU / `torch` required for training. Training is blocked in
CPU-only environments (e.g. the Wulver login node) and no usable checkpoint was
generated.

**Impact:** The learned retrieval fine-tuning path (`src/learning/`, `training/`)
is infrastructure-complete but not benchmark-validated. Results are documented as
future work.

**Status:** ⚠️ Future work. Infrastructure exists; see
[`docs/learning_runs/README.md`](docs/learning_runs/README.md).

---

## 4. Resolved Historical Issues

### 4.1 Optional runtime dependencies caused test failures

**Was:** `app.py` unconditionally imported `starlette.concurrency` and `gradio`,
causing `ModuleNotFoundError` in lightweight test environments.

**Fix:** Both imports wrapped in `try/except ImportError` blocks. `test_pdf_upload.py`
guarded with `pytest.importorskip("pypdf")`.

**Status:** ✅ Fixed.

---

### 4.2 Float type-match was near zero (five root causes)

**Was:** Overall `type_match ≈ 0.226`, but float-specific `type_match ≈ 0.029`
due to five bugs:
1. `_is_type_match("float", "int")` returned `False`
2. `_expected_type` misclassified quantity-constraint keywords as `"currency"`
3. `_is_type_match("currency", "int/float")` returned `False`
4. All four scoring functions missing the `currency × int/float` branch
5. `abs(val) >= 1000` size heuristic mis-tagged large non-monetary numbers as currency

**Fix:** All five root causes corrected. `_parse_num_token` currency classification
now requires an explicit `$` prefix or a MONEY_CONTEXT keyword. 43 targeted tests
in `tests/test_float_type_match.py` verify the fixes.

**Status:** ✅ Fixed — code-level. End-to-end TypeMatch improvement requires a
downstream rerun with HF gold data (blocker §2.1 above).

---

### 4.3 Short-query retrieval performance degraded for several problem families

**Was:** Short queries like `"lp"`, `"portfolio"`, `"matching"`, `"inventory"` received
only the generic fallback expansion, causing retrieval R@1 drop.

**Fix:** `retrieval/utils.py` `_DOMAIN_EXPANSION_MAP` extended with six new problem
families (LP/MIP/ILP, QP, portfolio/finance, bipartite matching, resource/inventory
capacity, cutting/layout/strip packing).

**Status:** ✅ Domain-expansion coverage improved.

---

### 4.4 Min/max ordering not enforced in partial assignments

**Was:** `_is_partial_admissible` did not check numeric ordering of min/max slot
pairs, allowing inverted assignments (e.g. `MinDemand > MaxDemand`) to propagate.

**Fix:** New `_slot_stem` helper pairs bound slots by quantity stem.
`_is_partial_admissible` now enforces `value(min_slot) ≤ value(max_slot)`.
22 tests in `tests/test_bound_role_layer.py` cover all cases.

**Status:** ✅ Fixed.

---

### 4.5 `_choose_token` scoring inconsistency for currency slots

**Was:** `_choose_token` gave `pref=0` to `int`/`float` tokens on currency slots,
identical to `unknown` tokens, causing plain-integer budget values to lose tiebreakers.

**Fix:** `pref=1` assigned to `int`/`float` tokens on currency slots (below `pref=2`
for explicit currency tokens). 9 tests in `tests/test_float_type_match.py` verify.

**Status:** ✅ Fixed.

---

### 4.6 LP structural consistency checks missing

**Was:** The `validate` checkbox in the UI had no implementation; structural errors
(invalid objective sense, missing variable symbols, duplicate symbols) were not caught.

**Fix:** `formulation/verify.py` new `verify_lp_consistency` function covers all
structural checks without requiring a solver. 16 tests in `tests/test_verify.py`.

**Status:** ✅ Fixed. Full solver-based feasibility validation remains future work.

---

### 4.7 Missing error handling and input validation in `answer()`

**Was:** `answer()` had no error handling around the main pipeline and no query length
cap; any exception produced an unhandled 500-style error.

**Fix:**
- `_MAX_QUERY_LEN = 5_000` character cap added; over-long queries return a styled
  warning card before model inference is triggered.
- Full pipeline wrapped in `try/except Exception`; errors surface as styled warning
  cards rather than unhandled exceptions.
- 5 new tests in `tests/test_app_validation_toggle.py`.

**Status:** ✅ Fixed.

---

### 4.8 `RuntimeWarning: invalid value encountered in divide` in LSA

**Was:** `sklearn`'s `TruncatedSVD` emitted a `RuntimeWarning` for tiny corpora
during `explained_variance_ratio_` calculation.

**Fix:** Both `retrieval/baselines.py` (`LSABaseline.fit`) and
`tools/run_overlap_analysis.py` now suppress the warning with
`warnings.catch_warnings() / warnings.simplefilter("ignore", RuntimeWarning)`.

**Status:** ✅ Fixed.
