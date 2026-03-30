# EAAI Copilot Handoff Report

**Generated:** 2026-03-30  
**Purpose:** Final handoff summary for human author after EAAI manuscript alignment pass

---

## 1. Summary of What Was Changed

This pass audited the repository for consistency with the current EAAI manuscript and made the following changes:

### Files Created (New)

| File | Description |
|------|-------------|
| `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` | Detailed audit of all overclaims, stale metrics, and missing artifact pointers |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Single canonical source-of-truth for the EAAI paper framing, metrics, and authoritative files |
| `docs/EAAI_COPILOT_HANDOFF_REPORT.md` | This file |
| `analysis/eaai_figures_reproduction_report.md` | Report documenting that figures were missing and have now been reproduced |
| `analysis/eaai_tables_reproduction_report.md` | Report verifying table presence and provenance |
| `analysis/eaai_repo_validation_report.md` | Test run results and dependency analysis |

### Files Modified

| File | Change |
|------|--------|
| `README.md` | Rewrote to align with EAAI framing: removed "solver-ready code" overclaim from title/intro, softened "Project vision", added "Paper artifacts" section, updated "Current status" to mention three EAAI validation subsets and the solver-backed result, corrected "solver-based output validation" status row |
| `requirements.txt` | Added `Pillow>=9.0.0` (required by figure build script) and `scipy>=1.9.0` (required by final solver attempt script) |

### Files Regenerated

| File | Description |
|------|-------------|
| `results/paper/eaai_camera_ready_figures/figure1_pipeline_overview.png` | Pipeline overview schematic |
| `results/paper/eaai_camera_ready_figures/figure1_pipeline_overview.pdf` | Same, PDF |
| `results/paper/eaai_camera_ready_figures/figure2_main_benchmark_comparison.png` | Main benchmark bar chart |
| `results/paper/eaai_camera_ready_figures/figure2_main_benchmark_comparison.pdf` | Same, PDF |
| `results/paper/eaai_camera_ready_figures/figure3_engineering_validation_comparison.png` | Engineering subset bar chart |
| `results/paper/eaai_camera_ready_figures/figure3_engineering_validation_comparison.pdf` | Same, PDF |
| `results/paper/eaai_camera_ready_figures/figure4_final_solver_backed_subset.png` | Solver-backed subset bar chart |
| `results/paper/eaai_camera_ready_figures/figure4_final_solver_backed_subset.pdf` | Same, PDF |
| `results/paper/eaai_camera_ready_figures/figure5_failure_breakdown.png` | Failure breakdown bar chart |
| `results/paper/eaai_camera_ready_figures/figure5_failure_breakdown.pdf` | Same, PDF |

---

## 2. Docs Rewritten or Marked Historical

### Rewritten

- **README.md** — Substantially revised. Key changes:
  - Title and intro now describe the system as "retrieval-assisted optimization schema grounding and scalar parameter instantiation" instead of "translates plain-English to solver-ready code"
  - "Project vision" softened to be benchmark-scoped
  - "Current evidence-based status" table updated: solver validation row now reflects real 20-instance solver-backed result
  - Added "Paper artifacts" section with links to camera-ready tables and figures
  - LLM-generation claim for unknown problems marked as "demo/outside paper scope"
  - EAAI validation subsets (engineering, executable, solver-backed) now mentioned

### Not Deleted (Historical, left as-is with note in audit doc)

The following docs are historical and should NOT be cited as authoritative for the EAAI submission. They were not deleted because they contain valuable experiment history, but their status is now documented in `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` and `docs/EAAI_SOURCE_OF_TRUTH.md`:

- `docs/JOURNAL_READINESS_AUDIT.md`
- `docs/Q1_JOURNAL_AUDIT.md`
- `docs/CURRENT_STATE_AUDIT.md`
- `docs/eswa_revision/`
- `current_repo_vs_manuscript_rerun.md`
- `publish_now_decision_report.md`

---

## 3. Figures: Were They Missing? Were They Reproduced?

**Yes, all 5 figures were missing** (PNG and PDF). Only source CSV files existed in `results/paper/eaai_camera_ready_figures/`.

**Yes, all 5 figures were successfully reproduced** using `tools/build_eaai_camera_ready_figures.py`.

- Command: `python tools/build_eaai_camera_ready_figures.py`
- Exit code: 0 (success)
- All 10 files (5 PNG + 5 PDF) now exist
- Values are sourced directly from the authoritative camera-ready tables in `results/paper/eaai_camera_ready_tables/`
- See `analysis/eaai_figures_reproduction_report.md` for full details

**Dependency fix required:** `Pillow` was missing from `requirements.txt`. It has been added as `Pillow>=9.0.0`.

---

## 4. Tables: Were They Reproduced?

**No regeneration was needed.** All 5 tables were already present and verified against their source EAAI experiment reports.

- See `analysis/eaai_tables_reproduction_report.md` for full details.
- All table values are consistent with `analysis/eaai_*_report.md` files.
- No silently-changed values.

---

## 5. Tests: Which Were Run and What Were the Outcomes?

**Command:** `python -m pytest tests/ -q`

**Results:**
- ✅ 1,466 passed
- ❌ 2 failed (pre-existing, unrelated to EAAI alignment)
- ⏭ 6 skipped (environment-conditional)

**Pre-existing failures (not introduced by this pass):**
1. `tests/test_bottlenecks_3_4.py::TestExtractNumTokensWordRecognition::test_hyphenated_written_word` — word recognition for hyphenated numbers
2. `tests/test_bottlenecks_3_4.py::TestExpectedTypeExtended::test_currency_slots_extended` — currency classification for "totalAmount" slot

See `analysis/eaai_repo_validation_report.md` for full details.

---

## 6. Remaining Blockers

| Blocker | Impact | Recommended action |
|---------|--------|-------------------|
| HuggingFace gated dataset (`udell-lab/NLP4LP`) requires `HF_TOKEN` | Cannot rerun engineering/executable subset experiments without HF access | Use existing committed reports; note in paper |
| `gurobipy` not available | Executable-attempt study always shows 0% solver success (this is the documented finding, not a bug) | Already documented in Table 3 and paper |
| 2 pre-existing test failures | No paper impact but minor code quality issue | Fix if time permits (low priority) |
| E5/BGE dense retrieval results not in EAAI main table | Potential reader confusion | Clarify in paper that dense retrieval is supplementary |
| "LLM generation for unknown problems" in codebase | Not an EAAI claim; could confuse manuscript reviewers | Mark as demo-only in code comments |

---

## 7. Recommended Next Steps for Human Author

### Immediate (before final submission)

1. **Verify README.md changes** match your intended paper framing. The revised README is aligned with the EAAI manuscript but you may want to adjust the "Project vision" wording.

2. **Check double-anonymization**: Ensure the manuscript body does not contain the GitHub URL, your name, or NJIT in any form that would de-anonymize the submission.

3. **Verify figures in paper**: Use the PNG/PDF files from `results/paper/eaai_camera_ready_figures/` as your manuscript figures. All were regenerated from verified table sources on 2026-03-30.

4. **Verify tables in paper**: Use CSVs from `results/paper/eaai_camera_ready_tables/`. All values verified against EAAI experiment reports.

5. **Generative AI declaration**: Place in the correct submission field (cover letter / designated form), not in the manuscript body.

### Short-term

6. **Fix 2 failing tests** (low priority): hyphenated word recognition and currency classification for "totalAmount".

7. **Add `[HISTORICAL]` headers** to the oldest pre-EAAI docs if you want to prevent confusion for future contributors.

8. **Clarify E5/BGE status** in the paper: are they in an appendix or dropped from the submission?

9. **Decide on "LLM generation for unknown problems"** claim: this exists in the web app but is not an EAAI manuscript result. Consider removing it from the paper-facing README or clearly labeling it as out-of-scope for the paper.

### Longer-term

10. **Full benchmark rerun** when HF dataset access is available: re-run `tools/run_eaai_engineering_subset_experiment.py` and `tools/run_eaai_executable_subset_experiment.py` to verify the committed reports against fresh runs.

11. **Solver subset expansion**: If you want to demonstrate more instances, expand the SciPy-compatible filter in `tools/run_eaai_final_solver_attempt.py` to include more NLP4LP problem families.

---

## File Map: Where to Find Everything

| What you need | Where to find it |
|---------------|-----------------|
| Camera-ready figures (PNG + PDF) | `results/paper/eaai_camera_ready_figures/` |
| Camera-ready tables (CSV + Markdown) | `results/paper/eaai_camera_ready_tables/` |
| Table provenance notes | `analysis/eaai_tables_build_report.md` |
| Figure provenance notes | `analysis/eaai_figures_build_report.md`, `analysis/eaai_figures_reproduction_report.md` |
| Engineering subset raw data | `analysis/eaai_engineering_subset_report.md` |
| Executable-attempt raw data | `analysis/eaai_executable_subset_report.md` |
| Final solver-backed raw data | `analysis/eaai_final_solver_attempt_report.md` |
| Source of truth for paper claims | `docs/EAAI_SOURCE_OF_TRUTH.md` |
| Repo alignment audit | `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` |
| Test run results | `analysis/eaai_repo_validation_report.md` |
| This handoff | `docs/EAAI_COPILOT_HANDOFF_REPORT.md` |
