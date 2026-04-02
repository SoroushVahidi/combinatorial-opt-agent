## Copilot → main porting plan (cursor-managed)

**Source branch:** `origin/copilot/analyze-combinatorial-optimization-bot`  \
**Target branch:** `main` (current HEAD `4e67ae5`)

---

## 1. Improvements present only on the Copilot branch

Based on `git diff --stat origin/main..origin/copilot/analyze-combinatorial-optimization-bot`:

- **Retrieval / embedding / UI**
  - `retrieval/utils.py` (new): short-query helpers `_is_short_query`, `expand_short_query`.
  - `retrieval/baselines.py`: all four baselines (BM25, TF-IDF, LSA, SBERT) call `expand_short_query()` inside `rank()`.
  - `retrieval/search.py`:
    - New `expand_short_queries` flag on `search()`.
    - Uses `expand_short_query()` before embedding.
    - `format_problem_and_ip()` handles **missing formulations** gracefully (displays a notice instead of empty collapsibles).
  - `retrieval/pdf_utils.py` (new): `extract_text_from_pdf()` using `pypdf`.
  - `app.py`:
    - Embedding **cache/index fix**: `build_index()` used once at startup; `EMBEDDINGS` reused in `search()`.
    - Gradio UI: **PDF upload** via `gr.File`, wired to `handle_pdf_upload()` (calls `extract_text_from_pdf`).
    - Slight UI text tweaks.

- **NLP4LP downstream / optimization-role / bottlenecks**
  - `tools/nlp4lp_downstream_utility.py`:
    - Written-word number recognition:
      - `_word_to_number`, `_ONES`, `_TENS`, `_LARGE`, `_WORD_TO_NUM`, `_WRITTEN_NUM_RE`.
      - `_extract_num_tokens` and `_extract_num_mentions` now recognise “two”, “twenty-five”, etc. as `NumTok`s.
    - (Global-consistency grounding is referenced in tests and docs but the concrete functions live in this file; the copilot diff around these is large and must be merged atop the already-enhanced version in `main`.)

- **Training data augmentation / metrics**
  - `training/generate_mention_slot_pairs.py`:
    - `_int_to_word` and paraphrase helpers (`_word_paraphrase`, `_parse_value`).
    - Emission of **written-word paraphrase** variants for each digit-based mention (“2 warehouses” → “two warehouses”).
    - Additional negative pairs to sharpen type-matching.
  - `training/metrics.py`: minor adjustments (e.g., better coverage/InstReady computation; small diff).

- **Tests / infra**
  - `pytest.ini`: minimal config (`testpaths = tests`, `addopts = -v`, `requires_network` marker).
  - `tests/conftest.py`: shared fixtures (tiny catalog, etc.).
  - New tests:
    - `tests/test_short_query.py` — short-query expansion and baseline coverage.
    - `tests/test_pdf_upload.py` — `extract_text_from_pdf`.
    - `tests/test_bottlenecks_3_4.py` — written-word number handling, `_int_to_word`, expected-type patterns, and graceful no-formulation output.
    - `tests/test_global_consistency_grounding.py` — GCG behaviour and metrics (requires GCG functions in the downstream utility).
    - `tests/test_bottlenecks_3_4.py`, `tests/test_metrics.py`, `tests/test_baselines.py` small tweaks.

- **Stage-3 / learning / audit tooling**
  - New Stage-3 analysis scripts:
    - `src/learning/analyze_pairwise_features.py`
    - `src/learning/audit_nlp4lp_bottlenecks.py`
    - `src/learning/check_nlp4lp_pairwise_data_quality.py`
    - `src/learning/export_manual_inspection_cases.py`
  - Batch wrappers:
    - `batch/learning/analyze_pairwise_features.sbatch`
    - `batch/learning/audit_nlp4lp_bottlenecks.sbatch`
    - `batch/learning/check_nlp4lp_pairwise_data_quality.sbatch`
    - `batch/learning/export_manual_inspection_cases.sbatch`
  - Shell wrappers:
    - `scripts/learning/run_analyze_pairwise_features.sh`
    - `scripts/learning/run_audit_nlp4lp_bottlenecks.sh`
    - `scripts/learning/run_check_nlp4lp_pairwise_data_quality.sh`
    - `scripts/learning/run_export_manual_inspection_cases.sh`
  - `src/learning/check_training_env.py`, `batch/learning/check_training_env.sbatch`, `scripts/learning/run_check_training_env.sh` on Copilot are more elaborate than the current versions (more robust env probing).

- **Docs / handoff**
  - `.github/agents/*.md`: catalog-agent, docs-agent, retrieval-agent, testing-agent, training-agent, etc.
  - `docs/AGENT_HANDOFF.md`, `docs/BOTTLENECK_ANALYSIS.md`, `docs/BRANCH_VS_MAIN_COMPARISON.md`.
  - `docs/FINAL_MAIN_MERGE_CHECKLIST.md`, `docs/FINAL_MERGE_SUMMARY.md`, `docs/FULL_REPO_SUMMARY.md`.
  - `docs/LEARNING_AUDIT_ANALYSIS.md`, `docs/LEARNING_FIRST_REAL_TRAINING_BLOCKER.md`, `docs/NEXT_TASK.md`, `docs/STRONGER_DETERMINISTIC_PIPELINE_PLAN.md`, `docs/STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md`.
  - Note-only / scratch-style docs such as `docs/COPY_PASTE_INFO.md`, `docs/GCG_CHATGPT_SUMMARY.md`.

---

## 2. Which improvements should be ported to main

**Definitely port (high-value, low risk, code-backed):**

1. **Embedding cache fix & retrieval improvements**
   - `app.py`: use `build_index()` at startup; reuse `EMBEDDINGS` in `search()`.
   - `retrieval/utils.py`: `_is_short_query`, `expand_short_query`.
   - `retrieval/baselines.py`: apply `expand_short_query()` in all four baselines.
   - `retrieval/search.py`: `expand_short_queries` flag and integration; graceful “no formulation” messaging in `format_problem_and_ip`.

2. **PDF upload / UI improvements**
   - `retrieval/pdf_utils.py` and `tests/test_pdf_upload.py`.
   - `app.py`: `handle_pdf_upload()` and Gradio `gr.File` accordion.

3. **Bottleneck-3 / Bottleneck-4 fixes (numbers + missing formulations)**
   - `tools/nlp4lp_downstream_utility.py`: written-word number recognition and docstring-level clarifications, merged atop the current enhanced version (which already has anchor/beam/entity-semantic logic).
   - `tests/test_bottlenecks_3_4.py`, plus any small related changes in `training/metrics.py`.

4. **Training data augmentation for mention–slot scorer**
   - `training/generate_mention_slot_pairs.py`: `_int_to_word`, written-word paraphrase augmentation, extra negatives.

5. **Short-query and retrieval tests**
   - `pytest.ini`, `tests/conftest.py`.
   - `tests/test_short_query.py`, the minimal touch in `tests/test_baselines.py`, and `tests/test_metrics.py`.

6. **Stage-3 analysis scripts and jobs (where aligned with the current Stage-3 layout)**
   - Add `src/learning/analyze_pairwise_features.py`, `src/learning/audit_nlp4lp_bottlenecks.py`, `src/learning/check_nlp4lp_pairwise_data_quality.py`, `src/learning/export_manual_inspection_cases.py` if they match the current Stage-3 IR and paths.
   - Add the corresponding `batch/learning/*.sbatch` and `scripts/learning/*.sh` wrappers, adjusting paths to the already-existing Stage-3 structure on `main`.

7. **Test infrastructure**
   - `pytest.ini` and `tests/conftest.py` to standardize tests and mark network-requiring tests.

**Potentially port (after careful review / light edit):**

8. **Global-consistency grounding (GCG)**
   - Implementation appears to live in `tools/nlp4lp_downstream_utility.py` with tests in `tests/test_global_consistency_grounding.py` and doc coverage in `docs/GCG_EVAL_REPORT.md` / `docs/GCG_FINAL_EVAL_REPORT.md`.
   - Should be ported **implementation + tests** but with clear doc status: implemented + unit-tested, **not fully benchmarked** on HF due to network/HPC constraints.

9. **Stage-3 audits / handoff docs**
   - `docs/AGENT_HANDOFF.md`, `docs/BOTTLENECK_ANALYSIS.md`, `docs/BRANCH_VS_MAIN_COMPARISON.md`, `docs/FULL_REPO_SUMMARY.md`, `docs/LEARNING_AUDIT_ANALYSIS.md`, etc., where they describe structure and bottlenecks honestly.

---

## 3. What should be excluded or down-scoped

**Exclude or de-prioritize:**

- **Note-only / scratch docs** that add noise without new signal:
  - `docs/COPY_PASTE_INFO.md`
  - `docs/GCG_CHATGPT_SUMMARY.md`
  - Very ephemeral “what I did today” style files.
- **Raw audit artifacts and JSONs under `artifacts/`** from the Copilot branch:
  - e.g. `artifacts/learning_audit/bottleneck_audit_summary.json`, per-instance MDs.
  - These are derived outputs, not inputs.
- **Any claim in docs that GCG (or other methods) achieves a specific Exact20 on gold HF** unless we can verify those runs now.

**Handled carefully (may keep but scrub language):**

- `docs/GCG_EVAL_REPORT.md`, `docs/GCG_FINAL_EVAL_REPORT.md`, `docs/STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md`:
  - Keep implementation details and qualitative insights.
  - Ensure they **do not overclaim** benchmark status (mark clearly as “preliminary” or “not yet run on full HF benchmark” if that’s the case).

---

## 4. Conflict / risk areas

- **`tools/nlp4lp_downstream_utility.py`**
  - Already heavily modified on `main` (anchor-linking, entity-semantic beam, relation-aware assignment, low-resource HF loader, etc.).
  - Copilot branch also touches the same file (written-word numbers, GCG, bottleneck fixes).
  - Plan: **manual merge**, keeping:
    - All new anchor/beam/entity-semantic logic from `main`.
    - All written-word recognition and GCG-related functions from Copilot (where not overlapping).
    - Tests (`test_bottlenecks_3_4.py`, `test_global_consistency_grounding.py`) as correctness checks.

- **`src/learning/check_training_env.py` and `batch/learning/check_training_env.sbatch`**
  - `main` already has a simpler environment check; Copilot version is more elaborate (conda-first, module fallback, more logging, different SLURM headers).
  - Plan: merge **behavioural improvements** (better diagnostics and activation strategy) without hardcoding a single account/qos; align with current Wulver conventions.

- **`docs/`**
  - Copilot branch adds many process docs that partially overlap with the new learning docs on `main`.
  - Plan: port only the **highest-signal** ones (handoff, bottleneck summary, full-repo summary), and keep Stage-3 docs consistent about what has/has not been benchmarked.

---

## 5. Implemented vs benchmarked status (per major item)

1. **Embedding cache fix**
   - Status: **Implemented** (build_index + EMBEDDINGS; used in `app.py`).
   - Benchmarked: only qualitatively (latency reduction); no accuracy change.

2. **Short-query expansion**
   - Status: **Implemented + unit-tested** (`expand_short_query`, `_is_short_query`, tests in `test_short_query.py`).
   - Benchmarked: some targeted examples; no large-scale metrics table yet. Safe to describe as a recall-oriented heuristic, not a fully quantified benchmark.

3. **Global-consistency grounding (GCG)**
   - Status: **Implemented in code + has unit tests** (`test_global_consistency_grounding.py`).
   - Benchmarked: **not fully benchmarked** on HF/NLP4LP due to blocking network/HPC runs. Docs must state this clearly.

4. **PDF upload / UI**
   - Status: **Implemented + unit-tested** for PDF text extraction; integrated into Gradio UI.
   - Benchmarked: UX only; no effect on quantitative metrics.

5. **Graceful empty-formulation display**
   - Status: **Implemented + unit-tested** in `format_problem_and_ip`.
   - Benchmarked: n/a (correctness and UX only).

6. **Training data augmentation (short-query templates, number-word paraphrases)**
   - Status: **Implemented in training code** (`generate_mention_slot_pairs.py`, GCG-related training helpers).
   - Benchmarked: Partial; unit tests cover correctness of generation and mapping, but downstream metrics vs baseline are not yet fully locked in.

7. **Test infrastructure**
   - Status: **Implemented** (`pytest.ini`, `conftest.py`, focused tests suite).
   - Benchmarked: n/a; improves reliability and signal but not metrics directly.

8. **Improved SLURM scripts**
   - Status: **Implemented** in Copilot branch with better env handling and resource annotations.
   - Benchmarked: Run-level evidence from Copilot branch; will be validated again after port (at least via dry-run and basic `sbatch` submission).

