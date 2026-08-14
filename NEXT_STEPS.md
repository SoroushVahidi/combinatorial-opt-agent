# Next Steps — Execution Queue

**Purpose:** a short, operational task queue so the next agent does not
have to decide from scratch what to do. Derived from
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`,
`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md`, and the 2026-08-13 method audit.
For full scientific context, read `docs/SCIENTIFIC_STATE.md` first.

---

## P0 — Completed: schema-correctness-gated readiness

- **Status:** `STRICT_METRIC_RECOMMENDED`.
- **Evidence:** `docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md`
  and `results/strict_instantiation_ready/`.
- **Key result:** fresh `tfidf_typed_greedy` ordinary readiness is
  257/331 but strict readiness is 247/331; `tfidf_selective_grounding_rerank`
  ordinary readiness is 265/331 but strict readiness is only 249/331.
  The selective reranker's +8 ordinary gain collapses to +2 strict-ready
  true schema rescues.
- **Instruction:** future method gates should use strict readiness as the
  primary native end-to-end proxy, while retaining ordinary
  InstantiationReady as a predicted-schema diagnostic.

## P1 — Next: complete external baseline empirical work

- **Status:** method development is `FROZEN_FOR_RESUBMISSION`.
- **Evidence:** `docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md` and
  `results/final_resubmission_method/`.
- **Final method result:** production ratio-word extraction validated the
  diagnostic projection exactly: strict readiness is 255/331, ordinary
  readiness is 265/331, Schema R@1 remains 301/331, and there are 0
  strict/ordinary readiness losses.
- **Task:** finish external baseline empirical completion for the
  resubmission comparison package.
- **Instruction:** do not start another algorithm experiment before
  resubmission.

## P2 — Completed: ratio-word extraction quick fix and method freeze

- **Status:** `QUICK_FIX_VALIDATED`; method state `FROZEN_FOR_RESUBMISSION`.
- **Evidence:** `docs/STRICT_FAILURE_QUICK_FIX_DIAGNOSTIC_2026-08-13.md`,
  `docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`, and
  `results/final_resubmission_method/`.
- **Implemented patch:** multiplicative ratio-word extraction for
  `twice`/`double`/`two times` and `triple`/`three times`.
- **Validated effect:** strict +8 (`247/331 -> 255/331`), ordinary +8
  (`257/331 -> 265/331`), 0 strict/ordinary losses, McNemar p=0.0078125.

## P3 — Completed: Stage-B selective grounding reranker

- **Status:** `STAGE_B_METRIC_ONLY_GAIN`.
- **Evidence:** `docs/SELECTIVE_GROUNDING_RERANK_STAGE_B_2026-08-13.md` and
  `results/selective_grounding_rerank/`.
- **Key result:** production `tfidf_selective_grounding_rerank` reaches
  265/331 InstantiationReady with 0 ready losses and 0 schema regressions, but
  only 2/8 readiness gains are semantically better true schema rescues.
- **Instruction:** do not promote this as the new main method without metric
  redesign or strict-readiness validation.

## P4 — Completed: TOP-2 Stage-A diagnostic

- **Status:** `TOP2_GO`.
- **Evidence:** `docs/TOPK_SCHEMA_RERANK_STAGE_A_2026-08-13.md` and
  `results/topk_schema_rerank_stage_a/`.
- **Key result:** true rescue ceiling is 8 queries at k=3, 9 at k=5, and
  13 at k=10. The recommended selective k=5/margin<=0.05/R5 rule reaches
  265/331 with 0 schema regressions.
- **Instruction:** implement only the minimal deterministic Stage-B rule
  first; do not add API, learned reranking, semantic parsing, or structured
  assignment.

## P5 — Completed: role-quantity Stage-A diagnostic

- **Status:** `STAGE_A_NO_GO`.
- **Evidence:** `docs/ROLE_QUANTITY_STAGE_A_DIAGNOSTIC_2026-08-13.md` and
  `results/role_quantity_stage_a/`.
- **Key result:** 49 targeted wrong assignments, 28 separable by
  deterministic role/quantity features, but 0 projected query-level rescues
  and +0.0 pp InstantiationReady upper-bound gain.
- **Instruction:** do not implement the role-quantity factorized scorer as
  the next main-method patch unless the target metric changes to numeric
  exactness rather than InstantiationReady.

## P6 — Verify/refresh the remaining stale method numbers

- **Task:** rerun `global_compat_full` (and `_local`/`_pairwise`),
  `relation_aware_full` (and `_basic`/`_ops`/`_semantic`),
  `ambiguity_aware_full` (and `_beam`/`_abstain`/`_candidate_greedy`) via
  `run_single_setting()`, same procedure as
  `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` §8, compared against a
  **freshly rerun** `tfidf_typed_greedy` (not the committed 0.5287).
- **Purpose:** these 3 method families were the only ones in
  `docs/METHOD_INVENTORY.md` Part 2 not re-verified fresh in Phase 4
  (time-bounded). Given the pattern in the 9 methods that were checked
  (all drifted upward by roughly the same ~0.2-0.25 magnitude), these are
  very likely also stale, but this is not confirmed.
- **Prerequisite:** none. Cost: ~1-2s per setting, ~10 settings total.
- **Expected artifact:** update to `docs/METHOD_INVENTORY.md` Part 2 rows
  currently marked "not yet regenerated fresh."
- **Stop/success criterion:** every row in that table has a fresh,
  same-code number.

## P6 — Decide the manuscript's path forward (requires the author)

- **Task:** read `docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md` and
  choose one of: (a) issue an erratum with fresh numbers, (b) pin/tag the
  exact commit the submitted numbers correspond to, (c) treat this as a
  "v2"/revision opportunity and regenerate all camera-ready tables, (d)
  some combination.
- **Purpose:** the submitted manuscript's headline InstantiationReady
  (0.5287) does not reproduce from current code (fresh: 0.7764). This is
  the single most consequential open item in the repository.
- **Prerequisite:** none — this is a decision, not an experiment. Cannot
  be made by an agent; requires the paper's author.
- **Expected artifact:** a recorded decision (update
  `docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md` with the choice
  and rationale) and, if regeneration is chosen, a fully re-run
  `training/external/run_full_downstream_benchmark.py` → `tools/build_camera_ready_table1.py`
  → `tools/run_confidence_intervals.py` → `tools/run_strict_instantiation_ready.py`
  → `tools/build_eaai_camera_ready_figures.py` chain, all 3 variants.
- **Stop/success criterion:** n/a — decision gate.

## P7 — PaMOP: PILOT VALIDATED (2026-08-12), scale-up pending

- **Status:** the fidelity diagnostic is complete —
  `results/pamop/fidelity_diagnostic_gpt5/README.md`. Gate: `B. MODEL_LIMITED`.
  A model swap alone (gpt-4.1-mini → gpt-5.4, same prompts) took semantic
  correctness from 1/6 to 4/5 evaluable.
- **Optional follow-up (not required):** a C2/C4 prompt-strengthening
  comparison was not run (scope-reduced). If more precision on the
  model-vs-prompt split is needed, this is a cheap (6-problem) follow-up.
- **If scaling to 18 or 269 cases is decided:** use `gpt-5.4`, not
  `gpt-4.1-mini` — this recommendation is evidence-backed but the scale-up
  itself was deliberately not launched in Phase 4 and remains a future
  decision.
- **Implementation hardening completed:** the runner now accepts an explicit
  config/deployment, preserves generated AMPL/correction artifacts in local
  traces, and labels objective equality as an objective-value proxy. Do not
  call that proxy the paper's full Accuracy metric.
- **Immediate prerequisite before scale-up:** avoid interference with the
  unrelated active AMPL/HiGHS computation, then use a fresh output directory
  and a pre-registered subset/configuration.

## P8 — Improve the local pairwise score's dominant error modes

- **Task:** target `_score_mention_slot_opt`'s residual same-type
  ambiguity (335 slot-level instances) and total/per-unit confusion (166)
  — see `results/max_weight_matching_validation/mechanism_and_error_analysis_summary.json`
  — with further targeted, deterministic fixes, in the style of the 49
  commits documented in the staleness audit.
- **Purpose:** this is the only lever with a demonstrated track record in
  this codebase (it produced the entire +0.2477 improvement already
  measured). No learned scorer, global-assignment method, or repair rule
  has beaten it.
- **Prerequisite:** P0/P1 (strict metric and current failure target defined)
  plus P5 if comparing against the still-stale method families.
- **Expected artifact:** a targeted fix (or documented negative result if
  attempted and it doesn't help) plus a fresh InstantiationReady number
  for `optimization_role_repair`/`max_weight_matching` after the fix.
- **Falsification criterion:** if the fix doesn't move those methods'
  numbers closer to typed greedy's, the local score is not the gating
  factor and this line should stop.

## P9 — ORLM baseline: pilot complete; common-18 running (2026-08-14)

- **Status:** `baselines/orlm/` is inference-ready for resource-available
  execution (adapter, official prompt, lazy runner, normalizer, static
  validator, safe harness, result schema, evaluator, resume store, and 11
  lightweight tests). Wraps
  `CardinalOperations/ORLM-LLaMA-3-8B` (the only confirmed-public
  checkpoint of the three the paper names — verify this before trusting
  any older claim of "multiple checkpoints public").
- **Checkpoint:** downloaded and cached at pinned revision
  `94fdc3c5738c6536d4880dc19a78f215529181c5` (16.1 GiB download).
- **Pilot:** `PILOT_COMPLETED_NORMALLY`; six IDs `[14, 23, 34, 59, 69, 72]`
  completed with unique valid JSONL records, 6/6 generation success, 6/6
  code extraction, and 6/6 static validation. No COPT execution was attempted
  because `coptpy` is missing.
- **Handoff:** session `orlm_pilot_official_20260813_corrected`, log
  `results/orlm/pilot_official_checkpoint/inference_corrected.log`, output
  `results/orlm/pilot_official_checkpoint/results.jsonl`, start
  `2026-08-13T23:06:54-04:00`, Git SHA `6bb75a4c4bed02c458ac30b4af206a2802fce095`.
- **Provenance:** the prompt is now locked to upstream `eval/generate.py`
  revision `33bc47d0a1d1710d24ab839118bdf4cb89b9e31b`.
- **Common-18:** session `orlm_common18_official_20260814`, PID `3807778`,
  log `results/orlm/common18_official_checkpoint/inference.log`, output
  `results/orlm/common18_official_checkpoint/results.jsonl`, start
  `2026-08-14T00:13:37-04:00`, Git SHA
  `1831396b0b3d4428415e354b0a4e1fcbc658df26`.
- **Next action:** inspect this same common-18 tmux job later. Do not launch a
  duplicate. `coptpy` is missing, so solver execution is blocked and must not
  be substituted with another solver.

## P10 — OptMATH lightweight implementation DONE (2026-08-12), inference pending

- **Status:** `baselines/optmath/` is ready for resource-available inference.
  The primary checkpoint is `Aurora-Gem/OptMATH-Qwen2.5-7B`; the official
  prompt and Gurobi target are provenance-locked.
- **Completed:** NLP4LP adapter, static validation, safe Gurobi harness,
  result/evaluation schema, resume store, fixed common manifest, provenance,
  and mocked end-to-end tests.
- **Remaining:** obtain model resources, run one inference smoke test, then
  evaluate the fixed pilot. Do not download weights or execute Gurobi during
  the current high-priority AMPL/HiGHS workload.

## P11 — Re-derive the typed-greedy bottleneck table against current code

- **Task:** `docs/CURRENT_BOTTLENECK_ANALYSIS.md`'s counts (82/331 type
  mismatch etc.) were derived from `per_instance_diagnostics.csv`, which
  has not been re-verified to reproduce from current code the way
  `postfix_main_metrics.csv` was found not to. Re-derive fresh.
- **Prerequisite:** P0.
- **Expected artifact:** updated bottleneck table with a note on whether
  the fresh counts differ from the ones currently documented.

## P12 — Cross-baseline comparison harness: infrastructure DONE (2026-08-13), empirical rows pending

- **Status:** `baselines/comparison/` implements a unified analysis view
  (`UnifiedRow`), adapters for all six systems, native/shared metric
  taxonomy, Wilson CI + exact McNemar statistics, mock-evidence exclusion,
  duplicate-run rejection, and a Markdown/CSV/JSON report generator. See
  `baselines/comparison/README.md` and
  `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` (frozen protocol).
- **Preliminary report generated:** `results/external_baseline_comparison/`
  (regenerate via `python -m baselines.comparison.cli`). Currently contains
  real rows only for `ours` (fresh common-18-subset rerun,
  `instantiation_ready` 10/18) and `pamop` (the existing 6-instance gpt-5.4
  diagnostic). ORLM/OptMATH/DeepOR/OR-R1 show `PENDING`/`UNAVAILABLE`, not
  fabricated numbers.
- **Remaining task:** once P9/P10 (ORLM/OptMATH inference) or a DeepOR/OR-R1
  checkpoint become available, ingest their real result files via
  `baselines/comparison/ingest.py` (add an explicit known-location entry;
  do not add filesystem crawling) and regenerate the report.
- **Do not** treat the current report as a final paper comparison — it is
  explicitly labeled `PRELIMINARY_EXTERNAL_BASELINE_STATUS`.

---

**Do not start from scratch on any of the above without reading the
referenced source documents first** — each one already contains the
detailed reasoning, exact commands, and prior findings needed to execute
the task correctly.
