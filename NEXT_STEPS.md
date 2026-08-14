# Next Steps — Execution Queue

**Purpose:** a short, operational task queue so the next agent does not
have to decide from scratch what to do. Derived from
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`,
`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md`, and the 2026-08-13 method audit.
For full scientific context, read `docs/SCIENTIFIC_STATE.md` first.

---

## P0 — Implement minimal Stage-B top-k schema + grounding reranker

- **Task:** implement only the Stage-B candidate selected by
  `docs/TOPK_SCHEMA_RERANK_STAGE_A_2026-08-13.md`: if TF-IDF top1-top2
  margin is `<= 0.05`, ground top-5 schemas with the unchanged typed-greedy
  decoder and choose by
  `0.50 * normalized_tfidf + 0.25 * coverage + 0.25 * type_match`; otherwise
  keep TF-IDF top1.
- **Purpose:** TOP-2 Stage A resulted in `TOP2_GO`. The diagnostic candidate
  reached 265/331 InstantiationReady, reranked only 27/331 queries, recovered
  2 schemas, introduced 0 schema regressions, and had 0 ready losses.
- **Prerequisite:** keep using a fresh same-code `tfidf_typed_greedy`
  reference (`0.7764` on 331 `orig`) and do not modify manuscript files.
- **Stop/success criterion:** production Stage B must reproduce at least
  264/331 on the same 331-query protocol, with no material schema regression,
  no eligibility change, paired transition reporting, and runtime still in the
  lightweight CPU-only regime.

## P1 — Completed: TOP-2 Stage-A diagnostic

- **Status:** `TOP2_GO`.
- **Evidence:** `docs/TOPK_SCHEMA_RERANK_STAGE_A_2026-08-13.md` and
  `results/topk_schema_rerank_stage_a/`.
- **Key result:** true rescue ceiling is 8 queries at k=3, 9 at k=5, and
  13 at k=10. The recommended selective k=5/margin<=0.05/R5 rule reaches
  265/331 with 0 schema regressions.
- **Instruction:** implement only the minimal deterministic Stage-B rule
  first; do not add API, learned reranking, semantic parsing, or structured
  assignment.

## P2 — Completed: role-quantity Stage-A diagnostic

- **Status:** `STAGE_A_NO_GO`.
- **Evidence:** `docs/ROLE_QUANTITY_STAGE_A_DIAGNOSTIC_2026-08-13.md` and
  `results/role_quantity_stage_a/`.
- **Key result:** 49 targeted wrong assignments, 28 separable by
  deterministic role/quantity features, but 0 projected query-level rescues
  and +0.0 pp InstantiationReady upper-bound gain.
- **Instruction:** do not implement the role-quantity factorized scorer as
  the next main-method patch unless the target metric changes to numeric
  exactness rather than InstantiationReady.

## P3 — Verify/refresh the remaining stale method numbers

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

## P4 — Decide the manuscript's path forward (requires the author)

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

## P5 — PaMOP: PILOT VALIDATED (2026-08-12), scale-up pending

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

## P6 — Improve the local pairwise score's dominant error modes

- **Task:** target `_score_mention_slot_opt`'s residual same-type
  ambiguity (335 slot-level instances) and total/per-unit confusion (166)
  — see `results/max_weight_matching_validation/mechanism_and_error_analysis_summary.json`
  — with further targeted, deterministic fixes, in the style of the 49
  commits documented in the staleness audit.
- **Purpose:** this is the only lever with a demonstrated track record in
  this codebase (it produced the entire +0.2477 improvement already
  measured). No learned scorer, global-assignment method, or repair rule
  has beaten it.
- **Prerequisite:** P0 (accurate current baselines for every method, so
  any improvement is measured against the right numbers).
- **Expected artifact:** a targeted fix (or documented negative result if
  attempted and it doesn't help) plus a fresh InstantiationReady number
  for `optimization_role_repair`/`max_weight_matching` after the fix.
- **Falsification criterion:** if the fix doesn't move those methods'
  numbers closer to typed greedy's, the local score is not the gating
  factor and this line should stop.

## P7 — ORLM baseline: lightweight implementation DONE (2026-08-12), inference pending

- **Status:** `baselines/orlm/` is inference-ready for resource-available
  execution (adapter, official prompt, lazy runner, normalizer, static
  validator, safe harness, result schema, evaluator, resume store, and 11
  lightweight tests). Wraps
  `CardinalOperations/ORLM-LLaMA-3-8B` (the only confirmed-public
  checkpoint of the three the paper names — verify this before trusting
  any older claim of "multiple checkpoints public").
- **Remaining task:** the actual smoke test — obtain weights, run one
  NLP4LP query through the verified upstream prompt template, and check the
  output at least parses and passes static validation. **Not done**
  — requires a single 24GB-class GPU (not provisioned on this
  workstation) and a COPT/`coptpy` solver license (ORLM's official
  pipeline generates COPT solver code, not Gurobi/Pyomo/plain LP).
- **Provenance:** the prompt is now locked to upstream `eval/generate.py`
  revision `33bc47d0a1d1710d24ab839118bdf4cb89b9e31b`.
- **Stop/success criterion:** do not download the 8B weights or attempt
  GPU inference without first confirming GPU/license availability.

## P8 — OptMATH lightweight implementation DONE (2026-08-12), inference pending

- **Status:** `baselines/optmath/` is ready for resource-available inference.
  The primary checkpoint is `Aurora-Gem/OptMATH-Qwen2.5-7B`; the official
  prompt and Gurobi target are provenance-locked.
- **Completed:** NLP4LP adapter, static validation, safe Gurobi harness,
  result/evaluation schema, resume store, fixed common manifest, provenance,
  and mocked end-to-end tests.
- **Remaining:** obtain model resources, run one inference smoke test, then
  evaluate the fixed pilot. Do not download weights or execute Gurobi during
  the current high-priority AMPL/HiGHS workload.

## P9 — Re-derive the typed-greedy bottleneck table against current code

- **Task:** `docs/CURRENT_BOTTLENECK_ANALYSIS.md`'s counts (82/331 type
  mismatch etc.) were derived from `per_instance_diagnostics.csv`, which
  has not been re-verified to reproduce from current code the way
  `postfix_main_metrics.csv` was found not to. Re-derive fresh.
- **Prerequisite:** P0.
- **Expected artifact:** updated bottleneck table with a note on whether
  the fresh counts differ from the ones currently documented.

## P10 — Cross-baseline comparison harness: infrastructure DONE (2026-08-13), empirical rows pending

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
- **Remaining task:** once P4/P5 (ORLM/OptMATH inference) or a DeepOR/OR-R1
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
