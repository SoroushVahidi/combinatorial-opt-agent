# Manuscript Integration Decision — 2026-08-12 (Phase 4)

**Scope:** this document records the M1/M2/M3 classification required by
Phase 4 for two related but distinct pieces of evidence, and explains
exactly why neither `manuscript/main.tex` nor
`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
were modified in this pass.

---

## What would need to change if either finding were integrated

### Finding 1: `max_weight_matching` / `search_structured_grounding` / `hierarchical_structured_grounding`

**Classification: M3 — SHOULD REMAIN FUTURE WORK.**

These are now negative results (`docs/NEGATIVE_RESULTS.md` NR12) — they
lose to a fresh typed-greedy rerun. There is nothing to integrate into the
manuscript. No manuscript sections are affected. This classification is
unambiguous and required no author judgment call.

### Finding 2: the `postfix_main_metrics.csv` / `tfidf_typed_greedy` staleness

**Classification: M2 — NEEDS ADDITIONAL VALIDATION, author decision
required. Not integrated.**

This is the consequential one. `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`
established that a fresh, same-code rerun of the manuscript's own headline
method (`tfidf_typed_greedy`) gives InstantiationReady 0.7764, not the
submitted 0.5287. If this were integrated, it would require rewriting:

- **Abstract** (line 30): "The strongest non-oracle method reaches an
  InstantiationReady score of $0.5287$... oracle control... reaches only
  $0.5680$."
- **Introduction / contributions** (line 88): repeats 0.5287 → 0.5680 as
  "the central benchmark-level empirical point."
- **§Main downstream results** (lines 348-372, including Table 4 /
  `tab:nlp4lp-downstream-main`): `tfidf_typed_greedy` Coverage/TypeMatch/
  InstantiationReady, the BM25/LSA comparison row, the "gap among TFIDF-TG,
  TFIDF-AR, BM25-TG, TFIDF-HAR is small" paragraph, the oracle-control
  paragraph, and every downstream table that cites these numbers
  (significance tables, StrictInstantiationReady, robustness-by-variant,
  Figure 2).
- Every other camera-ready table/figure in
  `results/paper/eaai_camera_ready_tables/`/`eaai_camera_ready_figures/`
  that was generated from or cross-checked against these numbers.

### An important nuance: the paper's *qualitative* thesis is not what's wrong

The fresh, current-code numbers were computed for the full method set
that appears in the manuscript's Table 4
(`results/baseline_staleness_audit_2026-08-12/`). Under fresh numbers:

- The oracle-vs-TF-IDF gap is still small in absolute terms (0.8248 −
  0.7764 = 0.0484, vs. the submitted 0.5680 − 0.5287 = 0.0393) — the
  paper's central claim ("schema retrieval is strong; the oracle gain is
  modest; downstream grounding remains the bottleneck") **still holds
  directionally** under fresh numbers.
- Every richer deterministic method still loses to (or ties) typed
  greedy — the paper's second central claim ("richer deterministic
  grounding does not reliably outperform typed greedy") **still holds,
  and is now evidenced by three additional methods** that were not in the
  submitted manuscript at all (`max_weight_matching`,
  `search_structured_grounding`, `hierarchical_structured_grounding`).

**In other words: this does not appear to be a case where the paper's
scientific conclusion was wrong.** It appears to be a case where the
paper's *reported absolute numbers* do not reproduce from the current
codebase, because 49 commits of legitimate grounding-accuracy fixes
landed after the numbers were measured and were never propagated into a
re-measurement before submission. This is a serious reproducibility
problem for a submitted paper, but a narrower and more mechanical one than
"the central finding was wrong."

## Why this is not resolved in this pass

- **Rewriting a submitted manuscript's headline empirical numbers is not
  a repository-polish action.** It changes what reviewers/readers of the
  already-submitted paper would see if they tried to reproduce Table 4
  today, and is exactly the kind of decision that requires the paper's
  author to choose a path, not an automated pass to resolve unilaterally.
- **Multiple defensible author responses exist, each with different
  consequences**, and none can be inferred from repository content alone:
  1. Issue a correction/erratum to the venue with the fresh numbers
     (0.7764/0.8248/etc.), keeping the same qualitative conclusions.
  2. Treat the current codebase as "post-submission work" and explicitly
     pin/tag the exact commit the submitted numbers correspond to (so the
     paper's numbers remain reproducible *at that commit*, even though
     `main` has since moved on) — this requires identifying the exact
     commit, which was not done in this pass (candidate: `3fffe68` or
     earlier; not confirmed as the precise submission commit).
  3. Treat this as a "v2"/extended-version opportunity: re-run the full
     benchmark suite on current code and report the (now higher, and if
     anything more favorable to the paper's thesis) numbers in a revision.
  4. Some combination — e.g., note the discrepancy in a public repository
     README/erratum without touching the already-submitted PDF text.
- **Do NOT blindly insert the fresh numbers into the manuscript** (Phase 4's
  own explicit instruction). The safest action is to surface the finding
  as clearly and completely as possible (this document,
  `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`, `PROJECT_STATUS.md`) and
  leave the actual manuscript/table-regeneration decision to the author.

## What WAS done in this pass

- `manuscript/main.tex`: **not modified.**
- `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
  and all other files under `results/paper/`: **not modified.**
- `results/eswa_revision/13_tables/postfix_main_metrics.csv`: **not
  modified** (left as the exact source the submitted manuscript was built
  from — regenerating it in place would silently change what
  `tools/build_camera_ready_table1.py` produces on a future run, without
  the explicit author decision this warrants).
- Fresh numbers are captured, clearly labeled as fresh/current-code and
  separate from the camera-ready artifacts, in
  `results/baseline_staleness_audit_2026-08-12/` and flagged in
  `results/canonical_results_manifest.json` (families B, C, N).

## Recommended next step for the author

1. Identify the exact commit `manuscript/main.tex`'s Table 4 numbers were
   generated at (cross-reference `results/eswa_revision/13_tables/postfix_main_metrics.csv`'s
   git history against the manuscript submission date).
2. Decide among the four response options above.
3. If choosing to regenerate: re-run
   `training/external/run_full_downstream_benchmark.py` (or the equivalent
   `run_single_setting()` calls used in this audit) for all 3 variants,
   then `tools/build_camera_ready_table1.py`, `tools/run_confidence_intervals.py`,
   `tools/run_strict_instantiation_ready.py`, and
   `tools/build_eaai_camera_ready_figures.py` in sequence, to keep every
   downstream table/figure consistent with the new baseline.
4. Re-verify the manuscript's qualitative claims (oracle-gap-is-modest,
   richer-methods-don't-help) still hold under the fully-regenerated
   numbers before finalizing any text changes — this audit checked this
   for the `orig` variant only; `noisy`/`short` and the significance
   tables were not fully re-verified in this pass.
