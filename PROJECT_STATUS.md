# Project Status

**Last verified:** 2026-08-13 (method novelty/efficiency audit added:
`docs/METHOD_NOVELTY_EFFICIENCY_AUDIT_2026-08-13.md`; Phase 4 previously
validated and RETRACTED Phase 3's headline claim after discovering the
comparison baseline was stale; full audit in
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`). This file is the
primary entry point for a new agent or contributor; it stays concise and
links out for detail. See
[`docs/KAIS_SOURCE_OF_TRUTH.md`](docs/KAIS_SOURCE_OF_TRUTH.md) for
manuscript-specific authority, [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) /
[`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) for the directory map, and
these documents for full scientific detail:

- [`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`](docs/BASELINE_STALENESS_AUDIT_2026-08-12.md) — **read this first.** The manuscript's headline typed-greedy number does not reproduce from current code; this changes everything below it.
- [`docs/SCIENTIFIC_STATE.md`](docs/SCIENTIFIC_STATE.md) — detailed scientific handoff (research question, current best method, why it works, full weakness/strength classification, open questions)
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — short, operational execution queue; start here for "what do I do right now"
- [`results/CANONICAL_RESULTS.md`](results/CANONICAL_RESULTS.md) — where the truth lives for every result family
- [`docs/METHOD_INVENTORY.md`](docs/METHOD_INVENTORY.md) — pipeline decomposition + every grounding method implemented
- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — everything already tried that didn't work (read before proposing a new method)
- [`docs/CURRENT_BOTTLENECK_ANALYSIS.md`](docs/CURRENT_BOTTLENECK_ANALYSIS.md) — ranked failure modes + weakness/strength classification
- [`docs/RESEARCH_HYPOTHESES.md`](docs/RESEARCH_HYPOTHESES.md) — testable hypotheses (H1-H5), status-tagged
- [`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md`](docs/ALGORITHM_IMPROVEMENT_ROADMAP.md) — literature review + prioritized roadmap (DONE/NEXT/CONDITIONAL)
- [`docs/LEARNED_GROUNDING_P0.md`](docs/LEARNED_GROUNDING_P0.md) — the P0 learned-scorer experiment (negative result, decision gate C)
- [`docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md`](docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md) — why the manuscript was NOT touched despite the staleness finding, and what the author should do next
- [`docs/BASELINE_IMPLEMENTATION_ROADMAP.md`](docs/BASELINE_IMPLEMENTATION_ROADMAP.md) — ORLM/OptMATH/DeepOR/OR-R1 implementation planning
- [`docs/METHOD_NOVELTY_EFFICIENCY_AUDIT_2026-08-13.md`](docs/METHOD_NOVELTY_EFFICIENCY_AUDIT_2026-08-13.md) — current method algorithm, fresh failure taxonomy, reviewer-gap matrix, novelty assessment, and ranked go/no-go plan for the next method improvement
- [`docs/ROLE_QUANTITY_STAGE_A_DIAGNOSTIC_2026-08-13.md`](docs/ROLE_QUANTITY_STAGE_A_DIAGNOSTIC_2026-08-13.md) — Stage-A diagnostic for the top-ranked role/quantity candidate; result `STAGE_A_NO_GO`
- [`docs/TOPK_SCHEMA_RERANK_STAGE_A_2026-08-13.md`](docs/TOPK_SCHEMA_RERANK_STAGE_A_2026-08-13.md) — Stage-A diagnostic for selective top-k schema + grounding reranking; result `TOP2_GO`
- [`docs/SELECTIVE_GROUNDING_RERANK_STAGE_B_2026-08-13.md`](docs/SELECTIVE_GROUNDING_RERANK_STAGE_B_2026-08-13.md) — frozen Stage-B implementation of `tfidf_selective_grounding_rerank`; result `STAGE_B_METRIC_ONLY_GAIN`
- [`docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md`](docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md) — fresh schema-correctness-gated readiness diagnostic; result `STRICT_METRIC_RECOMMENDED`
- [`docs/STRICT_FAILURE_QUICK_FIX_DIAGNOSTIC_2026-08-13.md`](docs/STRICT_FAILURE_QUICK_FIX_DIAGNOSTIC_2026-08-13.md) — strict-failure quick-fix diagnostic; result `QUICK_FIX_GO` for one ratio-word extraction patch, then freeze method development
- [`docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`](docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md) — final method freeze; result `FROZEN_FOR_RESUBMISSION` after validating multiplicative ratio-word extraction

**Headline change since Phase 3 (retraction, same day, 2026-08-12):**
Phase 3 believed `tfidf_typed_greedy` (committed at 0.5287) had been
beaten by `max_weight_matching` (0.7432). This was found to be an
apples-to-oranges comparison: 0.5287 is **stale** — it predates 49 commits
of grounding fixes and does not reproduce from current code. A fresh
rerun of plain typed greedy gives **0.7764**, which significantly beats
`max_weight_matching`, `search_structured_grounding`, and
`hierarchical_structured_grounding` (all p<0.05). **`tfidf_typed_greedy`
is the strongest known non-oracle method in this repository as of
2026-08-12** — see §3-4 below and
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` for the full audit.

---

## 1. Scientific Goal

This repository studies **retrieval-assisted instantiation of natural-language
optimization problems**: given a natural-language description of an optimization
problem, (a) retrieve the most compatible schema from a fixed catalog of
optimization problem templates, then (b) deterministically ground the schema's
scalar parameters from numeric evidence in the text. The manuscript frames this
as a knowledge-processing problem, not a full NL-to-solver compiler.

## 2. Current Pipeline

```
natural-language query
  → schema retrieval (TF-IDF / BM25 / LSA / Oracle control)
  → numeric mention extraction
  → schema-conditioned scalar grounding (typed greedy + extensions, see §5)
  → structural verification (formulation/verify.py, no live solver)
  → restricted solver-backed validation (SciPy HiGHS shim, 20-instance subset)
```

Core implementation: [`tools/nlp4lp_downstream_utility.py`](tools/nlp4lp_downstream_utility.py)
(retrieval-to-grounding pipeline); retrieval baselines in [`retrieval/`](retrieval/);
structural checks in [`formulation/verify.py`](formulation/verify.py).

## 3. Current Authoritative Results

**Primary benchmark:** NLP4LP (`udell-lab/NLP4LP`, gated HuggingFace dataset),
`orig` variant, 331 test queries.

**2026-08-11: a staleness issue was found in Phase 1 and fixed in Phase 2.**
`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
previously carried stale values from a pre-correction intermediate snapshot;
it is now regenerated by `tools/build_camera_ready_table1.py` and verified
against `manuscript/main.tex`. Tables 2-5 in the same directory were audited
and found **CURRENT** (verified byte-for-byte against the manuscript). Full
provenance: `results/CANONICAL_RESULTS.md`.

| Method | Schema R@1 | Coverage | TypeMatch | Exact20_on_hits | InstantiationReady |
|---|---|---|---|---|---|
| TF-IDF (typed greedy) | 0.9094 | 0.8609 | 0.7453 | 0.1834 | 0.5287 |
| BM25 (typed greedy) | 0.8822 | 0.8509 | 0.7336 | 0.1884 | 0.5196 |
| LSA (typed greedy) | 0.8459 | 0.8267 | 0.7054 | 0.1822 | 0.5076 |
| Oracle (typed greedy) | 1.0000 | 0.9151 | 0.7998 | 0.1745 | 0.5680 |

Source of these corrected numbers: `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv`
and `results/eswa_revision/13_tables/postfix_main_metrics.csv` (both currently
"live"/non-`.stale` files), cross-checked directly against `manuscript/main.tex`.

**StrictInstantiationReady** (adds a schema-match gate; from
`results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv`,
also matches the manuscript):

| Method | InstantiationReady | StrictInstantiationReady |
|---|---|---|
| TFIDF-TG | 0.5287 | 0.5045 |
| BM25-TG | 0.5196 | 0.4924 |
| LSA-TG | 0.5076 | 0.4864 |
| Oracle-TG | 0.5680 | 0.5680 |

**A disclosed, unresolved, small Schema R@1 offset exists** and should not be
"fixed" without more investigation: some diagnostic tables/scripts report
TF-IDF Schema R@1 = 0.9063 (not 0.9094) and BM25 = 0.8852 (not 0.8822). The
manuscript explicitly discloses this (§ "overlap analysis") and attributes it
to a 335- vs original-catalog-vintage document-count difference between the
diagnostic script and the canonical Table run. **Do not silently pick one
value** — report which script/table produced the number you cite.

**Other result families:** engineering structural subset (60 instances),
executable-attempt subset (269 instances), final solver-backed subset (20
instances) — all verified **CURRENT** against the manuscript in Phase 2; see
`results/CANONICAL_RESULTS.md` §G-I for exact values and provenance.
`results/eswa_revision/13_tables/robustness_by_variant.csv` was found
stale in Phase 3 (same root cause as table1) and regenerated by the new
`tools/build_robustness_by_variant.py`; `results/paper/eaai_camera_ready_figures/figure2_main_benchmark_comparison.{png,pdf}`
was likewise found visually stale (a Pillow plugin-registration bug
silently prevented regeneration in Phase 2) and has been fixed and
regenerated — see §14 "Things a New Agent Must NOT Do" for the specific
bug and its fix.

**IMPORTANT — table above is the manuscript's own headline table**, and
**it does not reproduce from the current codebase** (see the staleness
finding immediately below). It remains the manuscript's own reported
authority for the submitted paper, but a new agent should not assume a
fresh rerun will reproduce these exact numbers.

## 4. Main Scientific Finding

**Manuscript-level finding (qualitatively still supported by fresh
evidence, even though the exact numbers below have drifted — see the
staleness note):** retrieval is strong (TF-IDF Schema R@1 ≈ 0.91); the
oracle-vs-TF-IDF gap on InstantiationReady is modest under the
manuscript's `typed_greedy` method (0.5680 vs 0.5287 as submitted; 0.8248
vs 0.7764 under a fresh rerun — the gap stays small either way). **The
primary bottleneck for that specific method is downstream semantic
number-to-schema-slot grounding, not schema retrieval — this conclusion
holds under both the submitted and the fresh numbers.**

**CRITICAL FINDING (2026-08-12, Phase 4):
`results/eswa_revision/13_tables/postfix_main_metrics.csv` — the source of
the manuscript's headline `tfidf_typed_greedy` = 0.5287 — does not
reproduce from the current codebase.** A fresh, same-code rerun of the
identical method gives **InstantiationReady = 0.7764** (drift +0.2477),
because 49 commits of grounding-accuracy fixes landed after that table was
last generated and it was never regenerated. Full audit, root cause, and
exact reproduction commands:
[`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`](docs/BASELINE_STALENESS_AUDIT_2026-08-12.md).
**Consequence: `tfidf_typed_greedy`, freshly rerun, is the strongest known
non-oracle method in this repository — it beats every richer alternative
evaluated, including the three "unevaluated methods" below.**

**Phase 3 (earlier the same day) believed it had found the opposite** —
that three methods implemented but never before run against real NLP4LP
data (`max_weight_matching`, `search_structured_grounding`,
`hierarchical_structured_grounding`) dramatically outperformed typed
greedy. **This was retracted the same day** once the comparison baseline
was found stale:

| Method | InstantiationReady (raw, still reproducible) | vs. **fresh** typed greedy (0.7764) | p-value |
|---|---|---|---|
| `max_weight_matching` (exact Hungarian assignment) | 0.7432 | **−0.0332 (loses)** | 0.042 |
| `search_structured_grounding` (beam search) | 0.7039 | **−0.0725 (loses)** | <0.001 |
| `hierarchical_structured_grounding` (region-decomposed beam search) | 0.7039 | **−0.0725 (loses)** | <0.001 |

All three raw numbers are independently reproduced and leakage-free (the
*measurement* was sound) — only the *comparison* was invalid (fresh vs.
stale). See `docs/NEGATIVE_RESULTS.md` NR12 and
[`results/unevaluated_methods_evaluation/README.md`](results/unevaluated_methods_evaluation/README.md)
(now superseded interpretation, numbers still accurate). §8's "most
promising improvement direction" below reflects the corrected picture, not
the retracted one.

## 5. Current Grounding Methods Already Implemented

Do **not** reinvent these — full inventory with CLI dispatch names, mechanism,
and evaluated-or-not status in [`docs/METHOD_INVENTORY.md`](docs/METHOD_INVENTORY.md).
Evaluated and CANONICAL: **typed greedy — baseline, fresh InstantiationReady
0.7764 as of 2026-08-12, the strongest known non-oracle method** (committed
Table 1 shows 0.5287, which is stale — see §4). Evaluated and
NEGATIVE_RESULT: max-weight bipartite matching (0.7432), search-structured
grounding (0.7039), hierarchical-structured grounding (0.7039) — all lose
to fresh typed greedy; constrained matching, semantic IR repair,
optimization-role repair, acceptance/hierarchical-acceptance reranking,
global compatibility grounding (`global_compat_{local,pairwise,full}`),
relation-aware linking (`relation_aware_{basic,ops,semantic,full}`),
ambiguity-aware grounding (4 variants), P0 learned scorer (see §6).

## 6. What Did NOT Work / Negative Results

Full ledger with statistical evidence: [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md).
**Headline finding (strengthened 2026-08-12, Phase 4):** none of the
evaluated richer deterministic grounding families — greedy/repair-based
(GCG, relation-aware, ambiguity-aware) **or global-assignment-based**
(`max_weight_matching`, `search_structured_grounding`,
`hierarchical_structured_grounding`) — beat plain `tfidf_typed_greedy`
(freshly rerun: 0.7764) on `orig` InstantiationReady; most lose
significantly (p<0.05, paired bootstrap). **A same-day Phase 3 claim that
exact global assignment was the exception has been retracted** — it
compared against a stale typed-greedy baseline; see §4 and
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`.

**Critical, easy to miss:** a learned **local mention-slot scorer was
already tried twice**, both times unsuccessfully. NR10 (Phase pre-2): a
text-only `distilroberta-base` pairwise ranker, lost on every metric to
the rule baseline (exact-fill accuracy collapsed to 0.000). NR11 (Phase 3,
`docs/LEARNED_GROUNDING_P0.md`): a feature-augmented classifier over rich
hand-engineered features plus a frozen sentence embedding — still did not
beat canonical typed greedy or even a pure rule-only decode over the same
features (not statistically significant, p=0.44, but no configuration
showed a nominal gain either; this specific comparison used a fresh M0
baseline computed for that pass and is unaffected by the §4 staleness
finding). **Read NR10 AND NR11 in full before proposing "try a learned
scorer"** — two well-executed, differently-scoped attempts have both
failed to show a gain; any third attempt needs a concrete, evidenced
reason to expect a different outcome, and must be benchmarked against a
**freshly rerun** typed greedy (see §8).

## 7. Current Weaknesses

Full three-way classification (architectural / benchmark-data /
evaluation-evidence) and a ranked, per-query-derived bottleneck table:
[`docs/CURRENT_BOTTLENECK_ANALYSIS.md`](docs/CURRENT_BOTTLENECK_ANALYSIS.md).
**Headline bottleneck, `typed_greedy`-specific:** type mismatch on
otherwise-fully-covered decisions — 82/331 queries (24.8%) under the
originally-measured (now-flagged-stale) `per_instance_diagnostics.csv`;
schema retrieval miss (9.1%) and coverage gaps (6.6-9.1%) are real but
smaller contributors. **This bottleneck table has not been re-derived
against current code** (open item, see `NEXT_STEPS.md`) — treat category
*ranking* as informative, exact counts as unverified. The Phase 3 claim
that `max_weight_matching` "closes most of this gap" is **retracted**
(§4) — `max_weight_matching` is a negative result, not a superior method.

**Root-caused in Phase 3 (unaffected by the §4 staleness finding):** the
"20/331 zero-expected-scalar-slot" item previously listed as a fifth
bottleneck category is **not independent** — 19/20 (95%) are a downstream
artifact of schema-retrieval misses (the `n_expected_scalar` metric is
computed from the *predicted*, not gold, schema); only 1/331
(`nlp4lp_test_293`) is a genuine case, and it is a vector/matrix-parameter
problem the scalar-only architecture cannot represent regardless. See
`docs/CURRENT_BOTTLENECK_ANALYSIS.md` rank 5 for detail.

## 8. Most Promising Improvement Direction

**Reframed twice on 2026-08-12: Phase 3's reframing (toward
`max_weight_matching`) is retracted; the original Phase 2 direction is
restored and sharpened by Phase 4's evidence.** The most promising
direction is targeted, deterministic feature engineering on the *local*
pairwise score / typed-greedy's own candidate-selection logic
(`_choose_token`) — the same class of fix that produced the 49-commit,
+0.2477 InstantiationReady improvement documented in
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. This has a demonstrated
track record in this exact codebase; no richer-architecture alternative
(learned scorer, global assignment, beam search, repair rules) has beaten
it. Concretely (see `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` P1-P3):
1. Verify/refresh the remaining Phase-1/2 negative-result numbers
   (`global_compat_*`, `relation_aware_*`, `ambiguity_aware_*` were not
   regenerated fresh in Phase 4 — likely also stale, not yet confirmed).
2. Target `_score_mention_slot_opt`'s dominant residual error modes
   (same-type ambiguity, total/per-unit confusion — quantified in
   `results/max_weight_matching_validation/`) with further deterministic
   fixes, in the style of the 49 commits already in this repository's
   history.
3. Re-target H4 (richer semantic-role features) at `_choose_token` itself
   rather than a learned scorer.

**2026-08-13 novelty/efficiency audit update:** the next scientifically
defensible method-improvement line should not be another unchanged matching,
beam-search, repair, or generic learned-pair scorer. The audit selects a
pre-gated **role-quantity factorized grounding with an ambiguity cascade** as
the top candidate, with a strict success gate: beat fresh typed greedy
(`0.7764`) by at least +2 pp on the same 331-query protocol, improve a
predeclared error class, pass paired testing and ablations, and preserve
runtime. The single next action is a Stage-A lightweight per-slot diagnostic
over the 54 schema-hit/not-ready current typed-greedy cases.

**Stage-A result (2026-08-13): `STAGE_A_NO_GO`.** The diagnostic found
28 role/quantity-separable wrong assignments among 49 targeted schema-hit /
not-ready slot errors, but correcting all separable assignments would rescue
0 additional InstantiationReady queries because the current readiness metric
is gated by coverage/type compatibility rather than numeric exactness. Do not
implement the role-quantity factorized scorer as the next main-method patch;
move to the TOP-2 candidate, selective top-k schema + grounding reranking.

**TOP-2 Stage-A result (2026-08-13): `TOP2_GO`.** Selective top-k schema +
grounding reranking has enough query-level signal for a minimal Stage-B:
the recommended diagnostic rule reranks only 27/331 low-margin queries, grounds
top-5 schemas, selects by `0.50 * normalized_tfidf + 0.25 * coverage +
0.25 * type_match`, and reaches 265/331 InstantiationReady with 0 schema
regressions and 0 ready losses. Implement this minimal deterministic cascade
next; do not add API, learned, or semantic reranking before the Stage-B
regression test.

**TOP-2 Stage-B result (2026-08-13): `STAGE_B_METRIC_ONLY_GAIN`.** The frozen
production method `tfidf_selective_grounding_rerank` exactly reproduces the
265/331 InstantiationReady result, but semantic audit shows only 2/8 new ready
queries are true schema rescues; the other 6 are wrong-schema readiness gains.
Treat this as evidence that InstantiationReady can be gamed by easier wrong
schemas. Do not promote this as the new main method without schema-correctness
gating or metric redesign.

**Strict metric result (2026-08-13): `STRICT_METRIC_RECOMMENDED`.**
`docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md` adds a
schema-correctness gate to the fresh current-code readiness metric. Fresh
`tfidf_typed_greedy` is 257/331 under ordinary predicted-schema
InstantiationReady but 247/331 under strict readiness; fresh
`tfidf_selective_grounding_rerank` is 265/331 ordinary but only 249/331
strict. The selective reranker's +8 ordinary gain collapses to +2 true
schema-gated ready gains, confirming that ordinary InstantiationReady should
be treated as a predicted-schema proxy, not end-to-end correctness.

**Strict-failure quick-fix production result (2026-08-13):
`QUICK_FIX_VALIDATED`; method state `FROZEN_FOR_RESUBMISSION`.**
`docs/STRICT_FAILURE_QUICK_FIX_DIAGNOSTIC_2026-08-13.md` verified 54
schema-correct/not-strict-ready current failures and 58 oracle-schema
not-ready failures. The only small high-confidence candidate is
multiplicative ratio-word extraction (`twice`/`double`/`two times`,
`triple`/`three times`). Production implementation validated the diagnostic
projection exactly under `PYTHONHASHSEED=0`: strict readiness improves from
247/331 to 255/331, ordinary readiness improves from 257/331 to 265/331,
Schema R@1 remains 301/331, and there are 0 strict/ordinary readiness losses.
See `docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md` and
`results/final_resubmission_method/`. Do not perform further method
development for this resubmission.

The P0 feature-augmented learned scorer (the direction recommended in
Phase 2) WAS built and evaluated in Phase 3 and did **not** improve over
typed greedy (`docs/LEARNED_GROUNDING_P0.md`, decision gate C) — see §6.
Do not re-attempt it without first reading why it failed, and benchmark
any future attempt against a **freshly rerun** typed greedy, not a
committed number. Full detail and next steps:
[`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md`](docs/ALGORITHM_IMPROVEMENT_ROADMAP.md).

## 9. External Baseline Roadmap

| Baseline | Status |
|---|---|
| **PaMOP** (IJCAI 2025) | **PILOT VALIDATED**, fidelity gate RESOLVED (`B. MODEL_LIMITED`) — independent reconstruction, no official code available; larger run pending. See §10. |
| **ORLM** | **PILOT RUNNING HEALTHY (2026-08-13)** — pinned official checkpoint cached; six-instance official-checkpoint inference is healthy in tmux with CPU offload; no completed empirical rows yet; coptpy missing so solver execution is blocked |
| **OptMATH** | **IMPLEMENTED, READY FOR INFERENCE (2026-08-12)** — official prompt/checkpoint provenance, NLP4LP adapter, Gurobi parser/validator/harness, result schema, evaluator, manifest, and mocked tests complete; no inference or solver run |
| **DeepOR** | **PAPER RECONSTRUCTION READY (2026-08-12)** — mock-tested adapter, paper-level prompt, reasoning parser, Pyomo static validator, safe harness, schema, evaluator, and manifest; official code/checkpoint not found and no empirical result |
| **OR-R1** | **CODE INTEGRATED, CHECKPOINT BLOCKED (2026-08-13)** — official code verified (`SCUTE-ZZ/OR-R1`, cited directly by the arXiv paper); lightweight adapter/runner/TGRPO-control/majority-voting/normalizer/validator/harness/evaluator and mocked tests complete; no SFT/GRPO/merged checkpoint released anywhere; TGRPO training set is transductive (== union of all eval sets, incl. NLP4LP); see [`docs/ORR1_PROVENANCE.md`](docs/ORR1_PROVENANCE.md) |

Verified by directory listing: `baselines/` contains `baselines/pamop/`
(pilot executed), `baselines/orlm/`, `baselines/optmath/`, `baselines/deepor/`,
and `baselines/orr1/` (lightweight inference-preparation paths, 2026-08-12
through 2026-08-13). Full
research (citations, code/weight availability, GPU/environment
requirements, ranked order and rationale):
[`docs/BASELINE_IMPLEMENTATION_ROADMAP.md`](docs/BASELINE_IMPLEMENTATION_ROADMAP.md).

### 9a. Cross-baseline comparison harness

`baselines/comparison/` (2026-08-13) is a unified, lightweight analysis
layer over the five baselines above plus `ours`: a shared `UnifiedRow`
schema, per-baseline adapters, a native-vs-shared metric taxonomy, Wilson
CI/exact-McNemar statistics, mock-evidence exclusion, and a Markdown/CSV/JSON
report generator (`python -m baselines.comparison.cli`). Frozen protocol:
[`docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md`](docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md).
Generated report (status `PRELIMINARY_EXTERNAL_BASELINE_STATUS`, real rows
only for `ours` and PaMOP; ORLM has a running pilot but no completed rows and
remains pending in the report; OptMATH/DeepOR/OR-R1 are `PENDING`/
`UNAVAILABLE`, never fabricated):
[`results/external_baseline_comparison/comparison.md`](results/external_baseline_comparison/comparison.md).

## 10. PaMOP Reproduction Status

**Implemented stages** (verified present in `baselines/pamop/`): LLM extraction
(`G_extr`), partition tree (`partition.py`), self-augmented modeling merge
(`G_mod`, bottom-up), AMPL renderer/executor (`ampl_interface.py`), and the
`G_exe`/`G_rev`/`G_comp`/`G_remod` correction loop (`correction.py`).

**Known reproduction choices / gaps:**
- No official PaMOP code was found; this is an independent reproduction.
- Current LLM: `gpt-4.1-mini-2025-04-14` via Azure OpenAI (exact original
  PaMOP GPT-4 model/prompts are unknown/unavailable).
- The exact PaMOP 67-problem subset identity is **unresolved** — see
  `docs/PAMOP_REPRODUCTION_PLAN.md` §13.4-13.7 for the detailed investigation;
  a deterministic stratified 6-problem pilot subset was used instead for the
  forensics pass (problem IDs 14, 23, 34, 72, 84, 88).
- A major input-source/systematic bug was found and fixed prior to the
  current results (see `docs/PAMOP_PILOT_FAILURE_FORENSICS.md`).

**Current results** (verified against `results/pamop/forensics_targeted/summary.json`):
initial execution success 2/6 (0.333), final execution success 6/6 (1.0) after
the correction loop, semantic correctness 1/6 (0.167), mean correction
iterations = 1.0 among the 4 corrected problems (0.67 averaged across all 6),
total tokens = 24,194.

**Decision gate (RESOLVED 2026-08-12, Phase 4): `B. MODEL_LIMITED`.** The
fidelity diagnostic required by Phase 3's `FIDELITY_DIAGNOSTIC_REQUIRED`
gate has been run:
[`results/pamop/fidelity_diagnostic_gpt5/README.md`](results/pamop/fidelity_diagnostic_gpt5/README.md).
Same 6 problem ids, same reconstructed prompts, only the Azure deployment
changed from `gpt-4.1-mini-2025-04-14` to `gpt-5.4` (the strongest
deployment available on this workstation). Result: semantic correctness
jumped from 1/6 to **4/5 evaluable (0.8)** with zero prompt changes.
**This is model-limited, not prompt-limited** — the reconstructed
pipeline and prompts are not the primary bottleneck. `docs/PAMOP_PILOT_FAILURE_FORENSICS.md`'s
prior "C. IMPLEMENTATION SOUND, MODEL/PROMPT IS PRIMARY LIMITATION" reading
is now sharpened: it is the *model*, not the prompt, that was primarily
limiting.

**Recommendation, not yet executed:** any future scale-up (18- or
269-case rerun) should use `gpt-5.4` or the strongest available deployment,
not `gpt-4.1-mini`. **This diagnostic does not itself authorize a
scale-up** — per this phase's explicit instruction, no 18- or 269-case
PaMOP rerun was launched. That remains a deliberate future decision, now
informed by evidence about which deployment to use when it happens.

**Caveat:** n=5 evaluable problems is a small sample (4/5 vs 1/6 is
suggestive, not rigorously powered). A C2/C4 (prompt-strengthening)
comparison was not run — this diagnostic isolated the model axis only,
a deliberate scope reduction given time constraints.

**Current implementation status:** the independent reconstruction now has a
configurable pilot runner, structured per-instance traces preserving generated
AMPL and correction remodel outputs, and explicit labeling of objective
equality as an objective-value proxy rather than full PaMOP semantic accuracy.
No larger run is active; the existing unrelated AMPL/HiGHS computation must
not be disturbed.

## 11. Data / Solver / API Environment

High-level only, no secrets:

- **NLP4LP**: gated HuggingFace dataset (`udell-lab/NLP4LP`); a local gold
  cache exists (`results/eswa_revision/00_env/nlp4lp_gold_cache.json`); no
  `HF_TOKEN` needed to read that cache, but a token is needed to re-pull fresh.
- **Gurobi**: a license is present on this workstation
  (`~/gurobi.lic`, outside the repo) and `gurobipy` is importable, but **only
  inside a dedicated virtualenv** (`~/.venvs/gurobi`, referenced via the
  `PAMOP_AMPLPY_PYTHON` env var in `baselines/pamop/README.md`) — **not** in
  the default `python3` / repo `venv`. The main paper's solver-backed subset
  intentionally uses a SciPy HiGHS shim instead and does not require Gurobi.
- **AMPL / amplpy**: same dedicated virtualenv as Gurobi above; verified
  importable there, not importable in the default environment.
- **Azure OpenAI**: working per PaMOP pilot/forensics runs (see §10); no
  further detail here (no secrets).
- **Other optional LLM providers**: OpenAI, Gemini, Mistral wiring exists for
  auxiliary (non-paper-core) reruns — see `docs/GEMINI_RERUN_REPORT.md`,
  `docs/MISTRAL_RERUN_REPORT.md`. Mistral Wulver jobs 902367/902368 did not
  produce output (missing `MISTRAL_API_KEY` in job env, per
  `docs/provenance/mistral_wulver_submission_2026-04-03.md`) — treat as
  unresolved infra, not a completed rerun.

## 12. Canonical Files

- **Main downstream tool:** `tools/nlp4lp_downstream_utility.py`
- **Retrieval:** `retrieval/search.py`, `retrieval/baselines.py`
- **Structural verification:** `formulation/verify.py`
- **Manuscript source (authoritative):** `manuscript/main.tex` (compiles to
  `manuscript/main.pdf`, 36 pages) — see `docs/KAIS_SOURCE_OF_TRUTH.md`
- **Corrected downstream results:**
  `results/eswa_revision/13_tables/postfix_main_metrics.csv`,
  `results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv`,
  regenerated into `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
  by `tools/build_camera_ready_table1.py`
- **PaMOP implementation:** `baselines/pamop/` (see `baselines/pamop/README.md`)
- **PaMOP results:** `results/pamop/pilot/`, `results/pamop/forensics_targeted/`
- **Learned-grounding infra (existing, reusable):** `src/learning/`,
  `artifacts/learning_ranker_data/nlp4lp/` (leak-free 230/50/50 split, see
  `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` "Training supervision feasibility")
- **Strongest known method (2026-08-12, Phase 4):** plain `typed_greedy`
  (default `assignment_mode`), `tools/nlp4lp_downstream_utility.py` —
  freshly rerun InstantiationReady 0.7764 on `orig`, beats every richer
  alternative evaluated. See `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`.
  (`max_weight_matching` was briefly, incorrectly, believed to hold this
  title earlier the same day — see `results/unevaluated_methods_evaluation/`
  for the raw numbers, now reinterpreted as a negative result.)
- **P0 learned-scorer implementation (negative result):**
  `tools/learned_local_scorer.py`, `scripts/learning/build_p0_corpus.py`,
  `scripts/learning/train_p0_classifier.py`, `scripts/learning/eval_p0_grounding.py`;
  results in `results/learned_grounding_p0/`
- **Key docs:** `docs/KAIS_SOURCE_OF_TRUTH.md` (manuscript authority),
  `docs/REVIEWER_GUIDE.md` (reviewer orientation), `docs/KNOWN_ISSUES.md`,
  `docs/HOW_TO_REPRODUCE.md`, `docs/PAMOP_REPRODUCTION_PLAN.md`,
  `docs/PAMOP_PILOT_FAILURE_FORENSICS.md`, plus the Phase-2/3 documents
  linked at the top of this file

## 13. Immediate Next Steps

**See [`NEXT_STEPS.md`](NEXT_STEPS.md) for the short, operational execution
queue — this section gives the same picture at a higher level.** Full
prioritized roadmap with prerequisites/success/stop criteria:
[`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md`](docs/ALGORITHM_IMPROVEMENT_ROADMAP.md)
(P0-P6, rewritten 2026-08-12 Phase 4 after the `max_weight_matching`
retraction).

**P0 (algorithm improvement) — DONE, negative result.** P0 learned scorer
built and evaluated; did not improve over typed greedy (`docs/LEARNED_GROUNDING_P0.md`).

**P1 (was "understand max_weight_matching") — RETRACTED (2026-08-12,
Phase 4).** That finding was an artifact of a stale comparison baseline;
see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. Do not resume this
line of work as originally framed.

**P1 (NEW, replaces the retracted one) — verify/refresh the remaining
stale method numbers:**
- `global_compat_*`, `relation_aware_*`, `ambiguity_aware_*` were not
  regenerated fresh in Phase 4 (time-bounded) — rerun them via
  `run_single_setting()` against a freshly rerun typed greedy, the same
  procedure already used for the other 9 methods
  (`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` §8).

**P2 (PaMOP — DONE, resolved 2026-08-12, see §10):** fidelity diagnostic
run; gate is `B. MODEL_LIMITED`. **A future 18- or 269-case rerun, if
undertaken, should use `gpt-5.4` (or the strongest available deployment),
not `gpt-4.1-mini`** — this was not itself authorized or launched in this
phase.

**P3 (algorithm improvement):**
- Target `_score_mention_slot_opt`'s and `_choose_token`'s known residual
  error modes (same-type ambiguity, total/per-unit confusion — see
  `results/max_weight_matching_validation/`) with further deterministic
  fixes, in the style of the 49 commits documented in the staleness audit.
  This has a demonstrated track record; no richer architecture has beaten
  it so far.

**P4 (manuscript, requires author decision — see
`docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md`):**
- Decide how to handle the submitted manuscript's headline number not
  reproducing from current code (erratum, pin the exact submission
  commit, or a "v2" revision with regenerated numbers).

**P5 (baseline coverage):**
- Inspect the running ORLM pilot and validate its six completed rows before
  launching the fixed common-18 run. A COPT/`coptpy` license is required only
  for later generated-code execution, not for model inference.
- Evaluate Ner4Opt (Kadıoğlu et al. 2024, pretrained models on HuggingFace)
  against our numeric-extraction stage — an existing, published, locally-
  runnable model in the same problem family that we do not currently compare
  against (`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` literature table).

## 14. Things a New Agent Must NOT Do

- **Do not trust any committed InstantiationReady number without first
  checking `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`.** The
  manuscript's own headline number (0.5287) does not reproduce from
  current code (fresh: 0.7764). Always rerun `tfidf_typed_greedy` fresh
  before using it as a comparison baseline for any new method — see the
  audit doc §8 for the exact commands.
- **Do not re-propose `max_weight_matching` / `search_structured_grounding`
  / `hierarchical_structured_grounding` as positive results** — they are
  negative results (lose to fresh typed greedy, p<0.05 on `orig`; see
  NR12 in `docs/NEGATIVE_RESULTS.md`). A same-day Phase 3 claim to the
  contrary has been retracted.
- Do not treat archived metrics under `docs/archive/`, `docs/archive_internal_status/`,
  `docs/provenance/`, or `results/eswa_revision/` (except the specific files
  named in §3/§12 as corrected sources) as current authoritative numbers —
  and even the "corrected sources" named there
  (`postfix_main_metrics.csv`) are now known to not reproduce from current
  code; see the staleness audit.
- Do not reinvent GCG, relation-aware linking, or ambiguity-aware grounding
  — they exist and have been benchmarked (§5), with documented negative
  results (§6), though their exact numbers have not been re-verified fresh
  (see `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` P1).
- Do not claim exact PaMOP reproduction — this is an independent, admittedly
  imperfect reproduction with named unresolved details (model, prompts,
  67-problem subset identity).
- **Do not launch a PaMOP 18- or 269-case rerun without deciding to** —
  the fidelity diagnostic is now done (§10, gate `B. MODEL_LIMITED`), and
  a scale-up is technically informed (use `gpt-5.4`), but launching one
  was explicitly out of scope for this phase and remains a deliberate
  future decision, not something to do automatically just because the
  gate is resolved.
- Do not expose gated NLP4LP data or redistribute it outside HuggingFace's
  own access-control terms.
- Do not expose API keys, tokens, or the Gurobi/AMPL license contents in
  commits, logs, or chat.
- Do not run large/expensive experiments (full PaMOP larger run, full LLM
  provider reruns, full NLP4LP re-benchmark) before validating on a small
  pilot first, matching the existing pattern (`results/pamop/pilot/` before
  a larger run). Note: `max_weight_matching`/`search_structured_grounding`/
  `hierarchical_structured_grounding` were each confirmed to run in under 2
  seconds for the full 331-query benchmark — this class of experiment is
  cheap and does not need pilot-gating; the pilot-first discipline applies
  to genuinely expensive runs (LLM APIs, GPU training), not every experiment.
- Do not modify canonical result artifacts without recording provenance
  (generating script/commit, why the change was made) — follow the existing
  `.stale`-suffix convention (e.g. `downstream_comparison_all_methods.csv.stale`)
  when superseding a file rather than silently overwriting it. `table1_main_benchmark_summary.csv`
  and `results/eswa_revision/13_tables/robustness_by_variant.csv` both now
  have real generators (`tools/build_camera_ready_table1.py`,
  `tools/build_robustness_by_variant.py`) — re-run them, don't hand-edit
  the CSVs.
- **When rendering `tools/build_eaai_camera_ready_figures.py`, be aware of
  a Pillow gotcha already fixed once:** the module now calls `Image.init()`
  at import time because Pillow's plugin registry is lazy-populated per
  format, and `PdfImagePlugin` internally needs `Image.SAVE["JPEG"]`
  registered even though no JPEG file is ever explicitly saved — a PNG-only
  save does not trigger this, causing a `KeyError: 'JPEG'` mid-render that
  can leave a truncated PDF if saves aren't atomic (which they now are, via
  `_save_both`'s temp-file-then-rename pattern). Do not remove the
  `Image.init()` call or the atomic-write pattern without understanding why
  they're there; see `tests/test_camera_ready_figures.py` for the
  regression tests.
- Do not modify the manuscript's scientific claims without separately
  verifying the underlying data supports the change. Phases 1-4 of this
  repository-polish effort explicitly did not touch `manuscript/main.tex`
  or `results/paper/eaai_camera_ready_tables/`, **even after discovering
  that the manuscript's own headline number does not reproduce from
  current code** (§4) — that decision is deliberately left to the paper's
  author; see `docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md`.
- **Do not propose "train a learned local mention-slot scorer" as if it
  were untested** — two attempts (NR10, NR11/P0) have both failed to show
  a gain, for different, specific, documented reasons. Read both entries
  in `docs/NEGATIVE_RESULTS.md` and `docs/LEARNED_GROUNDING_P0.md` before
  building a third attempt, and benchmark any new attempt against a
  **freshly rerun** typed greedy (§8), not `max_weight_matching` (a
  negative result, not a working baseline to build on).
- **Do not implement `max_weight_matching` again** — it already exists
  (`_run_max_weight_matching_grounding`), was independently re-verified
  clean (no leakage, deterministic, correct metrics) in Phase 4, and is a
  documented negative result. Re-implementing it would be pure waste.
- **Do not manually edit generated numerical artifacts** —
  `results/eswa_revision/13_tables/postfix_main_metrics.csv` in particular
  should never be hand-edited to "fix" the staleness finding; regenerate
  it through its generator (`training/external/run_full_downstream_benchmark.py`
  or `run_single_setting()`) if and when the author decides to, per
  `docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md`.
- Do not rebuild the learned-grounding train/dev/test split from scratch —
  a leak-free, hash-verified 230/50/50 instance-level split already exists
  (`artifacts/learning_ranker_data/nlp4lp/`, see
  `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` "Training supervision feasibility").
  Phase 3 additionally confirmed schema-level leakage is structurally
  impossible for this dataset (330 instances, 330 unique schemas) — see
  `results/learned_grounding_p0/split_metadata.json`.
- Do not assume the `max_weight_matching` family's 0.70-0.74 InstantiationReady
  numbers are directly comparable to P0's 0.80-0.86 numbers in
  `docs/LEARNED_GROUNDING_P0.md` — the former is the full 331-query
  retrieval-conditioned benchmark; the latter is a 50-instance oracle-
  schema subsample. Always check the denominator before comparing numbers
  across this repository's result families.
