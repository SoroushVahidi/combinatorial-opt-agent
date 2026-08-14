# Scientific State

**Purpose:** the detailed scientific handoff behind `PROJECT_STATUS.md`.
Written so a new agent needs no chat history to continue this project.
Last verified 2026-08-12 (Phase 4). Read
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` first if you have not — it
changes the interpretation of almost every number below.

---

## 1. Research Question

Given a natural-language description of an optimization problem, can a
system (a) retrieve the most compatible schema from a fixed catalog of
optimization problem templates, then (b) deterministically ground that
schema's scalar parameters from numeric evidence in the text — without a
generative LLM at inference time, deterministically, and with every
intermediate decision inspectable? This is framed as a knowledge-
processing / schema-conditioned slot-filling problem, explicitly **not**
a full NL-to-solver compiler. Benchmark: NLP4LP (`udell-lab/NLP4LP`,
gated HuggingFace dataset), 331 `orig`-variant test queries against a
335-entry schema catalog.

## 2. Current Best Pipeline

```
NL query → Schema retrieval (TF-IDF cosine similarity, top-1)
         → Numeric mention extraction (regex + word-number parsing)
         → Schema-conditioned scalar grounding: TYPED GREEDY
           (_choose_token: type-preference-ranked greedy fill,
            tools/nlp4lp_downstream_utility.py)
         → Structural LP check (formulation/verify.py)
         → [Optional] Solver on restricted subset (SciPy HiGHS shim)
```

Plain typed greedy — the manuscript's own original baseline method — remains
the strongest semantically reliable non-oracle method as of 2026-08-13 (see
§3-4). A later selective top-k schema reranker improves the repository's
InstantiationReady metric to 265/331, but Stage-B audit classified it as
`STAGE_B_METRIC_ONLY_GAIN` because most new ready cases use incorrect schemas.
The follow-up strict-readiness diagnostic therefore recommends
schema-correctness-gated readiness as the primary native end-to-end proxy.

## 3. Current Best Verified Results

**Fresh, same-code (`0f0b24e`+), `orig`, 331 queries, 2026-08-12:**

| Method | InstantiationReady | Notes |
|---|---|---|
| `oracle_typed_greedy` | 0.8248 | retrieval upper-bound control |
| **`tfidf_typed_greedy`** | **0.7764** | **strongest non-oracle method** |
| `tfidf_selective_grounding_rerank` | 0.8006 | **metric-only gain**: 6/8 new ready cases use incorrect schemas |
| `bm25_typed_greedy` | 0.7644 | n.s. vs. tfidf, p=0.322 |
| `tfidf_acceptance_rerank` | 0.7644 | n.s. vs. tfidf, p=0.328 |
| `tfidf_constrained` | 0.7492 | borderline, p=0.050 |
| `max_weight_matching` | 0.7432 | **loses**, p=0.042 |
| `tfidf_optimization_role_repair` | 0.7372 | loses, p=0.020 |
| `lsa_typed_greedy` | 0.7341 | loses, p<0.001 |
| `tfidf_hierarchical_acceptance_rerank` | 0.7190 | loses, p<0.001 |
| `tfidf_semantic_ir_repair` | 0.7160 | loses, p<0.001 |
| `search_structured_grounding` | 0.7039 | loses, p<0.001 |
| `hierarchical_structured_grounding` | 0.7039 | loses, p<0.001 |

**IMPORTANT:** the manuscript's *submitted* Table 4 reports
`tfidf_typed_greedy` = 0.5287 (Coverage 0.8609, TypeMatch 0.7453). This is
what the paper says and it is unchanged; it just does not reproduce from
the current codebase (see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`).
Both numbers are "real" in the sense that they are correctly computed —
they were computed at different points in the codebase's history.

`global_compat_*`, `relation_aware_*`, `ambiguity_aware_*` have **not**
been re-verified fresh (`NEXT_STEPS.md` P5) — their committed numbers
(0.42-0.50 range) are presumed stale by pattern but not confirmed.

**Fresh strict-readiness diagnostic (2026-08-13):**

| Method | StrictInstantiationReady | False-ready count |
|---|---:|---:|
| `oracle_typed_greedy` | 273/331 = 0.8248 | 0 |
| `tfidf_selective_grounding_rerank` | 249/331 = 0.7523 | 16 |
| `tfidf_typed_greedy` | 247/331 = 0.7462 | 10 |

The selective reranker's ordinary +8 readiness gain becomes only +2 strict
ready gains (`nlp4lp_test_222`, `nlp4lp_test_268`). Future main-method gates
should use strict readiness, with ordinary InstantiationReady retained as a
predicted-schema diagnostic.

**Strict-failure quick-fix production validation (2026-08-13):**

`docs/STRICT_FAILURE_QUICK_FIX_DIAGNOSTIC_2026-08-13.md` verifies 54 current
schema-correct/not-strict-ready failures and 58 oracle-schema/not-ready
failures. The only small, high-confidence, deterministic candidate is
multiplicative ratio-word extraction: expose `twice`/`double`/`two times` as
2.0 and `triple`/`three times` as 3.0 ratio tokens. The diagnostic prototype
projected 247/331 -> 255/331 strict readiness with 0 simulated strict losses.
Production validation reproduced that projection exactly: strict readiness is
255/331 = 0.7704, ordinary InstantiationReady is 265/331 = 0.8006, Schema
R@1 remains 301/331, and there are 0 strict/ordinary readiness losses.
Method development is now `FROZEN_FOR_RESUBMISSION`; see
`docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`.

## 4. Why the Best Method Works

Typed greedy's simplicity is not the reason it wins — its *history* is.
49 commits (`git log 3fffe68..HEAD -- tools/nlp4lp_downstream_utility.py`)
made targeted, deterministic fixes to `_choose_token` and its supporting
functions (`_expected_type`, `_is_type_match`, bound-role annotation,
percent-vs-int/float disambiguation, distractor suppression, enumeration-
derived counts, etc.) between the manuscript's submission-era codebase
and today. None of the richer alternative methods (learned scorers,
repair rules, global assignment, beam search) received the same volume of
targeted refinement to their own scoring machinery
(`_score_mention_slot_opt` and its dependents), and/or their more complex
mechanisms did not benefit from the same class of fix as directly. The
net effect: the simple baseline overtook every richer alternative through
iterative, unglamorous, deterministic bug-fixing — not through a new
algorithmic idea. See §13 for why this is the recommended direction going
forward.

## 5. Methods Already Tried

Full inventory with CLI dispatch names: `docs/METHOD_INVENTORY.md`.
Summary: typed greedy (baseline, canonical); constrained matching,
semantic IR repair, optimization-role repair, acceptance/hierarchical-
acceptance reranking, global compatibility grounding (3 ablations),
relation-aware linking (4 ablations), ambiguity-aware grounding (4
variants), `max_weight_matching`, `search_structured_grounding`,
`hierarchical_structured_grounding` — all evaluated, all lose to fresh
typed greedy. Two learned local mention-slot scorers (NR10: text-only
transformer pairwise ranker; NR11/P0: feature-augmented classifier) —
both negative results, see §6.

## 6. Negative Results

Full ledger with statistical evidence: `docs/NEGATIVE_RESULTS.md` (NR1-NR12).
**Headline:** no evaluated method — deterministic-richer, repair-based,
global-assignment, or learned — beats plain, freshly-rerun typed greedy
on `orig` InstantiationReady. This is a **strengthened** version of the
original Phase 2 finding, not a new one: Phase 3 briefly (same day)
believed global assignment (`max_weight_matching` et al.) was an
exception; this was retracted after the comparison baseline was found
stale (NR12).

## 7. Current Error Taxonomy

For `max_weight_matching` specifically (full 331-query, slot-level,
`results/max_weight_matching_validation/mechanism_and_error_analysis_summary.json`):
same-type ambiguity (335 slot-level instances, dominant), total/per-unit
confusion (166), missing mentions (156), objective/constraint confusion
(124), type mismatch (65), min/max polarity (33), schema retrieval miss
(30), zero-expected-scalar (20), percent ambiguity (11). Since
`max_weight_matching` uses the *same* local score (`_score_mention_slot_opt`)
that several other tried methods depend on, this taxonomy is informative
about the local score's own weaknesses generally, independent of which
decode strategy sits on top of it.

For typed greedy specifically: `docs/CURRENT_BOTTLENECK_ANALYSIS.md`'s
counts (82/331 type mismatch, etc.) have **not** been re-verified against
current code (`NEXT_STEPS.md` P5) — treat as directionally informative,
not exact.

## 8. Current Bottleneck Ranking

1. Same-type ambiguity / total-per-unit confusion in the local pairwise
   score — the dominant residual failure mode across every method that
   depends on `_score_mention_slot_opt`, confirmed even under exact
   global optimization (§7).
2. Type mismatch on otherwise-covered decisions (typed-greedy-specific
   count not yet re-verified fresh, see §7).
3. Schema retrieval miss (~9% of queries) — top-k reranking can improve
   InstantiationReady, but wrong schemas can exploit the readiness metric.
4. Coverage gaps (missing numeric mentions) — moderate.
5. Zero-expected-scalar-slot queries — 95% are a downstream artifact of
   (3), not independent; 1/331 genuine (vector/matrix-valued parameters,
   architecturally out of scope for scalar-only grounding).

## 9. Architectural Weaknesses

- Hand-engineered pairwise mention-slot scoring — root cause of the
  dominant same-type/total-per-unit bottleneck (§7-8); every richer
  *deterministic* variant of this scoring also failed.
- Coarse numeric type system — cannot distinguish semantically different
  quantities sharing a coarse type.
- Lexicon-based role/unit/operator cue detection — brittle to unseen
  phrasing.
- Scalar-only grounding — cannot represent vector/matrix-valued
  parameters (the one genuine zero-expected-scalar case, §8).
- Fixed top-1 retrieval before grounding — retrieval errors are recoverable by
  selective top-k reranking, but the current readiness metric can reward wrong
  schemas with easier overlapping scalar slots.

## 10. Benchmark / Data Weaknesses

- Fixed schema catalog (335 entries) — cannot generalize beyond it.
- Gated dataset — reproduction requires HF approval; local gold cache
  mitigates but does not eliminate this.
- Single primary benchmark — no cross-benchmark generalization evidence
  (Text2Zinc/CP-Bench are adapter-only).
- 331-vs-335 catalog-vintage Schema R@1 offset (0.9094 vs 0.9063 family)
  — a disclosed measurement artifact, separate from and smaller than the
  §Evaluation-weaknesses staleness issue below.

## 11. Evaluation Weaknesses

- **NEW, dominant (2026-08-12):** the manuscript's own result-generation
  pipeline (`postfix_main_metrics.csv` via
  `training/external/run_full_downstream_benchmark.py`) was not
  re-run after 49 subsequent commits of grounding fixes, producing a
  49-commit-stale headline number. This was not, and structurally could
  not have been, caught by internal-consistency checks (does the CSV
  match the manuscript text?) — only by actually re-running the
  generator. See `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`.
- Limited solver-backed scale (20/331), compatibility-filtered.
- No broad comparison against LLM-based auto-formulation methods yet
  (ORLM/OptMATH complete for lightweight inference preparation; DeepOR is
  paper-reconstruction-ready; OR-R1's official code is verified and
  integrated but its checkpoint is unreleased — CODE_INTEGRATED_CHECKPOINT_BLOCKED).
- PaMOP reproduction fidelity uncertain (6/6 execution, 1/6 semantic
  correctness on the pilot).
- 3 of ~16 grounding-method families (`global_compat_*`,
  `relation_aware_*`, `ambiguity_aware_*`) have not been re-verified
  against current code (`NEXT_STEPS.md` P5).

## 12. Strengths of the Approach

- No external generative-LLM API required at inference for the core
  pipeline (classical IR + deterministic rules).
- Deterministic and reproducible inference — verified bit-exact across
  fresh reruns for `InstantiationReady`/`Coverage`/`TypeMatch` (minor
  hash-randomization-driven tie-breaking nondeterminism exists only in
  the secondary `exact5`/`exact20` diagnostics, fixed by
  `PYTHONHASHSEED=0`, see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` §6).
- CPU-friendly, no GPU dependency in the evaluated method set.
- Interpretable — every intermediate decision inspectable.
- Modular — retrieval, extraction, scoring, assignment are separable.
- Strong fixed-catalog retrieval (TF-IDF R@1 ≈ 0.91).
- The 49-commit improvement trajectory (§4) demonstrates this codebase
  responds well to targeted, evidence-driven, deterministic fixing — a
  real, demonstrated strength of the development process itself, not
  just the current snapshot.

## 13. Improvement Directions

Ranked (`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` has full detail,
prerequisites, falsification criteria):

1. **Complete external baseline execution and manuscript revision.** Method
   development is frozen for this resubmission after validating the
   multiplicative ratio-word extraction patch.
2. **Integrate strict readiness into the manuscript evaluation framing.**
   Ordinary InstantiationReady remains useful for historical comparison, but
   strict readiness is the primary native end-to-end proxy.
3. **Verify/refresh remaining stale numbers** (`global_compat_*` etc.) only
   if needed for evidence cleanup, not as a new method direction.
4. H5 (top-k retrieval + grounding joint reranking) — replicated as an
   ordinary-readiness metric artifact; only +2 strict-ready gains.
5. Confidence calibration/abstention — conditional, needs a working
   improved score first (NR7 shows current abstention badly miscalibrated).
6. A genuinely new combinatorial algorithm — only if 1-5 are exhausted
   and shown insufficient; the bar for this is high given nothing tried
   so far has beaten iterative deterministic refinement of the baseline.

## 14. External Baseline Status

| Baseline | Status | Next action |
|---|---|---|
| PaMOP | IN PROGRESS, fidelity gate RESOLVED | optional C2/C4 prompt follow-up, or decide on scale-up (§15) |
| ORLM | **IMPLEMENTED, READY FOR INFERENCE** (`baselines/orlm/`) | smoke test — needs checkpoint/GPU; COPT only for later solver execution, see `NEXT_STEPS.md` P9 |
| OptMATH | **IMPLEMENTED, READY FOR INFERENCE** (`baselines/optmath/`) | 7B checkpoint smoke test when resources allow, see `docs/OPTMATH_PROVENANCE.md` |
| DeepOR | PAPER RECONSTRUCTION READY; official code/checkpoint not found | use `baselines/deepor/`; do not claim empirical results until an official checkpoint is available |
| OR-R1 | **CODE INTEGRATED, CHECKPOINT BLOCKED** (`baselines/orr1/`) | no official SFT/GRPO/merged checkpoint exists anywhere; faithful reproduction requires training TGRPO from scratch, and TGRPO's official training data is transductive over the eval sets — see `docs/ORR1_PROVENANCE.md` |

ORLM verified 2026-08-12 (primary-source research): official code public
(`github.com/Cardinal-Operations/ORLM`, Apache-2.0), one confirmed public
HF checkpoint (`CardinalOperations/ORLM-LLaMA-3-8B`, 8B, llama3 license —
the paper's Mistral-7B/DeepSeek-Math-7B checkpoints are NOT independently
confirmed public), generates `coptpy` (COPT solver) code not
Pyomo/GurobiPy, not evaluated on NLP4LP in the original paper (NL4OPT/
MAMO/IndustryOR only), single 24GB-class GPU plausible for inference, no
fine-tuning required, fully local/offline once weights+COPT license
obtained.

**Cross-baseline comparison harness (2026-08-13):** `baselines/comparison/`
unifies all five baselines above plus `ours` into one analysis view without
conflating incomparable metrics (native vs. shared vs. resource vs.
availability, per `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md`).
Generated report: `results/external_baseline_comparison/comparison.md`
(status `PRELIMINARY_EXTERNAL_BASELINE_STATUS`; real rows only for `ours`
and PaMOP).

## 15. PaMOP Status

**Decision gate: `B. MODEL_LIMITED`** (resolved 2026-08-12, Phase 4 — see
`results/pamop/fidelity_diagnostic_gpt5/README.md` and `PROJECT_STATUS.md`
§10). 6-problem deterministic pilot (ids 14, 23, 34, 72, 84, 88), C1
(`gpt-4.1-mini-2025-04-14`): initial execution 2/6, final execution 6/6
(after correction loop), semantic correctness 1/6, 24,194 total tokens.
Fidelity diagnostic C3 (same 6 ids, same prompts, only the deployment
changed to `gpt-5.4`, the strongest available on this workstation):
semantic correctness 4/5 evaluable (0.8) — a large improvement from a
model swap alone, no prompt changes. **This is model-limited, not
prompt-limited.** **Recommendation (not executed):** any future scale-up
should use `gpt-5.4`. No 18- or 269-case rerun was launched in this
phase — that remains a deliberate future decision. Caveat: n=5 evaluable
is a small sample; a full C2/C4 prompt-strengthening comparison was not
run (scope-reduced given time constraints).

## 16. Manuscript Status

**Not modified in Phases 1-4.** The submitted manuscript's headline
InstantiationReady (0.5287) does not reproduce from current code (fresh:
0.7764) — see §3, §11, and
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. This is left to the
author to resolve; see `docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md`
for the four candidate responses and why none was chosen automatically.
The paper's *qualitative* central claims (oracle gain is modest; richer
deterministic methods don't help) appear to still hold under fresh
numbers — this is not a case of the science being wrong, but of the
reported absolute numbers not reproducing.

## 17. Open Scientific Questions

- Why did the 49 commits improve typed greedy so much more than they (or
  contemporaneous commits) improved the richer scoring methods? Not
  investigated at the individual-commit level (would require a full
  bisection across ~49 commits × several methods — not attempted, flagged
  as expensive and lower-value than the P0-P3 directions in §13).
- Are `global_compat_*`/`relation_aware_*`/`ambiguity_aware_*` also stale
  by a similar magnitude? Pattern-predicted but not confirmed
  (`NEXT_STEPS.md` P5).
- Does the same-type-ambiguity/total-per-unit-confusion bottleneck (§7-8)
  have a tractable deterministic fix, or does it require genuinely richer
  semantic understanding (dependency parsing, H3) or more training data
  than a rule-based approach can encode?
- What exact commit does the submitted manuscript's Table 4 correspond
  to? Not identified precisely in this pass (candidate: at or before
  `3fffe68`, not confirmed as the exact submission commit).

## 18. Immediate Next Experiments

See `NEXT_STEPS.md` for the full operational queue. Highest
priority: P1 (strict-readiness failure diagnosis) and P6 (manuscript
path decision, requires the author, not an agent-executable experiment).

## 19. Things Future Agents Must Not Repeat

- Do not trust a committed InstantiationReady number without first
  rerunning it fresh — this is the single most important lesson from
  Phase 4, see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`.
- Do not re-propose `max_weight_matching`/`search_structured_grounding`/
  `hierarchical_structured_grounding` as positive results (NR12).
- Do not propose a third learned local mention-slot scorer without a
  concrete, evidenced reason to expect a different outcome than NR10/NR11.
- Do not launch a PaMOP 18- or 269-case rerun before the fidelity
  diagnostic (§15).
- Do not treat solver-feasibility as semantic correctness (PaMOP's own
  pilot shows 6/6 execution but 1/6 semantic correctness — these are
  different things).
- Do not treat Oracle-TG as a hard upper bound (the manuscript itself
  documents a case, the 20-instance solver-backed subset, where it isn't).
- Do not treat archived/`.stale`-suffixed artifacts as canonical.
- Do not implement max-weight bipartite matching again — it exists, was
  independently re-verified clean (no leakage, deterministic, correct
  metrics), and is a documented negative result.
- Do not manually edit generated numerical artifacts
  (`postfix_main_metrics.csv` in particular) — regenerate through the
  generator script if and when the manuscript decision (§16) calls for it.
- Do not rebuild the learned-grounding train/dev/test split from scratch
  — a leak-free, hash-verified 230/50/50 split already exists.
- Do not modify `manuscript/main.tex` or the camera-ready tables without
  the author's explicit decision (§16) — this applies even to seemingly
  "corrective" changes.
