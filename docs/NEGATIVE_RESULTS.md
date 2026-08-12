# Negative Results Ledger

**Purpose:** prevent future agents from repeatedly rebuilding ideas that were
already tried and did not materially improve the primary metric
(`InstantiationReady`, `orig` variant, TF-IDF retrieval unless noted).

All values below are canonical, corrected values (see `results/CANONICAL_RESULTS.md`)
— none are hand-invented. Statistical evidence is from
`results/eswa_revision/15_significance/SIGNIFICANCE_SUMMARY.md` (paired
bootstrap, B=1000, seed=42, two-sided). Baseline (as submitted /
NR1-NR9): `tfidf_typed_greedy`, InstReady = **0.5287**.

**2026-08-12 (Phase 4) — read `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`
before citing the 0.5287 baseline above for any *new* comparison.** That
number does not reproduce from the current codebase (fresh rerun: 0.7764).
NR1-NR9's own *conclusions* (richer greedy/repair/global methods lose to
typed greedy) are not overturned by this — see NR12 below, which shows the
gap direction holds and if anything strengthens under fresh numbers — but
the exact margins/p-values for NR1-NR9 were computed against the stale
baseline and have not all been individually re-verified fresh. Any new
method proposed after 2026-08-12 must be compared against a **freshly
rerun** `tfidf_typed_greedy`, not the committed 0.5287.

---

## Central finding

**Richer deterministic grounding does not reliably outperform typed greedy.**
Every richer deterministic method that was actually evaluated on real NLP4LP
gold data either ties typed greedy (not statistically distinguishable) or
loses to it, several significantly (p<0.001). See `docs/METHOD_INVENTORY.md`
Part 2 for the full method-by-method table this ledger summarizes. **This
finding is now stronger than originally stated**: it also holds for the
three global-assignment methods (`max_weight_matching`,
`search_structured_grounding`, `hierarchical_structured_grounding`) that
were briefly (same day, 2026-08-12) believed to be exceptions — see NR12.

---

## NR1: Constrained (1-to-1) matching

- **Hypothesis:** enforcing a hard 1-to-1 constraint between mentions and
  slots (no mention reused across slots) reduces slot-disambiguation errors.
- **Implementation:** `_constrained_assignment` in `tools/nlp4lp_downstream_utility.py`.
- **Expected benefit:** higher Coverage/InstReady by preventing greedy
  double-assignment of the same numeric mention.
- **Actual result:** InstReady 0.4230 vs. 0.5287 baseline — a large drop.
- **Statistical evidence:** not directly in the paired-significance table
  (only InstReady deltas for TF-IDF vs. named comparators are tabulated
  there), but the magnitude (-0.1057) is far larger than any
  not-significant comparison in this ledger.
- **Interpretation:** the hard constraint apparently forces the assignment
  algorithm away from otherwise-good greedy choices more often than it
  prevents genuine double-assignment errors.
- **Worth revisiting?** Only as a *component* combined with a better local
  scorer (§ "P0" in `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md"), not as a
  standalone fix.

## NR2: Semantic IR repair

- **Hypothesis:** semantic tag-based repair (detecting operator/unit tags,
  re-scoring with `_score_mention_slot_ir`) resolves float/percent
  confusion better than plain typed greedy.
- **Implementation:** `_run_semantic_ir_repair`, `tools/nlp4lp_downstream_utility.py`.
- **Actual result:** InstReady 0.4864 vs. 0.5287 baseline.
- **Interpretation:** the hand-engineered semantic tag vocabulary does not
  capture enough of the actual ambiguity to net-improve over greedy.
- **Worth revisiting?** The mechanism (semantic tags) is a reasonable
  *feature* for a learned scorer, but the deterministic repair-rule version
  is not worth further tuning on its own.

## NR3: Optimization-role repair

- **Hypothesis:** lexicon-based role cues (capacity, cost, cardinality-bound
  language) resolve "total vs. per-unit" and role-confusion errors.
- **Actual result:** InstReady 0.4411 vs. 0.5287 baseline.
- **Interpretation:** role-cue lexicons are brittle; likely both false
  positives (misapplied role cues) and false negatives (missed cues) offset
  the intended gains.
- **Worth revisiting?** The *role-tag features* (`_compute_primary_role`,
  `_compute_bound_role`) remain available as input features for a learned
  scorer (H4 in `docs/RESEARCH_HYPOTHESES.md`), independent of whether the
  deterministic repair rule built on top of them is used.

## NR4: Acceptance reranking / hierarchical acceptance reranking

- **Hypothesis:** reranking top-k retrieved schemas by an acceptance score
  (optionally hierarchical) recovers cases where top-1 retrieval is wrong
  but a nearby candidate is right.
- **Actual result:** InstReady 0.5257 (acceptance rerank) and 0.5196
  (hierarchical), both near the 0.5287 baseline.
- **Statistical evidence:** TFIDF-TG vs. TFIDF-AR: diff=0.0030, p=0.89
  (not significant). TFIDF-TG vs. TFIDF-HAR: diff=0.0091, p=0.58 (not
  significant).
- **Interpretation:** statistically indistinguishable from the baseline —
  neither a proven win nor a proven loss. Given retrieval R@1 is already
  ~0.91, there is limited room for reranking to matter on `orig`.
- **Worth revisiting?** Possibly more informative on `noisy`/`short`
  variants where retrieval is weaker (not verified in this pass — check
  `results/eswa_revision/02_downstream_postfix/nlp4lp_downstream_{noisy,short}_tfidf_acceptance_rerank.json`
  before assuming either way).

## NR5: Global compatibility grounding (GCG, `global_compat_*`)

- **Hypothesis:** beam search with pairwise global-consistency penalties
  (duplicate reuse, bound inversion, total-vs-per-unit mismatch) jointly
  resolves errors that local scoring misses.
- **Actual result:** `global_compat_full` InstReady = 0.4320 vs. 0.5287.
- **Statistical evidence:** TFIDF-TG vs. GCG-Full: diff=0.0967, 95% CI
  [0.0544, 0.1420], **p<0.001 (robust)**.
- **Interpretation:** a statistically robust negative result — global
  consistency penalties, as currently scored, actively hurt more than they
  help on this benchmark.
- **Worth revisiting?** Not in its current hand-engineered-penalty form. A
  learned local scorer feeding into (not replacing) this global-assignment
  machinery is the recommended next test (H2 in `RESEARCH_HYPOTHESES.md`),
  since the failure may be in the local pairwise scores the beam search
  optimizes over, not the beam search itself.

## NR6: Relation-aware linking (basic/ops/semantic/full)

- **Hypothesis:** explicit mention-mention and slot-slot relation features
  (four increasing ablation levels) improve disambiguation beyond
  pairwise-only scoring.
- **Actual result:** `relation_aware_basic` InstReady = 0.4985 (closest
  competitor to baseline in this ledger); `relation_aware_full` = 0.4169
  (worst of the four levels).
- **Statistical evidence:** TFIDF-TG vs. RAL-Basic: diff=0.0302, p=0.15
  (not significant — the only relation-aware variant that isn't a proven
  loss). TFIDF-TG vs. RAL-Full: diff=0.1118, p<0.001 (robust loss).
  RAL-Basic vs. Oracle-TG: diff=-0.0695, p=0.006 (RAL-Basic is
  significantly worse than oracle too).
- **Interpretation:** more relation features monotonically hurt in this
  implementation (basic > ops > semantic > full in performance, worst to
  best inverted) — each added feature family introduces more noise than
  signal under the current hand-engineered scoring.
- **Worth revisiting?** Yes, with a caveat: the module's own docstring
  (`tools/relation_aware_linking.py`) states "a learned scorer can be
  plugged in" — this is the most natural integration point for the §P0
  local learned scorer (`RESEARCH_HYPOTHESES.md` H2), since the relation
  *feature extraction* infrastructure already exists and only the scoring
  function would change.

## NR7: Ambiguity-aware grounding (candidate-greedy/beam/abstain/full)

- **Hypothesis:** explicit modeling of competing candidates, ambiguity
  signals (margin/entropy), and confidence-gated abstention improves
  precision on ambiguous slots.
- **Actual result:** `ambiguity_aware_beam` InstReady = 0.4230;
  `ambiguity_aware_full` = 0.4199; `ambiguity_aware_abstain` collapses
  Coverage to 0.2207 (over-abstains far more than it helps).
- **Statistical evidence:** TFIDF-TG vs. AAG-Beam: diff=0.1057, p<0.001.
  TFIDF-TG vs. AAG-Full: diff=0.1088, p<0.001. TFIDF-TG vs. AAG-Abstain
  (Coverage): diff=0.6402, p<0.001 — an extreme, robust loss.
- **Interpretation:** the abstention threshold is far too conservative as
  tuned, and the beam/full variants lose more from search overhead or
  mis-scored competition than they gain from explicit ambiguity modeling.
- **Worth revisiting?** The abstention *mechanism* (confidence + margin
  gating) is architecturally relevant to future calibration work (P4 in
  `ALGORITHM_IMPROVEMENT_ROADMAP.md`), but only after a better base scorer
  exists — abstaining well on a bad score doesn't help much.

## NR8: Sample-size / benchmark-bias checks (not a grounding method, but a ruled-out explanation)

- **Hypothesis:** strong retrieval (R@1≈0.91) is driven by lexical overlap
  (queries reusing schema-description words) rather than genuine schema
  understanding.
- **Actual result:** retrieval performance is *preserved or improved* after
  stripping numbers/stopwords (LSA improves from 0.8459 to 0.9184 under
  stopword removal).
- **Interpretation:** this concern is **ruled out** — retrieval success is
  attributable to structural/domain-term overlap, not numeric-value
  leakage or superficial lexical shortcuts.
- **Worth revisiting?** No — this is a closed, resolved question, not an
  open negative result.

## NR9: Sample-size explanation for the retrieval R@1 offset

- **Not a grounding-method negative result** — flagged here to prevent
  confusion with NR1-NR7. The 0.9094-vs-0.9063 Schema R@1 offset (see
  `results/CANONICAL_RESULTS.md` §A) is a **disclosed, unresolved, minor**
  catalog-vintage artifact (331 vs. 335 documents), not a failed
  improvement attempt. Do not list it as a negative result in future
  updates to this file — it belongs in provenance notes, not here.

---

## NR10: Learned pairwise mention-slot ranker (real-data-only, text-only)

- **This is the single most important entry in this ledger for anyone
  planning to build a learned local scorer — read it before starting.**
- **Hypothesis:** a fine-tuned transformer pairwise ranker (`distilroberta-base`)
  over (mention context, slot) text pairs would outperform the hand-engineered
  rule scorer, trained and evaluated entirely on real NLP4LP data with a
  clean, leak-free split (no synthetic/GAMS auxiliary data).
- **Implementation:** `src/learning/` + `training/` infrastructure; corpus
  built via instance-level 70/15/15 split (230 train / 50 dev / 50 test
  instances; 9,729 / 2,230 / 2,339 pairwise pairs), split integrity verified
  by `verify_split_integrity` (distinct SHA-256 hashes per split, no
  instance-level overlap). Training: 500 steps, batch size 8, lr 2e-5, 1
  epoch-ish, seed 42. Full record: `docs/learning_runs/real_data_only_learning_check.md`.
- **Actual result (job 854626):** learned model **lost on every metric** to
  the same-split rule baseline: pairwise_accuracy 0.197 vs 0.247,
  slot_selection_accuracy 0.182 vs 0.229, exact_slot_fill_accuracy **0.000**
  vs 0.022, type_match_after_decoding 0.068 vs 0.125.
- **Two earlier, even worse variants** (also negative, do not revive as-is):
  GAMS weak-label auxiliary training (`docs/learning_runs/gams_aux_vs_nlp4lp_only.md`)
  — TypeMatch collapsed; targeted synthetic auxiliary training
  (`docs/learning_runs/targeted_synth_vs_nlp4lp_only.md`) — TypeMatch
  collapsed when scaled.
- **Interpretation — why this likely failed, not just "learning doesn't
  work":** (a) 500 steps × batch 8 ≈ 4,000 examples seen, well under one
  full epoch over 9,729 training pairs — likely severely undertrained; (b)
  **text-only** input — none of the rich hand-engineered features already
  computed elsewhere in this codebase (type tags, operator/unit cues,
  relation-aware features from `tools/relation_aware_linking.py`,
  optimization-role tags) were given to the learned model as auxiliary
  input, so it had to relearn from ~10K examples what the rule scorer
  encodes as a prior; (c) a feature-augmented variant
  (`nlp4lp_pairwise_text_plus_features`, planned in the "Stage 3" round,
  `docs/EXPERIMENTS.md` §5.4) was **never actually run** — that round
  reported "no learned runs completed (torch/transformers not available in
  run environment)," so the feature-augmented hypothesis remains untested,
  not falsified.
- **Statistical evidence:** none reported (single run, no significance test) — treat the negative result as directionally strong (loses on *every* metric, including a catastrophic exact-match collapse to 0.000) but not statistically characterized.
- **Explicit prior decision:** `docs/learning_runs/real_data_only_learning_check.md`
  §8 already recorded "[x] Stop and keep learning as future work" for *this
  specific formulation*.
- **Worth revisiting?** **Yes, but not by repeating this exact setup.** Any
  future learned-local-scorer attempt (see `docs/RESEARCH_HYPOTHESES.md` H1/H2)
  MUST differ from this one in at least: (1) inject existing hand-engineered
  features as auxiliary input rather than text-only; (2) train substantially
  longer / more data-efficiently (contrastive or hard-negative-mining
  objectives, not 500 steps of plain cross-entropy); (3) reuse the existing
  leak-free split/infra (`artifacts/learning_ranker_data/nlp4lp/`,
  `src/learning/verify_split_integrity`) rather than rebuilding it. A
  same-setup rerun should be treated as **already answered**.

## NR11: P0 feature-augmented local scorer (Phase 3, differs from NR10 as prescribed)

- **Hypothesis:** a small classifier (logistic regression / gradient-
  boosted trees) over ~24 already-computed hand-engineered pairwise
  features (from `_score_mention_slot_opt`) plus a frozen sentence-
  embedding similarity feature would outperform both the deterministic
  rule score and NR10's failed text-only transformer, addressing NR10's
  two diagnosed weaknesses (no structured features, undertrained).
- **Implementation:** `tools/learned_local_scorer.py` (new module),
  `scripts/learning/{build_p0_corpus,train_p0_classifier,eval_p0_grounding}.py`.
  Reused NR10's exact instance-level split (230/50/50, seed 42); verified
  schema-level leakage is impossible for this dataset (330 instances, 330
  unique schemas). Model selected on dev only (logistic regression over
  gradient-boosted trees, 0.4694 vs. 0.4653 dev slot-selection accuracy).
  Full detail: `docs/LEARNED_GROUNDING_P0.md`.
- **Actual result:** on a 50-instance oracle-schema test subset, canonical
  oracle+typed-greedy (M0) reaches InstantiationReady 0.86; P0's best
  configuration (M2, greedy decode) reaches 0.80. **Pure rule-only greedy
  decoding over the identical richer feature set (no learning at all)**
  reaches 0.84, beating every P0 (learned) configuration. Global assignment
  (M3) and validate/repair (M4) do not help on top of P0's scores (0.78
  each, below M2).
- **Statistical evidence:** paired bootstrap (B=1000, seed=42), M0 vs. M2:
  diff=0.06 (favors M0), 95% CI=[-0.06, 0.18], p=0.44 -- **not significant**
  at n=50.
- **Interpretation:** unlike NR10 (catastrophic collapse to 0.000 exact-
  match), P0 is a functioning method that trains normally and shows a real
  gain on the internal dev proxy metric (slot-selection accuracy) -- but
  that proxy-metric gain did not transfer to the downstream grounding-
  quality metrics. This is a genuine proxy/target mismatch, not a
  reproduction of NR10's specific failure mode. Error analysis
  (`docs/LEARNED_GROUNDING_P0.md` "Error Analysis") shows the learned
  model still struggles with the same multi-slot, weak-slot-name-cue
  disambiguation cases that motivated it in the first place.
- **Worth revisiting?** Not with this exact feature/classifier/decode
  combination (decision gate C). **Correction (2026-08-12, Phase 4):** the
  paragraph originally here claimed `max_weight_matching` (also over the
  `_score_mention_slot_opt` feature family, no learning) reached 0.7432,
  "dramatically higher than either P0 or typed greedy," and should be the
  new benchmark for future learned-scorer attempts. That comparison used a
  stale typed-greedy number (0.5287); see NR12 below.
  `max_weight_matching` (0.7432) in fact loses to a fresh typed-greedy
  rerun (0.7764, p=0.042). Any future learned-scorer attempt should be
  benchmarked against a **freshly rerun** `tfidf_typed_greedy`, which
  remains the strongest known non-oracle method in this repository as of
  2026-08-12.

## NR12: `max_weight_matching`, `search_structured_grounding`,
`hierarchical_structured_grounding` (Phase 3 claim, corrected same-day
in Phase 4)

- **Original claim (Phase 3, 2026-08-12, morning/midday):** these three
  global-assignment methods, evaluated for the first time, scored
  0.70-0.74 InstantiationReady on `orig` — compared against the committed
  `tfidf_typed_greedy` = 0.5287, this looked like a dramatic, statistically
  significant (p<0.001) breakthrough, "the strongest results ever found in
  this repository," exceeding even Oracle-TG.
- **What was actually wrong:** the 0.5287 comparison baseline is stale.
  `results/eswa_revision/13_tables/postfix_main_metrics.csv` (source of
  0.5287) was last generated at commit `3fffe68`; 49 subsequent commits to
  `tools/nlp4lp_downstream_utility.py` improved candidate extraction and
  type-matching (the machinery typed greedy's `_choose_token` depends on)
  without the table ever being regenerated. A fresh, same-code rerun of
  plain `tfidf_typed_greedy` gives **0.7764** — a drift of +0.2477, larger
  than the entire claimed gain.
- **Fair, same-code comparison (`orig`, 331 queries, paired bootstrap
  B=1000 seed=42):**

  | Method | Fresh InstReady | vs. fresh typed greedy (0.7764) |
  |---|---|---|
  | `max_weight_matching` | 0.7432 | −0.0332, p=0.042 (significantly worse) |
  | `search_structured_grounding` | 0.7039 | −0.0725, p<0.001 (significantly worse) |
  | `hierarchical_structured_grounding` | 0.7039 | −0.0725, p<0.001 (significantly worse) |

  All three also lose to fresh Oracle-TG (0.8248, p<0.001) — the "exceeds
  Oracle-TG" claim does not hold either.
- **Mechanism (why the original comparison looked so dramatic):** the
  three methods were evaluated fresh, against current code, in the same
  Phase-3 session (correct). They were then compared against a committed
  number that was 49 commits stale (incorrect) instead of a freshly rerun
  typed-greedy baseline. The magnitude of code drift (+0.2477) happened to
  exceed the methods' own (real, but negative) gap to fresh typed greedy
  (−0.03 to −0.07), producing an apparent positive result that was, in
  net, actually negative.
- **Interpretation:** this does not mean `max_weight_matching`'s exact
  bipartite-matching decode is a bad idea in the abstract — the underlying
  local score (`_score_mention_slot_opt`) is the *same* score typed
  greedy's richer sibling methods use, and exact assignment is a
  reasonable decode strategy. It means that, given how much stronger this
  particular local score's *rival*, `_choose_token`'s much simpler
  type-preference heuristic, has become after 49 rounds of targeted
  fixing, there is currently no realized benefit to the extra
  global-assignment machinery on this benchmark. Full mechanism/error
  analysis: `results/max_weight_matching_validation/`.
- **Worth revisiting?** Only if `_score_mention_slot_opt`'s own local
  accuracy is substantially improved first (its residual errors — same-
  type ambiguity, total/per-unit confusion — are the dominant failure
  modes even under exact assignment; see
  `results/max_weight_matching_validation/mechanism_and_error_analysis_summary.json`).
  Applying exact assignment on top of a stronger local score than
  currently exists is untested and could plausibly help; applying it to
  the *current* score does not.
- **Full record:** `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` (primary
  source for this entry).

## What this ledger does NOT cover

- **PaMOP fidelity** (semantic correctness 1/6 despite 6/6 execution
  success) is a *reproduction-fidelity* finding, not a negative result
  about our own grounding methods — see `PROJECT_STATUS.md` §10 instead.
- **`max_weight_matching`, `search_structured_grounding*`,
  `hierarchical_structured_grounding*`** are *not* negative results — they
  are simply unevaluated (see `docs/METHOD_INVENTORY.md`). Do not assume
  they would fail like their cousins; evaluating them is cheap (existing
  code + tests) and should happen before writing them off.
