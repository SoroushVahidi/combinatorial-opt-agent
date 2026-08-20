# P0 Feature-Augmented Local Grounding

**Status:** Phase 3 (2026-08-12). Decision gate: **C -- P0 does NOT improve**
InstantiationReady relative to canonical typed greedy, on this evaluation.
See "Decision Gate" below for the precise, non-overstated reading (the gap
is numerically real but not statistically significant at n=50). **Read this
alongside `results/unevaluated_methods_evaluation/README.md`**, a
significant, unrelated positive finding from the same phase (three
already-implemented, never-before-evaluated methods substantially beat
typed greedy) -- that finding, not P0, is now the strongest known local
baseline and should anchor any future learned-scorer comparison.

Raw artifacts: `results/learned_grounding_p0/`. This document is the
narrative; it does not restate every number in those files.

---

## Motivation

Phase 2's `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` P0 recommended: "a small
classifier over the already-computed hand-engineered features ... plus a
frozen sentence embedding," explicitly scoped to differ from the prior
failed learned ranker (NR10) by using features instead of text-only input,
and a small classifier instead of a fine-tuned transformer. This document
reports the result of actually building and evaluating that recommendation.

## Difference From NR10

| | NR10 (`docs/NEGATIVE_RESULTS.md`) | P0 (this document) |
|---|---|---|
| Input | Raw text: `[SLOT] name (role) [SEP] mention_surface context` | ~24 hand-engineered features from `_score_mention_slot_opt` (type compatibility, operator/role/unit cues, total-vs-per-unit signals, bound-direction signals, the raw hand-engineered score itself) + 1 frozen sentence-embedding cosine-similarity feature |
| Model | Fine-tuned `distilroberta-base` (82M parameters) | Logistic regression (selected over gradient-boosted trees via validation-only comparison) |
| Training objective | Binary cross-entropy over 500 steps, batch size 8 (~4,000 examples seen, well under one epoch of the 9,729-pair training set) | Same 230/50/50-instance split; the classifier sees the FULL training set each fit (not step-capped) |
| Negative sampling | All (mention, slot) pairs in the original 5-feature corpus | All (mention, slot) pairs in a re-extracted, richer corpus (see "Data Split"); only ~10-12% are trivially type-incompatible, so no aggressive downsampling was needed |
| Encoder | Fine-tuned end-to-end | **Frozen**, not fine-tuned (`sentence-transformers/all-MiniLM-L6-v2`, the same default already used by this repo's own SBERT retrieval baseline) |
| Inference | Local (no external API) | Local (no external API) |
| Result (same 4 diagnostic metrics NR10 reported, informal comparison only -- different corpus/extraction, see caveat below) | pairwise_accuracy 0.197, slot_selection_accuracy 0.182, exact_slot_fill_accuracy **0.000**, type_match_after_decoding 0.068 -- all below the rule baseline | Dev slot-selection accuracy 0.4694 vs. rule-only 0.3673 on the SAME corpus (a real gain on this proxy metric) |

**Caveat on the "difference" claim itself:** P0 did materially differ from
NR10 in every dimension above, and did show a real improvement on the
internal proxy metric (dev group/slot-selection accuracy). The downstream
grounding-quality result (Coverage/TypeMatch/InstantiationReady, see
"Results" below) did **not** show the same improvement -- this is itself an
important, reported finding (a proxy-metric/downstream-metric mismatch),
not evidence that P0 secretly repeated NR10's mistake.

## Data Split

**Reused, not rebuilt**, per phase instructions. The existing NR10 split
(`docs/learning_runs/real_data_only_learning_check.md`) was rebuilt
byte-reproducibly with its own script/seed (`src.learning.split_nlp4lp_corpus_for_benchmark`,
seed=42) to recover the exact instance-id partition -- this is the SAME
partition NR10 used, not a new one:

- 230 train / 50 dev / 50 test **instances** (verified: exact match to the
  documented counts)
- Original 5-feature corpus pair counts reproduced exactly: 9,729 / 2,230 / 2,339
- P0's richer corpus (see "Features"): 11,755 / 2,633 / 2,736 pairs (more
  pairs than the original because the canonical opt-role mention extractor
  finds a different mention set than the original regex-based extractor)

**Schema-level leakage: verified absent, more strongly than previously
known.** Phase 2 flagged schema-level leakage as "not yet verified." This
phase found: **every one of the 330 usable NLP4LP instances has a unique
`schema_name`** (no two queries share a template), so instance-level and
schema-level disjointness coincide exactly -- there is no additional
schema-overlap risk beyond the already-verified instance-level split, and
no separate schema-disjoint diagnostic split was needed. Full detail:
`results/learned_grounding_p0/split_metadata.json`.

**Important scope limitation of this split:** it is built under an
**oracle/gold-schema assumption** (`schema_name` = NLP4LP's own
`relevant_doc_id`, not a retrieval prediction). It supports a grounding-only
comparison; it does **not** support a direct comparison against the
manuscript's retrieval-conditioned 331-query InstantiationReady figures
(0.5287 etc.). See "Evaluation Protocol" for how M0 was made comparable.

## Features

26-dimensional vector per (mention, slot) pair, reusing
`tools/nlp4lp_downstream_utility.py::_score_mention_slot_opt`'s own
diagnostics dict (the same function `optimization_role_repair` already
uses) rather than a separately-maintained feature set:

`type_incompatible, derived_count_non_count, type_exact, type_loose,
opt_role_overlap, fragment_objective, fragment_bound, fragment_resource,
fragment_ratio, operator_match, ctx_overlap, sent_overlap, unit_match,
entity_resource_overlap, total_match, coefficient_match,
coeff_to_total_penalty, total_to_coeff_penalty, count_role_match,
count_to_non_count_penalty, lower_bound_match, upper_bound_match,
bound_direction_wrong, weak_penalty, hand_engineered_score` (the raw
opt-role score itself, included as one input feature among many) +
`embedding_similarity` (see below). Implementation:
`tools/learned_local_scorer.py` (new, additive module -- does not modify
`tools/nlp4lp_downstream_utility.py`). No gold-target information is used
as a feature (verified by construction: features come only from
`_score_mention_slot_opt`, which never sees the label).

## Frozen Encoder

`sentence-transformers/all-MiniLM-L6-v2` -- the exact same default model
this repository's own SBERT retrieval baseline already uses
(`retrieval/search.py::_default_model_path`), already cached locally
(`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`),
no new download or dependency. Not fine-tuned in this phase. Feature =
cosine similarity between (mention surface + local context tokens) and
(slot name + role tags + operator preference + expected type), both
L2-normalized, batched and deduplicated for efficiency.

## Classifier

Preregistered set of exactly two: logistic regression (scikit-learn,
`StandardScaler` + `LogisticRegression(class_weight="balanced")`) and
gradient-boosted trees (`HistGradientBoostingClassifier`). No MLP (not
justified given the small feature/data scale) and no broad hyperparameter
search. **Selected on dev only**, before any test-set access: logistic
regression, dev slot-selection accuracy 0.4694 vs. gradient-boosted trees'
0.4653 -- both trained on the combined (engineered + embedding) feature
set. Rule baseline (raw `hand_engineered_score`, no learning) on the same
dev set: 0.3673.

## Training Procedure

```bash
export NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json
python3 -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split test --output_dir artifacts/learning_corpus
python3 -m src.learning.split_nlp4lp_corpus_for_benchmark --corpus_dir artifacts/learning_corpus --seed 42
python3 scripts/learning/build_p0_corpus.py       # richer feature corpus, reusing the split above
python3 scripts/learning/train_p0_classifier.py   # model selection + ablation, dev-only
python3 scripts/learning/eval_p0_grounding.py      # M2/M3/M4 on test
```

Model artifact: `artifacts/learning_runs/p0/p0_model.joblib` (not committed
-- deterministically regenerable from the commands above given the fixed
seed=42 corpus and `random_state=42` classifier; see "Reproducibility").

## Evaluation Protocol

Coverage/TypeMatch/InstantiationReady computed with the exact canonical
definitions (`param_coverage = n_filled / n_expected_scalar`, `type_match =
type_matches / n_filled`, `instantiation_ready = 1 iff coverage >= 0.8 AND
type_match >= 0.8`), matching `tools/nlp4lp_downstream_utility.py`'s own
per-query computation.

- **M0** (canonical oracle + typed greedy): produced by running the
  **unmodified** canonical CLI directly
  (`python3 -m tools.nlp4lp_downstream_utility --baseline oracle
  --assignment-mode typed`) restricted to the identical 50 test-split
  instance ids, for a true apples-to-apples comparison under the same
  oracle-schema assumption the P0 corpus uses. This is **not** the
  manuscript's 331-query TF-IDF headline (0.5287) -- it is the same method
  on a 50-query oracle-schema subsample, which scores substantially higher
  (0.86) because both the subsample and the oracle-schema assumption are
  easier than the full retrieval-conditioned benchmark.
- **M1** (NR10): not rerun, per phase instructions -- reused as already
  documented in `docs/NEGATIVE_RESULTS.md` NR10. Its four metrics are not
  numerically comparable to M0/M2-M4 (different corpus, different metric
  set), so no direct table row mixes them; the comparison is qualitative
  (see "Difference From NR10" above).
- **M2** (P0 greedy): independent per-slot argmax over the trained
  classifier's scores.
- **M3** (P0 + global assignment): exact bipartite max-weight matching
  (`scipy.optimize.linear_sum_assignment`), the same algorithm the
  canonical `_run_max_weight_matching_grounding` already uses internally,
  applied here as a decoupled decode step over P0's own score matrix
  (implemented in `scripts/learning/eval_p0_grounding.py`, not by modifying
  the canonical function).
- **M4** (M3 + validate/repair): M3's assignment, then each filled slot is
  checked with the canonical single-assignment plausibility primitive
  (`_opt_role_validate_one`, the same function
  `_opt_role_validate_and_repair` itself calls per candidate); implausible
  fills are dropped and slots are refilled from the next-best remaining
  candidate that passes validation.

## Results

50-instance oracle-schema test subset (`results/learned_grounding_p0/test_results.csv`):

| Method | Coverage | TypeMatch | InstantiationReady |
|---|---|---|---|
| **M0** (canonical oracle + typed greedy) | 0.9700 | 0.9370 | **0.8600** |
| Rule-only opt-role greedy (informational; same extraction/features as P0, no learning) | 0.9800 | 0.9153 | 0.8400 |
| **M2** (P0 greedy) | 0.9800 | 0.8928 | 0.8000 |
| **M3** (P0 + max-weight assignment) | 0.9700 | 0.8912 | 0.7800 |
| **M4** (M3 + validate/repair) | 0.9800 | 0.8812 | 0.7800 |

No P0 configuration exceeds M0. Notably, plain **rule-only** greedy
decoding over the identical opt-role feature set (no learning at all)
already beats every P0 (learned) configuration -- the learned classifier's
higher *dev proxy-metric* accuracy (0.4694 vs. rule's 0.3673) did **not**
translate into better downstream grounding quality on test. This is a
genuine, reported proxy-metric/target-metric mismatch, not a training bug
(verified: the model trains normally, dev accuracy moves in the expected
direction, the corpus passed all integrity checks in "Data Split").

## Global-Assignment Ablation

M2 (greedy, 0.80) > M3 (max-weight, 0.78) ≈ M4 (max-weight + repair, 0.78)
on this test subset. **Global assignment did not help on top of P0's
learned scores** -- consistent with the established pattern in
`docs/NEGATIVE_RESULTS.md` NR5-NR7 (global/repair methods do not rescue a
weak local scorer). This directly answers Phase 2's H2/roadmap-P1 question
for *this specific* learned scorer: no, combining it with existing global
assignment was not worthwhile. (Separately and by contrast, global
assignment over the **hand-engineered, non-learned** opt-role score is
dramatically valuable -- see
`results/unevaluated_methods_evaluation/README.md`. The two findings
together suggest the limiting factor here is P0's learned score quality,
not the global-assignment mechanism itself.)

## Feature/Embedding Ablation

Dev slot-selection accuracy, logistic regression, same train/dev split
(`results/learned_grounding_p0/ablation_results.csv`):

| Feature set | Accuracy |
|---|---|
| Combined (engineered + embedding) | **0.4694** |
| Engineered only | 0.4449 |
| Embedding only | 0.2367 |

The frozen embedding feature contributes positively (+0.0245 over
engineered-only) but is far weaker alone than the engineered features
(0.2367 vs. 0.4449) -- consistent with the embedding being a single
similarity scalar versus ~24 structured signals.

## Error Analysis

Instance-level transitions, M0 -> M2 (`results/learned_grounding_p0/error_analysis.csv`):
3 instances improved (not-ready under M0 -> ready under M2), 6 regressed
(ready -> not-ready), 37 unchanged-ready, 4 unchanged-not-ready (net -3 of
50, matching the 0.86 -> 0.80 InstantiationReady drop exactly).

A representative regressed case (`nlp4lp_test_8`) illustrates a **multi-slot
disambiguation** failure matching the Phase 2 bottleneck's rank-1/2
categories: two distinct slots (`ShapingTimeAvailable`, gold=3000;
`BakingTimeAvailable`, gold=4000) both compete against a third,
higher-hand-engineered-score but wrong-role candidate (value 150) that the
rule score prefers for neither correct slot but which still displaces both
gold assignments under the learned model's ranking -- a same-numeric-type,
weak-slot-name-cue case the classifier did not learn to resolve better than
the hand-engineered score already did.

No dedicated breakdown by min/max, percent, coefficient-vs-total, etc. was
computed beyond this representative case, given the small (n=6) regressed
set; a future pass with a larger evaluation set would be needed for a
statistically meaningful per-category breakdown.

## Statistical Test

Paired bootstrap (B=1000, seed=42, same methodology as
`results/eswa_revision/15_significance/`), M0 vs. M2 (best P0 config) on
InstantiationReady, n=50:

**diff = 0.0600 (favors M0), 95% CI = [-0.0600, 0.1800], p = 0.44 (NOT
significant).**

The CI includes zero (and even includes small negative values, i.e. M2
numerically better), so the observed gap is **not statistically
distinguishable from no difference** at this sample size -- but no P0
configuration exceeded M0's point estimate either. Full record:
`results/learned_grounding_p0/significance.json`.

## Limitations

- n=50 test instances is small; the significance test's wide CI reflects
  this directly.
- The oracle-schema evaluation protocol (necessary for a clean grounding-
  only comparison) means these numbers are not directly comparable to the
  manuscript's retrieval-conditioned headline figures.
- Only 2 classifier types and 3 feature-set variants were tried, by design
  (avoid a broad hyperparameter search); a more thorough architecture
  search was explicitly out of scope for this phase.
- The `max_weight_matching`/`search_structured_grounding`/`hierarchical_structured_grounding`
  discovery (see `results/unevaluated_methods_evaluation/`) was made
  *after* P0's decision gate was reached and is not folded into this
  document's frozen M0-M4 comparison. **Correction (2026-08-12, Phase 4):**
  that discovery was itself later found to rest on a stale typed-greedy
  comparison baseline and is now a negative result — see
  `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` and NR12 in
  `docs/NEGATIVE_RESULTS.md`. A future P0-successor should be benchmarked
  against a **freshly rerun** `tfidf_typed_greedy` (0.7764 on the full
  331-query benchmark as of 2026-08-12), which remains the strongest known
  non-oracle method — not `max_weight_matching`.

## Decision Gate

**C -- P0 does NOT improve** grounding relative to the M0 baseline it was
evaluated against. Precise reading: no P0 configuration (M2/M3/M4)
numerically exceeded M0, and pure rule-only decoding over the same
richer feature set already beats every P0 configuration; the M0-vs-M2 gap
itself is not statistically significant at n=50, so a true null effect
cannot be ruled out, but there is no positive evidence for P0 either. This
is not decision D (data/supervision was not fundamentally broken -- the
corpus is clean, the classifier trained normally, dev metrics moved
sensibly) and not A/B (no configuration showed even a nominal gain).

## Next Step

Per the roadmap's decision-gate consequence for outcome C: **do not keep
stacking learned classifiers on this feature/architecture combination.**
Move to the next established family from
`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md`'s literature review (structured
cross-encoder with joint text+feature input, i.e. roadmap architecture #2,
or top-k retrieval-grounding joint reranking, H5) -- but **first**,
re-anchor against a **freshly rerun** `tfidf_typed_greedy` baseline
(InstantiationReady 0.7764 on the full 331-query benchmark as of
2026-08-12 — always rerun this, never trust a committed number without
checking `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` first), since it
is the strongest known non-oracle method and the correct target for any
future comparison. (An earlier version of this section pointed at
`max_weight_matching` as the bar to beat; that was a same-day error,
corrected in Phase 4 — see the staleness audit.) See
`docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` for the updated roadmap.
