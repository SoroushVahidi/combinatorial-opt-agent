# Research Hypotheses

**Purpose:** testable hypotheses to guide the next development phase, grounded
in `docs/CURRENT_BOTTLENECK_ANALYSIS.md`, `docs/NEGATIVE_RESULTS.md`, and
`docs/METHOD_INVENTORY.md`. **Read `docs/NEGATIVE_RESULTS.md` NR10 before
acting on H1/H2** — a naive version of the learned-scorer idea was already
tried and failed; these hypotheses are deliberately scoped to differ from
that attempt in specific, evidenced ways.

Novelty status legend: **ESTABLISHED_ADAPTATION** (applying a known technique
to this task), **MODERATE_NOVELTY** (a non-obvious combination of known
techniques), **POTENTIALLY_NOVEL** (no close precedent found in this pass's
literature search — treat as a weak claim, not a strong one).

---

## H1: A feature-augmented local scorer will beat both the rule baseline and the prior text-only learned scorer

**Status (2026-08-12, Phase 3): NOT SUPPORTED.** Implemented and evaluated
as P0 (`docs/LEARNED_GROUNDING_P0.md`). P0's best configuration (greedy
decode) did NOT beat either the rule baseline (0.80 vs. rule-only 0.84) or
canonical typed greedy (0.80 vs. 0.86) on InstantiationReady, though it did
beat both on the internal dev proxy metric (slot-selection accuracy). The
M0-vs-P0 gap itself was not statistically significant at n=50 (p=0.44), so
this is a soft, not a decisive, non-support — but the falsification
criterion specified ("must beat the rule baseline on pairwise_accuracy AND
type_match_after_decoding on the same held-out test split") was not met on
the downstream grounding metrics that matter. See NR11 in
`docs/NEGATIVE_RESULTS.md`.

- **Motivation:** `docs/NEGATIVE_RESULTS.md` NR10 shows a text-only learned
  pairwise ranker lost to the rule baseline; it never had access to the
  hand-engineered features (type tags, operator/unit cues, relation-aware
  features) the rule scorer implicitly encodes. `docs/METHOD_INVENTORY.md`
  Part 1 stage 11 notes `tools/relation_aware_linking.py`'s own docstring
  states "a learned scorer can be plugged in." Schema-guided dialogue-state-
  tracking work (BERT cross-encoders jointly encoding context + slot name +
  slot value with cross-attention; see roadmap §Literature) establishes that
  local contextual scoring over (context, slot-description) pairs is a
  standard, working pattern in an adjacent task.
- **Current evidence:** NR10 (negative, text-only); `docs/CURRENT_BOTTLENECK_ANALYSIS.md`
  rank 1 (type mismatch on fully-covered decisions, 82/331 = 24.8% of all
  queries) is exactly the error class a better local scorer should reduce.
- **Implementation concept:** extend the existing pairwise ranker
  (`src/learning/`) to take the already-computed hand-engineered features
  (from `_score_mention_slot_opt`, `relation_aware_linking.py`) as
  structured input features alongside (or instead of) raw text embeddings —
  i.e., `nlp4lp_pairwise_text_plus_features` from the never-run Stage 3 plan
  (`docs/EXPERIMENTS.md` §5.4), executed for the first time.
- **Primary metric:** InstantiationReady, `orig`, on the existing held-out
  test split (`artifacts/learning_ranker_data/nlp4lp/test.jsonl`, 50
  instances, 2,339 pairs).
- **Secondary metrics:** pairwise_accuracy, slot_selection_accuracy,
  exact_slot_fill_accuracy, type_match_after_decoding (same four metrics
  NR10 already reports, for direct comparability).
- **Falsification criterion:** if the feature-augmented learned scorer does
  not beat the rule baseline on pairwise_accuracy AND type_match_after_decoding
  on the same held-out test split, this hypothesis is falsified — do not
  proceed to H2's global-assignment integration.
- **Expected risk:** the 230-instance/9,729-pair training set is still small
  for fine-tuning a full transformer; a lighter classifier (gradient-boosted
  trees or a small MLP) over the engineered features plus a frozen sentence
  embedding may be more data-efficient than fine-tuning `distilroberta-base`
  end-to-end again.
- **Novelty status:** **ESTABLISHED_ADAPTATION** (feature-augmented
  cross-encoder / classifier is standard in schema-guided slot filling; the
  adaptation to optimization-schema grounding is the only new part).

## H2: Combining a working local scorer (from H1) with the existing global assignment will outperform either alone

**Status (2026-08-12, Phase 4 — supersedes the Phase 3 version of this
status): NOT SUPPORTED for either learned or non-learned scores.** H2
as literally framed presupposes "a working local scorer from H1" — since H1
was not supported, H2's own precondition was not met, so the learned-score
half of this hypothesis remains untested in the form it was framed (P0 + a
frozen embedding + global assignment (M3/M4) did not beat P0 + greedy (M2)
either, but this is a weaker test since H1's scorer itself was not strong).
**Phase 3 briefly (same day) believed** it had found that combining the
EXISTING NON-LEARNED opt-role score with exact global assignment
(`max_weight_matching`) was dramatically valuable (InstantiationReady 0.7432
vs. typed greedy's 0.5287, p<0.001). **This was retracted the same day**:
the 0.5287 comparison baseline was found to be stale relative to current
code (see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`); against a fresh
typed-greedy rerun (0.7764), `max_weight_matching` (0.7432) loses
significantly (p=0.042). H2's underlying intuition (global assignment adds
value once the local score is good enough) is **not supported** by any
evidence gathered so far — the existing hand-engineered local score,
combined with exact global assignment, still does not beat a strong
greedy baseline. If H2 is re-tested in a future phase, it must be against
a **freshly rerun** typed-greedy baseline, and would need either a
genuinely improved local score (not yet available) or evidence that the
current local score's own error modes (same-type ambiguity, total/per-unit
confusion — see `results/max_weight_matching_validation/`) are the actual
bottleneck for global assignment specifically.

- **Motivation:** `docs/METHOD_INVENTORY.md` Part 1 already separates local
  scoring (stages 8-12) from global assignment (stages 13-16); the negative
  results in `docs/NEGATIVE_RESULTS.md` NR5-NR7 show that improving *only*
  the global-assignment/search machinery on top of the *existing*
  hand-engineered local scores does not help (GCG, relation-aware, and
  ambiguity-aware all lose or tie). This suggests the local scores
  themselves are the limiting input to those global methods, not the global
  search logic.
- **Current evidence:** NR5/NR6/NR7 (global methods on top of hand-engineered
  local scores fail); `docs/METHOD_INVENTORY.md` stages 14-16 (max-weight
  matching, search-structured, hierarchical-structured) are implemented but
  **never evaluated on real data** — evaluating them with the *existing*
  local scores first is a cheap, informative control before attributing any
  future gain to the learned scorer specifically.
- **Implementation concept:** (a) first, evaluate `max_weight_matching`,
  `search_structured_grounding`, `hierarchical_structured_grounding` with
  the *existing* hand-engineered local scores on real NLP4LP data (near-zero
  new code, just running the existing CLI dispatch) to establish a clean
  baseline; (b) only then substitute H1's learned local scorer as the score
  source for the best-performing of these global methods.
- **Primary metric:** InstantiationReady, `orig`.
- **Secondary metrics:** Coverage, TypeMatch (to see whether gains are
  concentrated in Coverage, TypeMatch, or both).
- **Falsification criterion:** if learned-local + best-global does not beat
  both learned-local + typed-greedy (H1 alone) and hand-engineered-local +
  best-global (part (a) above) by a margin outside the paired-bootstrap 95%
  CI, the combination hypothesis is falsified — the components are not
  complementary and should be pursued separately, not jointly.
- **Expected risk:** compounding two experimental changes (new scorer +
  previously-unevaluated global method) makes failure attribution harder;
  the staged design above (a before b) mitigates this.
- **Novelty status:** **MODERATE_NOVELTY** (the individual pieces are
  established; combining a learned local scorer with exact bipartite
  matching or beam search over schema-slot assignment is a reasonable but
  not extensively precedented combination for this specific task).

## H3: Dependency/syntactic features will particularly improve min/max and same-sentence multi-number cases

**Status (2026-08-12, Phase 4 note): NOT TESTED.** Out of scope for this
phase (requires a new dependency, spaCy or similar, and a prerequisite
per-type error breakdown that was not computed). The Phase 3 note
de-prioritizing this in favor of `max_weight_matching`-based decode-strategy
changes has been retracted (`max_weight_matching` is a negative result —
see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`); this hypothesis's
priority should be assessed on its own merits, not relative to that
retracted claim.

- **Motivation:** user-identified failure categories (bound/polarity
  confusion, multi-numeric ambiguity) map to `_compute_bound_role`,
  `_find_range_annotations` in the existing code — currently pattern-based,
  not syntactic/dependency-aware.
- **Current evidence:** `docs/CURRENT_BOTTLENECK_ANALYSIS.md` rank-1 bucket
  (type mismatch, 82/331) is not broken down by sub-category in this pass
  (flagged as not independently re-derived); this hypothesis requires that
  finer breakdown as a prerequisite, not yet available.
- **Implementation concept:** add dependency-parse features (e.g. spaCy
  dependency arcs between numeric tokens and role-indicating words) as
  additional input to the H1 scorer, ablated against the non-dependency
  version.
- **Primary metric:** TypeMatch and Exact20_on_hits, restricted to a
  min/max-annotated or multi-numeric-mention subset (to be defined by first
  computing the finer breakdown this hypothesis depends on).
- **Secondary metrics:** overall InstantiationReady (to check no regression
  elsewhere).
- **Falsification criterion:** no improvement on the min/max/multi-numeric
  subset specifically, even if overall metrics are flat.
- **Expected risk:** requires a new dependency-parsing dependency (spaCy or
  similar) not currently in `requirements.txt` — a new environment cost to
  budget for.
- **Novelty status:** **ESTABLISHED_ADAPTATION** (dependency-aware slot
  filling is a known NLP technique; not yet applied here).

## H4: Richer semantic role typing will improve float/int/percent/coefficient-vs-total cases

**Status (2026-08-13): NOT SUPPORTED AS A NEXT MAIN METHOD FOR
InstantiationReady.** A dedicated Stage-A diagnostic was run after the method
novelty/efficiency audit:
`docs/ROLE_QUANTITY_STAGE_A_DIAGNOSTIC_2026-08-13.md`. It found real
slot-level signal — 28 of 49 targeted schema-hit/not-ready wrong assignments
were separable by deterministic role/quantity features — but the conservative
query-level upper bound rescued **0 additional InstantiationReady queries**
because the current readiness metric is gated by coverage/type compatibility,
not numeric exactness. This does not prove role/quantity features are useless;
it means they are not a justified next implementation target for the current
main metric and +2 pp manuscript gate. If revisited, scope them to numeric
exactness/Exact20 or a changed metric, not as another InstantiationReady patch.

**Prior status (2026-08-12, Phase 3): NOT TESTED.** Out of scope for that
phase (a follow-up ablation contingent on H1, which was not supported). Note
the semantic-role features H4 targets ARE already present in P0's feature set
and did not, in combination, produce a working scorer (see H1) — a standalone
ablation isolating just these features had not been run at that point.

- **Motivation:** `docs/NEGATIVE_RESULTS.md` NR3 (optimization-role repair)
  and NR2 (semantic IR repair) both failed as *deterministic rules*, but
  their underlying *feature extraction* (`_compute_primary_role`,
  `_context_to_semantic_tags`) remains available as input to a learned
  model rather than a hand-tuned decision rule.
- **Current evidence:** NR2/NR3 negative as rules; not yet tested as
  learned-model input features (this is the same reframing H1 already
  proposes generally — H4 specifically isolates the role/semantic-tag
  feature family for ablation).
- **Implementation concept:** ablate H1's feature set with/without the
  role/semantic-tag features specifically, to isolate their marginal
  contribution once wrapped in a learned model instead of a hand-written
  rule.
- **Primary metric:** TypeMatch on the percent/float/currency-typed slot
  subset specifically (per-type breakdown, e.g. as attempted pre-fix in the
  now-outdated `error_taxonomy.md` — must be recomputed post-fix).
- **Secondary metrics:** overall InstantiationReady.
- **Falsification criterion:** removing role/semantic-tag features from
  H1's feature set does not change per-type TypeMatch — would mean these
  features are not carrying the signal claimed here.
- **Expected risk:** feature ablation requires H1 to already be working
  (i.e., this hypothesis is a follow-up to H1, not independent of it).
- **Novelty status:** **ESTABLISHED_ADAPTATION**.

## H5: Joint top-k schema + grounding reranking may improve strict readiness more than top-1 retrieval alone

**Status (2026-08-13): STRICT-METRIC SECONDARY RESULT.**
`docs/SELECTIVE_GROUNDING_RERANK_STAGE_B_2026-08-13.md` implemented the
frozen Stage-A candidate as `tfidf_selective_grounding_rerank` and reproduced
265/331 InstantiationReady exactly. However, semantic audit showed only 2/8
new ready queries are true schema rescues; 6/8 are wrong-schema readiness
gains. H5 is therefore supported as a metric-improvement mechanism, but not as
a main-method semantic improvement under the current InstantiationReady
definition. The strict metric follow-up
(`docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md`) confirms this:
the ordinary +8 readiness gain collapses to +2 strict-ready gains
(`247/331 -> 249/331`, McNemar `p=0.5`).

**Prior Stage-A status (2026-08-13): SUPPORTED / READY FOR MINIMAL STAGE-B.**
`docs/TOPK_SCHEMA_RERANK_STAGE_A_2026-08-13.md` recomputed this hypothesis
against the fresh 257/331 typed-greedy baseline. The true gold+ready oracle
ceiling was 8 rescued queries at k=3, 9 at k=5, and 13 at k=10. A deterministic
selective cascade crossed the +2 pp gate diagnostically: margin `<=0.05`,
top-5 grounding, and
`0.50 * normalized_tfidf + 0.25 * coverage + 0.25 * type_match`.

**Prior status (2026-08-12, Phase 4 note): NOT TESTED.** Out of scope for
that phase. H5's upper-bound estimate (≤8/331, 2.4%) was computed against
typed greedy's committed 0.5287; that number is stale (fresh rerun: 0.7764,
see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`), so the upper-bound
estimate needed recomputation against the fresh baseline before this
hypothesis could be prioritized. The Phase 3 note reprioritizing this relative
to `max_weight_matching` was retracted along with that claim.

- **Motivation:** `docs/CURRENT_BOTTLENECK_ANALYSIS.md` rank 3 (schema
  retrieval miss, 30/331 = 9.1%) is a comparatively small but nonzero
  contributor; `docs/NEGATIVE_RESULTS.md` NR4 (acceptance/hierarchical
  reranking) already reranks top-k retrieval by an *acceptance score*, with
  a statistically indistinguishable result from top-1 — but that reranking
  does not use grounding-quality signal, only schema-acceptance signal.
- **Current evidence:** Stage B replicates the ordinary InstantiationReady
  gain but strict readiness shows only a small true schema-gated gain. NR4
  remains relevant as a negative result for schema-only acceptance reranking,
  but it does not invalidate retrieval-grounding consistency selection as a
  secondary diagnostic.
- **Implementation concept:** for each of the top-k retrieved schemas,
  actually run grounding and pick the schema whose grounding result scores
  highest (not just whose retrieval/acceptance score is highest) —
  distinct from NR4's schema-only reranking.
- **Primary metric for any future claim:** StrictInstantiationReady, `orig`.
- **Secondary metrics:** ordinary InstantiationReady, Schema R@1, false-ready
  count, and wall-clock cost.
- **Falsification criterion:** as a main-method candidate, fails because most
  readiness gains are incorrect-schema gains. As a metric diagnostic, it is
  useful and should motivate schema-correctness-gated readiness.
- **Expected risk:** confirmed. Readiness-oriented reranking can select wrong
  schemas; schema transitions must be reported alongside InstantiationReady.
- **Novelty status:** **MODERATE_NOVELTY** (joint retrieval-grounding
  reranking is known in retrieve-and-generate architectures broadly;
  specific application here, and distinction from the already-tried
  schema-only acceptance reranking, is the adaptation).

---

## Explicitly deprioritized (not a hypothesis to test next)

- **Repeating NR10's exact setup** (text-only pairwise ranker, 500 steps) —
  already answered, see `docs/NEGATIVE_RESULTS.md` NR10.
- **Any of NR1-NR9's deterministic-only repair rules** as standalone fixes —
  already shown not to beat typed greedy; only worth revisiting as *feature
  sources* for a learned model (per H1/H4), not as deterministic rules.
