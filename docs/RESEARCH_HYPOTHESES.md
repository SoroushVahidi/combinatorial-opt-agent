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

- **Motivation:** `docs/CURRENT_BOTTLENECK_ANALYSIS.md` rank 3 (schema
  retrieval miss, 30/331 = 9.1%) is a comparatively small but nonzero
  contributor; `docs/NEGATIVE_RESULTS.md` NR4 (acceptance/hierarchical
  reranking) already reranks top-k retrieval by an *acceptance score*, with
  a statistically indistinguishable result from top-1 — but that reranking
  does not use grounding-quality signal, only schema-acceptance signal.
- **Current evidence:** NR4 (not significant either way); StrictInstantiationReady
  (`results/CANONICAL_RESULTS.md` §D) shows the schema-match gate removes
  8/331 TF-IDF queries from the ready count — an upper bound on what
  fixing retrieval-grounding joint decisions could plausibly recover.
- **Implementation concept:** for each of the top-k retrieved schemas,
  actually run grounding and pick the schema whose grounding result scores
  highest (not just whose retrieval/acceptance score is highest) —
  distinct from NR4's schema-only reranking.
- **Primary metric:** InstantiationReady, `orig`.
- **Secondary metrics:** wall-clock cost (this is strictly more expensive
  than top-1, running grounding k times per query).
- **Falsification criterion:** gain smaller than the ≤8/331 (2.4%) upper
  bound implied by the StrictInstantiationReady schema-match-gate analysis
  would mean this is not worth its k-fold inference cost.
- **Expected risk:** k-fold inference cost; already-small upper bound on
  achievable gain given how strong top-1 retrieval already is.
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
