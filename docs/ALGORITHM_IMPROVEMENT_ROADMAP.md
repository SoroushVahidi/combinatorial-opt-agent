# Algorithm Improvement Roadmap

Companion to `docs/RESEARCH_HYPOTHESES.md` (testable hypotheses),
`docs/NEGATIVE_RESULTS.md` (what already failed — **read NR10 first**), and
`docs/CURRENT_BOTTLENECK_ANALYSIS.md` (where errors actually concentrate).

**Current resubmission decision (2026-08-13): `FROZEN_FOR_RESUBMISSION`.**
The strict-readiness metric audit and strict-failure quick-fix diagnostic led
to exactly one production patch: multiplicative ratio-word extraction for
`twice`/`double`/`two times` and `triple`/`three times`. Production validation
reproduced the diagnostic projection exactly: 247/331 -> 255/331 strict
readiness, 257/331 -> 265/331 ordinary readiness, and 0 strict/ordinary
readiness losses (`docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`).
Do not start another broad algorithm family for this resubmission.

---

## Literature review: established techniques closest to our numeric-mention → schema-slot problem

Targeted primary-source research (not a generic LLM survey), focused on the
specific gap: local contextual scoring of a numeric mention against a
candidate schema slot.

| Technique | Closest source | What it does | Similarity to our task | Already implemented here? | Training required? | External LLM API? | Locally runnable? | Relevance to our actual errors |
|---|---|---|---|---|---|---|---|---|
| NL4Opt entity extraction | Ramamonjison et al., "NL4Opt Competition: Formulating Optimization Problems Based on Their Natural Language Descriptions," PMLR v220, 2023 ([proceedings.mlr.press/v220/ramamonjison23a.html](https://proceedings.mlr.press/v220/ramamonjison23a.html)) | NER-style tagging of optimization entities (objective, decision variables, constraint limits) from NL text, feeding a meaning-representation generator | Directly the source task family NLP4LP descends from; same two-stage decomposition (entity/schema recognition → structured generation) we use | Partially — our numeric extraction (`_extract_num_tokens` etc.) is a hand-rolled equivalent, not the NL4Opt-trained model itself | Yes (their models are trained) | No | Yes | High — this is the closest published entity-extraction precedent for our extraction stage specifically |
| Ner4Opt | Kadıoğlu et al., "Ner4Opt: named entity recognition for optimization modelling from natural language," *Constraints* 29, 261-299 (2024); pretrained models on HuggingFace ([github.com/skadio/ner4opt](https://github.com/skadio/ner4opt)) | Fine-tuned NER for optimization-problem entities, published/pretrained, directly comparable in scope to our extraction stage | Very high — same problem family (LP word problems), published, pretrained, evaluable off-the-shelf | **No** — not currently compared against or integrated | No (pretrained weights available) | No | Yes | **High and actionable** — this is an existing, published, locally-runnable model we are not currently using or comparing to at all; a natural near-term evaluation target, independent of the learned-scorer hypotheses |
| Schema-guided dialogue state tracking (cross-encoder slot-value matching) | Multiple DSTC8/SGD-era works; BERT-based cross-attention over (context, slot name, slot value) | Jointly encodes context + slot description + candidate value with cross multi-head attention, outputs a match score | Structurally almost identical to our (mention-context, slot) scoring problem, different domain (dialogue vs. optimization) | No | Yes | No | Yes | High — this is the architecture pattern H1/H2 propose adapting |
| MeasEval (SemEval-2021 Task 8) | Harper & Cox et al., "SemEval-2021 Task 8: MeasEval," ACL Anthology ([aclanthology.org/2021.semeval-1.38](https://aclanthology.org/2021.semeval-1.38.pdf)); QA-based approach: UPB team, arXiv:2104.04549 | Quantity/unit/entity/relation extraction from text, including a multi-turn QA-based extraction variant | Directly relevant to our numeric-mention extraction and (implicitly) mention-to-role linking | No | Yes (both the base task and the QA variant require training/fine-tuning) | No (QA variant uses a fine-tuned QA model, not necessarily a generative LLM) | Yes | Medium — most relevant to extraction (stage 3-5 in `docs/METHOD_INVENTORY.md`), less directly to slot assignment |
| QA-based slot extraction | UPB SemEval-2021 Task 8 system (above) | Frames measurement-attribute extraction as multi-turn question answering over the source text | Could reframe "which mention fills slot X?" as an extractive QA problem per slot | No | Yes | No | Yes | Medium — an alternative to pairwise ranking, not yet compared; worth a small pilot if H1 underperforms |
| Local cross-encoder / bi-encoder pair scoring (sentence-transformers) | sentence-transformers documentation and Augmented SBERT (Thakur et al., arXiv:2010.08240) | Cross-encoders jointly encode a pair and output one score (higher quality, slower); bi-encoders embed independently then compare (faster, lower quality); standard retrieve-then-rerank combines both | Directly the mechanism for H1's local scorer; informs the architecture ranking below | Partially — our TF-IDF/BM25/LSA retrieval is a classical bi-encoder-like stage; no cross-encoder exists for mention-slot scoring | Cross-encoders: yes; bi-encoders: pretrained usable directly | No | Yes | High — directly informs the top-3 ranking below |
| Constrained structured prediction (CRF / structured SVM style) | Standard NLP structured-prediction literature | Enforces global structural constraints (e.g. no duplicate slot fills) jointly with local scoring, typically via dynamic programming or ILP | Similar in spirit to our existing bipartite matching / beam search (already implemented, just not learned) | Partially — `_run_max_weight_matching_grounding`, `search_structured_grounding.py` are the deterministic-score version of this pattern | Depends on formulation | No | Yes | Medium — informs H2 (combine learned local score with existing exact/beam global assignment) rather than a new technique to add |

**Deliberately not surveyed as a "generic LLM approach":** end-to-end
generative LLM prompting for grounding was excluded from this research
pass per the task's own instruction to research established *local,
non-generative-API* techniques specifically — PaMOP (`baselines/pamop/`)
already covers the generative-LLM-based auto-formulation family as a
separate baseline-comparison effort (`PROJECT_STATUS.md` §9-10), and mixing
the two research threads would blur the "preserve no-external-LLM-at-inference"
property this pipeline currently has (`docs/CURRENT_BOTTLENECK_ANALYSIS.md`
strengths section).

---

## Top-3 candidate local-scorer architectures (ranked)

All three are scoped to avoid repeating `docs/NEGATIVE_RESULTS.md` NR10's
failure mode (text-only, undertrained, no structured features).

### 1st: Small classifier over engineered features + frozen sentence embedding (RECOMMENDED FIRST)

- **Architecture:** gradient-boosted trees or a small MLP over (a) the
  already-computed hand-engineered features from `_score_mention_slot_opt`
  / `relation_aware_linking.py`, concatenated with (b) a frozen
  sentence-embedding similarity (e.g. off-the-shelf `sentence-transformers`
  bi-encoder, not fine-tuned) between mention context and slot description.
- **Expected benefit:** highest data efficiency of the three — does not
  require fine-tuning a transformer on ~10K pairs, which NR10 showed is
  likely insufficient.
- **Training requirement:** yes, but cheap (classical ML, CPU-only,
  minutes not hours).
- **Data requirement:** the existing 9,729 training pairs are plausibly
  *sufficient* for a low-parameter-count classifier, unlike for fine-tuning
  a full transformer.
- **Local inference cost:** very low (CPU, milliseconds per pair).
- **Determinism/reproducibility:** high (fixed seed, no sampling at
  inference for tree models; near-deterministic for a small fixed MLP).
- **Ease of integration:** high — same call site as the existing
  `_score_mention_slot*` functions, same output shape (a score).
- **Scientific novelty:** **ESTABLISHED_ADAPTATION** (feature-based
  learning-to-rank is a mature technique; the novelty is purely in the
  application).

### 2nd: Cross-encoder over (context, slot description, slot type) text, fine-tuned WITH structured features injected

- **Architecture:** a compact pretrained encoder (e.g. `distilroberta-base`,
  same as NR10, or a smaller model) taking the concatenated
  (mention-context text, slot name + description + type) as input, PLUS the
  engineered features injected as additional token-type or auxiliary input
  (not text-only as NR10 was) — i.e., finally running the never-executed
  `nlp4lp_pairwise_text_plus_features` configuration from `docs/EXPERIMENTS.md` §5.4.
- **Expected benefit:** potentially higher ceiling than architecture #1 if
  there is genuine textual nuance the engineered features miss, but only
  realizable if training is adequate (more steps/epochs than NR10's 500).
- **Training requirement:** yes, GPU preferred (per `docs/KNOWN_ISSUES.md`,
  CPU-only training was the blocker for the never-run Stage 3 variants).
- **Data requirement:** same 9,729 pairs; higher risk of underfitting or
  overfitting than architecture #1 given more parameters.
- **Local inference cost:** low-moderate (CPU-feasible for a distilled
  model, but slower than architecture #1).
- **Determinism/reproducibility:** moderate (seeded, but transformer
  fine-tuning has more run-to-run variance than tree/small-MLP methods).
- **Ease of integration:** moderate — needs the feature-injection plumbing
  that was planned but never built for Stage 3.
- **Scientific novelty:** **ESTABLISHED_ADAPTATION** (schema-guided DST
  precedent), but this exact repo has already partially attempted it
  (infra exists, run was blocked) — completing it is lower-risk than
  architecture #3.

### 3rd: Bi-encoder (sentence-transformer) fine-tuned for mention-slot similarity, used as a fast pre-filter before existing deterministic repair

- **Architecture:** fine-tune a bi-encoder to embed mention-context and
  slot-description into a shared space, use cosine similarity as one
  additional signal feeding into the *existing* deterministic pipeline
  (typed greedy or repair stages), rather than replacing local scoring
  outright.
- **Expected benefit:** cheapest at inference (embed once, compare via dot
  product), most naturally composable with the existing global-assignment
  stages (H2) since it's just another score to combine.
- **Training requirement:** yes (contrastive/triplet loss), moderate data
  needs — bi-encoders are typically less sample-efficient than
  classifiers for narrow domains, per Augmented-SBERT's own finding that
  bi-encoders need data augmentation to match cross-encoder quality on
  small datasets.
- **Data requirement:** the existing pairs likely need augmentation
  (Augmented SBERT-style, e.g. using the existing rule scorer to
  soft-label additional silver pairs) to reach competitive quality — an
  extra engineering step not needed for architectures #1/#2.
- **Local inference cost:** lowest of the three at inference time.
- **Determinism/reproducibility:** moderate.
- **Ease of integration:** moderate — cleanest as an *additional feature*
  (like architecture #1) rather than a scorer replacement.
- **Scientific novelty:** **MODERATE_NOVELTY** (bi-encoder + rule-based
  silver-labeling combination is not extensively precedented for this
  specific task).

**Recommendation: start with #1.** It is the cheapest to build and test
against H1's falsification criterion, reuses 100% of existing feature
infrastructure, and directly avoids NR10's specific failure mode (data
starvation for a full transformer fine-tune). Escalate to #2 only if #1's
ablation (H4) shows the *text itself* (not just engineered features) is
carrying signal the classifier can't capture from features alone.

---

## Training supervision feasibility

**Answer: YES — clean, leak-free supervision already exists and should be
reused, not rebuilt.**

- NLP4LP's local gold cache used throughout this repo covers 331 `test`-
  split records (`results/eswa_revision/00_env/nlp4lp_gold_cache.json`).
  Separately, `docs/EXPERIMENTS.md` §5.3 documents an existing 330-record
  →230/50/50 instance-level train/dev/test split (9,729/2,230/2,339
  pairwise pairs) built specifically for learned-grounding experiments.
- **(mention, slot, match/non-match) pair construction:** already
  implemented — gold parameter values per problem
  (`nlp4lp_gold_cache.json`'s `gold_by_id[*].parameters`) give the true
  numeric value for each slot; a candidate mention is a positive pair for a
  slot if its extracted value matches the gold value (within the existing
  pipeline's tolerance), negative otherwise. This is exactly what
  `artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl` already
  contains.
- **Leakage risk:** addressed — the split is at the **instance level**
  (`verify_split_integrity` confirms distinct SHA-256 hashes per split,
  `docs/learning_runs/real_data_only_learning_check.md` §2), not just
  pair-level, so no instance's pairs leak across splits. **Not yet verified
  in this pass:** whether any two *different* instances in different splits
  share the *same underlying schema* (schema-level leakage, distinct from
  instance-level leakage) — check `predicted_doc_id`/`gold_doc_id`
  consistency across splits before treating this as fully leak-proof for a
  schema-level generalization claim, though for a mention-slot scorer
  (which doesn't need to generalize across schemas, only across mention
  phrasings) this risk is lower than it would be for a schema-retrieval
  model.
- **Sufficiency:** 9,729 training pairs is small for full transformer
  fine-tuning (NR10's failure is consistent with this) but plausibly
  adequate for the top-ranked architecture #1 (feature-based classifier)
  above.
- **Do NOT train anything as part of this phase** — this section answers
  feasibility only, per the task's explicit Phase 2 scope limit.

---

## Prioritized roadmap

**Updated 2026-08-12 (Phase 4) — supersedes the Phase 3 version of this
section.** Phase 3 (same day, earlier) believed it had found two major
pieces of new evidence: (1) P0 implemented and evaluated — DONE, decision
gate C (does not improve), unaffected by this update, see
`docs/LEARNED_GROUNDING_P0.md`; (2) three previously-unevaluated methods
(`max_weight_matching`, `search_structured_grounding`,
`hierarchical_structured_grounding`) were the "strongest methods in this
repository's history." **Finding (2) has been retracted** — it compared
fresh numbers against a typed-greedy baseline (0.5287) that turned out to
be stale relative to current code by 49 commits. A fresh, same-code typed
greedy rerun scores 0.7764, significantly beating all three. Full
correction: `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. **The entire
former P1-P6 restructuring built around the max-weight-matching finding is
withdrawn.** The roadmap below returns to (and sharpens) the original
Phase 2 prioritization, now informed by the mechanism/error analysis that
was done for `max_weight_matching` before the retraction (it remains
useful evidence about the local score's own weaknesses, even though the
method built on top of it does not win).

### P0 — Establish a working learned local scorer, cheaply — **DONE (2026-08-12)**

- **Result:** implemented (`tools/learned_local_scorer.py`, feature-
  augmented classifier per architecture #1 above). Decision gate:
  **C — does not improve.** Best P0 configuration (greedy decode) reached
  InstantiationReady 0.80 on a 50-instance oracle-schema test subset vs.
  canonical M0's 0.86 (not statistically significant, p=0.44) and vs. a
  **pure rule-only decode over the same feature set's 0.84** (P0's learned
  scores underperformed the unlearned rule score on this test set, despite
  a real gain on the internal dev proxy metric). Full record:
  `docs/LEARNED_GROUNDING_P0.md`, ledger entry NR11 in
  `docs/NEGATIVE_RESULTS.md`.
- **Consequence per stop criterion:** do not escalate to roadmap
  architecture #2/#3 (cross-encoder, bi-encoder) without first re-examining
  the feature set / decode strategy.

### P1 (RETRACTED, 2026-08-12 Phase 4) — ~~Understand and extend the max-weight-matching finding~~

- Withdrawn. `max_weight_matching`, `search_structured_grounding`, and
  `hierarchical_structured_grounding` are negative results (lose to fresh
  typed greedy, p<0.05 on `orig`); see NR12 in `docs/NEGATIVE_RESULTS.md`.
  The mechanism/error analysis that was already done for this line of work
  (`results/max_weight_matching_validation/`) is retained as evidence for
  P2 below, since it characterizes the *local score's* own weaknesses
  independent of which decode strategy sits on top of it.

### P1 (was P3) — Verify and, if needed, refresh the remaining Phase-1/2 negative-result numbers — **NEW, highest priority**

- **Why:** the staleness audit regenerated fresh numbers for 12 of the ~16
  methods in `docs/METHOD_INVENTORY.md` Part 2. Three families —
  `global_compat_*` (GCG), `relation_aware_*`, `ambiguity_aware_*` — were
  **not** regenerated (time-bounded in Phase 4) and still show only their
  original, now-confirmed-stale numbers (0.42-0.50 range) compared against
  the also-stale 0.5287 typed-greedy baseline. It is very likely (given the
  pattern in the 12 already-checked methods) that these three families'
  numbers have also drifted upward, but this has not been confirmed.
- **Prerequisite:** none — this is a cheap rerun (~1-2s per setting per
  the timings in `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` §8),
  identical procedure to what was just done for the other 9 methods.
- **Expected output:** fresh `orig` (and ideally `noisy`/`short`) numbers
  for `global_compat_full`, `relation_aware_full` (and the other ablation
  levels), and `ambiguity_aware_full`/`beam`/`abstain`, each compared via
  paired bootstrap against a **freshly rerun** `tfidf_typed_greedy`
  (0.7764), not the stale 0.5287.
- **Success criterion:** every row in `docs/METHOD_INVENTORY.md` Part 2 has
  a fresh, same-code, mutually comparable number, closing the "not yet
  regenerated fresh" gaps this phase left open.
- **Stop criterion:** n/a — low-cost, high-value data-quality task.

### P2 (was P2, re-scoped) — Improve the local pairwise score directly, targeting its known error modes

- **Why:** the retracted `max_weight_matching` work still produced a
  useful byproduct: a full 331-query slot-level error taxonomy for
  `_score_mention_slot_opt`, the local score every richer method (learned
  and non-learned) has depended on and failed to beat typed greedy with.
  Dominant residual categories (`results/max_weight_matching_validation/mechanism_and_error_analysis_summary.json`):
  same-type ambiguity (335 slot-level instances), total/per-unit confusion
  (166), missing mentions (156), objective/constraint confusion (124).
  These are the same categories P0's feature set targeted and failed to
  resolve through *learning* — the open question is whether they are
  resolvable through further *deterministic* feature engineering (in the
  style of the 49 commits that improved `_choose_token`) instead.
- **Prerequisite:** P1 (confirm the full current method-comparison picture
  first, so any local-score improvement is evaluated against accurate
  baselines).
- **Falsification criterion:** if targeted fixes to same-type-ambiguity /
  total-per-unit-confusion detection in `_score_mention_slot_opt` do not
  move `optimization_role_repair`/`max_weight_matching`'s own numbers
  (freshly rerun) closer to typed greedy's, the local score is not the
  gating factor for those methods and this line should stop.
- **Novelty status:** ESTABLISHED_ADAPTATION (same class of fix as the 49
  commits already in this repository's history; not a new technique).

**2026-08-13 Stage-A update:** the specific role/quantity-factorized version
of this direction was tested diagnostically and should not be implemented next
for InstantiationReady. See
`docs/ROLE_QUANTITY_STAGE_A_DIAGNOSTIC_2026-08-13.md`: 28/49 targeted wrong
assignments were role/quantity-separable, but correcting all separable
assignments rescued 0 currently not-ready queries under the coverage/type
InstantiationReady gate. The next method diagnostic should move to selective
top-k schema + grounding reranking rather than another local role/quantity
scorer.

**2026-08-13 strict-metric update:** after schema-gated readiness became the
primary native end-to-end proxy, a quick-fix diagnostic found and production
validation confirmed one acceptable small extraction patch: multiplicative
ratio-word extraction. Broader local expected-type repair has a high oracle
ceiling (24 strict-ready rescues) but is not a quick fix because overloaded
slot names would require global rule redesign and regression control. Do not
use this roadmap section to reopen role/quantity reranking, learned pair
scoring, matching, or search for the current resubmission.

### P3 (was P3/H4) — Add richer semantic-role/unit/domain features to `_choose_token` itself (H4, re-targeted)

- **Status:** FROZEN FOR RESUBMISSION (`docs/RESEARCH_HYPOTHESES.md` H4).
  The validated ratio-word extraction patch supersedes this as near-term work.
  Originally
  framed as input to a learned scorer (P0, negative result); **re-targeted**
  here at `_choose_token` (typed greedy's own simple heuristic), given that
  49 commits of exactly this kind of incremental, deterministic feature
  refinement are what took typed greedy from 0.5287 to 0.7764 in the first
  place. This is now the most evidence-backed lever in the roadmap: the
  technique (targeted deterministic fixes to the simple baseline) has a
  demonstrated track record in this exact codebase, unlike any of the
  richer-architecture alternatives tried so far. However, it is broader than
  the single ratio-word patch and should not be pursued before resubmission.
- **Prerequisite:** P1 (accurate current baselines).

### P4 (was P4/H5) — Top-k retrieval + grounding joint reranking (H5)

- **Status:** STRICT-METRIC SECONDARY RESULT
  (`docs/SELECTIVE_GROUNDING_RERANK_STAGE_B_2026-08-13.md`,
  `docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md`). The frozen
  production method `tfidf_selective_grounding_rerank` reaches 265/331
  ordinary InstantiationReady with 0 ready losses, but only 2/8 new ready
  cases are true schema rescues. Under strict readiness, the method improves
  only 247/331 -> 249/331 (McNemar `p=0.5`). This is useful evidence about
  metric design and a small retrieval diagnostic, not a main-method result.
- **Prerequisite:** P1.
- **Next implementation:** do not extend this reranker unless a future design
  explicitly optimizes strict readiness and controls false-ready artifacts.
- **Risk:** confirmed. Readiness-only improvement can be driven by wrong
  schemas with easier overlapping scalar slots.

### P5 (was P5) — Confidence calibration / abstention — **CONDITIONAL**

- **Why:** `docs/NEGATIVE_RESULTS.md` NR7 shows the current abstention
  threshold (ambiguity-aware grounding) is badly miscalibrated. Frame
  against a **freshly rerun** typed-greedy baseline, not
  `max_weight_matching`'s scores (that method is a negative result, not a
  "known-good" score source).
- **Prerequisite:** P1.

### P6 (was P6) — Only then, design a genuinely new combinatorial grounding algorithm, if established methods plateau

- **Why:** per the task's own framing — established techniques should be
  exhausted first. Nothing in Phase 3-4's evidence lowers this bar: every
  richer scoring, repair, and global-assignment technique tried so far has
  lost to a simple, iteratively-refined greedy baseline. If anything this
  strengthens the case for continuing to invest in P2/P3-style incremental
  deterministic refinement before considering a new algorithm class.
- **Prerequisite:** P1-P5 all attempted with documented stop/success
  outcomes.
- **Expected output:** a written case for *why* a new algorithm class is
  needed, citing which specific established technique failed and why,
  before any new implementation begins.
- **Success criterion:** n/a — this stage is a decision gate, not an
  experiment.
- **Stop criterion:** n/a.
