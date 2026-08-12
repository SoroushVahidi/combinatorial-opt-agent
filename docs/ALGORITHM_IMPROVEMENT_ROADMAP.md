# Algorithm Improvement Roadmap

Companion to `docs/RESEARCH_HYPOTHESES.md` (testable hypotheses),
`docs/NEGATIVE_RESULTS.md` (what already failed — **read NR10 first**), and
`docs/CURRENT_BOTTLENECK_ANALYSIS.md` (where errors actually concentrate).

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

### P0 — Establish a working learned local scorer, cheaply

- **Why:** `docs/NEGATIVE_RESULTS.md` NR10 is the only prior attempt, and
  it was both undertrained and text-only. Architecture #1 above directly
  addresses both weaknesses at minimal cost.
- **Prerequisite:** none blocking — training data, split, and feature
  infrastructure all already exist.
- **Expected output:** a trained classifier + its H1 falsification-criterion
  result (beats or does not beat rule baseline on pairwise_accuracy and
  type_match_after_decoding, same held-out test split as NR10 for direct
  comparability).
- **Success criterion:** beats the rule baseline (pairwise_accuracy > 0.247,
  type_match_after_decoding > 0.125) on the same test split NR10 used.
- **Stop criterion:** if architecture #1 does not beat the rule baseline
  after reasonable hyperparameter search, do not escalate to architecture
  #2/#3 without first re-examining whether the feature set itself (not the
  model class) is the limiting factor.

### P1 — Learned score + existing global assignment (H2)

- **Why:** isolates whether the bottleneck is local scoring or global
  assignment, per H2's staged design.
- **Prerequisite:** P0 succeeds (a working local scorer exists).
- **Expected output:** InstantiationReady comparison across {typed-greedy,
  max-weight-matching, search-structured, hierarchical-structured} × {rule
  scores, learned scores from P0}.
- **Success criterion:** learned-local + best-global beats both learned-local
  + typed-greedy and hand-engineered-local + best-global, per H2's
  falsification criterion.
- **Stop criterion:** if no combination beats P0 alone, treat local scoring
  and global assignment as non-complementary for now and do not pursue
  further global-assignment engineering.

### P2 — Add richer semantic-role/unit/domain features, only if P0 error analysis motivates them (H4)

- **Why:** avoid adding complexity speculatively; only justified by a
  concrete post-P0 error analysis showing a specific feature-attributable
  failure pattern.
- **Prerequisite:** P0 complete, with per-slot-type error breakdown computed
  (the finer breakdown flagged as not yet computed in
  `docs/CURRENT_BOTTLENECK_ANALYSIS.md`).
- **Expected output:** ablation of role/semantic-tag features in/out of
  the P0 classifier.
- **Success criterion:** measurable per-type TypeMatch improvement from the
  ablated features, per H4.
- **Stop criterion:** no measurable per-type improvement — features are not
  carrying the claimed signal.

### P3 — Top-k retrieval + grounding joint reranking (H5)

- **Why:** rank-3 bottleneck (schema miss, 9.1%) is real but small; only
  worth the k-fold inference cost if P0-P2 have plateaued on the larger
  rank-1/2 bottleneck.
- **Prerequisite:** P0 (need a working grounding scorer worth reranking
  against) — this is why it's P3, not P0, despite retrieval being
  "upstream" architecturally.
- **Expected output:** InstantiationReady with joint top-k reranking vs.
  top-1.
- **Success criterion:** gain approaching the ≤8/331 (2.4%) StrictInstantiationReady
  upper bound (H5's falsification criterion).
- **Stop criterion:** gain far below that bound relative to added k-fold
  inference cost.

### P4 — Confidence calibration / abstention

- **Why:** `docs/NEGATIVE_RESULTS.md` NR7 shows the current abstention
  threshold (ambiguity-aware grounding) is badly miscalibrated (Coverage
  collapses to 0.2207). A properly calibrated version, built on a *working*
  scorer from P0, is architecturally motivated even though the current
  version failed.
- **Prerequisite:** P0 (calibrating a bad scorer's confidence doesn't help).
- **Expected output:** a precision/coverage trade-off curve for abstention
  on top of the P0 scorer.
- **Success criterion:** a usable operating point exists where abstaining
  on low-confidence predictions improves precision without collapsing
  coverage the way NR7's abstain variant did.
- **Stop criterion:** no such operating point exists at any reasonable
  threshold.

### P5 — Only then, design a genuinely new combinatorial grounding algorithm, if established methods plateau

- **Why:** per the task's own framing — established techniques should be
  exhausted first. `docs/NEGATIVE_RESULTS.md` already shows 9 deterministic
  ideas and 1 learned idea (NR1-NR10) have not beaten typed greedy; P0-P4
  represent the next established techniques to try before concluding a
  fundamentally new algorithm is warranted.
- **Prerequisite:** P0-P4 all attempted with documented stop/success
  outcomes.
- **Expected output:** a written case for *why* a new algorithm class is
  needed, citing which specific established technique failed and why,
  before any new implementation begins.
- **Success criterion:** n/a — this stage is a decision gate, not an
  experiment.
- **Stop criterion:** n/a.
