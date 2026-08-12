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

**Updated 2026-08-12 (Phase 3) with two major pieces of new evidence:**
(1) P0 was implemented and evaluated -- **DONE, decision gate C (does not
improve)**, see `docs/LEARNED_GROUNDING_P0.md`; (2) three previously-
unevaluated methods were run for the first time and turned out to be the
**strongest methods in this repository's history**
(`max_weight_matching`: InstantiationReady 0.7432 vs. typed greedy's
0.5287, p<0.001 -- see `results/unevaluated_methods_evaluation/`). This
second finding was not anticipated by the roadmap as originally written
and materially reprioritizes everything below it: **the highest-value next
step is no longer "build a better learned scorer" but "understand and
extend the exact-global-assignment finding."**

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
  the feature set / decode strategy — and given P1 below now supersedes
  this whole line of investigation in expected value, that re-examination
  should happen only after P1 is complete.

### P1 (NEW, was not in the original roadmap) — Understand and extend the max-weight-matching finding — **NEXT, highest priority**

- **Why:** `max_weight_matching` (exact Hungarian assignment over the
  existing, non-learned `_score_mention_slot_opt` scores) reaches
  InstantiationReady 0.7432 on the full 331-query benchmark — a +0.2145
  absolute, p<0.001 gain over typed greedy, and the best result ever
  recorded in this repository, including beating Oracle-TG (0.5680).
  `search_structured_grounding` and `hierarchical_structured_grounding`
  independently corroborate this (0.7039 each) using a different decode
  mechanism (beam search) over related scores. This is now unambiguously
  the highest-value open thread.
- **Prerequisite:** none — the result already exists
  (`results/unevaluated_methods_evaluation/`).
- **Expected output (three sub-tasks, roughly in order):**
  1. **Mechanism analysis:** why does exact assignment help so much more
     here than it helped GCG/relation-aware/ambiguity-aware (which also
     used global/beam search but performed poorly, NR5-NR7)? Hypothesis to
     test: those methods added *extra* consistency-penalty terms on top of
     already-adequate local scores, actively hurting; `max_weight_matching`
     uses the *unmodified* opt-role score, letting the assignment algorithm
     alone do the work. Verify by inspecting whether GCG's pairwise-penalty
     terms are net-negative in isolation.
  2. **Error analysis:** what does `max_weight_matching` still get wrong,
     on the ~26% of queries where it isn't InstantiationReady? Apply the
     same per-query methodology `docs/LEARNED_GROUNDING_P0.md` "Error
     Analysis" used, at full 331-query scale this time.
  3. **Manuscript-integration decision:** this result is not yet in
     `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
     or the manuscript text — a future phase must explicitly decide
     whether/how to integrate it (new table row? revised headline claim?),
     which is a manuscript-writing decision this phase deliberately did not
     make (`results/unevaluated_methods_evaluation/README.md`).
- **Success criterion:** a documented, evidence-backed explanation for the
  mechanism, plus a decision recorded on manuscript integration.
- **Stop criterion:** n/a — this is now the default next task, not a
  conditional experiment.

### P2 (was P1) — Learned score + existing global assignment (H2) — **CONDITIONAL, re-scoped**

- **Why:** H2's original framing (learned-local + global assignment) is
  now supersedable by testing the *existing, non-learned* score + global
  assignment first (which is exactly `max_weight_matching`, already done —
  see P1 above). The learned-score version of H2 remains open but lower
  priority, since P0's learned score did not clear the bar the non-learned
  score already clears easily.
- **Prerequisite:** P1's mechanism analysis (to know whether a *better*
  learned score, not just any learned score, is the missing ingredient).
- **Stop criterion (updated):** do not pursue a learned-score + global-
  assignment combination until P0's specific learned-score weaknesses
  (identified in P1's mechanism analysis and `docs/LEARNED_GROUNDING_P0.md`
  "Error Analysis") are understood well enough to target them directly,
  rather than repeating H1/P0's architecture with minor variations.

### P3 (was P2) — Add richer semantic-role/unit/domain features (H4) — **CONDITIONAL, deprioritized**

- **Status:** NOT TESTED (`docs/RESEARCH_HYPOTHESES.md` H4). Deprioritized
  further by P1's discovery — richer features were never the bottleneck for
  `max_weight_matching`, which uses the SAME feature set P0 already had
  access to and still wins by a wide margin through decode strategy alone.
- **Prerequisite:** P1 and P2 (only revisit if a future learned scorer,
  informed by P1/P2, shows a feature-attributable gap).

### P4 (was P3) — Top-k retrieval + grounding joint reranking (H5) — **CONDITIONAL, deprioritized further**

- **Status:** NOT TESTED. `max_weight_matching`'s 0.7432 makes the
  remaining schema-retrieval-miss share of the gap even smaller in
  relative terms than when H5 was first framed against typed greedy's
  0.5287 (see H5's updated status in `docs/RESEARCH_HYPOTHESES.md`).
- **Prerequisite:** P1 complete first — no reason to invest in retrieval-
  side reranking before understanding why the grounding-side gain was so
  large.

### P5 (was P4) — Confidence calibration / abstention — **CONDITIONAL**

- **Why:** `docs/NEGATIVE_RESULTS.md` NR7 shows the current abstention
  threshold (ambiguity-aware grounding) is badly miscalibrated. Now more
  naturally framed as calibration on top of `max_weight_matching`'s scores
  (which are known-good) rather than on top of a not-yet-working learned
  scorer.
- **Prerequisite:** P1 (use `max_weight_matching`'s error analysis to
  target calibration where it would actually help).

### P6 (was P5) — Only then, design a genuinely new combinatorial grounding algorithm, if established methods plateau

- **Why:** per the task's own framing — established techniques should be
  exhausted first. This bar is now considerably higher than when the
  roadmap was first written: `max_weight_matching` alone closed most of
  the gap to a perfect score using entirely established techniques (exact
  bipartite matching, no learning). A new algorithm is even less clearly
  warranted now than before P1's discovery.
- **Prerequisite:** P1-P5 all attempted with documented stop/success
  outcomes, AND `max_weight_matching`'s remaining ~26% failure mode
  (from P1's error analysis) is shown to require something beyond
  established assignment/reranking/calibration techniques.
- **Expected output:** a written case for *why* a new algorithm class is
  needed, citing which specific established technique failed and why,
  before any new implementation begins.
- **Success criterion:** n/a — this stage is a decision gate, not an
  experiment.
- **Stop criterion:** n/a.
