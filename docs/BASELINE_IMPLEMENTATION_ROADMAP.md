# External Baseline Implementation Roadmap

**Status:** Phase 3 (2026-08-12), research/planning only. None of ORLM,
OptMATH, DeepOR, or OR-R1 are implemented in this repository. This document
records what implementing each would require and in what order, so a
future phase can act without repeating this research. See `PROJECT_STATUS.md`
§9 for the current status summary and `baselines/pamop/` for the one
baseline that IS implemented (in progress).

---

## 1. PaMOP (already in progress -- included here for ranking context only)

- **Citation:** PaMOP, IJCAI 2025 (independent reproduction in this repo;
  no official code was found -- see `docs/PAMOP_REPRODUCTION_PLAN.md`).
- **Official code available:** NO.
- **Model weights available:** N/A (uses an LLM API, not open weights;
  this reproduction uses Azure OpenAI `gpt-4.1-mini-2025-04-14`, not the
  original paper's unspecified "GPT-4").
- **Task overlap with ours:** High -- full NL-to-solver-code generation via
  LLM + partitioning + correction loop, on the same class of LP/MILP word
  problems.
- **NLP4LP support:** Direct -- this reproduction already runs against
  NLP4LP-derived problems (`results/pamop/`).
- **Environment requirements:** Azure OpenAI access (already configured);
  `amplpy` + Gurobi in a dedicated virtualenv (already configured, see
  `PROJECT_STATUS.md` §11).
- **GPU requirement:** No (API-based).
- **API requirement:** Yes (Azure OpenAI) -- this is the one baseline in
  this roadmap that inherently requires an external generative-LLM API,
  which is why it is tracked separately from the main (no-external-LLM)
  grounding pipeline.
- **Licensing/access issues:** None known beyond standard Azure OpenAI
  terms.
- **Expected implementation difficulty:** Already implemented; remaining
  work is fidelity diagnosis and scale-up, not new implementation.
- **Fairness concerns:** Model/prompt fidelity to the original paper is
  unresolved (`gpt-4.1-mini-2025-04-14` substituted for an unspecified
  "GPT-4"; exact original prompts unavailable) -- see
  `docs/PAMOP_PILOT_FAILURE_FORENSICS.md` "Model-Fidelity Risk: HIGH".
- **Comparable metrics:** Execution success rate, solver feasibility,
  semantic (objective-value) correctness, correction-loop iteration count,
  token cost.
- **Priority:** Already in progress -- see PROJECT_STATUS.md §10 for
  current decision gate.
- **First implementation milestone:** N/A (past this stage) -- current
  milestone is the fidelity diagnostic recommended in
  `docs/PAMOP_PILOT_FAILURE_FORENSICS.md`.

## 2. ORLM

- **Citation:** Huang et al., "ORLM: A Customizable Framework in Training
  Large Language Models for Automated Optimization Modeling," arXiv:2405.17743
  (2024); accepted at *Operations Research*.
- **Official code available:** **YES** -- [github.com/Cardinal-Operations/ORLM](https://github.com/Cardinal-Operations/ORLM), Apache-2.0.
- **Model weights available:** **YES** -- multiple fine-tuned checkpoints
  on HuggingFace (~7-8B parameter models: Mistral-7B, DeepSeek-Math-7B,
  LLaMA-3-8B variants fine-tuned via their "OR-Instruct" synthetic-data
  pipeline).
- **Task overlap with ours:** High -- same task family (NL optimization
  problem -> formal model), but ORLM targets full model+code generation,
  not schema retrieval + scalar grounding; comparison would be at the
  final-instantiation level only.
- **NLP4LP support:** **NOT direct.** ORLM was evaluated on NL4OPT, MAMO,
  and their own IndustryOR benchmark -- NLP4LP is not one of their reported
  benchmarks. Running ORLM against NLP4LP would require adapting our own
  evaluation harness to score ORLM's generated models/code, not a
  plug-and-play comparison.
- **Environment requirements:** `transformers`/`torch`, HuggingFace model
  download (~15-30GB depending on checkpoint).
- **GPU requirement:** **Yes** -- 7-8B parameter local inference is not
  practical CPU-only at reasonable latency; this is a meaningful
  environment-cost difference from the current CPU-only pipeline.
- **API requirement:** No (self-hosted weights).
- **Licensing/access issues:** Apache-2.0 code; base-model licenses
  (Mistral/DeepSeek-Math/LLaMA-3) vary and would need individual review
  before redistribution of any fine-tuned checkpoint, though local
  research use is standard.
- **Expected implementation difficulty:** Medium -- code and weights are
  public and documented, but adapting NLP4LP problems into ORLM's expected
  input format and adapting its code-generation output into our
  Coverage/TypeMatch/InstantiationReady metric family (or an ORLM-native
  metric) is nontrivial glue work.
- **Fairness concerns:** ORLM is a full generative pipeline (LLM writes
  the entire model + solver code), fundamentally different in
  computational cost and scope from this repo's retrieval + scalar-
  grounding pipeline -- any comparison must clearly caveat this is
  "full auto-formulation" vs. "grounding into a fixed catalog," not an
  apples-to-apples method comparison.
- **Comparable metrics:** Execution/solve success rate, objective
  accuracy vs. gold, wall-clock/token cost (directly comparable to
  PaMOP's existing metric family).
- **Priority:** **1st** among the four non-PaMOP baselines.
- **First implementation milestone:** Stand up ORLM inference (a single
  published checkpoint, e.g. `ORLM-LLaMA-3-8B`) on a small (10-20 instance)
  NLP4LP pilot subset, GPU available, and manually verify at least one
  end-to-end generated model executes -- matching the same pilot-before-
  scale discipline already used for PaMOP.

## 3. OptMATH

- **Citation:** Lu et al., "OptMATH: A Scalable Bidirectional Data
  Synthesis Framework for Optimization Modeling," ICML 2025.
- **Official code available:** **YES** -- [github.com/optsuite/OptMATH](https://github.com/optsuite/OptMATH), Apache-2.0.
- **Model weights available:** **YES** -- released on HuggingFace
  (`Aurora-Gem/models` per the repository).
- **Task overlap with ours:** High -- same full auto-formulation task
  family as ORLM.
- **NLP4LP support:** **NOT direct.** Evaluated on their own
  OptMATH-Bench (200K synthetic training triplets, GPT-synthesized,
  longer/more complex descriptions than NLP4LP) plus comparisons against
  MAMO EasyLP; NLP4LP not among their reported benchmarks.
- **Environment requirements:** Same class as ORLM (`transformers`/`torch`,
  HuggingFace download).
- **GPU requirement:** **Yes**, same caveat as ORLM.
- **API requirement:** No (self-hosted weights).
- **Licensing/access issues:** Apache-2.0; base-model license review
  needed as with ORLM.
- **Expected implementation difficulty:** Medium, same class of glue work
  as ORLM (NLP4LP -> OptMATH input adaptation, output -> our metrics).
- **Fairness concerns:** Same as ORLM -- full generative pipeline, not a
  like-for-like method comparison with grounding-only approaches.
- **Comparable metrics:** Same as ORLM; OptMATH additionally reports a
  solver-verified equivalence-accuracy metric (99.6% on their own training
  data) that could inform a stricter semantic-correctness check if adapted.
- **Priority:** **2nd** -- essentially tied with ORLM in maturity and
  access; ranked second only because ORLM's benchmark set (NL4OPT, MAMO)
  is closer in spirit to NLP4LP's word-problem style than OptMATH-Bench's
  longer, more complex synthetic descriptions, making ORLM's results
  slightly more directly interpretable against our benchmark's difficulty
  level.
- **First implementation milestone:** Same pilot-subset approach as ORLM,
  after ORLM's pilot is complete (to reuse the NLP4LP-adaptation glue code
  built for ORLM rather than duplicating it).

## 4. DeepOR

- **Citation:** "DeepOR: A Deep Reasoning Foundation Model for
  Optimization Modeling," AAAI 2026 (per the AAAI proceedings listing
  found; very recent, published at/after this repository's knowledge
  cutoff window).
- **Official code available:** **UNCONFIRMED** -- no public GitHub
  repository was located during this pass's research; AAAI 2026
  publication suggests code may not yet be released or may be
  released concurrently with/after the conference.
- **Model weights available:** **UNCONFIRMED**, same caveat.
- **Task overlap with ours:** High in spirit (optimization modeling from
  NL) but architecturally very different -- DeepOR is a reasoning-LLM
  (long chain-of-thought, RL-trained with solver-feedback reward shaping),
  not a retrieval+grounding pipeline.
- **NLP4LP support:** Unknown -- benchmark set not confirmed in this pass.
- **Environment requirements:** Unknown pending code release; reasoning-
  RL-trained LLMs of this class typically require substantial GPU
  resources for both training and often for inference (long CoT generation).
- **GPU requirement:** Likely yes, probably more demanding than ORLM/OptMATH
  given the long-CoT reasoning generation pattern.
- **API requirement:** Unknown (depends on whether a hosted or open-weight
  release accompanies the paper).
- **Licensing/access issues:** Unknown pending release.
- **Expected implementation difficulty:** **Currently not implementable**
  -- blocked on code/weight availability, not on our own effort.
- **Fairness concerns:** N/A until implementable.
- **Comparable metrics:** Presumably the same execution/feasibility/
  objective-accuracy family as ORLM/OptMATH/PaMOP, pending confirmation.
- **Priority:** **3rd** (behind ORLM/OptMATH on availability grounds
  alone, ahead of OR-R1 only because DeepOR's AAAI 2026 venue suggests a
  slightly more mature/reviewed artifact than OR-R1's November 2025 arXiv
  preprint, though neither is currently actionable).
- **First implementation milestone:** **Monitor for code release** (check
  the paper's eventual camera-ready / AAAI proceedings page for a code
  link); no implementation action possible until then.

## 5. OR-R1

- **Citation:** Zhu, Ma, Wang, Bi, et al., "OR-R1: Automating Modeling and
  Solving of Operations Research Optimization Problem via Test-Time
  Reinforcement Learning," arXiv:2511.09092 (2025).
- **Official code available:** **UNCONFIRMED** -- not located during this
  pass's research; very recent (November 2025) preprint.
- **Model weights available:** **UNCONFIRMED**, same caveat.
- **Task overlap with ours:** High in spirit, same reasoning-LLM /
  test-time-RL architecture family as DeepOR, not a retrieval+grounding
  pipeline.
- **NLP4LP support:** Unknown.
- **Environment requirements:** Unknown pending release; test-time RL
  methods often require nontrivial inference-time compute (multiple
  reasoning rollouts per problem), separate from training cost.
- **GPU requirement:** Likely yes.
- **API requirement:** Unknown.
- **Licensing/access issues:** Unknown pending release.
- **Expected implementation difficulty:** **Currently not implementable**
  -- blocked on code/weight availability.
- **Fairness concerns:** N/A until implementable.
- **Comparable metrics:** Presumably the same family as above, pending
  confirmation.
- **Priority:** **4th (last)** -- most recent preprint of the five, least
  mature evidence of a stable public artifact as of this research pass.
- **First implementation milestone:** **Monitor for code release**, same
  as DeepOR; re-check both together on a periodic basis rather than as a
  dedicated task, since neither currently supports any concrete next step.

---

## Recommended order and rationale

**PaMOP (in progress) -> ORLM -> OptMATH -> DeepOR -> OR-R1**, matching the
task's originally-assumed order. This was re-verified against actual
evidence in this pass, not merely assumed by publication recency:

1. **Direct comparability**: ORLM and OptMATH are both full
   auto-formulation systems like PaMOP, so all three baselines share a
   comparable metric family (execution/feasibility/objective-accuracy),
   letting future work build one shared harness rather than three
   incompatible ones.
2. **Availability**: ORLM and OptMATH have confirmed public code AND
   weights today; DeepOR and OR-R1 do not (as of this research pass) --
   this is the dominant factor separating the first two from the last two,
   not a scientific-value judgment.
3. **Implementation cost**: ORLM and OptMATH require the same class of
   new infrastructure (GPU-hosted 7-8B model inference), which is a real,
   one-time cost this repository does not currently pay (the whole
   grounding pipeline is deliberately CPU-only, no-external-LLM, per
   `docs/CURRENT_BOTTLENECK_ANALYSIS.md`'s documented strengths) --
   building that infrastructure once and reusing it for both ORLM and
   OptMATH is more efficient than any other ordering.
4. **Reviewer relevance**: ORLM (Operations Research journal) and OptMATH
   (ICML 2025) are both peer-reviewed/accepted at their respective venues
   already; DeepOR (AAAI 2026) and OR-R1 (November 2025 arXiv, venue
   unconfirmed) are newer and less established in the literature as of
   this pass.
5. **Recency is deliberately NOT the ranking criterion** -- DeepOR and
   OR-R1 are the most recent methods but rank last, precisely because
   recency without confirmed code/weight availability is not actionable;
   re-ranking should happen automatically once either publishes usable
   artifacts, independent of any further recency judgment.

## What must NOT happen next

Per phase scope: do not begin implementing ORLM, OptMATH, DeepOR, or OR-R1
in this phase. This document is planning-only. The concrete next action
for external baselines is the ORLM pilot-subset milestone above, to be
picked up as a dedicated future task once GPU resources are confirmed
available for this workstation or an alternative compute environment.
