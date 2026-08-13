# External Baseline Implementation Roadmap

**Status:** Phase 5 (2026-08-13), PaMOP independent reconstruction
pilot-validated with larger evaluation still pending. ORLM and OptMATH are
implemented for inference preparation (`baselines/orlm/`, `baselines/optmath/`
— no model call, GPU/weights, or solver execution). DeepOR is a paper-level,
mock-tested reconstruction in `baselines/deepor/` (no official code found).
**OR-R1's official code is now verified and integrated**
(`baselines/orr1/`, `docs/ORR1_PROVENANCE.md`): the repository
`SCUTE-ZZ/OR-R1` is cited directly by the arXiv paper as its code release,
but no SFT/GRPO/merged checkpoint has been published anywhere, so this is
`ORR1_CODE_INTEGRATED_CHECKPOINT_BLOCKED`, not an empirical result. All five
baselines are now at some lightweight-implementation stage; none has a
runnable empirical result. This document records what completing each would
require, so a future phase can act without repeating this research. See
`PROJECT_STATUS.md` §9 for the current status summary and `baselines/pamop/`
for the pilot-validated reconstruction.

---

## 1. PaMOP (pilot validated; larger evaluation pending)

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
- **Expected implementation difficulty:** The independent reconstruction is
  implemented and pilot-validated; remaining work is a preregistered,
  non-interfering scale-up and fuller semantic evaluation.
- **Fairness concerns:** Model/prompt fidelity to the original paper is
  unresolved (`gpt-4.1-mini-2025-04-14` substituted for an unspecified
  "GPT-4"; exact original prompts unavailable) -- see
  `docs/PAMOP_PILOT_FAILURE_FORENSICS.md` "Model-Fidelity Risk: HIGH".
- **Comparable metrics:** Execution success rate, solver feasibility,
  semantic (objective-value) correctness, correction-loop iteration count,
  token cost.
- **Priority:** Pilot validated; larger evaluation pending -- see
  `PROJECT_STATUS.md` §10 for the current decision gate.
- **First implementation milestone:** N/A (past this stage) -- current
  milestone is the fidelity diagnostic recommended in
  `docs/PAMOP_PILOT_FAILURE_FORENSICS.md`.

## 2. ORLM — **IMPLEMENTED READY FOR INFERENCE 2026-08-12, see `baselines/orlm/`**

- **Citation:** Huang et al., "ORLM: A Customizable Framework in Training
  Large Language Models for Automated Optimization Modeling," arXiv:2405.17743
  (2024, v5 April 2025); accepted at *Operations Research* (2025).
- **Official code available:** **YES, re-verified 2026-08-12** —
  [github.com/Cardinal-Operations/ORLM](https://github.com/Cardinal-Operations/ORLM),
  public, active (272 stars, last push Sept 2025), Apache-2.0, default
  branch `master`.
- **Model weights available: PARTIALLY, corrected 2026-08-12.** Only
  **`CardinalOperations/ORLM-LLaMA-3-8B`** (8.03B, bf16, `llama3` license)
  is confirmed publicly retrievable on HuggingFace. The Mistral-7B and
  DeepSeek-Math-7B checkpoints named in the paper/README do **not**
  resolve as public HF repos under any checked path — the original
  "multiple fine-tuned checkpoints... public" claim in this document was
  not independently verified when first written; it is confirmed for one
  of three, not all three.
- **Task overlap with ours:** High -- same task family (NL optimization
  problem -> formal model), but ORLM targets full model+code generation,
  not schema retrieval + scalar grounding; comparison would be at the
  final-instantiation level only.
- **NLP4LP support:** **NOT direct.** ORLM was evaluated on NL4OPT, MAMO,
  and their own IndustryOR benchmark -- NLP4LP is not one of their reported
  benchmarks. Running ORLM against NLP4LP would require adapting our own
  evaluation harness to score ORLM's generated models/code, not a
  plug-and-play comparison.
- **Environment requirements:** `transformers`/`torch`, upstream pins
  `vllm==0.3.2` (old — likely needs an isolated env) and legacy
  `openai==0.28.1` (only for optional GPT-baseline comparison scripts,
  not ORLM inference itself). **`coptpy`** (Cardinal Operations' own COPT
  solver) is an additional, previously-undocumented dependency — ORLM's
  official generation target is COPT solver code, not Pyomo/GurobiPy/
  plain LP. A COPT license (community/academic tier likely available, not
  independently verified) is required to *execute* generated code,
  separate from running the LLM.
- **GPU requirement:** **Yes** — 8B params in bf16 ≈16GB weights; a single
  24GB-class GPU (RTX 3090/4090, A5000) is plausible for inference (not
  training). **Not currently provisioned on this workstation.**
- **API requirement:** No (self-hosted weights, fully local/offline once
  downloaded).
- **Fine-tuning required:** No — published weights usable directly.
- **Licensing/access issues:** Apache-2.0 code; `llama3` license for the
  one confirmed checkpoint (a Meta license term, review before any
  redistribution — local research use is standard).
- **Expected implementation difficulty:** Medium — the lightweight
  inference-preparation and evaluation glue is complete. The actual model
  call remains unexecuted because no GPU/weights are available; COPT remains
  required only for solver execution.
- **Fairness concerns:** ORLM is a full generative pipeline (LLM writes
  the entire model + solver code), fundamentally different in
  computational cost and scope from this repo's retrieval + scalar-
  grounding pipeline -- any comparison must clearly caveat this is
  "full auto-formulation" vs. "grounding into a fixed catalog," not an
  apples-to-apples method comparison. See `baselines/orlm/README.md`
  "Fair comparison caveats" for the full dimension-by-dimension table.
- **Comparable metrics:** Execution/solve success rate, objective
  accuracy vs. gold, wall-clock/token cost (directly comparable to
  PaMOP's existing metric family) — **not** directly comparable to
  InstantiationReady/Coverage/TypeMatch.
- **Priority:** **1st** among the four non-PaMOP baselines.
- **First inference milestone (prepared, not executed):** provision a GPU
  and checkpoint, then run one query through the local lazy runner and
  static-validate the output before any COPT execution. The fixed pilot
  manifest and mocked end-to-end path are already committed.

## 3. OptMATH — **IMPLEMENTED READY FOR INFERENCE 2026-08-12, see `baselines/optmath/`**

- **Citation:** Lu et al., "OptMATH: A Scalable Bidirectional Data
  Synthesis Framework for Optimization Modeling," ICML 2025.
- **Official code available:** **YES, re-verified 2026-08-12** -- [github.com/optsuite/OptMATH](https://github.com/optsuite/OptMATH), upstream `main` revision `f15bbc4477c70db85ad148df8bcc1b780bca0f8c`, Apache-2.0.
- **Model weights available:** **YES** -- `Aurora-Gem/OptMATH-Qwen2.5-7B`
  and `Aurora-Gem/OptMATH-Qwen2.5-32B-Instruct` are released; the 7B model
  is the primary size-comparable checkpoint for this repository.
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
- **Expected implementation difficulty:** Lightweight inference preparation
  is complete; actual model/Gurobi execution remains resource-gated.
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
- **First inference milestone:** Run one 7B checkpoint query through the
  locked official prompt and static-validate generated Gurobi code before
  any solver execution. See `docs/OPTMATH_PROVENANCE.md`.

## 4. DeepOR (paper reconstruction ready; checkpoint pending)

- **Citation:** "DeepOR: A Deep Reasoning Foundation Model for
  Optimization Modeling," AAAI 2026 (per the AAAI proceedings listing
  found; very recent, published at/after this repository's knowledge
  cutoff window).
- **Official code available:** **NOT FOUND** after a fresh exact-title,
  author, GitHub, Hugging Face, and ModelScope search (2026-08-12).
- **Model weights available:** **NOT FOUND**. The paper specifies Qwen3-8B
  as the base model but does not release a DeepOR checkpoint identifier.
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
- **Expected implementation difficulty:** the inference/evaluation interface
  is implemented as a paper-level reconstruction; empirical execution is
  blocked on the trained checkpoint and exact upstream details.
- **Fairness concerns:** N/A until implementable.
- **Comparable metrics:** Presumably the same execution/feasibility/
  objective-accuracy family as ORLM/OptMATH/PaMOP, pending confirmation.
- **Priority:** **3rd** (behind ORLM/OptMATH on availability grounds).
  DeepOR and OR-R1 are now both checkpoint-blocked for different reasons:
  DeepOR has no confirmed official code at all, while OR-R1's official code
  is confirmed but its checkpoint is unreleased (see §5) -- neither is
  currently actionable for an empirical result.
- **First implementation milestone:** completed lightweight reconstruction;
  monitor for an official checkpoint/code release before claiming empirical
  DeepOR results. See `docs/DEEPOR_PROVENANCE.md`.

## 5. OR-R1

- **Citation:** Ding, Tan, Zhang, Chen, "OR-R1: Automating Modeling and
  Solving of Operations Research Optimization Problem via Test-Time
  Reinforcement Learning," AAAI-26, Vol. 40, No. 1, DOI
  10.1609/aaai.v40i1.36983, arXiv:2511.09092 (2025). (An earlier pass of
  this document cited the authors as "Zhu, Ma, Wang, Bi, et al." -- that
  was incorrect and has been corrected here and in
  `docs/ORR1_PROVENANCE.md`.)
- **Official code available:** **CONFIRMED** -- `SCUTE-ZZ/OR-R1`
  (`master`, commit `9de48e3b22555e729ec032e7efd00ebaaa8e78d5`), cited
  directly by the arXiv paper's own HTML/LaTeX source as "Code". No
  LICENSE file is present.
- **Model weights available:** **NOT RELEASED**. Searched Hugging Face,
  ModelScope (web search), GitHub releases/tags/wiki/issues, and the
  repository itself -- no SFT, TGRPO LoRA, or merged checkpoint found
  anywhere. `CHECKPOINT_NOT_RELEASED`.
- **Task overlap with ours:** High in spirit; full NL-to-model generation
  with a `coptpy` target, architecturally close to ORLM (shares the same
  `TEMPLATE_q2mc_en` prompt structure and evaluation-harness lineage --
  `eval/generate.py` even special-cases `ORLM-LLaMA-3-8B` model paths),
  not a retrieval+grounding pipeline.
- **NLP4LP support:** Yes -- one of nine official evaluation benchmarks
  (`eval.NLP4LP.pass1.sh`, `eval.NLP4LP.pass8.sh`), 242 official test rows.
- **Environment requirements:** vLLM for inference, coptpy for solver
  execution; DeepSpeed ZeRO-3 + TRL + PEFT for training (SFT and TGRPO),
  multi-GPU per the official shell scripts. No requirements file is
  published, so exact versions are unpinned.
- **GPU requirement:** Yes -- 24GB+ for inference-only; multi-GPU for any
  training (SFT or TGRPO), since neither checkpoint is released.
- **API requirement:** No.
- **Licensing/access issues:** No LICENSE file in the repository at all;
  treat as all-rights-reserved pending clarification. The SFT dataset
  (`OR-Instruct-Data-3K`) is ORLM's own release under cc-by-nc-4.0.
- **Expected implementation difficulty:** Lightweight integration (adapter,
  prompt, TGRPO control-flow modeling, static validation, execution
  harness, evaluator) is **complete** (`baselines/orr1/`). A faithful
  empirical result additionally requires training SFT and TGRPO from
  scratch -- substantially more expensive than ORLM/OptMATH's
  inference-only path, since OR-R1 has no released checkpoint at any
  stage.
- **Fairness concerns:** **Significant and specific.** The official TGRPO
  training set (`datasets/trainset/train_all.jsonl`) is verified (by
  direct file inspection, not just the paper's prose) to be exactly the
  union of all nine official evaluation test sets, including all 242
  NLP4LP rows. OR-R1's headline Pass@1/Pass@8 numbers therefore come from
  a model trained (via label-free self-consistency RL) directly on the
  questions being scored -- a fundamentally different evaluation protocol
  from every other baseline here, none of which train on the evaluation
  set. See `docs/ORR1_PROVENANCE.md` for the full analysis.
- **Comparable metrics:** Official Pass@1/Pass@8/mj@8 (execution +
  tolerance-based objective agreement), same family as ORLM/OptMATH; kept
  distinct from this repository's own proxy metrics.
- **Priority:** Lightweight integration is now complete alongside the
  other four baselines; further work is checkpoint-blocked, identically to
  DeepOR, but for a different reason (DeepOR: no official code at all;
  OR-R1: official code confirmed, but no checkpoint at any training
  stage).
- **First implementation milestone reached:** adapter, official prompt,
  TGRPO control-flow/reward-breakdown model, majority-voting/Pass@k
  scoring, static validator, execution harness, result schema, evaluator,
  fixed manifest, and mocked end-to-end tests (`baselines/orr1/`,
  2026-08-13). Next milestone is checkpoint-gated (see
  `docs/ORR1_PROVENANCE.md` "Future inference/TGRPO prerequisites").

---

## Recommended order and rationale

**PaMOP (pilot validated) -> ORLM -> OptMATH -> DeepOR -> OR-R1** for
lightweight-implementation order, which is what actually happened
(2026-08-12 through 2026-08-13); all five now have *some* lightweight
artifact. For future *empirical* (checkpoint-requiring) work the ranking is
no longer availability-only, since OR-R1's code turned out to be confirmed
public (unlike when this document was first written):

1. **Direct comparability**: ORLM and OptMATH are both full
   auto-formulation systems like PaMOP, so all three baselines share a
   comparable metric family (execution/feasibility/objective-accuracy),
   letting future work build one shared harness rather than three
   incompatible ones.
2. **Checkpoint availability now separates the baselines, not code
   availability**: ORLM and OptMATH have confirmed public code AND
   weights today. DeepOR has neither official code nor a checkpoint.
   **OR-R1 has confirmed official code (`SCUTE-ZZ/OR-R1`, cited directly by
   its arXiv paper) but no checkpoint at any training stage** -- a
   different and more specific blocker than DeepOR's. Empirical work on
   ORLM/OptMATH remains the cheapest path (inference-only); OR-R1's
   empirical path additionally requires running SFT and TGRPO from
   scratch, and TGRPO's official training data is transductive over every
   evaluation set (see `docs/ORR1_PROVENANCE.md`), which is a real
   scientific-fairness cost, not merely an engineering one.
3. **Implementation cost**: ORLM and OptMATH require the same class of
   new infrastructure (GPU-hosted 7-8B model inference), which is a real,
   one-time cost this repository does not currently pay (the whole
   grounding pipeline is deliberately CPU-only, no-external-LLM, per
   `docs/CURRENT_BOTTLENECK_ANALYSIS.md`'s documented strengths) --
   building that infrastructure once and reusing it for both ORLM and
   OptMATH is more efficient than any other ordering. OR-R1's inference
   path can reuse the same infrastructure (vLLM, coptpy) once a checkpoint
   exists.
4. **Reviewer relevance**: ORLM (Operations Research journal) and OptMATH
   (ICML 2025) are both peer-reviewed/accepted at their respective venues
   already; DeepOR and OR-R1 are both AAAI-26, newer and less established
   in the literature as of this pass, but OR-R1's DOI
   (10.1609/aaai.v40i1.36983) and code are both now confirmed.
5. **Recency is deliberately NOT the ranking criterion** -- DeepOR and
   OR-R1 are the most recent methods but rank last for *empirical* work,
   because they are checkpoint-blocked, not because of recency itself;
   re-ranking should happen automatically once either publishes usable
   checkpoints, independent of any further recency judgment.

## What must NOT happen next

All five baselines (PaMOP, ORLM, OptMATH, DeepOR, OR-R1) now have a
lightweight implementation, reconstruction, or integration in
`baselines/`. Do not start a sixth baseline. Do not run GPU-heavy inference,
model downloads, training (SFT or TGRPO), or solver benchmarks against any
of the five without first confirming compute resources are available and
without checking for conflicts with any unrelated higher-priority workload
running on the same machine. The concrete next actions per baseline are the
checkpoint/GPU-gated milestones recorded in each baseline's own README and
provenance document (`docs/ORLM_PROVENANCE.md`, `docs/OPTMATH_PROVENANCE.md`,
`docs/DEEPOR_PROVENANCE.md`, `docs/ORR1_PROVENANCE.md`) and in
`docs/PAMOP_REPRODUCTION_PLAN.md`.
