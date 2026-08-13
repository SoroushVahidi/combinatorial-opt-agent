# OR-R1 baseline (official code verified, checkpoint blocked)

**Status: `ORR1_CODE_INTEGRATED_CHECKPOINT_BLOCKED`, 2026-08-13.** No model
weights were downloaded, no training was run, and no coptpy execution was
performed. The lightweight adapter, official prompt/format checklist, lazy
vLLM/Transformers runner, TGRPO control-flow model, majority-vote/Pass@k
scoring, output normalizer, static validator, safe execution harness, result
schema, evaluator, fixed manifest, and mocked end-to-end tests are complete.

## Primary sources

- Paper: Ding, Tan, Zhang, Chen, "OR-R1: Automating Modeling and Solving of
  Operations Research Optimization Problem via Test-Time Reinforcement
  Learning," *AAAI-26*, Vol. 40, No. 1. DOI:
  [10.1609/aaai.v40i1.36983](https://doi.org/10.1609/aaai.v40i1.36983).
  arXiv: [2511.09092](https://arxiv.org/abs/2511.09092).
- Official code: [SCUTE-ZZ/OR-R1](https://github.com/SCUTE-ZZ/OR-R1),
  commit `9de48e3b22555e729ec032e7efd00ebaaa8e78d5` (repository's only two
  commits, both `2025-11-12`; no releases/tags).
- Evidence the repository is official: the arXiv paper's own HTML/LaTeX
  source cites this exact URL as "Code" in its abstract. The repository was
  created 2025-11-11 and pushed 2025-11-12T02:47:45Z, one day before the
  arXiv submission timestamp (2025-11-12T08:05:31Z) — consistent with an
  author-controlled camera-ready drop. The GitHub account (`SCUTE-ZZ`) itself
  has no bio/affiliation confirming author identity independently, so this is
  paper-self-citation evidence, not GitHub-identity evidence; see
  `docs/ORR1_PROVENANCE.md` for the full writeup.
- Checkpoint: **not released anywhere** — searched Hugging Face, ModelScope
  (via web search), GitHub releases/tags/wiki/issues, and the repository
  itself. No LICENSE file is present in the repository.

## What OR-R1 actually is (training vs. inference)

OR-R1 has three offline stages plus pure-inference evaluation — it does
**not** perform per-instance online weight updates at test time, despite the
paper's "Test-Time Reinforcement Learning" title:

1. **SFT** (`01_sft_train.py`) on `OR-Instruct-Data-3K` — the same dataset
   released by ORLM (Tang et al. 2024, cc-by-nc-4.0), not new OR-R1 data.
2. **TGRPO** (`02_grpo_train.py`, TRL `GRPOTrainer`, LoRA) — group-of-8
   rollouts per question, reward = format reward + valid-code reward +
   majority-voting reward. **The ground-truth answer is never read by the
   reward function** (`kwargs['answer']` is logged to CSV only) — this is
   the paper's "test-time" framing: self-consistency RL without labels.
3. **Merge** (`03_combine_lora.py`) — LoRA adapter merged into the SFT model.
4. **Evaluation** (`04_eval.sh` → `eval/generate.py` + `eval/execute.py`) —
   plain vLLM inference against the merged checkpoint (greedy for Pass@1,
   `temperature=0.7`/`top_p=0.95` sampling for Pass@8). No training happens
   during evaluation.

**Critical finding, verified by direct inspection of the released data
files:** `datasets/trainset/train_all.jsonl` (2634 rows) is *exactly* the
concatenation of every official `datasets/testset/*.jsonl` file
(18+100+211+652+230+605+166+410+242 = 2634), including all 242 rows of the
official NLP4LP test split verbatim. So step 2 above is trained directly on
the questions later scored in step 4, using only self-consistency (no
labels) as signal. OR-R1's headline Pass@1/Pass@8 numbers are **transductive**
— this is the officially published design, not an error in this
reconstruction — and it is a first-order fairness difference from every
other baseline in this repository (ORLM, OptMATH, DeepOR, PaMOP), none of
which train on the evaluation questions. See `tgrpo_controller.py` and
`docs/ORR1_PROVENANCE.md`.

## Chosen primary baseline configuration

The paper's main reported result is the SFT+TGRPO model evaluated at
**Pass@8** (with mj@8 as the majority-vote variant); Pass@1 is reported as a
secondary/ablation number in the same table from the same checkpoint. This
package's `config.pass8_config()` is therefore the primary comparison point;
`pass1_config()` remains available and is not silently substituted for it.

## Lightweight implementation

- `config.py` — official prompt, hyperparameters (SFT/TGRPO/eval), upstream
  revision, and the transductive-training-set finding, all with inline
  citations to the exact upstream file/line source.
- `data_adapter.py` — deterministic NLP4LP→`{"question","answer"}` conversion
  and the exact `str.replace("{Question}", ...)` prompt substitution (not
  `str.format`, matching upstream precisely).
- `runner.py` — lazy `VLLMBackend` mirroring `eval/generate.py` exactly, plus
  a non-official `TransformersBackend` fallback and an injectable mock.
- `tgrpo_controller.py` — the distinguishing component: reward-component
  breakdown ported from `reward_with_reference`, a `CheckpointState` state
  machine (`BASE → SFT → GRPO_LORA → MERGED`), and a cross-group leakage
  guard for any future non-official isolated-TGRPO experiment.
- `rollout.py` — exact port of `eval/execute.py`'s `majority_voting` and its
  pass@k/mj@k tolerance logic (including the "No Best Solution" and
  zero-gold special cases).
- `output_normalizer.py` — first-fenced-python-block extraction (matching
  upstream's `find`/`find` logic, not longest-block selection) plus the
  six-field format-reward checklist.
- `static_validation.py` — coptpy shape checks plus a check specific to
  OR-R1: the generated code must assign a variable literally named `model`,
  because the official execution harness appends `model.status`/
  `model.objval` verbatim.
- `execution_harness.py` — opt-in isolated subprocess harness appending the
  official `ORR1_ADD_SCRIPT` suffix; dry-run by default.
- `result_schema.py` / `evaluator.py` — per-rollout JSON-friendly records and
  official pass@k/mj@k group scoring, kept separate from this repository's
  own proxy metrics.
- `pipeline.py` — mock 8-rollout end-to-end path exercising every stage
  above with an injectable backend; no GPU/vLLM/coptpy/network required.
- `manifests/nlp4lp_common_manifest.json` — the fixed six-instance pilot and
  18-instance future subset shared with ORLM/OptMATH/DeepOR, plus an
  explicit caveat that the official `nlp4lp.jsonl`'s `ori` field is a
  constant string on all 242 rows and therefore cannot be used to
  cross-reference these pilot IDs against the official split by inspection.

## Future inference/TGRPO prerequisites

Separate two regimes explicitly:

**`INFERENCE_ONLY`** (base Qwen3-8B or a locally-produced SFT/merged
checkpoint, no TGRPO): a single GPU with ≥24GB VRAM, vLLM, and coptpy is
sufficient to exercise the runner and execution harness end to end.

**`FULL_ORR1_TGRPO_EVALUATION`** (faithful reproduction of the paper's
primary number): requires running SFT then TGRPO from scratch — no official
checkpoint at any stage is released — which means multi-GPU DeepSpeed ZeRO-3
training (`config/sft_config.json`, `config/grpo_config.json`), a working
coptpy solver for reward computation during training, and, per the
transductive-training finding above, training directly on whatever manifest
is evaluated. This is substantially more expensive than ORLM/OptMATH
inference-only reproduction and is not attempted in this pass.

No GPU-heavy inference, model download, training, or solver benchmark was
performed in this implementation pass.
