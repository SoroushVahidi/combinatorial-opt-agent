# OR-R1 provenance

Verified 2026-08-13 against the paper, the arXiv HTML source, the AAAI
proceedings record, and the official GitHub repository at commit
`9de48e3b22555e729ec032e7efd00ebaaa8e78d5`.

## Publication

- **Title:** "OR-R1: Automating Modeling and Solving of Operations Research
  Optimization Problem via Test-Time Reinforcement Learning"
- **Authors:** Zezhen Ding, Zhen Tan, Jiheng Zhang, Tianlong Chen (confirmed
  identical on arXiv and the AAAI proceedings record)
- **Venue:** Proceedings of the AAAI Conference on Artificial Intelligence,
  Vol. 40, No. 1 (AAAI-26)
- **DOI:** [10.1609/aaai.v40i1.36983](https://doi.org/10.1609/aaai.v40i1.36983)
- **arXiv:** [2511.09092](https://arxiv.org/abs/2511.09092), submitted
  2025-11-12T08:05:31Z, v1
- **Publication date:** 2026-03-14

Note: an earlier pass of `docs/BASELINE_IMPLEMENTATION_ROADMAP.md` recorded
the authors as "Zhu, Ma, Wang, Bi, et al." That citation was incorrect — the
paper's actual authors, confirmed independently on both arXiv and the AAAI
proceedings page, are Ding, Tan, Zhang, and Chen. This document and the
roadmap have been corrected.

## Official implementation

- **Repository:** [SCUTE-ZZ/OR-R1](https://github.com/SCUTE-ZZ/OR-R1)
- **Default branch:** `master`
- **Reference/latest commit:** `9de48e3b22555e729ec032e7efd00ebaaa8e78d5`
  ("init", 2025-11-12T02:47:31Z). The repository has exactly two commits,
  both authored by `SCUTE-ZZ` on 2025-11-12, and no releases or tags.
- **License:** none. No `LICENSE` file exists anywhere in the repository
  (checked via the GitHub license API, which returned 404).
- **Stars/forks:** 17 stars, 3 forks as of this check (informational only).

### Evidence this repository is official

1. **Direct self-citation (primary evidence).** The arXiv paper's own
   HTML/LaTeX source (rendered at `arxiv.org/html/2511.09092`) contains the
   line "Code — https://github.com/SCUTE-ZZ/OR-R1" in its abstract section.
   This is the strongest available evidence: the authors' own paper names
   this exact repository as their code release.
2. **Timeline consistency.** The GitHub repository was created
   2025-11-11T10:13:58Z and pushed 2025-11-12T02:47:45Z — about six hours
   before the arXiv submission timestamp (2025-11-12T08:05:31Z). This is
   the pattern expected of an author uploading code immediately before
   submitting the paper, not an unrelated third party.
3. **Content match.** The repository's evaluation suite
   (`eval/eval.NLP4LP.pass1.sh`, `eval.NLP4LP.pass8.sh`, plus seven other
   benchmark families: NL4OPT, MAMO_EasyLP/ComplexLP, IndustryOR, ComplexOR,
   OptiBench, OptMath, ICMLTEST) matches the benchmarks named in the paper,
   and the base model (`Qwen/Qwen3-8B`, hardcoded in `01_sft_train.sh`)
   matches the paper's stated base model.
4. **What is *not* independently confirmed:** the GitHub account `SCUTE-ZZ`
   itself carries no bio, real name, affiliation, or link back to any of the
   four paper authors (checked via the GitHub Users API). "SCUTE" plausibly
   abbreviates a South China University of Technology-affiliated group, but
   this is not verified. The evidence above is paper-self-citation evidence,
   not GitHub-account-identity evidence. Classification below reflects this:
   code reused verbatim is `EXACT_OFFICIAL` on the strength of the paper's
   own citation, while any claim resting on the GitHub account's identity
   alone would need the more cautious `ADAPTED_OFFICIAL`/`PAPER_SPECIFIED`
   label instead. No such claim is made in this integration.

## Released artifacts

| Artifact | Status | Evidence |
|---|---|---|
| Training/eval code | AVAILABLE | `SCUTE-ZZ/OR-R1`, all 7 top-level scripts + `config/`, `eval/`, `utils/` present |
| SFT dataset (`OR-Instruct-Data-3K`) | AVAILABLE | `datasets/OR-Instruct-Data-3K/`; this is the ORLM dataset (Tang et al. 2024, arXiv:2405.17743), cc-by-nc-4.0, not new OR-R1 data |
| TGRPO training data (`trainset/train_all.jsonl`) | AVAILABLE | See leakage finding below — it is exactly the union of the test sets |
| Evaluation test sets (9 benchmarks incl. NLP4LP) | AVAILABLE | `datasets/testset/*.jsonl` |
| DeepSpeed configs | AVAILABLE | `config/sft_config.json`, `config/grpo_config.json` (ZeRO-3, CPU offload) |
| Base checkpoint | AVAILABLE (external) | `Qwen/Qwen3-8B` on Hugging Face — not an OR-R1 artifact |
| SFT checkpoint | **NOT RELEASED** | No HF/ModelScope/GitHub-release model found under any OR-R1/SCUTE-ZZ/author naming |
| TGRPO LoRA adapter | **NOT RELEASED** | Same search, no result |
| Merged final checkpoint | **NOT RELEASED** | Same search, no result |
| README / usage docs | **ABSENT** | No `README.md` exists at the repository root (only inside `datasets/OR-Instruct-Data-3K/`) |
| LICENSE | **ABSENT** | No license file anywhere in the repository |

Checkpoint search covered: Hugging Face Hub (model + dataset search for
"OR-R1" and "OR-Instruct"), a general web search for "OR-R1 Qwen3-8B
huggingface checkpoint", a web search combining the paper's exact author
names with "modelscope checkpoint", the GitHub repository's own contents
tree (`config/`, `eval/`, `utils/` — no model files), GitHub Releases API
(empty), GitHub Tags API (empty), GitHub Issues API (empty, zero issues
ever filed), and the repository's wiki (redirects to a login page, no
public content). **Conclusion: `CHECKPOINT_NOT_RELEASED`.**

## What OR-R1 actually is

### A. What is baked into the final trained model

The final evaluated checkpoint is `SFT(Qwen3-8B on OR-Instruct-Data-3K)` with
a `TGRPO`-trained LoRA adapter merged in (`03_combine_lora.py`). Both stages
are complete before evaluation begins; nothing about the model changes
during evaluation itself.

### B. Is TGRPO actually performed during evaluation on each test problem?

**No.** `04_eval.sh` calls `eval/eval.all.sh`, which calls
`eval/generate.py` (vLLM inference) then `eval/execute.py` (coptpy
execution + scoring) per benchmark. Neither script imports a trainer,
computes a gradient, or updates any weights. TGRPO is exclusively an
offline training stage (`02_grpo_train.py`), run once, before evaluation.

### C. Does inference require online/test-time gradient updates?

**No**, contrary to what the paper's title might suggest in isolation. See
B. "Test-time" describes the *reward design* (self-consistency without
labels — see the paper's own statement that "the Majority Voting Reward is
derived from the Test-Time Reinforcement Learning (TTRL) framework"), not a
literal test-time optimization loop.

### D. Are multiple candidate rollouts generated?

**Yes, in two different contexts:**
- During TGRPO training: `num_generations=8` per training question (GRPO
  group size), used to compute the majority-voting reward.
- During evaluation: `topk=8` sampled completions per question for the
  Pass@8/mj@8 metric (`eval.<dataset>.pass8.sh`); `topk=1` greedy for Pass@1.

### E. Does majority voting or solver reward select candidates?

**Both, in different places.** Training reward = format reward + valid-code
reward (1.0 iff generated code executes and yields a numeric best solution)
+ majority-voting reward (1.0 iff a completion's own answer matches its
group's majority vote) — see `02_grpo_train.py`'s `reward_with_reference`.
At evaluation, `eval/execute.py` reports Pass@8 (any of 8 rollouts within
5% of gold) and separately mj@8 (the majority-voted rollout within 5% of
gold) as two distinct metrics — solver execution feeds both.

### F. Is the test problem itself used as unlabeled data for TGRPO?

**Yes — confirmed by direct file inspection, not just the paper's prose.**
`datasets/trainset/train_all.jsonl` contains exactly 2634 lines. The nine
official `datasets/testset/*.jsonl` files contain, respectively: NL4OPT 230,
MAMO_EasyLP 652, MAMO_ComplexLP 211, IndustryOR 100, NLP4LP 242, ComplexOR
18, OptiBench 605, OptMath 166, and a `task3_test` file of 410 rows — summing
to exactly 2634. A set-based comparison of question text confirms **100% of
every one of these 2634 test rows, across all nine files, appears verbatim
in `train_all.jsonl`**, including all 242 NLP4LP questions. The reward
function never reads the ground-truth answer (`kwargs['answer']` in
`reward_with_reference` is written to a CSV log only, never added to any
reward term). This is precisely the paper's own description: "the model
undergoes further training with TGRPO on unlabeled test set data."

**Implication:** OR-R1's officially reported Pass@1/Pass@8/mj@8 numbers are
obtained from a checkpoint that has already been trained (via label-free RL)
directly on the exact questions being scored. This is not a leak or a bug —
it is the published methodology — but it means OR-R1's evaluation protocol
is fundamentally different from a conventional held-out test split, and from
every other baseline in this repository. See "Cross-baseline fairness"
below.

## Chosen primary baseline configuration

The paper's headline number is SFT+TGRPO evaluated at **Pass@8** (with mj@8
reported alongside); Pass@1 is a secondary column in the same results table
from the same checkpoint, not a different training regime. This repository's
`baselines/orr1/config.pass8_config()` is accordingly the primary comparison
configuration; `pass1_config()` is retained and available but not silently
substituted as "the" OR-R1 result.

## Fidelity matrix

| Component | Primary evidence | Local implementation | Fidelity |
|---|---|---|---|
| Prompt template (`TEMPLATE_q2mc_en`) | `02_grpo_train.py`, `eval/generate.py` | `config.ORR1_PROMPT_TEMPLATE` + `data_adapter.build_orr1_prompt` (uses `str.replace`, matching upstream exactly) | EXACT_OFFICIAL |
| Format-reward field checklist | `reward_with_reference` | `config.ORR1_FORMAT_FIELDS`, `output_normalizer.py` | EXACT_OFFICIAL |
| Code-fence extraction (first block only) | `run_code` / `eval/execute.py`'s code loader | `output_normalizer._extract_first_python_block` | EXACT_OFFICIAL |
| Post-execution suffix (`ORR1_ADD_SCRIPT`) | `reward_with_reference`, `eval/execute.py` | `config.ORR1_ADD_SCRIPT`, `execution_harness.py` | EXACT_OFFICIAL |
| Majority voting / pass@k / mj@k | `eval/execute.py` | `rollout.py` (`majority_voting`, `score_group`) | EXACT_OFFICIAL |
| SFT/TGRPO hyperparameters | `01_sft_train.sh`, `02_grpo_train.sh` | `config.py` constants | EXACT_OFFICIAL (recorded, not executed) |
| Base model | `01_sft_train.sh` | `config.BASE_MODEL = "Qwen/Qwen3-8B"` | EXACT_OFFICIAL |
| Decoding params (pass@1 greedy, pass@8 sampling) | `eval/generate.py` argparse defaults | `config.pass1_config`/`pass8_config` | EXACT_OFFICIAL |
| Generation backend | `eval/generate.py` (vLLM, local path only) | `runner.VLLMBackend` | ADAPTED_OFFICIAL (lazy import, injectable for tests) |
| Generation backend (fallback) | none upstream | `runner.TransformersBackend` | LOCAL_ENGINEERING |
| SFT/TGRPO/merge state machine | `01_sft_train.py`→`02_grpo_train.py`→`03_combine_lora.py` control flow | `tgrpo_controller.CheckpointState` + `mock_*` transitions | ADAPTED_OFFICIAL (state modeled, no gradients run) |
| Reward-component breakdown | `reward_with_reference` | `tgrpo_controller.reward_component_breakdown` | EXACT_OFFICIAL |
| NLP4LP adaptation (`question`/`answer` shape) | `datasets/testset/nlp4lp.jsonl` schema | `data_adapter.py` | ADAPTED_OFFICIAL |
| Static coptpy validation | No official equivalent (upstream only executes) | `static_validation.py`, incl. the OR-R1-specific `model` variable-name check | LOCAL_ENGINEERING |
| Result/evaluator schema | Upstream `generated.jsonl`/`executed.jsonl`/`metrics.json` shapes | `result_schema.py`, `evaluator.py` | LOCAL_ENGINEERING, kept distinct from official pass@k/mj@k |
| Transductive-training-set finding | Not stated as a caveat anywhere upstream; derived by this repository from direct file inspection | `config.GRPO_TRANSDUCTIVE_LEAKAGE_NOTE`, this document | LOCAL_ENGINEERING (a finding about official data, not a reconstruction) |

## Training

- **Base model:** `Qwen/Qwen3-8B`.
- **SFT:** `01_sft_train.py`/`.sh`. Data:
  `datasets/OR-Instruct-Data-3K/OR-Instruct-Data-{sample}.json` (variants at
  1/10/100/1000/3000 examples; the released ORLM instruction corpus, not new
  OR-R1-authored data). `lr=2e-5`, `max_seq_length=8192`,
  `gradient_accumulation_steps=16`, linear schedule, `warmup_ratio=0.03`,
  DeepSpeed ZeRO-3 (`config/sft_config.json`), `bf16=True`.
- **TGRPO:** `02_grpo_train.py`/`.sh`, TRL `GRPOTrainer`. Data:
  `datasets/trainset/train_all.jsonl` (== union of all 9 eval test sets, see
  finding F above). `num_generations=8`, LoRA `r=16, alpha=16,
  target_modules=all-linear`, `lr=1e-4` cosine, `num_train_epochs=1`,
  `warmup_steps=10`, `weight_decay=0.01`, `adam_beta2=0.95`,
  `max_prompt_length=2048`, `max_completion_length=6144`, DeepSpeed ZeRO-3
  (`config/grpo_config.json`).
- **Reward components:** format reward (fraction of 6 required section
  headers present), valid-code reward (binary: code executes and yields a
  numeric best solution), majority-voting reward (binary: own answer matches
  the group-of-8 majority vote). No ground-truth term.
- **Merge:** `03_combine_lora.py` — `PeftModel.merge_and_unload()` on top of
  the SFT checkpoint.

## Inference / evaluation

- **Prompt:** `TEMPLATE_q2mc_en`, wrapped in the model's chat template via
  `tokenizer.apply_chat_template` at generation time (not reproducible
  without the actual tokenizer; recorded as `requires_chat_template=True`).
- **Decoding:** Pass@1 = greedy (`temperature=0, top_p=1`); Pass@8 = sampling
  (`temperature=0.7, top_p=0.95`, `eval/generate.py` argparse defaults).
- **Rollout count:** 1 or 8 depending on the metric being computed.
- **Test-time optimization:** none (see B above).
- **Solver:** `coptpy`, via the official `ORR1_ADD_SCRIPT` suffix appended
  to generated code, requiring a literal `model` variable.
- **Timeout:** 600s (`eval/execute.py --timeout`); the training-time reward
  path uses a separate, shorter 10s default (`reward_with_reference`'s
  `compile_script`).
- **Metrics:** `pass@k` (any of k rollouts within 5% relative tolerance of
  gold, or exact match on the "No Best Solution" sentinel; absolute
  tolerance ≤5% when gold is exactly 0) and `mj@k` (majority-voted rollout
  within the same tolerance). Both are official `eval/execute.py` outputs,
  not derived metrics.

## Cross-baseline fairness note

Unlike ORLM, OptMATH, DeepOR, and PaMOP, OR-R1's primary reported metric
comes from a model that has already been trained (via label-free RL) on the
exact evaluation questions. Any future empirical OR-R1 number reported
against this repository's NLP4LP manifest must state explicitly whether it
follows the official transductive protocol (train TGRPO on the manifest
itself before scoring it — the only way to reproduce the paper's headline
number faithfully) or a held-out variant (skip TGRPO / use only the SFT
checkpoint, which is not what the paper reports as its main result). Do not
present a held-out-only number as "the" OR-R1 result without this caveat.

## Unresolved ambiguities

- The exact chat template/tokenizer configuration for `Qwen/Qwen3-8B` at the
  specific revision OR-R1 trained against is not pinned in the repository
  (`model_revision` defaults to `"main"` throughout).
- No requirements/environment file is published; TRL, PEFT, DeepSpeed, and
  vLLM versions are unspecified.
- Whether this repository's own PaMOP-derived NLP4LP pilot IDs
  (`baselines/orr1/manifests/nlp4lp_common_manifest.json`) overlap with the
  official 242-row NLP4LP test split cannot currently be determined: the
  official file's `ori` provenance field is the constant string
  `"14_nlp4lp"` on all 242 rows, not a per-instance identifier, so it gives
  no cross-reference signal. A text-level comparison against this
  repository's own raw NLP4LP source would be needed to settle this before
  any faithful-TGRPO run against the shared manifest.
