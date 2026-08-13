# OptMATH provenance and fidelity

Rechecked 2026-08-12 against the [ICML/OpenReview paper](https://openreview.net/forum?id=9P5e6iE4WK),
the [official `optsuite/OptMATH` repository](https://github.com/optsuite/OptMATH),
and the [Aurora-Gem released models](https://huggingface.co/Aurora-Gem/models).
The official repository `main` revision inspected is
`f15bbc4477c70db85ad148df8bcc1b780bca0f8c`.

**Citation:** Hongliang Lu, Zhonglin Xie, Yaoyu Wu, Can Ren, Yuxuan Chen,
and Zaiwen Wen, “OptMATH: A Scalable Bidirectional Data Synthesis Framework
for Optimization Modeling,” *ICML 2025*, PMLR 267, pp. 40769–40802;
arXiv:2502.11102 (v1, 2025-02-16). The current paper record is the ICML 2025
version; the repository's current `main` revision is the implementation
reference.

## What the baseline represents

OptMATH is primarily a bidirectional data-synthesis and validation framework:
seed mathematical formulations produce problem data, backtranslation creates
natural language, and forward modeling plus rejection sampling verifies the
triplets. The external inference baseline here is the released fine-tuned
OptMATH checkpoint, not a reproduction of the data-synthesis/training run.

Primary checkpoint selection:

- `Aurora-Gem/OptMATH-Qwen2.5-7B` is the primary baseline because it is a
  released OptMATH model and is closest in scale to the existing ORLM 8B
  baseline. The inspected Hugging Face model revision is `617fe77`.
- `Aurora-Gem/OptMATH-Qwen2.5-32B-Instruct` is recorded as an optional larger
  sensitivity model, not silently substituted for the primary baseline.
- The primary 7B checkpoint is a Qwen2/Qwen2.5-7B causal language model in
  bf16; its model card/config identifies the Qwen2 architecture and does not
  require OptMATH-specific preprocessing at inference.

## Fidelity matrix

| Component | Primary source | Local implementation | Fidelity | Notes |
|---|---|---|---|---|
| Data-synthesis contribution | Paper and official README | Documentation only | PAPER_RECONSTRUCTED | Training is not reproduced |
| Released inference model | Official HF organization | `OptmathConfig.model_id` | EXACT_OFFICIAL | 7B selected; 32B optional |
| System prompt | Official `eval/evaluator.py` | `prompt.py` | EXACT_OFFICIAL | Exact system string preserved |
| User prompt | Official `_build_cot_prompt` | `prompt.py` | EXACT_OFFICIAL | Gurobi code-only instructions preserved |
| Model generation | Official evaluator/LLM abstraction | Lazy Transformers backend | ADAPTED_OFFICIAL | No model loaded in this task |
| Input fields | Official `en_question`, `en_answer` | NLP4LP adapter | ADAPTED_OFFICIAL | IDs/source/gold metadata added |
| Output format | Official `en_math_model_code` and fenced Python | Output normalizer | ADAPTED_OFFICIAL | Raw output always preserved |
| Solver | Official `gurobipy` | Gurobi harness | EXACT_OFFICIAL target | Execution opt-in and unavailable locally |
| Conversion fallback | Official evaluator/executor | Disabled by default in local harness | LOCAL_ENGINEERING | Avoids silent semantic source mutation |
| Correctness metric | Official rounded relative tolerance, 5% | Evaluator proxy | ADAPTED_OFFICIAL | Full solver-verified evaluation remains execution-dependent |
| NLP4LP benchmark | No official OptMATH NLP4LP evaluation | Fixed common manifest | LOCAL_ENGINEERING | Not an official OptMATH benchmark claim |

## Official settings and environment

The official evaluation defaults inspected were temperature `0.8`, max tokens
`8192`, timeout `100` seconds, and numerical tolerance `0.05`. The official
pipeline uses Python 3.11, `gurobipy`, and API-backed LLM clients for its
generation/evaluation utility; released checkpoint inference is prepared here
through a lazy local Transformers backend.

The repository `LICENSE` is Apache-2.0. Its `pyproject.toml` contains a stale
MIT metadata field; the actual LICENSE file is treated as authoritative.
No API keys, Gurobi license files, or model weights are stored here.

## Metric boundary

Official evaluation executes generated Gurobi code, extracts the printed
objective, rounds values, and checks relative error against the reference.
That objective agreement is implemented as a named proxy in this repository;
it is not renamed semantic equivalence. The official OptMATH triplet
equivalence/rejection-sampling claim is a training-data validation concept,
not automatically transferable to an NLP4LP adaptation.
