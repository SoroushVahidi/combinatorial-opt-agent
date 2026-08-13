# ORLM provenance and fidelity

Rechecked 2026-08-12 against the official [paper](https://arxiv.org/abs/2405.17743),
[Cardinal-Operations/ORLM](https://github.com/Cardinal-Operations/ORLM), and
the official [LLaMA-3-8B checkpoint](https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B).
The repository revision used for source inspection is
`33bc47d0a1d1710d24ab839118bdf4cb89b9e31b`.

| Component | Primary source | Local implementation | Fidelity | Action/status |
|---|---|---|---|---|
| Prompt wording | `eval/generate.py`, `TEMPLATE_q2mc_en` | `config.py` + `data_adapter.py` | EXACT_OFFICIAL | Locked and snapshot-tested |
| Input shape | Upstream `en_question` + `prompt` fields | `OrlmInputRecord.to_upstream_example()` | ADAPTED_OFFICIAL | NLP4LP ID/text/gold metadata retained |
| Generation backend | Upstream vLLM `eval/generate.py` | Lazy Transformers backend with injectable backend | ADAPTED_OFFICIAL | No weights loaded; local-path inference ready |
| Greedy decoding | Upstream `topk=1`, temperature `0`, top-p `1` | `OrlmConfig` | EXACT_OFFICIAL | Recorded per result |
| Model/checkpoint | Official HF model card | `CardinalOperations/ORLM-LLaMA-3-8B` | EXACT_OFFICIAL | Revision remains configurable |
| Output format | Upstream `en_math_model_coptpy_code` / fenced code | `output_normalizer.py` | ADAPTED_OFFICIAL | Robust parser, raw output preserved |
| Execution | Upstream `eval/execute.py`, COPT-only | `execution_harness.py` | ADAPTED_OFFICIAL | Safer isolated subprocess, opt-in only |
| Static validation | No official equivalent | `static_validation.py` | LOCAL_ENGINEERING | Never substitutes for execution or semantics |
| Result/evaluation schema | Upstream generated/execution JSONL | `result_schema.py`, `evaluator.py` | LOCAL_ENGINEERING | Objective agreement explicitly proxy-only |
| NLP4LP adaptation | No upstream NLP4LP path | `data_adapter.py` and manifest | LOCAL_ENGINEERING | New cross-system adaptation; no original benchmark claim |

## Model and environment limits

The official repository requires a local model path for its vLLM evaluation
script and executes generated programs with COPT. The checkpoint is public,
but no weights were downloaded in this pass. COPT/coptpy availability and
license status were not changed or exposed. No ORLM inference or solver
execution has been performed locally.

## Fair comparison boundary

ORLM generates a complete optimization formulation and COPT program. This
repository's primary system grounds scalar values into a fixed catalog. The
common manifest is intended for parallel reporting of executable formulation,
feasibility, and gold-objective agreement only; it does not make ORLM's native
full-formulation accuracy equal to `InstantiationReady`.
