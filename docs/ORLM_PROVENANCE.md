# ORLM provenance and fidelity

Rechecked 2026-08-13 against the official [paper](https://arxiv.org/abs/2405.17743),
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
| Generation length | Upstream `max_tokens=None` -> model max length | `OrlmConfig.max_new_tokens=8192` | EXACT_OFFICIAL | Checkpoint `max_position_embeddings=8192` |
| Model/checkpoint | Official HF model card | `CardinalOperations/ORLM-LLaMA-3-8B` | EXACT_OFFICIAL | Pinned revision `94fdc3c5738c6536d4880dc19a78f215529181c5` |
| Output format | Upstream `en_math_model_coptpy_code` / fenced code | `output_normalizer.py` | ADAPTED_OFFICIAL | Robust parser, raw output preserved |
| Execution | Upstream `eval/execute.py`, COPT-only | `execution_harness.py` | ADAPTED_OFFICIAL | Safer isolated subprocess, opt-in only |
| Static validation | No official equivalent | `static_validation.py` | LOCAL_ENGINEERING | Never substitutes for execution or semantics |
| Result/evaluation schema | Upstream generated/execution JSONL | `result_schema.py`, `evaluator.py` | LOCAL_ENGINEERING | Objective agreement explicitly proxy-only |
| NLP4LP adaptation | No upstream NLP4LP path | `data_adapter.py` and manifest | LOCAL_ENGINEERING | New cross-system adaptation; no original benchmark claim |

## Model and environment limits

The official repository requires a local model path for its vLLM evaluation
script and executes generated programs with COPT. The pinned checkpoint is
cached at the Hugging Face snapshot for revision
`94fdc3c5738c6536d4880dc19a78f215529181c5`. The current host has an RTX 5060
Ti with 16.3 GiB VRAM, so the Transformers adaptation uses CPU offload. The
six-instance pilot is running in tmux; no completed row or solver execution
exists yet. `coptpy` is not installed, so COPT execution is currently blocked.

Pilot handoff: session `orlm_pilot_official_20260813_corrected`, started
`2026-08-13T23:06:54-04:00`, Git SHA
`6bb75a4c4bed02c458ac30b4af206a2802fce095`, log
`results/orlm/pilot_official_checkpoint/inference_corrected.log`, output
`results/orlm/pilot_official_checkpoint/results.jsonl`. The approximately
three-minute health check found the model loaded, stable GPU memory below
15.3 GiB, and no OOM or generation exception.

## Fair comparison boundary

ORLM generates a complete optimization formulation and COPT program. This
repository's primary system grounds scalar values into a fixed catalog. The
common manifest is intended for parallel reporting of executable formulation,
feasibility, and gold-objective agreement only; it does not make ORLM's native
full-formulation accuracy equal to `InstantiationReady`.
