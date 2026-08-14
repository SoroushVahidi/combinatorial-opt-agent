# ORLM baseline (implemented, inference-ready)

**Status: `ORLM_PILOT_RUNNING_HEALTHY`, 2026-08-13.** The pinned checkpoint
is cached and the official six-instance pilot is running in tmux; no
completed empirical row or COPT execution exists yet. The
lightweight adapter, official prompt builder, lazy Transformers runner,
normalizer, static validator, safe execution harness, result schema,
evaluator, resume store, fixed manifest, and mocked end-to-end tests are
complete.

## Primary sources

- Paper: Huang et al., [ORLM](https://arxiv.org/abs/2405.17743).
- Official code: [Cardinal-Operations/ORLM](https://github.com/Cardinal-Operations/ORLM).
- Verified upstream revision: `33bc47d0a1d1710d24ab839118bdf4cb89b9e31b`.
- Official checkpoint: [CardinalOperations/ORLM-LLaMA-3-8B](https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B).
- Run checkpoint revision: `94fdc3c5738c6536d4880dc19a78f215529181c5`.
- Code license: Apache-2.0. The checkpoint has a Llama 3 license; review it
  before redistribution or deployment.

The upstream code uses vLLM with a local model path for generation and COPT
(`coptpy`) for execution. This repository uses its documented Transformers
adaptation for the pinned checkpoint; the weights remain in the Hugging Face
cache and are not repository artifacts.

The component-by-component evidence table is maintained in
[`docs/ORLM_PROVENANCE.md`](../../docs/ORLM_PROVENANCE.md).

## Official prompt and decoding

The prompt is copied structurally from upstream `eval/generate.py` and
versioned as `upstream-eval-generate-TEMPLATE_q2mc_en-v1`:

```text
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}
# Response:
```

The official greedy generation path uses `topk=1`, `temperature=0`,
`top_p=1`, `max_tokens` resolved to the checkpoint's 8192-token model limit,
and stops on `</s>`. The local configuration records these settings
explicitly. The runner also supports an injected backend for tests and a lazy
Transformers backend; importing it does not load the 8B checkpoint.

## NLP4LP adaptation and evaluation boundary

ORLM was evaluated by its authors on NL4OPT, MAMO, and IndustryOR, not
NLP4LP. The adapter therefore performs a transparent task adaptation: each
NLP4LP raw problem becomes the upstream `en_question`/`prompt` shape while
preserving its ID, raw text, metadata, and text hash. Unsupported or malformed
records are retained as explicit exclusions with reasons.

ORLM-native metrics are kept separate from this repository's scalar-grounding
metrics:

| ORLM-native | Shared/common outcome |
|---|---|
| generated model/code, parseability, static validity, COPT execution, feasibility, objective agreement | valid executable formulation and gold-objective agreement where both systems and gold data support it |

Objective agreement is an objective-value proxy, never semantic accuracy.
`InstantiationReady` is not directly comparable to full ORLM formulation
accuracy.

## Lightweight implementation

- `config.py` — official prompt, upstream revision, checkpoint, and decoding metadata.
- `data_adapter.py` — deterministic NLP4LP record conversion and explicit exclusion reasons.
- `runner.py` — lazy Transformers backend plus injectable mock backend.
- `output_normalizer.py` — fenced/unfenced extraction with raw-output preservation and warnings.
- `static_validation.py` — non-executing Python/coptpy shape and safety checks.
- `execution_harness.py` — opt-in isolated subprocess harness; dry-run by default.
- `result_schema.py` — JSON-friendly per-instance provenance/result record.
- `evaluator.py` — generation, parsing, static, execution, and objective-proxy metrics.
- `pipeline.py` — mocked-ready path and append-only problem-ID resume store.
- `scripts/run_orlm_inference.py` — resumable pilot/common-18 launcher with run metadata.
- `manifests/nlp4lp_common_manifest.json` — fixed six-instance pilot and 18-instance future subset.

## Future inference prerequisites

1. Provision an isolated environment with compatible PyTorch/Transformers or
   the upstream vLLM path.
2. Obtain the checkpoint and verify its revision/license terms. The pinned
   revision is already cached for the current pilot.
3. Provision and verify COPT/coptpy separately; do not silently substitute
   Gurobi or another solver for ORLM-native results.
4. Run one local inference instance and static-validate its generated code.
5. Only then consider the fixed pilot manifest and larger evaluation set.

## Current pilot handoff

- tmux session: `orlm_pilot_official_20260813_corrected`
- command: `export CUDA_VISIBLE_DEVICES=0; python -u -m scripts.run_orlm_inference --subset pilot --output /home/soroush/combinatorial-opt-agent/results/orlm/pilot_official_checkpoint/results.jsonl --model-id CardinalOperations/ORLM-LLaMA-3-8B --model-revision 94fdc3c5738c6536d4880dc19a78f215529181c5 --max-new-tokens 8192 --device-map auto --dtype bfloat16 --device cuda:0`
- start: `2026-08-13T23:06:54-04:00`
- log: `results/orlm/pilot_official_checkpoint/inference_corrected.log`
- output: `results/orlm/pilot_official_checkpoint/results.jsonl`
- Git SHA: `6bb75a4c4bed02c458ac30b4af206a2802fce095`
- health check: passed after approximately three minutes; checkpoint loaded
  with CPU offload and GPU memory remained below 15.3 GiB. At the later
  handoff inspection, one real row (`problem_id=14`) had completed with
  `CODE_EXTRACTED` and `STATIC_VALID`; the other five rows remain pending.
