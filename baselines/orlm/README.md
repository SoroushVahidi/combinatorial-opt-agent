# ORLM baseline (implemented, inference-ready)

**Status: `ORLM_IMPLEMENTED_READY_FOR_INFERENCE`, 2026-08-12.** No model
weights were downloaded and no inference or COPT execution was run. The
lightweight adapter, official prompt builder, lazy Transformers runner,
normalizer, static validator, safe execution harness, result schema,
evaluator, resume store, fixed manifest, and mocked end-to-end tests are
complete.

## Primary sources

- Paper: Huang et al., [ORLM](https://arxiv.org/abs/2405.17743).
- Official code: [Cardinal-Operations/ORLM](https://github.com/Cardinal-Operations/ORLM).
- Verified upstream revision: `33bc47d0a1d1710d24ab839118bdf4cb89b9e31b`.
- Official checkpoint: [CardinalOperations/ORLM-LLaMA-3-8B](https://huggingface.co/CardinalOperations/ORLM-LLaMA-3-8B).
- Code license: Apache-2.0. The checkpoint has a Llama 3 license; review it
  before redistribution or deployment.

The upstream code uses vLLM with a local model path for generation and COPT
(`coptpy`) for execution. The current repository does not vendor upstream
code, download weights, or expose any license contents.

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
`top_p=1`, and stops on `</s>`. The local configuration records these
settings explicitly. The runner also supports an injected backend for tests
and a lazy Transformers backend for future local inference; importing it does
not load the 8B checkpoint.

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
- `manifests/nlp4lp_common_manifest.json` — fixed six-instance pilot and 18-instance future subset.

## Future inference prerequisites

1. Provision an isolated environment with compatible PyTorch/Transformers or
   the upstream vLLM path.
2. Obtain the checkpoint and verify its revision/license terms.
3. Provision and verify COPT/coptpy separately; do not silently substitute
   Gurobi or another solver for ORLM-native results.
4. Run one local inference instance and static-validate its generated code.
5. Only then consider the fixed pilot manifest and larger evaluation set.

No GPU-heavy inference, model download, or solver benchmark was performed in
the current implementation pass.
