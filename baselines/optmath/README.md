# OptMATH baseline (implemented, inference-ready)

Status: `OPTMATH_IMPLEMENTED_READY_FOR_INFERENCE`.

This package prepares the released `Aurora-Gem/OptMATH-Qwen2.5-7B` checkpoint
for a deterministic NLP4LP adaptation without downloading weights, running
inference, or executing Gurobi code. The official upstream target is Python
`gurobipy`, not COPT.

Primary sources and the component fidelity table are in
[`docs/OPTMATH_PROVENANCE.md`](../../docs/OPTMATH_PROVENANCE.md).

Implemented modules:

- `config.py`, `prompt.py`: official checkpoint, prompt, decoding, and provenance.
- `data_adapter.py`: stable NLP4LP records with explicit unsupported rows.
- `runner.py`: lazy Transformers backend with injectable mock and batch interface.
- `output_normalizer.py`: raw-preserving Gurobi code extraction.
- `static_validation.py`: non-executing Python/Gurobi shape and safety checks.
- `execution_harness.py`: opt-in isolated Gurobi subprocess harness, dry-run by default.
- `result_schema.py`, `evaluator.py`, `pipeline.py`: results, proxy metrics, and resume support.
- `manifests/nlp4lp_common_manifest.json`: fixed six-instance pilot and 18-instance future set.

ORLM/PaMOP scalar metrics are not conflated with OptMATH full formulation
accuracy. No empirical OptMATH result exists yet.
