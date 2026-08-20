"""Mockable end-to-end path used for validation and future inference."""
from __future__ import annotations
from .config import DeepORConfig
from .data_adapter import adapt_record
from .prompt import build_prompt
from .runner import DeepORRunner
from .output_normalizer import parse_output
from .static_validation import validate_code
from .result_schema import DeepORResult

def run_mock_pipeline(record, *, config=None, output=None):
    config=config or DeepORConfig(model_id="DEEPOR_PIPELINE_MOCK_OR_PROXY")
    adapted=adapt_record(record)
    if not adapted.supported: raise ValueError(adapted.reason)
    prompt=build_prompt(adapted.record.raw_text, config)
    class Mock:
        def generate(self, prompt, config): return (output or "<think>Identify variables, objective, and constraints.</think>\n```python\nfrom pyomo.environ import *\nmodel=ConcreteModel()\nmodel.x=Var()\nmodel.obj=Objective(expr=model.x)\nmodel.c=Constraint(expr=model.x <= 1)\nSolverFactory('highs').solve(model)\n```", {"prompt_tokens": 10, "generated_tokens": 30})
    generation=DeepORRunner(config, Mock()).generate(prompt); parsed=parse_output(generation.raw_output); validation=validate_code(parsed.generated_code)
    failure=None if validation.status=="STATIC_VALID" else "static_validation_failure"
    return DeepORResult(adapted.record.problem_id, adapted.record.dataset, adapted.record.input_sha256, config.model_id, config.model_revision, config.paper_revision, prompt.version, prompt.sha256, generation, parsed, validation, failure_category=failure)
