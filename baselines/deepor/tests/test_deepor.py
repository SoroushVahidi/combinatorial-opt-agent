from baselines.deepor.data_adapter import adapt_record
from baselines.deepor.config import DeepORConfig
from baselines.deepor.prompt import build_prompt
from baselines.deepor.reasoning_parser import parse_reasoning
from baselines.deepor.output_normalizer import parse_output
from baselines.deepor.static_validation import validate_code
from baselines.deepor.pipeline import run_mock_pipeline

CODE="""from pyomo.environ import *
model=ConcreteModel()
model.x=Var()
model.obj=Objective(expr=model.x)
model.c=Constraint(expr=model.x <= 1)
SolverFactory('highs').solve(model)
"""

def test_adapter_hash_and_preserves_multiline():
    r=adapt_record({"problem_id": 14, "en_question": "line 1\nline 2", "gold_objective": 3})
    assert r.supported and r.record.raw_text == "line 1\nline 2" and len(r.record.input_sha256)==64

def test_adapter_rejects_missing_fields():
    assert not adapt_record({"problem_id": 1}).supported

def test_prompt_is_deterministic():
    assert build_prompt("x").sha256 == build_prompt("x").sha256

def test_reasoning_and_code_parse():
    out=parse_output("<think>choose a variable</think>\n```python\n"+CODE+"```")
    assert out.reasoning.reasoning == "choose a variable" and out.generated_code
    assert validate_code(out.generated_code).status == "STATIC_VALID"

def test_malformed_and_empty_outputs():
    assert parse_reasoning("").status == "EMPTY"
    assert parse_output("plain formulation").status == "FORMULATION_ONLY"

def test_mock_end_to_end_round_trip():
    result=run_mock_pipeline({"problem_id": "x", "text": "maximize production"})
    assert result.generation.status == "COMPLETED"
    assert result.static_validation.status == "STATIC_VALID"
    assert result.to_json()
