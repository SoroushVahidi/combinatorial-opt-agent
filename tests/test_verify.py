"""
Tests for formulation/verify.py: schema, formulation structure, Python syntax.
"""
from __future__ import annotations

import pytest


def test_missing_required_fields_triggers_errors():
    """Missing id, name, description, or formulation triggers schema errors."""
    from formulation.verify import verify_problem_schema, verify_formulation_structure

    # Empty dict
    assert "missing or empty 'id'" in verify_problem_schema({})
    assert "missing 'name'" in verify_problem_schema({})
    assert "missing 'formulation'" in verify_problem_schema({})

    # Missing formulation
    assert "missing 'formulation'" in verify_problem_schema({"id": "x", "name": "y", "description": "z"})

    # Formulation structure: missing variables/objective/constraints
    p = {"id": "x", "name": "y", "description": "z", "formulation": {}}
    assert verify_problem_schema(p) == []
    assert "formulation.variables missing" in verify_formulation_structure(p)
    assert "formulation.objective missing" in verify_formulation_structure(p)
    assert "formulation.constraints missing" in verify_formulation_structure(p)


def test_well_formed_toy_problem_passes():
    """A minimal valid problem passes schema and formulation checks."""
    from formulation.verify import verify_problem_schema, verify_formulation_structure

    problem = {
        "id": "toy",
        "name": "Toy Problem",
        "description": "A toy.",
        "formulation": {
            "variables": [{"symbol": "x", "description": "var", "domain": "x in [0,1]"}],
            "objective": {"sense": "minimize", "expression": "x"},
            "constraints": [{"expression": "x >= 0", "description": "nonneg"}],
        },
    }
    assert verify_problem_schema(problem) == []
    assert verify_formulation_structure(problem) == []


def test_python_syntax_validator_catches_errors():
    """verify_python_syntax returns errors for invalid Python."""
    from formulation.verify import verify_python_syntax

    assert verify_python_syntax("x = 1") == []
    assert verify_python_syntax("") == []
    errs = verify_python_syntax("def f(  \n  syntax error here")
    assert len(errs) >= 1
    assert "syntax" in errs[0].lower() or "line" in errs[0].lower()


def test_python_syntax_validator_robust():
    """verify_python_syntax never raises; handles None and non-string."""
    from formulation.verify import verify_python_syntax

    assert verify_python_syntax(None)  # returns list
    assert isinstance(verify_python_syntax(123), list)


def test_run_all_problem_checks():
    """run_all_problem_checks returns schema_errors and formulation_errors."""
    from formulation.verify import run_all_problem_checks

    out = run_all_problem_checks({})
    assert "schema_errors" in out and "formulation_errors" in out
    assert isinstance(out["schema_errors"], list)
    assert isinstance(out["formulation_errors"], list)
