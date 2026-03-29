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
    """run_all_problem_checks returns schema_errors, formulation_errors, and lp_consistency_errors."""
    from formulation.verify import run_all_problem_checks

    out = run_all_problem_checks({})
    assert "schema_errors" in out and "formulation_errors" in out
    assert "lp_consistency_errors" in out
    assert isinstance(out["schema_errors"], list)
    assert isinstance(out["formulation_errors"], list)
    assert isinstance(out["lp_consistency_errors"], list)


# ---------------------------------------------------------------------------
# verify_lp_consistency tests
# ---------------------------------------------------------------------------

def _toy_problem(
    sense: str = "minimize",
    variables: list | None = None,
    constraints: list | None = None,
) -> dict:
    if variables is None:
        variables = [{"symbol": "x", "description": "decision variable", "domain": "x >= 0"}]
    if constraints is None:
        constraints = [{"expression": "x >= 0", "description": "nonneg"}]
    return {
        "id": "toy",
        "name": "Toy",
        "description": "Toy.",
        "formulation": {
            "variables": variables,
            "objective": {"sense": sense, "expression": "x"},
            "constraints": constraints,
        },
    }


class TestVerifyLpConsistency:
    """verify_lp_consistency catches LP/ILP structural inconsistencies."""

    def test_valid_minimize_problem_passes(self):
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency(_toy_problem("minimize")) == []

    def test_valid_maximize_problem_passes(self):
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency(_toy_problem("maximize")) == []

    def test_min_abbreviation_passes(self):
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency(_toy_problem("min")) == []

    def test_max_abbreviation_passes(self):
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency(_toy_problem("max")) == []

    def test_invalid_sense_flagged(self):
        from formulation.verify import verify_lp_consistency
        errs = verify_lp_consistency(_toy_problem("optimise"))  # British spelling not accepted
        assert any("sense" in e for e in errs), f"Expected sense error, got: {errs}"

    def test_empty_sense_flagged(self):
        from formulation.verify import verify_lp_consistency
        errs = verify_lp_consistency(_toy_problem(""))
        assert any("sense" in e for e in errs)

    def test_duplicate_variable_symbols_flagged(self):
        from formulation.verify import verify_lp_consistency
        variables = [
            {"symbol": "x", "description": "first"},
            {"symbol": "x", "description": "duplicate"},
        ]
        errs = verify_lp_consistency(_toy_problem(variables=variables))
        assert any("duplicate" in e for e in errs), f"Expected duplicate error, got: {errs}"

    def test_missing_variable_symbol_flagged(self):
        from formulation.verify import verify_lp_consistency
        variables = [{"description": "no symbol here", "domain": "x >= 0"}]
        errs = verify_lp_consistency(_toy_problem(variables=variables))
        assert any("symbol" in e for e in errs), f"Expected symbol error, got: {errs}"

    def test_empty_variable_symbol_flagged(self):
        from formulation.verify import verify_lp_consistency
        variables = [{"symbol": "   ", "description": "whitespace only"}]
        errs = verify_lp_consistency(_toy_problem(variables=variables))
        assert any("symbol" in e for e in errs)

    def test_constraint_missing_expression_flagged(self):
        from formulation.verify import verify_lp_consistency
        constraints = [{"description": "no expression field"}]
        errs = verify_lp_consistency(_toy_problem(constraints=constraints))
        assert any("expression" in e for e in errs), f"Expected expression error, got: {errs}"

    def test_constraint_empty_expression_flagged(self):
        from formulation.verify import verify_lp_consistency
        constraints = [{"expression": "", "description": "empty expression"}]
        errs = verify_lp_consistency(_toy_problem(constraints=constraints))
        assert any("expression" in e for e in errs)

    def test_empty_constraints_list_passes(self):
        """An empty constraints list is valid (unconstrained problem)."""
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency(_toy_problem(constraints=[])) == []

    def test_non_dict_problem_handled(self):
        from formulation.verify import verify_lp_consistency
        assert isinstance(verify_lp_consistency("not a dict"), list)
        assert len(verify_lp_consistency("not a dict")) > 0

    def test_missing_formulation_gracefully_skipped(self):
        """If formulation is missing, lp_consistency returns empty (structural check handles it)."""
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency({"id": "x", "name": "y", "description": "z"}) == []

    def test_run_all_checks_includes_lp_errors(self):
        """run_all_problem_checks surfaces lp_consistency_errors."""
        from formulation.verify import run_all_problem_checks
        bad = _toy_problem(sense="optimise")
        out = run_all_problem_checks(bad)
        assert any("sense" in e for e in out.get("lp_consistency_errors", [])), (
            f"Expected LP sense error in run_all_problem_checks; got: {out}"
        )

    def test_run_all_checks_clean_problem_has_no_errors(self):
        """A fully correct problem produces empty error lists for all three checks."""
        from formulation.verify import run_all_problem_checks
        out = run_all_problem_checks(_toy_problem())
        assert out["schema_errors"] == []
        assert out["formulation_errors"] == []
        assert out["lp_consistency_errors"] == []


# ---------------------------------------------------------------------------
# New tests: objective expression emptiness check
# ---------------------------------------------------------------------------

class TestObjectiveExpressionCheck:
    """verify_lp_consistency now also checks that objective.expression is non-empty."""

    def _make(self, expression) -> dict:
        return {
            "id": "t",
            "formulation": {
                "variables": [{"symbol": "x"}],
                "objective": {"sense": "minimize", "expression": expression},
                "constraints": [],
            },
        }

    def test_empty_string_expression_flagged(self):
        from formulation.verify import verify_lp_consistency
        errs = verify_lp_consistency(self._make(""))
        assert any("expression" in e and "objective" in e for e in errs), (
            f"Expected objective.expression error; got: {errs}"
        )

    def test_whitespace_only_expression_flagged(self):
        from formulation.verify import verify_lp_consistency
        errs = verify_lp_consistency(self._make("   "))
        assert any("expression" in e and "objective" in e for e in errs)

    def test_none_expression_flagged(self):
        from formulation.verify import verify_lp_consistency
        errs = verify_lp_consistency(self._make(None))
        assert any("expression" in e and "objective" in e for e in errs)

    def test_valid_expression_passes(self):
        from formulation.verify import verify_lp_consistency
        assert verify_lp_consistency(self._make("x + y")) == []

    def test_run_all_checks_surfaces_empty_expression(self):
        from formulation.verify import run_all_problem_checks
        out = run_all_problem_checks(self._make(""))
        lp_errs = out.get("lp_consistency_errors", [])
        assert any("expression" in e and "objective" in e for e in lp_errs), (
            f"run_all_problem_checks should surface objective.expression error; got: {out}"
        )

    def test_optmath_bench_stub_pattern_flagged(self):
        """Reproduces the optmath_bench stub pattern: sense present but expression empty."""
        from formulation.verify import verify_lp_consistency
        p = {
            "id": "optmath_bench_001",
            "formulation": {
                "variables": [],
                "objective": {
                    "sense": "minimize",
                    "expression": "",
                },
                "constraints": [],
            },
        }
        errs = verify_lp_consistency(p)
        # Empty expression must be flagged
        assert any("expression" in e for e in errs)

    def test_case_insensitive_sense_still_passes(self):
        """MINIMIZE and MAXIMIZE are normalised to lowercase before checking."""
        from formulation.verify import verify_lp_consistency
        p = self._make("3*x + 2*y")
        p["formulation"]["objective"]["sense"] = "MINIMIZE"
        errs = verify_lp_consistency(p)
        # Sense is case-insensitively valid and expression is present → no errors
        assert errs == []


