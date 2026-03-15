"""
Lightweight validation for problem schema, formulation structure, Python code
syntax, and LP/ILP logical consistency.

Robust to missing optional fields; never raises — returns list of error strings.
"""
from __future__ import annotations

import ast
from typing import Any

# Valid optimization directions for the objective.
_VALID_SENSES = frozenset({"minimize", "maximize", "min", "max"})


def verify_problem_schema(problem: dict) -> list[str]:
    """
    Validate problem against unified schema: id, name, description, formulation
    (and formulation.variables, .objective, .constraints). Optional: formulation_latex, complexity, source.
    Returns list of error strings; empty list = pass. Never raises.
    """
    errors: list[str] = []
    if not isinstance(problem, dict):
        return ["problem must be a dict"]
    # Required top-level
    if not problem.get("id"):
        errors.append("missing or empty 'id'")
    if "name" not in problem:
        errors.append("missing 'name'")
    if "description" not in problem:
        errors.append("missing 'description'")
    if "formulation" not in problem:
        errors.append("missing 'formulation'")
    else:
        form = problem.get("formulation")
        if not isinstance(form, dict):
            errors.append("'formulation' must be a dict")
    return errors


def verify_formulation_structure(problem: dict) -> list[str]:
    """
    Check formulation has variables (non-empty list), objective (dict with sense/expression),
    constraints (list). Returns list of error strings; empty = pass. Never raises.
    """
    errors: list[str] = []
    if not isinstance(problem, dict):
        return ["problem must be a dict"]
    form = problem.get("formulation")
    if form is None:
        errors.append("formulation missing")
        return errors
    if not isinstance(form, dict):
        errors.append("formulation must be a dict")
        return errors
    # variables: list, non-empty when we care (some problems may have placeholder)
    if "variables" not in form:
        errors.append("formulation.variables missing")
    else:
        v = form.get("variables")
        if not isinstance(v, list):
            errors.append("formulation.variables must be a list")
        elif len(v) == 0:
            errors.append("formulation.variables is empty")
    # objective: dict with sense and expression
    if "objective" not in form:
        errors.append("formulation.objective missing")
    else:
        obj = form.get("objective")
        if not isinstance(obj, dict):
            errors.append("formulation.objective must be a dict")
        else:
            if "sense" not in obj:
                errors.append("formulation.objective.sense missing")
            if "expression" not in obj:
                errors.append("formulation.objective.expression missing")
    # constraints: list (can be empty for unconstrained)
    if "constraints" not in form:
        errors.append("formulation.constraints missing")
    else:
        c = form.get("constraints")
        if not isinstance(c, list):
            errors.append("formulation.constraints must be a list")
    return errors


def verify_lp_consistency(problem: dict) -> list[str]:
    """Check LP/ILP logical consistency of a well-structured formulation.

    Assumes ``verify_formulation_structure`` already passes (i.e. the required
    keys are present and have the right types).  Checks:

    * Objective sense is one of the accepted direction words (``minimize``,
      ``maximize``, ``min``, ``max``).
    * Every variable entry (if a list of dicts) has a non-empty ``symbol``
      field.
    * No two variables share the same ``symbol`` (duplicate-variable check).
    * Every constraint entry (if a list of dicts) has a non-empty
      ``expression`` field.

    Returns a list of error strings; empty list = all checks pass.
    Never raises.
    """
    errors: list[str] = []
    if not isinstance(problem, dict):
        return ["problem must be a dict"]
    try:
        form = problem.get("formulation")
        if not isinstance(form, dict):
            return []  # structural check not passed — skip LP consistency

        # --- objective sense ---
        obj = form.get("objective")
        if isinstance(obj, dict):
            sense = obj.get("sense", "")
            if isinstance(sense, str) and sense.strip().lower() not in _VALID_SENSES:
                errors.append(
                    f"formulation.objective.sense '{sense}' is not a recognized "
                    f"direction; expected one of {sorted(_VALID_SENSES)}"
                )

        # --- variables ---
        variables = form.get("variables")
        if isinstance(variables, list):
            symbols_seen: set[str] = set()
            for idx, var in enumerate(variables):
                if not isinstance(var, dict):
                    continue
                symbol = var.get("symbol", "")
                if not isinstance(symbol, str) or not symbol.strip():
                    errors.append(
                        f"formulation.variables[{idx}] has a missing or empty 'symbol'"
                    )
                else:
                    key = symbol.strip()
                    if key in symbols_seen:
                        errors.append(
                            f"formulation.variables: duplicate variable symbol '{key}'"
                        )
                    else:
                        symbols_seen.add(key)

        # --- constraints ---
        constraints = form.get("constraints")
        if isinstance(constraints, list):
            for idx, con in enumerate(constraints):
                if not isinstance(con, dict):
                    continue
                expr = con.get("expression", "")
                if not isinstance(expr, str) or not expr.strip():
                    errors.append(
                        f"formulation.constraints[{idx}] has a missing or empty "
                        f"'expression'"
                    )
    except Exception as exc:
        errors.append(f"verify_lp_consistency: unexpected error: {exc}")
    return errors


def verify_python_syntax(code: str) -> list[str]:
    """
    Check Python code for syntax errors using ast.parse.
    Returns list of error strings (e.g. "line 3: invalid syntax"); empty = pass. Never raises.
    """
    errors: list[str] = []
    if code is None:
        return ["code is None"]
    if not isinstance(code, str):
        return ["code must be a string"]
    s = code.strip()
    if not s:
        return []  # empty is valid (no code)
    try:
        ast.parse(s)
    except SyntaxError as e:
        msg = str(e.msg) if e.msg else "invalid syntax"
        if e.lineno is not None:
            errors.append(f"line {e.lineno}: {msg}")
        else:
            errors.append(msg)
    except Exception as e:
        errors.append(f"parse error: {e}")
    return errors


def run_all_problem_checks(problem: dict) -> dict[str, list[str]]:
    """Run schema, formulation-structure, and LP-consistency checks.

    Returns a dict with keys:
    * ``schema_errors`` — required-field and type checks.
    * ``formulation_errors`` — structural checks (variables/objective/constraints present).
    * ``lp_consistency_errors`` — LP/ILP logical consistency (valid sense, no duplicate
      variables, non-empty constraint expressions).

    Convenience for ``search()`` integration — the ``validate`` flag in
    ``answer()`` surfaces these errors directly to the user.
    """
    schema_errors: list[str] = []
    formulation_errors: list[str] = []
    lp_consistency_errors: list[str] = []
    try:
        schema_errors = verify_problem_schema(problem)
    except Exception:
        schema_errors = ["verify_problem_schema raised an exception"]
    try:
        formulation_errors = verify_formulation_structure(problem)
    except Exception:
        formulation_errors = ["verify_formulation_structure raised an exception"]
    try:
        lp_consistency_errors = verify_lp_consistency(problem)
    except Exception:
        lp_consistency_errors = ["verify_lp_consistency raised an exception"]
    return {
        "schema_errors": schema_errors,
        "formulation_errors": formulation_errors,
        "lp_consistency_errors": lp_consistency_errors,
    }


