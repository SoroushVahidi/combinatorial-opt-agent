"""Static validation for inferred Pyomo output; generated code is never run."""
from __future__ import annotations
import ast
from dataclasses import dataclass

@dataclass(frozen=True)
class StaticValidation:
    status: str
    syntax_valid: bool
    pyomo_import_present: bool
    model_creation_present: bool
    objective_present: bool
    constraints_present: bool
    solve_call_present: bool
    dangerous_operations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    def to_dict(self): return {"status": self.status, "syntax_valid": self.syntax_valid, "pyomo_import_present": self.pyomo_import_present, "model_creation_present": self.model_creation_present, "objective_present": self.objective_present, "constraints_present": self.constraints_present, "solve_call_present": self.solve_call_present, "dangerous_operations": list(self.dangerous_operations), "errors": list(self.errors), "warnings": list(self.warnings)}

def validate_code(code: str | None) -> StaticValidation:
    if not code or not code.strip(): return StaticValidation("EMPTY", False, False, False, False, False, False, errors=("empty_code",))
    try: tree = ast.parse(code)
    except SyntaxError as exc: return StaticValidation("SYNTAX_INVALID", False, False, False, False, False, False, errors=(str(exc),))
    text = code.lower(); dangerous=[]
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"system", "popen", "run", "remove", "rmtree"}: dangerous.append(n.func.attr)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"eval", "exec", "compile", "__import__"}: dangerous.append(n.func.id)
    flags = ("pyomo" in text, "concreteModel".lower() in text or "abstractmodel" in text, "objective(" in text or ".obj" in text, "constraint(" in text or "add_constraint" in text, ".solve(" in text or ".optimize(" in text)
    names=("missing_pyomo_import", "missing_model_creation_signal", "missing_objective_signal", "missing_constraint_signal", "missing_solve_call")
    errors=tuple(name for name, ok in zip(names, flags) if not ok)
    if dangerous: errors += ("dangerous_operation_present",)
    return StaticValidation("STATIC_VALID" if not errors else "STATIC_INVALID", True, *flags, tuple(sorted(set(dangerous))), errors, ())
