# Validation layer and paper-grade analysis — paths, signatures, commands

## 1) Formulation schema validation

### File: `formulation/verify.py`

**Signatures:**

```python
def verify_problem_schema(problem: dict) -> list[str]:
    """Returns list of error strings; empty = pass. Validates id, name, description, formulation."""

def verify_formulation_structure(problem: dict) -> list[str]:
    """Returns list of error strings; empty = pass. Checks formulation.variables (non-empty list),
       formulation.objective (dict with sense, expression), formulation.constraints (list)."""

def verify_python_syntax(code: str) -> list[str]:
    """Uses ast.parse; returns syntax error strings; empty = pass. No Pyomo dependency."""

def run_all_problem_checks(problem: dict) -> dict[str, list[str]]:
    """Returns {"schema_errors": [...], "formulation_errors": [...]} for search() integration."""
```

- Schema: required `id`, `name`, `description`, `formulation`; optional `formulation_latex`, `complexity`, `source`.
- Robust to missing optional fields; never raises (returns list of errors).

---

## 2) Code validation stub

- **`formulation/verify.py`**: `verify_python_syntax(code: str) -> list[str]` — uses `ast.parse` only; no Pyomo.

---

## 3) Integration into user-facing path

### `retrieval/search.py`

- **`search(..., validate: bool = False)`**. When `validate=True`, each returned problem dict gets:
  - `problem["_validation"]` = `{"schema_errors": [...], "formulation_errors": [...]}` (from `run_all_problem_checks`).

### `app.py`

- **Checkbox** "Validate outputs" (`validate_in`) added; passed as third argument to `answer(query, top_k, validate)`.
- **Display**: For each result, if `validate` is True and `problem.get("_validation")` exists:
  - If errors: `**Validation:** ⚠ N issue(s): ...` (first 3 errors).
  - If no errors: `**Validation:** ✓ Schema and formulation OK`.

---

## 4) Analysis scripts (paper artifacts)

### File: `formulation/run_verify_catalog.py`

- Iterates all problems in catalog; runs `verify_problem_schema` and `verify_formulation_structure`.
- **Writes:** `results/validation_catalog.jsonl` — one JSON object per line: `id`, `source`, `num_errors`, `error_types`.
- **Prints:** Pass rate overall, pass rate by source, top-10 most common errors.

### File: `training/plot_baselines.py`

- Reads `results/baselines_test.csv`.
- **Writes:** `results/baselines_barplot.png`, `results/baselines_table.tex` (LaTeX tabular).

---

## 5) Tests

- **`tests/test_verify.py`**: Missing required fields → errors; well-formed toy problem → pass; Python syntax validator catches invalid code and is robust to None/non-string; `run_all_problem_checks` returns both error lists.
- **`tests/test_app_validation_toggle.py`**: `answer()` has three parameters including `validate`.

---

## Commands to produce artifacts

### Validation catalog + summary

```bash
# Default catalog (from retrieval.search._load_catalog)
python -m formulation.run_verify_catalog --results-dir results

# Full catalog
python -m formulation.run_verify_catalog --catalog data/processed/all_problems.json --results-dir results
```

**Outputs:** `results/validation_catalog.jsonl` + printed summary (pass rate overall, by source, top-10 errors).

### Baselines bar plot + LaTeX table

```bash
python -m training.plot_baselines --csv results/baselines_test.csv --results-dir results
```

**Outputs:** `results/baselines_barplot.png`, `results/baselines_table.tex`.

(Requires `matplotlib` for the bar plot; if missing, only the `.tex` file is written.)

### Run tests

```bash
python -m pytest tests/test_verify.py tests/test_app_validation_toggle.py -v
```
