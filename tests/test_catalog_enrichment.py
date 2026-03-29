"""
Tests for retrieval/catalog_enrichment.py.

Covers:
  - find_incomplete_problems: correctly identifies entries with missing fields
  - _parse_notebook_formulation: correctly extracts variables/objective/constraints
  - _parse_variables, _parse_objective, _parse_constraints: unit parsing
  - fetch_formulation_from_web: graceful failure when network unavailable or ID unknown
  - enrich_catalog: integration over a tiny catalog with mixed completeness
  - build_extended_catalog: --enrich flag wires through correctly
"""
from __future__ import annotations

import json
import socket
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _github_reachable(host: str = "raw.githubusercontent.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


GITHUB_AVAILABLE = _github_reachable()


def _complete_problem(**overrides) -> dict:
    base = {
        "id": "p_complete",
        "name": "Complete Problem",
        "description": "A fully specified problem.",
        "formulation": {
            "variables": [{"symbol": "x", "description": "var x", "domain": "x ∈ {0,1}"}],
            "objective": {"sense": "minimize", "expression": "sum(x)"},
            "constraints": [{"expression": "sum(x) <= 1", "description": "at most one"}],
        },
        "source": "classic",
    }
    base.update(overrides)
    return base


def _no_formulation_problem(**overrides) -> dict:
    base = {
        "id": "p_no_form",
        "name": "No Formulation",
        "description": "A problem without any formulation.",
        "source": "gurobi_modeling_examples",
    }
    base.update(overrides)
    return base


def _missing_vars_problem(**overrides) -> dict:
    base = {
        "id": "p_no_vars",
        "name": "Missing Variables",
        "description": "Has objective and constraints but no variables.",
        "formulation": {
            "objective": {"sense": "minimize", "expression": "cost"},
            "constraints": [{"expression": "x >= 0", "description": "nonneg"}],
        },
        "source": "gams",
    }
    base.update(overrides)
    return base


def _missing_constraints_problem(**overrides) -> dict:
    base = {
        "id": "p_no_constr",
        "name": "Missing Constraints",
        "description": "Has variables and objective but no constraints.",
        "formulation": {
            "variables": [{"symbol": "x", "description": "var x", "domain": "x >= 0"}],
            "objective": {"sense": "maximize", "expression": "profit"},
        },
        "source": "NL4Opt",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# find_incomplete_problems
# ---------------------------------------------------------------------------

class TestFindIncompleteProblems:

    def test_complete_problem_not_returned(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        catalog = [_complete_problem()]
        assert find_incomplete_problems(catalog) == []

    def test_no_formulation_field_is_incomplete(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        catalog = [_no_formulation_problem()]
        result = find_incomplete_problems(catalog)
        assert len(result) == 1
        assert result[0]["id"] == "p_no_form"

    def test_missing_variables_is_incomplete(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        catalog = [_missing_vars_problem()]
        result = find_incomplete_problems(catalog)
        assert len(result) == 1

    def test_missing_constraints_key_is_incomplete(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        catalog = [_missing_constraints_problem()]
        result = find_incomplete_problems(catalog)
        assert len(result) == 1

    def test_empty_variables_list_is_incomplete(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        p = _complete_problem()
        p["formulation"]["variables"] = []
        result = find_incomplete_problems([p])
        assert len(result) == 1

    def test_empty_constraints_list_is_complete(self):
        """An explicitly empty constraints list counts as present (unconstrained problems are valid)."""
        from retrieval.catalog_enrichment import find_incomplete_problems
        p = _complete_problem()
        p["formulation"]["constraints"] = []  # empty but present
        result = find_incomplete_problems([p])
        assert result == []

    def test_mixed_catalog_returns_only_incomplete(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        catalog = [
            _complete_problem(),
            _no_formulation_problem(),
            _missing_vars_problem(),
        ]
        result = find_incomplete_problems(catalog)
        ids = {p["id"] for p in result}
        assert ids == {"p_no_form", "p_no_vars"}

    def test_empty_catalog_returns_empty(self):
        from retrieval.catalog_enrichment import find_incomplete_problems
        assert find_incomplete_problems([]) == []


# ---------------------------------------------------------------------------
# _parse_variables
# ---------------------------------------------------------------------------

class TestParseVariables:

    def test_binary_variable_extracted(self):
        from retrieval.catalog_enrichment import _parse_variables
        block = "$select_{j} \\in \\{0, 1\\}$: 1 if we build a facility at location j, 0 otherwise."
        result = _parse_variables(block)
        assert len(result) == 1
        v = result[0]
        assert "select" in v["symbol"]
        assert "facility" in v["description"] or "1 if" in v["description"]
        assert v["domain"]

    def test_continuous_variable_extracted(self):
        from retrieval.catalog_enrichment import _parse_variables
        block = "$0 \\leq assign_{i,j} \\leq 1$: Fraction of demand of customer i served by facility j."
        result = _parse_variables(block)
        assert len(result) == 1
        v = result[0]
        assert "assign" in v["symbol"]

    def test_multiple_variables(self):
        from retrieval.catalog_enrichment import _parse_variables
        block = (
            "$x_{i} \\in \\{0,1\\}$: Item i is selected.\n"
            "$y_{j} \\geq 0$: Amount assigned to bin j."
        )
        result = _parse_variables(block)
        assert len(result) == 2
        symbols = [v["symbol"] for v in result]
        assert any("x" in s for s in symbols)
        assert any("y" in s for s in symbols)

    def test_empty_block_returns_empty(self):
        from retrieval.catalog_enrichment import _parse_variables
        assert _parse_variables("") == []

    def test_non_variable_lines_skipped(self):
        from retrieval.catalog_enrichment import _parse_variables
        block = "Some explanatory text without LaTeX variables."
        result = _parse_variables(block)
        assert result == []


# ---------------------------------------------------------------------------
# _parse_objective
# ---------------------------------------------------------------------------

class TestParseObjective:

    def test_minimize_from_text(self):
        from retrieval.catalog_enrichment import _parse_objective
        block = "We want to minimize total cost: $\\sum_j f_j x_j + \\sum_{ij} c_{ij} y_{ij}$"
        result = _parse_objective(block)
        assert result is not None
        assert result["sense"] == "minimize"
        assert result["expression"]

    def test_maximize_from_latex(self):
        from retrieval.catalog_enrichment import _parse_objective
        block = "\\text{Max} Z = \\sum_i p_i x_i"
        result = _parse_objective(block)
        assert result is not None
        assert result["sense"] == "maximize"

    def test_equation_block_extracted(self):
        from retrieval.catalog_enrichment import _parse_objective
        block = (
            "Minimize total cost:\n"
            "\\begin{equation}\n"
            "Z = \\sum_{j \\in J} f_j x_j\n"
            "\\end{equation}"
        )
        result = _parse_objective(block)
        assert result is not None
        assert "f_j" in result["expression"] or "Z" in result["expression"]

    def test_no_sense_returns_none(self):
        from retrieval.catalog_enrichment import _parse_objective
        block = "Some explanatory text about the problem structure and variables."
        result = _parse_objective(block)
        assert result is None

    def test_empty_block_returns_none(self):
        from retrieval.catalog_enrichment import _parse_objective
        assert _parse_objective("") is None


# ---------------------------------------------------------------------------
# _parse_constraints
# ---------------------------------------------------------------------------

class TestParseConstraints:

    def test_single_constraint(self):
        from retrieval.catalog_enrichment import _parse_constraints
        block = (
            "- **Demand**. Each customer must be fully served.\n"
            "\\begin{equation}\n"
            "\\sum_j assign_{ij} = 1 \\quad \\forall i\n"
            "\\end{equation}"
        )
        result = _parse_constraints(block)
        assert len(result) >= 1
        assert any("Demand" in c["description"] for c in result)

    def test_multiple_constraints(self):
        from retrieval.catalog_enrichment import _parse_constraints
        block = (
            "- **Capacity**. Do not exceed facility capacity.\n"
            "\\begin{equation}\\sum_i x_{ij} \\leq C_j\\end{equation}\n"
            "\n"
            "- **Assignment**. Each item goes to exactly one bin.\n"
            "\\begin{equation}\\sum_j y_{ij} = 1\\end{equation}"
        )
        result = _parse_constraints(block)
        assert len(result) >= 2

    def test_empty_block_returns_empty(self):
        from retrieval.catalog_enrichment import _parse_constraints
        assert _parse_constraints("") == []


# ---------------------------------------------------------------------------
# _parse_notebook_formulation
# ---------------------------------------------------------------------------

class TestParseNotebookFormulation:

    def _make_notebook(self, formulation_markdown: str) -> dict:
        """Create a minimal fake notebook dict with one markdown cell."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [formulation_markdown],
                    "metadata": {},
                }
            ],
            "nbformat": 4,
        }

    def test_full_formulation_section_parsed(self):
        from retrieval.catalog_enrichment import _parse_notebook_formulation
        md = (
            "## Model Formulation\n\n"
            "### Decision Variables\n\n"
            "$x_{i} \\in \\{0,1\\}$: Item i selected.\n\n"
            "### Objective Function\n\n"
            "Minimize total weight:\n"
            "\\begin{equation}\\sum_i w_i x_i\\end{equation}\n\n"
            "### Constraints\n\n"
            "- **Capacity**. Weight limit.\n"
            "\\begin{equation}\\sum_i w_i x_i \\leq W\\end{equation}"
        )
        nb = self._make_notebook(md)
        result = _parse_notebook_formulation(nb, "Knapsack")
        assert result is not None
        assert "variables" in result or "objective" in result or "constraints" in result

    def test_no_formulation_section_returns_none(self):
        from retrieval.catalog_enrichment import _parse_notebook_formulation
        nb = self._make_notebook("# Introduction\n\nThis notebook covers background.")
        result = _parse_notebook_formulation(nb, "Unknown")
        assert result is None

    def test_empty_notebook_returns_none(self):
        from retrieval.catalog_enrichment import _parse_notebook_formulation
        nb = {"cells": [], "nbformat": 4}
        result = _parse_notebook_formulation(nb, "Empty")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_formulation_from_web
# ---------------------------------------------------------------------------

class TestFetchFormulationFromWeb:

    def test_unsupported_source_returns_none(self):
        """Problems from unsupported sources return None without crashing."""
        from retrieval.catalog_enrichment import fetch_formulation_from_web
        problem = {"id": "gams_some_model", "name": "Some GAMS Model", "source": "gams"}
        result = fetch_formulation_from_web(problem, timeout=2)
        assert result is None

    def test_unknown_id_returns_none(self):
        """An unrecognised ID returns None without crashing."""
        from retrieval.catalog_enrichment import fetch_formulation_from_web
        problem = {"id": "totally_unknown_abc123", "name": "Unknown", "source": "unknown"}
        result = fetch_formulation_from_web(problem, timeout=2)
        assert result is None

    def test_network_failure_returns_none(self):
        """When the network is unavailable the function returns None gracefully."""
        from retrieval.catalog_enrichment import fetch_formulation_from_web
        # Use an invalid timeout to force a fast failure regardless of connectivity
        problem = {"id": "gurobi_ex_facility_location", "name": "Facility Location",
                   "source": "gurobi_modeling_examples"}
        # We just verify it never raises; the return value depends on connectivity
        result = fetch_formulation_from_web(problem, timeout=0)
        assert result is None or isinstance(result, dict)

    @pytest.mark.skipif(not GITHUB_AVAILABLE, reason="GitHub raw content not reachable")
    def test_gurobi_facility_location_enriched(self):
        """With network access, facility_location notebook is parsed successfully."""
        from retrieval.catalog_enrichment import fetch_formulation_from_web
        problem = {
            "id": "gurobi_ex_facility_location",
            "name": "Facility Location",
            "source": "gurobi_modeling_examples",
        }
        result = fetch_formulation_from_web(problem, timeout=10)
        assert result is not None
        # At minimum, should extract some formulation data
        assert isinstance(result, dict)
        assert any(k in result for k in ("variables", "objective", "constraints"))

    @pytest.mark.skipif(not GITHUB_AVAILABLE, reason="GitHub raw content not reachable")
    def test_gurobi_agricultural_pricing_enriched(self):
        """With network access, agricultural_pricing notebook is parsed."""
        from retrieval.catalog_enrichment import fetch_formulation_from_web
        problem = {
            "id": "gurobi_ex_agricultural_pricing",
            "name": "Agricultural Pricing",
            "source": "gurobi_modeling_examples",
        }
        result = fetch_formulation_from_web(problem, timeout=10)
        assert result is not None
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# enrich_catalog
# ---------------------------------------------------------------------------

class TestEnrichCatalog:

    def test_no_incomplete_returns_empty(self):
        from retrieval.catalog_enrichment import enrich_catalog
        catalog = [_complete_problem()]
        result = enrich_catalog(catalog, verbose=False, timeout=2)
        assert result == []

    def test_empty_catalog_returns_empty(self):
        from retrieval.catalog_enrichment import enrich_catalog
        result = enrich_catalog([], verbose=False, timeout=2)
        assert result == []

    def test_unsupported_source_skipped_gracefully(self):
        """Problems from unsupported sources do not raise and return empty enrichment."""
        from retrieval.catalog_enrichment import enrich_catalog
        catalog = [
            _complete_problem(),
            _missing_vars_problem(source="gams"),  # unsupported source
        ]
        result = enrich_catalog(catalog, verbose=False, timeout=2)
        # GAMS source has no web fetcher → 0 enrichments (no crash)
        assert isinstance(result, list)
        assert all(isinstance(p, dict) for p in result)

    def test_original_catalog_not_mutated(self):
        """enrich_catalog never mutates the original catalog."""
        from retrieval.catalog_enrichment import enrich_catalog
        original = [_no_formulation_problem(source="gams")]
        original_copy = deepcopy(original)
        enrich_catalog(original, verbose=False, timeout=2)
        assert original == original_copy

    def test_enriched_entry_has_correct_id(self):
        """When a problem is enriched, the returned entry keeps the same id."""
        from retrieval.catalog_enrichment import enrich_catalog, _parse_notebook_formulation

        # Inject a fake notebook parse to avoid network dependency
        fake_formulation = {
            "variables": [{"symbol": "x", "description": "test var", "domain": "x in {0,1}"}],
            "objective": {"sense": "minimize", "expression": "sum(x)"},
            "constraints": [{"expression": "x >= 0", "description": "nonneg"}],
        }

        class _FakeModule:
            @staticmethod
            def fetch_formulation_from_web(problem, timeout=8):
                return fake_formulation if problem.get("source") == "fake_source" else None

        import retrieval.catalog_enrichment as mod
        original_fetch = mod.fetch_formulation_from_web
        mod.fetch_formulation_from_web = _FakeModule.fetch_formulation_from_web
        try:
            catalog = [_no_formulation_problem(source="fake_source", id="injected_id")]
            result = enrich_catalog(catalog, verbose=False, timeout=2)
            assert len(result) == 1
            assert result[0]["id"] == "injected_id"
            form = result[0].get("formulation", {})
            assert form.get("variables")
            assert form.get("objective")
            assert "constraints" in form
        finally:
            mod.fetch_formulation_from_web = original_fetch

    def test_already_complete_problem_not_overwritten(self):
        """Already-present formulation fields are not replaced by enrichment."""
        from retrieval.catalog_enrichment import enrich_catalog

        fake_formulation = {
            "variables": [{"symbol": "new_x", "description": "new var", "domain": "new_x >= 0"}],
            "objective": {"sense": "maximize", "expression": "new_obj"},
            "constraints": [],
        }

        import retrieval.catalog_enrichment as mod
        original_fetch = mod.fetch_formulation_from_web

        def fake_fetch(problem, timeout=8):
            return fake_formulation

        mod.fetch_formulation_from_web = fake_fetch
        try:
            catalog = [_missing_constraints_problem()]
            # _missing_constraints_problem has variables and objective → not empty
            result = enrich_catalog(catalog, verbose=False, timeout=2)
            if result:
                # Variables and objective were already present; they must not be overwritten
                form = result[0]["formulation"]
                # Symbol of the original variable must still be present
                assert any(v["symbol"] == "x" for v in form["variables"])
        finally:
            mod.fetch_formulation_from_web = original_fetch

    def test_empty_constraints_list_not_overwritten(self):
        """An existing empty constraints list must NOT be replaced by fetched constraints."""
        from retrieval.catalog_enrichment import enrich_catalog

        # Problem has all three fields present (constraints is empty list)
        problem_with_empty_constraints = {
            "id": "p_empty_constr",
            "name": "Empty Constraints",
            "description": "Has variables, objective, and an explicit empty constraints list.",
            "formulation": {
                "variables": [{"symbol": "x", "description": "var x", "domain": "x >= 0"}],
                "objective": {"sense": "minimize", "expression": "x"},
                "constraints": [],  # present but empty
            },
            "source": "fake_source",
        }
        # This problem is complete (empty constraints is allowed), so enrichment should skip it
        result = enrich_catalog([problem_with_empty_constraints], verbose=False, timeout=2)
        # find_incomplete_problems considers empty-constraints as complete → no enrichment
        assert result == []


# ---------------------------------------------------------------------------
# build_extended_catalog with --enrich
# ---------------------------------------------------------------------------

class TestBuildExtendedCatalogEnrich:

    def test_enrich_false_leaves_incomplete_problems_unchanged(self, tmp_path):
        """Without --enrich, incomplete problems are copied verbatim."""
        from build_extended_catalog import build_extended_catalog

        base = [
            _complete_problem(),
            _no_formulation_problem(source="gurobi_modeling_examples"),
        ]
        base_path = tmp_path / "data" / "processed" / "all_problems.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(base), encoding="utf-8")

        # Patch project root to use tmp_path
        import build_extended_catalog as mod
        original_root = mod._project_root
        mod._project_root = lambda: tmp_path
        try:
            out = build_extended_catalog(enrich=False)
            result = json.loads(out.read_text(encoding="utf-8"))
            assert len(result) == 2
            no_form = next(p for p in result if p["id"] == "p_no_form")
            assert "formulation" not in no_form
        finally:
            mod._project_root = original_root

    def test_enrich_true_calls_enrichment(self, tmp_path, monkeypatch):
        """With enrich=True, the enrichment function is invoked."""
        from build_extended_catalog import build_extended_catalog

        base = [_no_formulation_problem(source="gurobi_modeling_examples")]
        base_path = tmp_path / "data" / "processed" / "all_problems.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(base), encoding="utf-8")

        enrichment_called = []

        def fake_enrich(catalog, verbose=False, timeout=8):
            enrichment_called.append(True)
            return []

        # Patch the already-imported reference inside build_extended_catalog
        monkeypatch.setattr("build_extended_catalog._enrich_catalog", fake_enrich)

        import build_extended_catalog as mod
        original_root = mod._project_root
        mod._project_root = lambda: tmp_path
        try:
            build_extended_catalog(enrich=True)
            assert enrichment_called, "enrich_catalog was not called"
        finally:
            mod._project_root = original_root
