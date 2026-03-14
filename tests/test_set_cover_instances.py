"""
Tests for set cover problem instances.

Creates several concrete instances of the Set Cover Problem and verifies that
the program handles them correctly:
  - Schema and formulation-structure validation pass
  - format_problem_and_ip produces well-structured output
  - Short-query expansion activates for set-cover keyword queries
  - Text baselines can rank set-cover descriptions above unrelated problems
  - TestSetCoverProgramOutput records the actual rendered output the program
    produces for each instance (the "result" of running the program on them)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Concrete set cover instances used across all test classes
# ---------------------------------------------------------------------------

def _make_minimum_set_cover() -> dict:
    """Classic unweighted set cover: cover universe {1..5} with minimum sets."""
    return {
        "id": "set_cover_instance_minimum",
        "name": "Minimum Set Cover Instance",
        "description": (
            "Universe U = {1, 2, 3, 4, 5}. "
            "Available subsets: S1 = {1, 2, 3}, S2 = {2, 4}, S3 = {3, 4, 5}. "
            "Choose the minimum number of subsets so that every element in U "
            "belongs to at least one chosen subset."
        ),
        "formulation": {
            "variables": [
                {
                    "symbol": "x_S",
                    "description": "1 if subset S is selected, 0 otherwise",
                    "domain": "x_S ∈ {0,1} for each subset S ∈ {S1, S2, S3}",
                }
            ],
            "objective": {
                "sense": "minimize",
                "expression": "x_S1 + x_S2 + x_S3  (total number of subsets chosen)",
            },
            "constraints": [
                {
                    "expression": "x_S1 >= 1",
                    "description": "Element 1 must be covered (only S1 contains 1)",
                },
                {
                    "expression": "x_S1 + x_S2 >= 1",
                    "description": "Element 2 must be covered (S1 or S2)",
                },
                {
                    "expression": "x_S1 + x_S3 >= 1",
                    "description": "Element 3 must be covered (S1 or S3)",
                },
                {
                    "expression": "x_S2 + x_S3 >= 1",
                    "description": "Element 4 must be covered (S2 or S3)",
                },
                {
                    "expression": "x_S3 >= 1",
                    "description": "Element 5 must be covered (only S3 contains 5)",
                },
            ],
        },
        "complexity": "NP-hard",
        "source": "instance",
    }


def _make_weighted_set_cover() -> dict:
    """Weighted set cover: cover universe {a, b, c, d} at minimum cost."""
    return {
        "id": "set_cover_instance_weighted",
        "name": "Weighted Set Cover Instance",
        "description": (
            "Universe U = {a, b, c, d}. "
            "Available subsets with costs: S1 = {a, b} cost 3, S2 = {b, c} cost 2, "
            "S3 = {c, d} cost 4, S4 = {a, d} cost 5. "
            "Choose subsets to cover all elements at minimum total cost."
        ),
        "formulation": {
            "variables": [
                {
                    "symbol": "x_S",
                    "description": "1 if subset S is chosen, 0 otherwise",
                    "domain": "x_S ∈ {0,1} for S ∈ {S1, S2, S3, S4}",
                }
            ],
            "objective": {
                "sense": "minimize",
                "expression": "3 x_S1 + 2 x_S2 + 4 x_S3 + 5 x_S4  (total cost)",
            },
            "constraints": [
                {
                    "expression": "x_S1 + x_S4 >= 1",
                    "description": "Element a must be covered (S1 or S4)",
                },
                {
                    "expression": "x_S1 + x_S2 >= 1",
                    "description": "Element b must be covered (S1 or S2)",
                },
                {
                    "expression": "x_S2 + x_S3 >= 1",
                    "description": "Element c must be covered (S2 or S3)",
                },
                {
                    "expression": "x_S3 + x_S4 >= 1",
                    "description": "Element d must be covered (S3 or S4)",
                },
            ],
        },
        "formulation_latex": (
            r"\min \; 3x_{S1}+2x_{S2}+4x_{S3}+5x_{S4} "
            r"\quad \text{s.t.} \quad "
            r"x_{S1}+x_{S4}\ge1,\; x_{S1}+x_{S2}\ge1,\; "
            r"x_{S2}+x_{S3}\ge1,\; x_{S3}+x_{S4}\ge1,\; "
            r"x_{S_i}\in\{0,1\}."
        ),
        "complexity": "NP-hard",
        "source": "instance",
    }


def _make_hospital_coverage() -> dict:
    """Real-world scenario: open minimum hospitals to cover all cities."""
    return {
        "id": "set_cover_instance_hospital",
        "name": "Hospital Coverage Instance",
        "description": (
            "A regional health authority must ensure that every one of 6 cities "
            "(C1–C6) is served by at least one hospital. Three candidate hospital "
            "sites are available: H1 can serve {C1, C2, C3}, H2 can serve {C2, C4, C5}, "
            "H3 can serve {C3, C5, C6}. Minimize the number of hospitals opened."
        ),
        "formulation": {
            "variables": [
                {
                    "symbol": "y_h",
                    "description": "1 if hospital h is opened, 0 otherwise",
                    "domain": "y_h ∈ {0,1} for h ∈ {H1, H2, H3}",
                }
            ],
            "objective": {
                "sense": "minimize",
                "expression": "y_H1 + y_H2 + y_H3  (number of hospitals opened)",
            },
            "constraints": [
                {
                    "expression": "y_H1 >= 1",
                    "description": "City C1 must be covered (only H1 reaches C1)",
                },
                {
                    "expression": "y_H1 + y_H2 >= 1",
                    "description": "City C2 must be covered (H1 or H2)",
                },
                {
                    "expression": "y_H1 + y_H3 >= 1",
                    "description": "City C3 must be covered (H1 or H3)",
                },
                {
                    "expression": "y_H2 >= 1",
                    "description": "City C4 must be covered (only H2 reaches C4)",
                },
                {
                    "expression": "y_H2 + y_H3 >= 1",
                    "description": "City C5 must be covered (H2 or H3)",
                },
                {
                    "expression": "y_H3 >= 1",
                    "description": "City C6 must be covered (only H3 reaches C6)",
                },
            ],
        },
        "complexity": "NP-hard",
        "source": "instance",
    }


ALL_INSTANCES = [
    _make_minimum_set_cover(),
    _make_weighted_set_cover(),
    _make_hospital_coverage(),
]


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

class TestSetCoverInstanceSchema:
    """Every instance must pass the schema checker with no errors."""

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_schema_valid(self, instance):
        from formulation.verify import verify_problem_schema
        errors = verify_problem_schema(instance)
        assert errors == [], f"Schema errors for {instance['id']}: {errors}"

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_formulation_structure_valid(self, instance):
        from formulation.verify import verify_formulation_structure
        errors = verify_formulation_structure(instance)
        assert errors == [], f"Formulation errors for {instance['id']}: {errors}"

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_run_all_checks_clean(self, instance):
        from formulation.verify import run_all_problem_checks
        result = run_all_problem_checks(instance)
        assert result["schema_errors"] == []
        assert result["formulation_errors"] == []

    def test_instances_have_minimize_objective(self):
        """All set cover instances should minimize (not maximize) the objective."""
        for inst in ALL_INSTANCES:
            sense = inst["formulation"]["objective"]["sense"]
            assert sense == "minimize", (
                f"Instance '{inst['id']}' has sense='{sense}'; set cover should minimize"
            )

    def test_instances_have_binary_variables(self):
        """Each instance's decision variables must be binary (domain contains {0,1})."""
        for inst in ALL_INSTANCES:
            for var in inst["formulation"]["variables"]:
                domain = var.get("domain", "")
                assert "{0,1}" in domain, (
                    f"Instance '{inst['id']}' variable '{var['symbol']}' "
                    f"does not declare binary domain: {domain!r}"
                )

    def test_instances_have_coverage_constraints(self):
        """Every instance must have at least one covering constraint (>= 1)."""
        for inst in ALL_INSTANCES:
            constraints = inst["formulation"]["constraints"]
            assert any(">= 1" in c["expression"] for c in constraints), (
                f"Instance '{inst['id']}' has no covering constraint (>= 1)"
            )


# ---------------------------------------------------------------------------
# 2. Formulation content correctness
# ---------------------------------------------------------------------------

class TestSetCoverInstanceContent:
    """Spot-check that the concrete instance data is self-consistent."""

    def test_minimum_cover_has_five_covering_constraints(self):
        """Universe {1..5} requires exactly 5 covering constraints (one per element)."""
        inst = _make_minimum_set_cover()
        assert len(inst["formulation"]["constraints"]) == 5

    def test_weighted_cover_has_four_covering_constraints(self):
        """Universe {a,b,c,d} requires exactly 4 covering constraints."""
        inst = _make_weighted_set_cover()
        assert len(inst["formulation"]["constraints"]) == 4

    def test_hospital_cover_has_six_covering_constraints(self):
        """Six cities require exactly 6 covering constraints."""
        inst = _make_hospital_coverage()
        assert len(inst["formulation"]["constraints"]) == 6

    def test_weighted_instance_has_latex(self):
        """Weighted set cover instance must include a formulation_latex entry."""
        inst = _make_weighted_set_cover()
        assert inst.get("formulation_latex"), "formulation_latex is missing or empty"

    def test_all_instances_have_complexity_field(self):
        """Set cover is NP-hard; all instances should record that."""
        for inst in ALL_INSTANCES:
            assert inst.get("complexity") == "NP-hard", (
                f"Instance '{inst['id']}' missing or wrong complexity field"
            )


# ---------------------------------------------------------------------------
# 3. format_problem_and_ip output
# ---------------------------------------------------------------------------

class TestSetCoverFormatOutput:
    """Verify that format_problem_and_ip produces sensible markdown output."""

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_format_returns_string(self, instance):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(instance)
        assert isinstance(output, str) and len(output) > 0

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_format_includes_name(self, instance):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(instance)
        assert instance["name"] in output

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_format_includes_minimize(self, instance):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(instance)
        assert "minimize" in output.lower()

    def test_format_with_score_includes_relevance(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(_make_minimum_set_cover(), score=0.87)
        assert "0.87" in output and "relevance" in output.lower()

    def test_format_includes_variables_section(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(_make_minimum_set_cover())
        assert "Variables" in output

    def test_format_includes_constraints_section(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(_make_minimum_set_cover())
        assert "Constraints" in output

    def test_weighted_format_includes_latex(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(_make_weighted_set_cover())
        assert "LaTeX" in output or "$$" in output


# ---------------------------------------------------------------------------
# 4. Short-query expansion for set cover
# ---------------------------------------------------------------------------

class TestSetCoverQueryExpansion:
    """Set-cover keyword queries should be expanded with covering-domain context."""

    def test_cover_keyword_triggers_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("cover")
        assert result.startswith("cover")
        assert len(result) > len("cover"), "Short 'cover' query should be expanded"

    def test_set_cover_two_words_triggers_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("set cover")
        assert result.startswith("set cover")
        assert len(result) > len("set cover")

    def test_covering_keyword_triggers_expansion(self):
        from retrieval.utils import expand_short_query
        result = expand_short_query("covering")
        assert result.startswith("covering")
        assert len(result) > len("covering")

    def test_expansion_includes_covering_context(self):
        """The expansion for 'set cover' should include coverage-related terms."""
        from retrieval.utils import expand_short_query
        result = expand_short_query("set cover").lower()
        assert any(term in result for term in ("cover", "subset", "element", "minimum"))

    def test_long_set_cover_query_not_expanded(self):
        """A six-word set cover query bypasses the expansion entirely."""
        from retrieval.utils import expand_short_query
        q = "find minimum subsets to cover all elements"
        result = expand_short_query(q)
        assert result == q, "Long query must not be modified by expand_short_query"


# ---------------------------------------------------------------------------
# 5. Baseline retrieval for set cover queries
# ---------------------------------------------------------------------------

class TestSetCoverBaselineRetrieval:
    """Text baselines should surface set-cover problems for set-cover queries."""

    @staticmethod
    def _baseline(cls_name: str):
        """Return a freshly constructed baseline instance by class name."""
        from retrieval.baselines import BM25Baseline, TfidfBaseline, LSABaseline
        return {
            "BM25Baseline": BM25Baseline,
            "TfidfBaseline": TfidfBaseline,
            "LSABaseline": LSABaseline,
        }[cls_name]()

    def _catalog(self) -> list[dict]:
        """Small catalog that contains the set cover problem alongside distractors."""
        return [
            {
                "id": "set_cover",
                "name": "Set Cover Problem",
                "aliases": ["set covering", "hitting set (dual)"],
                "description": (
                    "Given a universe of elements and a family of subsets, choose the "
                    "minimum number of subsets such that every element is contained in "
                    "at least one chosen subset."
                ),
            },
            {
                "id": "knapsack",
                "name": "0-1 Knapsack",
                "aliases": ["binary knapsack"],
                "description": (
                    "Select a subset of items with weights and values to maximize total "
                    "value without exceeding a weight capacity."
                ),
            },
            {
                "id": "tsp",
                "name": "Traveling Salesman Problem",
                "aliases": ["TSP"],
                "description": (
                    "Find the shortest possible route that visits each city exactly once "
                    "and returns to the origin city."
                ),
            },
        ]

    @pytest.mark.parametrize("cls_name", ["BM25Baseline", "TfidfBaseline", "LSABaseline"])
    def test_set_cover_query_ranks_set_cover_first(self, cls_name):
        """'set cover' query should rank the Set Cover problem first (index 0).

        Baselines return (problem_id_str, score) tuples.
        """
        baseline = self._baseline(cls_name)
        baseline.fit(self._catalog())
        results = baseline.rank("set cover", top_k=3)
        assert results, f"{cls_name}: no results returned"
        top_id = results[0][0]  # (problem_id_str, score)
        assert top_id == "set_cover", (
            f"{cls_name}: expected 'set_cover' at rank 1 (index 0), got '{top_id}'"
        )

    @pytest.mark.parametrize("cls_name", ["BM25Baseline", "TfidfBaseline", "LSABaseline"])
    def test_covering_query_returns_set_cover(self, cls_name):
        """'covering subsets elements' should retrieve set_cover in top-2."""
        baseline = self._baseline(cls_name)
        baseline.fit(self._catalog())
        results = baseline.rank("covering subsets elements", top_k=2)
        ids = [r[0] for r in results]  # r is (problem_id_str, score)
        assert "set_cover" in ids, (
            f"{cls_name}: 'set_cover' not in top-2 for 'covering subsets elements'; got {ids}"
        )

    @pytest.mark.parametrize("cls_name", ["BM25Baseline", "TfidfBaseline", "LSABaseline"])
    def test_minimum_subsets_query_returns_set_cover(self, cls_name):
        """Descriptive query mentioning subsets and elements retrieves set_cover in top-2."""
        baseline = self._baseline(cls_name)
        baseline.fit(self._catalog())
        results = baseline.rank(
            "choose minimum number of subsets so every element is covered", top_k=2
        )
        ids = [r[0] for r in results]  # r is (problem_id_str, score)
        assert "set_cover" in ids, (
            f"{cls_name}: 'set_cover' not in top-2 for descriptive query; got {ids}"
        )


# ---------------------------------------------------------------------------
# 6. Program output snapshots — the actual "result" for each instance
#
# These tests record exactly what format_problem_and_ip renders for each
# concrete set cover instance.  They answer the question "What was the
# result?" by asserting on the full structure of the rendered output:
#   • header with instance name
#   • collapsible Variables / Objective / Constraints sections
#   • correct objective sense and expression
#   • each covering constraint with its human-readable description
#   • LaTeX block (where present)
#   • complexity footer
# ---------------------------------------------------------------------------

class TestSetCoverProgramOutput:
    """Snapshot the actual program output for every set cover instance."""

    # ------------------------------------------------------------------
    # Minimum Set Cover
    # ------------------------------------------------------------------

    def test_minimum_output_header(self):
        """Output starts with the instance name as a Markdown heading."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_minimum_set_cover())
        assert "## Minimum Set Cover Instance" in out

    def test_minimum_output_variables_section(self):
        """Variables section lists the binary decision variable x_S."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_minimum_set_cover())
        assert "<details><summary><strong>Variables</strong></summary>" in out
        assert "x_S" in out
        assert "1 if subset S is selected, 0 otherwise" in out

    def test_minimum_output_objective(self):
        """Objective section shows 'minimize' and the summed expression."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_minimum_set_cover())
        assert "**Sense:** minimize" in out
        assert "x_S1 + x_S2 + x_S3" in out

    def test_minimum_output_all_five_constraints(self):
        """All five covering constraints appear with their descriptions."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_minimum_set_cover())
        assert "x_S1 >= 1" in out
        assert "Element 1 must be covered" in out
        assert "x_S1 + x_S2 >= 1" in out
        assert "Element 2 must be covered" in out
        assert "x_S1 + x_S3 >= 1" in out
        assert "Element 3 must be covered" in out
        assert "x_S2 + x_S3 >= 1" in out
        assert "Element 4 must be covered" in out
        assert "x_S3 >= 1" in out
        assert "Element 5 must be covered" in out

    def test_minimum_output_complexity_footer(self):
        """Complexity footer is rendered for the minimum set cover instance."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_minimum_set_cover())
        assert "*Complexity: NP-hard*" in out

    def test_minimum_output_no_latex_block(self):
        """Minimum set cover has no formulation_latex so no LaTeX block is shown."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_minimum_set_cover())
        assert "LaTeX" not in out

    # ------------------------------------------------------------------
    # Weighted Set Cover
    # ------------------------------------------------------------------

    def test_weighted_output_header(self):
        """Output starts with the weighted instance name as a Markdown heading."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_weighted_set_cover())
        assert "## Weighted Set Cover Instance" in out

    def test_weighted_output_objective_expression(self):
        """Objective expression includes the four cost coefficients."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_weighted_set_cover())
        assert "3 x_S1 + 2 x_S2 + 4 x_S3 + 5 x_S4" in out

    def test_weighted_output_all_four_constraints(self):
        """All four element-covering constraints appear with descriptions."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_weighted_set_cover())
        assert "x_S1 + x_S4 >= 1" in out
        assert "Element a must be covered" in out
        assert "x_S1 + x_S2 >= 1" in out
        assert "Element b must be covered" in out
        assert "x_S2 + x_S3 >= 1" in out
        assert "Element c must be covered" in out
        assert "x_S3 + x_S4 >= 1" in out
        assert "Element d must be covered" in out

    def test_weighted_output_latex_block(self):
        """LaTeX block is rendered for the weighted instance."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_weighted_set_cover())
        assert "<details><summary><strong>LaTeX (rendered)</strong></summary>" in out
        assert "$$" in out
        assert "3x_{S1}" in out

    def test_weighted_output_complexity_footer(self):
        """Complexity footer is rendered for the weighted set cover instance."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_weighted_set_cover())
        assert "*Complexity: NP-hard*" in out

    # ------------------------------------------------------------------
    # Hospital Coverage
    # ------------------------------------------------------------------

    def test_hospital_output_header(self):
        """Output starts with the hospital coverage instance name."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_hospital_coverage())
        assert "## Hospital Coverage Instance" in out

    def test_hospital_output_variables_section(self):
        """Variables section lists the binary decision variable y_h."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_hospital_coverage())
        assert "y_h" in out
        assert "1 if hospital h is opened, 0 otherwise" in out

    def test_hospital_output_objective(self):
        """Objective section shows minimize over opened hospital count."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_hospital_coverage())
        assert "**Sense:** minimize" in out
        assert "y_H1 + y_H2 + y_H3" in out

    def test_hospital_output_all_six_constraints(self):
        """All six city-covering constraints appear with their descriptions."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_hospital_coverage())
        assert "y_H1 >= 1" in out
        assert "City C1 must be covered" in out
        assert "y_H1 + y_H2 >= 1" in out
        assert "City C2 must be covered" in out
        assert "y_H1 + y_H3 >= 1" in out
        assert "City C3 must be covered" in out
        assert "y_H2 >= 1" in out
        assert "City C4 must be covered" in out
        assert "y_H2 + y_H3 >= 1" in out
        assert "City C5 must be covered" in out
        assert "y_H3 >= 1" in out
        assert "City C6 must be covered" in out

    def test_hospital_output_complexity_footer(self):
        """Complexity footer is rendered for the hospital coverage instance."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(_make_hospital_coverage())
        assert "*Complexity: NP-hard*" in out

    # ------------------------------------------------------------------
    # Cross-instance output shape
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_all_instances_have_collapsible_sections(self, instance):
        """Every instance output has all three collapsible HTML sections."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(instance)
        assert "<details>" in out
        assert "</details>" in out
        assert "<summary><strong>Variables</strong></summary>" in out
        assert "<summary><strong>Objective</strong></summary>" in out
        assert "<summary><strong>Constraints</strong></summary>" in out

    @pytest.mark.parametrize("instance", ALL_INSTANCES, ids=[p["id"] for p in ALL_INSTANCES])
    def test_all_instances_output_with_score(self, instance):
        """Output with score=0.95 includes a relevance line at the top."""
        from retrieval.search import format_problem_and_ip
        out = format_problem_and_ip(instance, score=0.95)
        assert "(relevance: 0.950)" in out or "0.95" in out
