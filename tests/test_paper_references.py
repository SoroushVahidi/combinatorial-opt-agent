"""
Tests for paper-references support added to the catalog and UI.

Validates:
- _load_catalog() merges problem_references.json into catalog entries
- format_problem_and_ip() renders a 📚 Papers section for problems with references
- _format_result_html() renders a 📚 Papers collapsible section in the HTML card
- References without a URL still render (no broken links)
- Problems without references are unaffected
- The schema accepts the new 'references' array field
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers to build minimal problem dicts for format tests
# ---------------------------------------------------------------------------

def _minimal_problem(problem_id: str = "test_prob", references: list | None = None) -> dict:
    p = {
        "id": problem_id,
        "name": "Test Problem",
        "description": "A test optimization problem.",
        "formulation": {
            "variables": [
                {"symbol": "x", "description": "decision variable", "domain": "x ∈ {0,1}"}
            ],
            "objective": {"sense": "minimize", "expression": "c^T x"},
            "constraints": [{"expression": "Ax ≤ b", "description": "resource constraints"}],
        },
        "complexity": "NP-hard",
    }
    if references is not None:
        p["references"] = references
    return p


_SAMPLE_REFS = [
    {
        "title": "A Classic Paper",
        "authors": "Doe, J.; Smith, A.",
        "year": 1971,
        "venue": "Journal of OR, 10(2), 50–70",
        "url": "https://doi.org/10.1287/example",
    },
    {
        "title": "No URL Paper",
        "authors": "Roe, R.",
        "year": 1985,
        "venue": "Some Conference",
    },
]


# ---------------------------------------------------------------------------
# 1. Catalog loading merges references sidecar
# ---------------------------------------------------------------------------

class TestCatalogReferencesMerge:
    """_load_catalog() injects references from problem_references.json."""

    def test_refs_sidecar_file_exists(self):
        refs_path = ROOT / "data" / "processed" / "problem_references.json"
        assert refs_path.exists(), "problem_references.json must exist"

    def test_refs_sidecar_is_valid_json(self):
        refs_path = ROOT / "data" / "processed" / "problem_references.json"
        with open(refs_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "references sidecar must be a JSON object"

    def test_refs_sidecar_each_value_is_list(self):
        refs_path = ROOT / "data" / "processed" / "problem_references.json"
        with open(refs_path, encoding="utf-8") as f:
            data = json.load(f)
        for key, val in data.items():
            assert isinstance(val, list), f"references for '{key}' must be a list"

    def test_refs_each_reference_has_required_fields(self):
        refs_path = ROOT / "data" / "processed" / "problem_references.json"
        with open(refs_path, encoding="utf-8") as f:
            data = json.load(f)
        for prob_id, refs in data.items():
            for ref in refs:
                assert "title" in ref, f"ref for '{prob_id}' missing 'title'"
                assert "authors" in ref, f"ref for '{prob_id}' missing 'authors'"
                assert "year" in ref, f"ref for '{prob_id}' missing 'year'"

    def test_load_catalog_injects_references_for_tsp(self):
        from retrieval.search import _load_catalog
        catalog = _load_catalog()
        tsp = next((p for p in catalog if p.get("id") == "tsp"), None)
        assert tsp is not None, "TSP must be in catalog"
        refs = tsp.get("references")
        assert refs and len(refs) > 0, "TSP should have references after loading"
        titles = [r["title"] for r in refs]
        assert any("traveling" in t.lower() for t in titles), (
            "TSP references should mention 'traveling' in at least one title"
        )

    def test_load_catalog_injects_references_for_assignment(self):
        from retrieval.search import _load_catalog
        catalog = _load_catalog()
        prob = next((p for p in catalog if p.get("id") == "assignment"), None)
        assert prob is not None
        refs = prob.get("references")
        assert refs and len(refs) > 0
        assert any("Hungarian" in r["title"] for r in refs)

    def test_load_catalog_problems_without_refs_are_unaffected(self):
        from retrieval.search import _load_catalog
        catalog = _load_catalog()
        # NL4Opt and optmath_bench problems are not in the references sidecar
        no_ref_probs = [
            p for p in catalog
            if p.get("source") == "NL4Opt"
        ]
        assert len(no_ref_probs) > 0, "There should be NL4Opt problems"
        # Spot-check: at most a small fraction should have references
        with_refs = [p for p in no_ref_probs if p.get("references")]
        assert len(with_refs) == 0, "NL4Opt problems should have no references"

    def test_refs_sidecar_known_ids_present(self):
        refs_path = ROOT / "data" / "processed" / "problem_references.json"
        with open(refs_path, encoding="utf-8") as f:
            data = json.load(f)
        expected_ids = {
            "tsp", "knapsack_01", "assignment", "max_flow", "cvrp",
            "set_cover", "maximum_matching", "shortest_path",
        }
        for pid in expected_ids:
            assert pid in data, f"'{pid}' should be in problem_references.json"


# ---------------------------------------------------------------------------
# 2. format_problem_and_ip() includes Papers section
# ---------------------------------------------------------------------------

class TestFormatProblemAndIpReferences:
    """format_problem_and_ip() renders a 📚 Papers section when references exist."""

    def test_no_refs_no_papers_section(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem()
        output = format_problem_and_ip(p)
        assert "📚 Papers" not in output

    def test_with_refs_papers_section_present(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem(references=_SAMPLE_REFS)
        output = format_problem_and_ip(p)
        assert "📚 Papers" in output

    def test_papers_section_contains_title(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem(references=_SAMPLE_REFS)
        output = format_problem_and_ip(p)
        assert "A Classic Paper" in output

    def test_papers_section_contains_url_as_link(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem(references=_SAMPLE_REFS)
        output = format_problem_and_ip(p)
        assert "https://doi.org/10.1287/example" in output

    def test_papers_section_contains_author(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem(references=_SAMPLE_REFS)
        output = format_problem_and_ip(p)
        assert "Doe, J." in output

    def test_papers_section_contains_year(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem(references=_SAMPLE_REFS)
        output = format_problem_and_ip(p)
        assert "1971" in output

    def test_reference_without_url_still_renders(self):
        from retrieval.search import format_problem_and_ip
        p = _minimal_problem(references=_SAMPLE_REFS)
        output = format_problem_and_ip(p)
        assert "No URL Paper" in output

    def test_tsp_catalog_entry_papers_section(self):
        from retrieval.search import _load_catalog, format_problem_and_ip
        catalog = _load_catalog()
        tsp = next(p for p in catalog if p.get("id") == "tsp")
        output = format_problem_and_ip(tsp)
        assert "📚 Papers" in output
        assert "Dantzig" in output  # canonical TSP paper author


# ---------------------------------------------------------------------------
# 3. _format_result_html() renders Papers section in HTML card
# ---------------------------------------------------------------------------

class TestFormatResultHtmlReferences:
    """_format_result_html() renders a collapsible Papers section."""

    def test_no_refs_no_papers_section_in_html(self):
        from app import _format_result_html
        p = _minimal_problem()
        html = _format_result_html(p, 0.9, 1)
        assert "📚 Papers" not in html

    def test_with_refs_papers_section_in_html(self):
        from app import _format_result_html
        p = _minimal_problem(references=_SAMPLE_REFS)
        html = _format_result_html(p, 0.9, 1)
        assert "📚 Papers" in html

    def test_papers_section_is_details_element(self):
        from app import _format_result_html
        p = _minimal_problem(references=_SAMPLE_REFS)
        html = _format_result_html(p, 0.9, 1)
        assert '<details class="coa-section">' in html
        assert "coa-ref-list" in html

    def test_papers_section_has_hyperlinks_for_refs_with_url(self):
        from app import _format_result_html
        p = _minimal_problem(references=_SAMPLE_REFS)
        html = _format_result_html(p, 0.9, 1)
        assert 'href="https://doi.org/10.1287/example"' in html
        assert 'target="_blank"' in html

    def test_papers_section_renders_title(self):
        from app import _format_result_html
        p = _minimal_problem(references=_SAMPLE_REFS)
        html = _format_result_html(p, 0.9, 1)
        assert "A Classic Paper" in html

    def test_papers_section_renders_authors_and_year(self):
        from app import _format_result_html
        p = _minimal_problem(references=_SAMPLE_REFS)
        html = _format_result_html(p, 0.9, 1)
        assert "Doe, J." in html
        assert "1971" in html

    def test_papers_section_shows_count(self):
        from app import _format_result_html
        p = _minimal_problem(references=_SAMPLE_REFS)
        html = _format_result_html(p, 0.9, 1)
        assert f"({len(_SAMPLE_REFS)})" in html

    def test_tsp_html_card_has_papers_section(self):
        from retrieval.search import _load_catalog
        from app import _format_result_html
        catalog = _load_catalog()
        tsp = next(p for p in catalog if p.get("id") == "tsp")
        html = _format_result_html(tsp, 0.95, 1)
        assert "📚 Papers" in html
        assert "Dantzig" in html

    def test_html_escaping_in_title(self):
        """Titles with HTML-special characters are escaped in the HTML output."""
        from app import _format_result_html
        ref = {
            "title": "Title with <special> & 'quotes'",
            "authors": "Author, A.",
            "year": 2000,
        }
        p = _minimal_problem(references=[ref])
        html = _format_result_html(p, 0.9, 1)
        assert "&lt;special&gt;" in html
        assert "&amp;" in html


# ---------------------------------------------------------------------------
# 4. Schema validates references field
# ---------------------------------------------------------------------------

class TestSchemaReferencesField:
    """JSON Schema should now accept the 'references' array."""

    def test_schema_has_references_property(self):
        schema_path = ROOT / "schema" / "problem_schema.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        assert "references" in schema["properties"], (
            "'references' must be defined in problem_schema.json"
        )

    def test_schema_references_is_array(self):
        schema_path = ROOT / "schema" / "problem_schema.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        assert schema["properties"]["references"]["type"] == "array"

    def test_schema_reference_items_require_title_authors_year(self):
        schema_path = ROOT / "schema" / "problem_schema.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        items = schema["properties"]["references"]["items"]
        required = items.get("required", [])
        for field in ("title", "authors", "year"):
            assert field in required, f"'{field}' must be required in reference items"
