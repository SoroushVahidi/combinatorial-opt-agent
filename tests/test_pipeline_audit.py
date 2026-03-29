"""
Tests for retrieval/pipeline_audit.py and the staleness guard in _load_catalog.

Covers:
  - audit_catalog_health: detects stale extended catalog (critical)
  - audit_catalog_health: detects missing base catalog (critical)
  - audit_catalog_health: detects incomplete formulations (warning)
  - audit_catalog_health: detects empty catalog (critical)
  - audit_catalog_health: returns empty list when everything is healthy
  - _load_catalog: falls back to base with warning when extended is stale
  - _load_catalog: returns extended catalog when it is current or larger
  - print_audit_report: returns 0 on healthy catalog, >0 on critical issues
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_catalog(path: Path, problems: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(problems), encoding="utf-8")


def _make_complete_problem(pid: str) -> dict:
    return {
        "id": pid,
        "name": f"Problem {pid}",
        "description": f"Description for {pid}.",
        "formulation": {
            "variables": [{"symbol": "x", "description": "var x", "domain": "x >= 0"}],
            "objective": {"sense": "minimize", "expression": "x"},
            "constraints": [{"expression": "x >= 0", "description": "nonneg"}],
        },
        "source": "test",
    }


def _make_incomplete_problem(pid: str) -> dict:
    return {
        "id": pid,
        "name": f"Incomplete {pid}",
        "description": f"Description for incomplete problem {pid}.",
        "source": "test",
    }


# ---------------------------------------------------------------------------
# audit_catalog_health tests
# ---------------------------------------------------------------------------

class TestAuditCatalogHealth:

    def test_healthy_catalog_no_issues(self, tmp_path):
        """A base-only catalog with all complete problems reports no issues."""
        from retrieval.pipeline_audit import audit_catalog_health
        probs = [_make_complete_problem(str(i)) for i in range(5)]
        _write_catalog(tmp_path / "all_problems.json", probs)

        issues = audit_catalog_health(tmp_path)
        assert issues == []

    def test_missing_base_catalog_is_critical(self, tmp_path):
        """When all_problems.json does not exist, we get a critical issue."""
        from retrieval.pipeline_audit import audit_catalog_health
        issues = audit_catalog_health(tmp_path)
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert issues[0].code == "missing_base_catalog"

    def test_stale_extended_catalog_is_critical(self, tmp_path):
        """Extended catalog with fewer problems than base → critical issue."""
        from retrieval.pipeline_audit import audit_catalog_health
        base = [_make_complete_problem(str(i)) for i in range(10)]
        stale_ext = base[:3]  # only 3 of 10

        _write_catalog(tmp_path / "all_problems.json", base)
        _write_catalog(tmp_path / "all_problems_extended.json", stale_ext)

        issues = audit_catalog_health(tmp_path)
        critical = [i for i in issues if i.code == "stale_extended_catalog"]
        assert len(critical) == 1
        assert critical[0].severity == "critical"
        assert "3" in critical[0].message
        assert "10" in critical[0].message

    def test_stale_catalog_reports_coverage_percentage(self, tmp_path):
        """Coverage percentage appears in the stale-catalog message."""
        from retrieval.pipeline_audit import audit_catalog_health
        base = [_make_complete_problem(str(i)) for i in range(100)]
        stale_ext = base[:2]

        _write_catalog(tmp_path / "all_problems.json", base)
        _write_catalog(tmp_path / "all_problems_extended.json", stale_ext)

        issues = audit_catalog_health(tmp_path)
        stale = next(i for i in issues if i.code == "stale_extended_catalog")
        assert "2.0%" in stale.message  # 2/100 = 2.0%

    def test_current_extended_catalog_no_stale_issue(self, tmp_path):
        """Extended catalog with same or more problems than base → no stale issue."""
        from retrieval.pipeline_audit import audit_catalog_health
        base = [_make_complete_problem(str(i)) for i in range(5)]
        ext = [_make_complete_problem(str(i)) for i in range(7)]  # larger

        _write_catalog(tmp_path / "all_problems.json", base)
        _write_catalog(tmp_path / "all_problems_extended.json", ext)

        issues = audit_catalog_health(tmp_path)
        stale = [i for i in issues if i.code == "stale_extended_catalog"]
        assert stale == []

    def test_incomplete_formulations_warning(self, tmp_path):
        """Problems with missing formulation fields produce a warning issue."""
        from retrieval.pipeline_audit import audit_catalog_health
        probs = [
            _make_complete_problem("p0"),
            _make_incomplete_problem("p1"),
            _make_incomplete_problem("p2"),
        ]
        _write_catalog(tmp_path / "all_problems.json", probs)

        issues = audit_catalog_health(tmp_path)
        incomplete = [i for i in issues if i.code == "incomplete_formulations"]
        assert len(incomplete) == 1
        assert incomplete[0].severity == "warning"
        assert "2 of 3" in incomplete[0].message

    def test_no_incomplete_formulation_warning_when_all_complete(self, tmp_path):
        """No warning when all problems have complete formulations."""
        from retrieval.pipeline_audit import audit_catalog_health
        probs = [_make_complete_problem(str(i)) for i in range(3)]
        _write_catalog(tmp_path / "all_problems.json", probs)

        issues = audit_catalog_health(tmp_path)
        incomplete = [i for i in issues if i.code == "incomplete_formulations"]
        assert incomplete == []

    def test_empty_catalog_is_critical(self, tmp_path):
        """An empty base catalog (no problems) produces a critical issue."""
        from retrieval.pipeline_audit import audit_catalog_health
        _write_catalog(tmp_path / "all_problems.json", [])

        issues = audit_catalog_health(tmp_path)
        empty = [i for i in issues if i.code == "empty_catalog"]
        assert len(empty) == 1
        assert empty[0].severity == "critical"

    def test_stale_catalog_check_uses_extended_count_for_incomplete(self, tmp_path):
        """When extended is stale, incomplete check runs on the stale extended, not base."""
        from retrieval.pipeline_audit import audit_catalog_health
        base = [_make_complete_problem(str(i)) for i in range(10)]
        stale_ext = [_make_incomplete_problem("stale_p")]  # 1 incomplete problem

        _write_catalog(tmp_path / "all_problems.json", base)
        _write_catalog(tmp_path / "all_problems_extended.json", stale_ext)

        issues = audit_catalog_health(tmp_path)
        codes = [i.code for i in issues]
        # Both stale_extended_catalog and incomplete_formulations should be present
        assert "stale_extended_catalog" in codes
        assert "incomplete_formulations" in codes

    def test_all_issue_objects_have_fix(self, tmp_path):
        """Every CatalogIssue returned by the audit must have a non-empty fix."""
        from retrieval.pipeline_audit import audit_catalog_health
        base = [_make_complete_problem(str(i)) for i in range(5)]
        stale_ext = base[:1]

        _write_catalog(tmp_path / "all_problems.json", base)
        _write_catalog(tmp_path / "all_problems_extended.json", stale_ext)

        issues = audit_catalog_health(tmp_path)
        for issue in issues:
            assert issue.fix, f"Issue {issue.code} has empty fix"


# ---------------------------------------------------------------------------
# CatalogIssue __str__ and __repr__
# ---------------------------------------------------------------------------

class TestCatalogIssueFormatting:

    def test_str_includes_severity_and_code(self):
        from retrieval.pipeline_audit import CatalogIssue
        issue = CatalogIssue("critical", "stale_extended_catalog", "Test message.", "Fix it.")
        s = str(issue)
        assert "CRITICAL" in s
        assert "stale_extended_catalog" in s
        assert "Test message." in s
        assert "Fix it." in s

    def test_str_without_fix(self):
        from retrieval.pipeline_audit import CatalogIssue
        issue = CatalogIssue("warning", "some_code", "A warning.", "")
        s = str(issue)
        assert "WARNING" in s
        assert "Fix:" not in s

    def test_repr_round_trip(self):
        from retrieval.pipeline_audit import CatalogIssue
        issue = CatalogIssue("info", "test_code", "An info message.", "Do something.")
        r = repr(issue)
        assert "info" in r
        assert "test_code" in r


# ---------------------------------------------------------------------------
# print_audit_report
# ---------------------------------------------------------------------------

class TestPrintAuditReport:

    def test_returns_zero_on_healthy_catalog(self, tmp_path, capsys):
        from retrieval.pipeline_audit import print_audit_report
        probs = [_make_complete_problem(str(i)) for i in range(3)]
        _write_catalog(tmp_path / "all_problems.json", probs)

        rc = print_audit_report(tmp_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "no issues" in captured.out.lower()

    def test_returns_positive_on_critical_issue(self, tmp_path, capsys):
        from retrieval.pipeline_audit import print_audit_report
        # Missing base catalog → critical
        rc = print_audit_report(tmp_path)
        assert rc > 0

    def test_stale_catalog_returns_positive(self, tmp_path, capsys):
        from retrieval.pipeline_audit import print_audit_report
        base = [_make_complete_problem(str(i)) for i in range(10)]
        stale = base[:2]
        _write_catalog(tmp_path / "all_problems.json", base)
        _write_catalog(tmp_path / "all_problems_extended.json", stale)

        rc = print_audit_report(tmp_path)
        assert rc > 0
        captured = capsys.readouterr()
        assert "CRITICAL" in captured.out


# ---------------------------------------------------------------------------
# _load_catalog staleness guard
# ---------------------------------------------------------------------------

class TestLoadCatalogStalenessGuard:

    def _patch_root(self, monkeypatch, tmp_path):
        """Monkey-patch retrieval.search._project_root to return tmp_path."""
        import retrieval.search as mod
        monkeypatch.setattr(mod, "_project_root", lambda: tmp_path)

    def test_stale_extended_triggers_warning_and_returns_base(self, tmp_path, monkeypatch):
        """When extended < base, _load_catalog warns and returns the base."""
        self._patch_root(monkeypatch, tmp_path)
        import retrieval.search as mod

        base = [_make_complete_problem(str(i)) for i in range(50)]
        stale_ext = base[:5]
        data_dir = tmp_path / "data" / "processed"
        _write_catalog(data_dir / "all_problems.json", base)
        _write_catalog(data_dir / "all_problems_extended.json", stale_ext)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            catalog = mod._load_catalog()

        assert len(catalog) == 50, "Should fall back to base catalog"
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert "stale" in str(caught[0].message).lower()

    def test_current_extended_no_warning(self, tmp_path, monkeypatch):
        """When extended >= base, _load_catalog uses the extended without warning."""
        self._patch_root(monkeypatch, tmp_path)
        import retrieval.search as mod

        base = [_make_complete_problem(str(i)) for i in range(5)]
        ext = [_make_complete_problem(str(i)) for i in range(8)]  # larger
        data_dir = tmp_path / "data" / "processed"
        _write_catalog(data_dir / "all_problems.json", base)
        _write_catalog(data_dir / "all_problems_extended.json", ext)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            catalog = mod._load_catalog()

        assert len(catalog) == 8, "Should use extended catalog"
        stale_warnings = [w for w in caught if "stale" in str(w.message).lower()]
        assert stale_warnings == []

    def test_no_extended_loads_base(self, tmp_path, monkeypatch):
        """When no extended file exists, _load_catalog loads the base."""
        self._patch_root(monkeypatch, tmp_path)
        import retrieval.search as mod

        base = [_make_complete_problem(str(i)) for i in range(10)]
        data_dir = tmp_path / "data" / "processed"
        _write_catalog(data_dir / "all_problems.json", base)

        catalog = mod._load_catalog()
        assert len(catalog) == 10

    def test_neither_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """When neither catalog file exists, _load_catalog raises FileNotFoundError."""
        self._patch_root(monkeypatch, tmp_path)
        import retrieval.search as mod
        (tmp_path / "data" / "processed").mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            mod._load_catalog()
