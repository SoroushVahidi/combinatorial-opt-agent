"""
Pipeline audit utilities: surface catalog health issues before they silently
degrade search quality.

The single most damaging known issue is a **stale extended catalog**: when
``all_problems_extended.json`` exists but was built from an older, smaller
version of ``all_problems.json``, the search service silently searches only
the (tiny) extended file — missing the vast majority of the catalog.

Example use::

    from retrieval.pipeline_audit import audit_catalog_health
    issues = audit_catalog_health()
    for issue in issues:
        print(issue)
"""
from __future__ import annotations

import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CatalogIssue:
    """A single diagnosed catalog health problem.

    Attributes:
        severity: ``"critical"``, ``"warning"``, or ``"info"``.
        code: short machine-readable tag (e.g. ``"stale_extended_catalog"``).
        message: human-readable description of the issue.
        fix: suggested remediation command or action.
    """

    def __init__(self, severity: str, code: str, message: str, fix: str = "") -> None:
        self.severity = severity
        self.code = code
        self.message = message
        self.fix = fix

    def __str__(self) -> str:
        parts = [f"[{self.severity.upper()}] {self.code}: {self.message}"]
        if self.fix:
            parts.append(f"  Fix: {self.fix}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"CatalogIssue(severity={self.severity!r}, code={self.code!r}, "
            f"message={self.message!r})"
        )


def audit_catalog_health(data_dir: Path | None = None) -> list[CatalogIssue]:
    """Check for known catalog health issues and return a list of :class:`CatalogIssue`.

    Currently detects:

    * **stale_extended_catalog** (critical) — ``all_problems_extended.json``
      exists but contains *fewer* problems than ``all_problems.json``.  The
      search function prefers the extended file, so a stale extended catalog
      silently restricts search to an arbitrarily small subset of the catalog.

    * **incomplete_formulations** (warning) — some catalog entries lack one or
      more required formulation fields (``variables``, ``objective``,
      ``constraints``).  Retrieval still works, but retrieved problems display
      "Formulation not yet available."

    * **missing_base_catalog** (critical) — ``all_problems.json`` does not
      exist; the catalog cannot be loaded at all.

    * **empty_catalog** (critical) — the active catalog (extended if
      available, otherwise base) is empty.

    Args:
        data_dir: path to ``data/processed/``.  Defaults to the standard
            project-relative location.

    Returns:
        List of :class:`CatalogIssue` objects (may be empty if all checks
        pass).
    """
    if data_dir is None:
        data_dir = _project_root() / "data" / "processed"

    issues: list[CatalogIssue] = []

    base_path = data_dir / "all_problems.json"
    extended_path = data_dir / "all_problems_extended.json"

    # --- missing base catalog ---
    if not base_path.exists():
        issues.append(CatalogIssue(
            severity="critical",
            code="missing_base_catalog",
            message=f"Base catalog not found: {base_path}",
            fix="Run `python pipeline/run_collection.py` to build the catalog.",
        ))
        return issues  # can't do further checks without the base

    with open(base_path, encoding="utf-8") as f:
        base_catalog: list[dict] = json.load(f)

    base_count = len(base_catalog)

    # --- stale extended catalog ---
    if extended_path.exists():
        with open(extended_path, encoding="utf-8") as f:
            ext_catalog: list[dict] = json.load(f)
        ext_count = len(ext_catalog)
        if ext_count < base_count:
            issues.append(CatalogIssue(
                severity="critical",
                code="stale_extended_catalog",
                message=(
                    f"all_problems_extended.json has {ext_count} problems but "
                    f"all_problems.json has {base_count}. "
                    f"The search service is restricted to {ext_count} of "
                    f"{base_count} problems "
                    f"({100 * ext_count / base_count:.1f}% coverage)."
                ),
                fix="Run `python build_extended_catalog.py` to rebuild the extended catalog.",
            ))
        active_catalog = ext_catalog
    else:
        active_catalog = base_catalog

    # --- empty catalog ---
    if not active_catalog:
        issues.append(CatalogIssue(
            severity="critical",
            code="empty_catalog",
            message="The active catalog is empty — no problems will be returned by search.",
            fix="Run `python pipeline/run_collection.py` to populate the catalog.",
        ))
        return issues

    # --- incomplete formulations ---
    incomplete = _count_incomplete(active_catalog)
    if incomplete > 0:
        pct = 100.0 * incomplete / len(active_catalog)
        issues.append(CatalogIssue(
            severity="warning",
            code="incomplete_formulations",
            message=(
                f"{incomplete} of {len(active_catalog)} problems "
                f"({pct:.1f}%) have missing formulation fields "
                "(variables, objective, or constraints). "
                "Retrieval works, but these problems display "
                "'Formulation not yet available'."
            ),
            fix=(
                "Run `python build_extended_catalog.py --enrich` to fetch "
                "missing formulations from public web sources."
            ),
        ))

    return issues


def _count_incomplete(catalog: list[dict]) -> int:
    """Return the number of catalog entries with incomplete formulations."""
    count = 0
    for problem in catalog:
        form = problem.get("formulation") or {}
        has_vars = bool(form.get("variables") or [])
        has_obj = bool(form.get("objective"))
        # Key-presence check (not truthiness): an explicit empty constraints list
        # is valid for unconstrained problems and counts as "present".  This is
        # intentionally consistent with find_incomplete_problems() in
        # retrieval/catalog_enrichment.py and verify_formulation_structure()
        # in formulation/verify.py, which both allow constraints: [].
        has_constraints = "constraints" in form
        if not (has_vars and has_obj and has_constraints):
            count += 1
    return count


def print_audit_report(data_dir: Path | None = None) -> int:
    """Print a human-readable audit report to stdout.

    Returns the number of **critical** issues found (0 = healthy).
    """
    issues = audit_catalog_health(data_dir)
    if not issues:
        print("✓ Catalog health: no issues found.")
        return 0

    n_critical = sum(1 for i in issues if i.severity == "critical")
    print(f"Catalog health: {len(issues)} issue(s) found ({n_critical} critical).\n")
    for issue in issues:
        print(issue)
        print()
    return n_critical


if __name__ == "__main__":
    import sys
    sys.exit(print_audit_report())
