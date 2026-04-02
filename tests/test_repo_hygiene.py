"""Repository hygiene sanity tests.

These tests verify that the key source-of-truth documents, paper artifacts,
experiment scripts, requirements.txt health, and the canonical validation
script are all present and structurally sound.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# ── 1. Source-of-truth file existence ────────────────────────────────────────

@pytest.mark.parametrize("rel_path", [
    "docs/EAAI_SOURCE_OF_TRUTH.md",
    "README.md",
    "REPO_STRUCTURE.md",
    "CONTRIBUTING.md",
])
def test_source_of_truth_file_exists(rel_path: str) -> None:
    assert (ROOT / rel_path).exists(), f"Missing required file: {rel_path}"


# ── 2. Paper artifact existence ───────────────────────────────────────────────

@pytest.mark.parametrize("rel_path", [
    "results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv",
    "results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv",
    "results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv",
    "results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv",
    "results/paper/eaai_camera_ready_tables/table5_failure_taxonomy.csv",
])
def test_paper_artifact_exists(rel_path: str) -> None:
    p = ROOT / rel_path
    assert p.exists(), f"Missing paper artifact: {rel_path}"
    # Also verify it's a non-empty valid CSV
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) >= 2, f"{rel_path} has fewer than 2 rows (header + 1 data row)"


# ── 3. Canonical experiment script existence ──────────────────────────────────

@pytest.mark.parametrize("rel_path", [
    "tools/run_eaai_engineering_subset_experiment.py",
    "tools/run_eaai_executable_subset_experiment.py",
    "tools/run_eaai_final_solver_attempt.py",
])
def test_experiment_script_exists(rel_path: str) -> None:
    assert (ROOT / rel_path).exists(), f"Missing experiment script: {rel_path}"


# ── 4. requirements.txt has no duplicate packages ─────────────────────────────

def _pkg_name(line: str) -> str | None:
    """Extract bare package name from a requirements line, or None if non-package."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    # Strip extras e.g. "package[extra]>=1.0"
    name = stripped.split("[")[0].split(">=")[0].split("==")[0].split("<=")[0].split("!=")[0]
    return name.strip().lower() or None


def test_requirements_no_duplicate_packages() -> None:
    req_path = ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt missing"
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    with req_path.open() as fh:
        for lineno, raw_line in enumerate(fh, 1):
            pkg = _pkg_name(raw_line)
            if pkg is None:
                continue
            if pkg in seen:
                duplicates.append(f"'{pkg}' at lines {seen[pkg]} and {lineno}")
            else:
                seen[pkg] = lineno
    assert not duplicates, "Duplicate packages in requirements.txt:\n" + "\n".join(duplicates)


# ── 5. Validation script existence and syntax ─────────────────────────────────

def test_validation_script_exists() -> None:
    p = ROOT / "scripts" / "paper" / "run_repo_validation.py"
    assert p.exists(), "scripts/paper/run_repo_validation.py missing"


def test_validation_script_syntax() -> None:
    p = ROOT / "scripts" / "paper" / "run_repo_validation.py"
    source = p.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"Syntax error in run_repo_validation.py: {exc}")
