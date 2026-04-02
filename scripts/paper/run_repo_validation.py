#!/usr/bin/env python3
"""Canonical repository validation script for EAAI paper support.

Checks that all required paper artifacts, source-of-truth documents, and
key result files are present and structurally sound.  Does NOT rerun the
full benchmark (that requires the gated NLP4LP dataset on Hugging Face).

Usage
-----
    python scripts/paper/run_repo_validation.py
    python scripts/paper/run_repo_validation.py --run-tests

Exit codes
----------
    0  all checks passed
    1  one or more checks failed
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"

_passed: list[str] = []
_failed: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")
    _passed.append(msg)


def fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")
    _failed.append(msg)


def warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")
    _warnings.append(msg)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_source_of_truth_docs() -> None:
    print("\n[1] Source-of-truth documents")
    required = [
        "docs/EAAI_SOURCE_OF_TRUTH.md",
        "README.md",
        "REPO_STRUCTURE.md",
        "CONTRIBUTING.md",
    ]
    for rel in required:
        p = ROOT / rel
        if p.exists():
            ok(rel)
        else:
            fail(f"{rel}  (missing)")


def check_paper_tables() -> None:
    print("\n[2] EAAI camera-ready tables")
    tables_dir = ROOT / "results" / "paper" / "eaai_camera_ready_tables"
    required_tables = [
        "table1_main_benchmark_summary.csv",
        "table2_engineering_structural_subset.csv",
        "table3_executable_attempt_with_blockers.csv",
        "table4_final_solver_backed_subset.csv",
        "table5_failure_taxonomy.csv",
    ]
    for name in required_tables:
        p = tables_dir / name
        if not p.exists():
            fail(f"results/paper/eaai_camera_ready_tables/{name}  (missing)")
            continue
        try:
            rows = list(csv.reader(p.open()))
        except Exception as exc:
            fail(f"{name}: unreadable CSV — {exc}")
            continue
        if len(rows) < 2:
            fail(f"{name}: CSV has fewer than 2 rows")
        else:
            ok(f"{name}  ({len(rows) - 1} data rows)")


def check_paper_figures() -> None:
    print("\n[3] EAAI camera-ready figures")
    figs_dir = ROOT / "results" / "paper" / "eaai_camera_ready_figures"
    for fig_num in range(1, 5):
        matches = list(figs_dir.glob(f"figure{fig_num}_*.png"))
        if matches:
            ok(f"figure{fig_num} PNG: {matches[0].name}")
        else:
            warn(f"figure{fig_num} PNG not found in {figs_dir.relative_to(ROOT)}")


def check_analysis_reports() -> None:
    print("\n[4] EAAI analysis reports")
    required = [
        "analysis/eaai_engineering_subset_report.md",
        "analysis/eaai_executable_subset_report.md",
        "analysis/eaai_final_solver_attempt_report.md",
        "analysis/eaai_tables_build_report.md",
        "analysis/eaai_figures_build_report.md",
    ]
    for rel in required:
        p = ROOT / rel
        if p.exists():
            ok(rel)
        else:
            fail(f"{rel}  (missing)")


def check_experiment_scripts() -> None:
    print("\n[5] Canonical experiment scripts")
    scripts = [
        "tools/run_eaai_engineering_subset_experiment.py",
        "tools/run_eaai_executable_subset_experiment.py",
        "tools/run_eaai_final_solver_attempt.py",
        "tools/build_eaai_camera_ready_figures.py",
    ]
    for rel in scripts:
        p = ROOT / rel
        if p.exists():
            ok(rel)
        else:
            fail(f"{rel}  (missing)")


def check_downstream_utility() -> None:
    print("\n[6] Core downstream utility (import check)")
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0,''); import tools.nlp4lp_downstream_utility"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            ok("tools/nlp4lp_downstream_utility  (importable)")
        else:
            fail(f"tools/nlp4lp_downstream_utility  import failed:\n{result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        warn("tools/nlp4lp_downstream_utility  import timed out (30 s)")


def check_requirements_no_duplicates() -> None:
    print("\n[7] requirements.txt sanity")
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        fail("requirements.txt  (missing)")
        return
    seen: dict[str, int] = {}
    any_dup = False
    with req_path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pkg = stripped.split(">=")[0].split("==")[0].split("<=")[0].strip().lower()
            if pkg in seen:
                fail(
                    f"requirements.txt: duplicate entry '{pkg}' "
                    f"(lines {seen[pkg]} and {lineno})"
                )
                any_dup = True
            else:
                seen[pkg] = lineno
    if not any_dup:
        ok(f"requirements.txt  ({len(seen)} unique packages, no duplicates)")


def check_hf_dataset_note() -> None:
    print("\n[8] Hugging Face dataset dependency (info only)")
    warn(
        "Full benchmark rerun requires the gated NLP4LP dataset.\n"
        "      Set HF_TOKEN in your environment before running benchmark scripts.\n"
        "      See README.md § 'HuggingFace dataset access' for details."
    )


def run_unit_tests() -> None:
    print("\n[9] Unit tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q",
         "--tb=short",
         "--ignore=tests/test_pdf_upload.py",
         "--ignore=tests/test_experiment_rerun.py"],
        cwd=ROOT,
        timeout=300,
    )
    if result.returncode == 0:
        ok("pytest test suite passed")
    else:
        fail("pytest test suite had failures (see output above)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the EAAI companion repository for paper use."
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Also run the pytest unit-test suite (~15 s).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  EAAI Repository Validation")
    print(f"  Root: {ROOT}")
    print("=" * 60)

    check_source_of_truth_docs()
    check_paper_tables()
    check_paper_figures()
    check_analysis_reports()
    check_experiment_scripts()
    check_downstream_utility()
    check_requirements_no_duplicates()
    check_hf_dataset_note()

    if args.run_tests:
        run_unit_tests()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results: {len(_passed)} passed, {len(_failed)} failed, {len(_warnings)} warnings")
    if _failed:
        print(f"\n  {_RED}FAILED checks:{_RESET}")
        for f in _failed:
            print(f"    • {f}")
    if _warnings:
        print(f"\n  {_YELLOW}Warnings:{_RESET}")
        for w in _warnings:
            print(f"    • {w}")
    if not _failed:
        print(f"\n  {_GREEN}All required checks passed.{_RESET}")
    else:
        print(f"\n  {_RED}Repository validation FAILED — see failures above.{_RESET}")
    print("=" * 60)

    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
