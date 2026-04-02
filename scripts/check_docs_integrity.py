#!/usr/bin/env python3
"""
scripts/check_docs_integrity.py

Lightweight documentation integrity checker for the EAAI companion repo.

Checks:
  1. Conflict markers (<<<<<<, =======, >>>>>>>) in tracked files
  2. References to files that were moved/archived (stale path patterns)
  3. Broken relative markdown links in key canonical docs

No network calls, no experiments, no side effects.

Usage:
    python scripts/check_docs_integrity.py
    python scripts/check_docs_integrity.py --verbose
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files to scan for conflict markers (relative to REPO_ROOT)
CONFLICT_SCAN_GLOBS = ["**/*.md", "**/*.py", "**/*.yaml", "**/*.yml"]

# Directories to skip entirely
SKIP_DIRS = {".git", "__pycache__", ".cursor", "node_modules", "venv", ".venv"}

# Directories to skip for stale-reference checks (archive files may legitimately
# reference old paths; checking them would produce noise, not signal)
STALE_CHECK_SKIP_DIRS = {"docs/archive", "docs/archive_internal_status", "analysis/archive"}

# Stale path patterns: (old_path_fragment, suggested_replacement)
STALE_PATTERNS: List[Tuple[str, str]] = [
    ("analysis/FIGURE_REPRODUCTION_REPORT", "analysis/archive/FIGURE_REPRODUCTION_REPORT"),
    ("analysis/REPO_VALIDATION_REPORT", "analysis/archive/REPO_VALIDATION_REPORT"),
    ("analysis/TABLE_REPRODUCTION_REPORT", "analysis/archive/TABLE_REPRODUCTION_REPORT"),
    ("analysis/binary_cleanup_report", "analysis/archive/binary_cleanup_report"),
    ("analysis/classic_problem_family_performance", "analysis/archive/classic_problem_family_performance"),
    ("analysis/dataset_parallel_work_audit", "analysis/archive/dataset_parallel_work_audit"),
    ("analysis/grounding_failure_examples", "analysis/archive/grounding_failure_examples"),
    ("analysis/missing_normalized_sources_audit", "analysis/archive/missing_normalized_sources_audit"),
    ("analysis/new_dataset_integration_audit", "analysis/archive/new_dataset_integration_audit"),
    ("docs/DATASET_PARALLEL_INTEGRATION_REPORT", "docs/archive/DATASET_PARALLEL_INTEGRATION_REPORT"),
    ("docs/MIGRATION_GOOGLE_GENAI", "docs/archive/MIGRATION_GOOGLE_GENAI"),
    ("docs/NORMALIZED_SOURCE_MATRIX", "docs/archive/NORMALIZED_SOURCE_MATRIX"),
    ("docs/gemini_api_quota", "docs/archive/gemini_api_quota"),
    # Pre-archive stale references (files that no longer exist at docs/ root)
    ("docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS", "docs/archive/NLP4LP_ACCEPTANCE_RERANK_RESULTS"),
    ("docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS", "docs/archive/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS"),
    ("docs/NLP4LP_RELATION_AWARE_METHOD_RESULTS", "docs/archive/NLP4LP_RELATION_AWARE_METHOD_RESULTS"),
    ("docs/global_consistency_grounding", "docs/archive/global_consistency_grounding"),
    ("docs/copilot_vs_our_model_comparison", "docs/archive/copilot_vs_our_model_comparison"),
]

# Files to check for broken relative links (relative to REPO_ROOT)
LINK_CHECK_FILES = [
    "README.md",
    "REPO_STRUCTURE.md",
    "KNOWN_ISSUES.md",
    "HOW_TO_REPRODUCE.md",
    "HOW_TO_RUN_BENCHMARK.md",
    "CONTRIBUTING.md",
    "EXPERIMENTS.md",
    "docs/CURRENT_STATUS.md",
    "docs/EAAI_SOURCE_OF_TRUTH.md",
    "docs/RESULTS_PROVENANCE.md",
    "docs/REVIEWER_GUIDE.md",
    "docs/README.md",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_tracked_files() -> List[Path]:
    """Return all text files in the repo, skipping ignored dirs."""
    results: List[Path] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            p = Path(root) / fname
            results.append(p)
    return results


def _md_files() -> List[Path]:
    results: List[Path] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.endswith(".md"):
                results.append(Path(root) / fname)
    return results


# ---------------------------------------------------------------------------
# Check 1: Conflict markers
# ---------------------------------------------------------------------------


def check_conflict_markers(verbose: bool = False) -> List[str]:
    """Return list of 'file:line: message' strings for conflict markers found."""
    issues: List[str] = []
    marker_re = re.compile(r"^(<{7}|={7}|>{7})", re.MULTILINE)
    for p in _all_tracked_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue
        for m in marker_re.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            rel = p.relative_to(REPO_ROOT)
            issues.append(f"{rel}:{lineno}: conflict marker '{m.group(0)[:7]}'")
    return issues


# ---------------------------------------------------------------------------
# Check 2: Stale path references
# ---------------------------------------------------------------------------


def check_stale_references(verbose: bool = False) -> List[str]:
    """Return issues for references to paths that were moved/archived.

    Archive folders are excluded: stale cross-references inside historical/provenance
    files are expected and are not actionable for a reviewer.
    """
    issues: List[str] = []
    for p in _md_files():
        # Skip archive directories
        rel_str = str(p.relative_to(REPO_ROOT))
        if any(rel_str.startswith(skip) for skip in STALE_CHECK_SKIP_DIRS):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue
        rel = p.relative_to(REPO_ROOT)
        for old_frag, suggestion in STALE_PATTERNS:
            # Only flag if the old path itself no longer exists
            old_abs = REPO_ROOT / old_frag
            exists_with_md = old_abs.with_suffix(".md").exists()
            exists_with_csv = old_abs.with_suffix(".csv").exists()
            exists_exact = old_abs.exists()
            if exists_exact or exists_with_md or exists_with_csv:
                # File still exists at old location; not stale
                continue
            # Check if the text references this old fragment
            if old_frag in text:
                # Find line numbers
                for i, line in enumerate(text.splitlines(), start=1):
                    if old_frag in line:
                        issues.append(
                            f"{rel}:{i}: stale reference '{old_frag}' "
                            f"(now at '{suggestion}')"
                        )
    return issues


# ---------------------------------------------------------------------------
# Check 3: Broken relative markdown links
# ---------------------------------------------------------------------------


def check_broken_links(verbose: bool = False) -> List[str]:
    """Check relative file links in key canonical docs."""
    issues: List[str] = []
    link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

    for rel_path in LINK_CHECK_FILES:
        p = REPO_ROOT / rel_path
        if not p.exists():
            if verbose:
                print(f"  [skip] {rel_path} (file not found)")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            for m in link_re.finditer(line):
                target = m.group(2)
                # Skip anchors, absolute URLs, mailto
                if target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                # Strip inline anchor
                target_path = target.split("#")[0]
                if not target_path:
                    continue
                # Resolve relative to the containing file's directory
                resolved = (p.parent / target_path).resolve()
                if not resolved.exists():
                    issues.append(
                        f"{rel_path}:{i}: broken link '{target}' "
                        f"(resolved: {resolved.relative_to(REPO_ROOT)})"
                    )
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Check docs integrity")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Docs Integrity Check")
    print(f"Repo root: {REPO_ROOT}")
    print("=" * 60)

    total_issues = 0

    # Check 1
    print("\n[1/3] Checking for conflict markers ...")
    conflict_issues = check_conflict_markers(args.verbose)
    if conflict_issues:
        for issue in conflict_issues:
            print(f"  ERROR: {issue}")
        total_issues += len(conflict_issues)
    else:
        print("  OK — no conflict markers found")

    # Check 2
    print("\n[2/3] Checking for stale path references ...")
    stale_issues = check_stale_references(args.verbose)
    if stale_issues:
        for issue in stale_issues:
            print(f"  WARN: {issue}")
        total_issues += len(stale_issues)
    else:
        print("  OK — no stale references found")

    # Check 3
    print("\n[3/3] Checking for broken relative links in canonical docs ...")
    link_issues = check_broken_links(args.verbose)
    if link_issues:
        for issue in link_issues:
            print(f"  WARN: {issue}")
        total_issues += len(link_issues)
    else:
        print("  OK — no broken links found")

    print("\n" + "=" * 60)
    if total_issues == 0:
        print("All checks passed.")
        return 0
    else:
        print(f"{total_issues} issue(s) found. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
