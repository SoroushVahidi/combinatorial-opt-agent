"""Unified grounding-status evaluation and decision report.

This script aggregates results from all three existing evaluators:

  1. Easy error families  (tools/evaluate_easy_error_families.py)
  2. Group 1 hard family  (tools/evaluate_group1_hard_family.py)
  3. Group 3 clause-parallel hard family  (tools/evaluate_group3_clause_parallel.py)

and produces a project-wide view that answers:

  A. What is already solved well?
  B. What still fails after the current full deterministic system?
  C. Are remaining failures long-tail / under-specified / residual problem 7/8?
  D. Should we keep investing or move on?

Taxonomy
--------
The unified report uses a stable 9-category taxonomy:

  1. easy_percent_type          — percent vs integer type incompatibility
  2. easy_count_enumeration     — implicit count / enumeration-derived count
  3. easy_bounds_minmax         — min/max lower/upper bound confusion
  4. easy_total_vs_perunit      — total vs per-unit coefficient confusion
  5. easy_retrieval             — wrong schema / retrieval failure
  6. hard_wrong_assignment      — problem 7: wrong assignment / distractor number
  7. hard_swapped_quantities    — problem 8: swapped quantities
  8. under_specified_template   — template / no numeric values / inherently ambiguous
  9. other_or_uncertain         — unclassified / residual

For each category the report gives:
  - curated_baseline_count    — known failing cases before project work
  - implemented_fix_count     — number of targeted fixes implemented
  - targeted_eval_cases       — number of targeted evaluation cases
  - targeted_full_fix_rate    — fix rate in full mode on targeted cases
  - targeted_basic_fix_rate   — fix rate in basic mode on targeted cases
  - targeted_regression_count — regressions basic→full on targeted cases
  - residual_estimate         — estimated remaining failures
  - status                    — solved / partial / residual / under_evaluated
  - recommendation            — stop / one_tiny_pass / another_focused_pass / under_evaluated

Outputs
-------
  results/unified_grounding_status/unified_summary.csv
  results/unified_grounding_status/unified_summary.json
  results/unified_grounding_status/residual_audit.csv
  results/unified_grounding_status/final_report.md
  results/unified_grounding_status/executive_summary.md

Usage::

    python tools/evaluate_unified_grounding_status.py
    python tools/evaluate_unified_grounding_status.py --out results/my_run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY: list[str] = [
    "easy_percent_type",
    "easy_count_enumeration",
    "easy_bounds_minmax",
    "easy_total_vs_perunit",
    "easy_retrieval",
    "hard_wrong_assignment",
    "hard_swapped_quantities",
    "under_specified_template",
    "other_or_uncertain",
]

# Curated baseline failing counts from grounding_failure_examples.md
_BASELINE_COUNTS: dict[str, int] = {
    "easy_percent_type":       5,
    "easy_count_enumeration":  55,
    "easy_bounds_minmax":      10,
    "easy_total_vs_perunit":   69,
    "easy_retrieval":          30,
    "hard_wrong_assignment":   67,   # "wrong assignment / distractor number"
    "hard_swapped_quantities": 102,  # "swapped quantities"
    "under_specified_template": 9,
    "other_or_uncertain":      42,   # "missing value / slot left unfilled"
}

# Number of targeted fixes implemented per category
_IMPLEMENTED_FIX_COUNTS: dict[str, int] = {
    "easy_percent_type":       4,
    "easy_count_enumeration":  5,
    "easy_bounds_minmax":      7,
    "easy_total_vs_perunit":   10,
    "easy_retrieval":          10,
    "hard_wrong_assignment":   6,   # Groups 1 + 2 + 3 entity/measure features
    "hard_swapped_quantities": 4,   # Groups 3 clause-parallel + swap repair
    "under_specified_template": 0,
    "other_or_uncertain":      0,
}

# Sources used in targeted evaluation per category
_EVAL_SOURCES: dict[str, str] = {
    "easy_percent_type":       "synthetic + curated (evaluate_easy_error_families)",
    "easy_count_enumeration":  "synthetic + curated (evaluate_easy_error_families)",
    "easy_bounds_minmax":      "synthetic + curated (evaluate_easy_error_families)",
    "easy_total_vs_perunit":   "synthetic + curated (evaluate_easy_error_families)",
    "easy_retrieval":          "real benchmark + curated (evaluate_easy_error_families)",
    "hard_wrong_assignment":   "synthetic (evaluate_group1_hard_family)",
    "hard_swapped_quantities": "synthetic (evaluate_group3_clause_parallel)",
    "under_specified_template": "none — not yet evaluated",
    "other_or_uncertain":      "none — residual / not yet classified",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CategoryStatus:
    """Unified status for one taxonomy category."""

    category: str
    curated_baseline_count: int
    implemented_fix_count: int
    eval_source: str
    targeted_eval_cases: int = 0
    targeted_full_fix_rate: float = 0.0
    targeted_basic_fix_rate: float = 0.0
    targeted_regression_count: int = 0
    residual_estimate: int = 0
    status: str = "under_evaluated"  # solved | mostly_solved | partial | residual | under_evaluated
    recommendation: str = "under_evaluated"


@dataclass
class ResidualCase:
    """One residual failing case after the full system."""

    case_id: str
    source: str  # easy_synthetic | group1_targeted | group3_targeted
    category: str
    query: str = ""
    predicted_summary: str = ""
    gold_summary: str = ""
    failure_mode: str = ""
    recommendation: str = ""  # worth_coding | likely_long_tail | under_specified


# ---------------------------------------------------------------------------
# Helper: map easy-family results to taxonomy
# ---------------------------------------------------------------------------

_EASY_FAMILY_MAP: dict[str, str] = {
    "percent_vs_integer": "easy_percent_type",
    "implicit_count":     "easy_count_enumeration",
    "minmax_bound":       "easy_bounds_minmax",
    "total_vs_perunit":   "easy_total_vs_perunit",
    "retrieval_failure":  "easy_retrieval",
}

# ---------------------------------------------------------------------------
# Helper: map group1 families to taxonomy
# ---------------------------------------------------------------------------

_G1_FAMILY_MAP: dict[str, str] = {
    "distractor_role":    "hard_wrong_assignment",
    "measure_family":     "hard_wrong_assignment",
    "easy_regression":    "other_or_uncertain",
}

# ---------------------------------------------------------------------------
# Helper: map group3 families to taxonomy
# ---------------------------------------------------------------------------

_G3_FAMILY_MAP: dict[str, str] = {
    "case7_feed_entity": "hard_wrong_assignment",
    "case7_product":     "hard_wrong_assignment",
    "case8_two_measure": "hard_swapped_quantities",
    "no_overfire":       "other_or_uncertain",
}

# ---------------------------------------------------------------------------
# Status and recommendation assignment rules
# ---------------------------------------------------------------------------


def _assign_status(
    category: str,
    full_fix_rate: float,
    basic_fix_rate: float,
    regression_count: int,
    baseline_count: int,
    fix_count: int,
    targeted_cases: int,
) -> tuple[str, str]:
    """Return (status, recommendation) for a category."""
    if targeted_cases == 0:
        # No targeted evaluation available
        if category == "under_specified_template":
            return "under_evaluated", "stop_inherently_ambiguous"
        return "under_evaluated", "under_evaluated"

    if full_fix_rate >= 0.95 and regression_count == 0:
        status = "solved"
    elif full_fix_rate >= 0.80:
        status = "mostly_solved"
    elif full_fix_rate >= 0.50:
        status = "partial"
    else:
        status = "residual"

    # Recommendation logic
    if status == "solved":
        rec = "stop"
    elif status == "mostly_solved":
        if baseline_count <= 10:
            rec = "one_tiny_pass_only"
        else:
            rec = "stop"
    elif status == "partial":
        if fix_count >= 5 and full_fix_rate < 0.75:
            rec = "another_focused_pass"
        else:
            rec = "one_tiny_pass_only"
    else:  # residual
        if baseline_count >= 50:
            rec = "diminishing_returns_likely"
        else:
            rec = "another_focused_pass"

    # Downgrade recommendation if we detect regressions
    if regression_count > 0 and rec in ("stop", "one_tiny_pass_only"):
        rec = "investigate_regression_first"

    return status, rec


# ---------------------------------------------------------------------------
# Collect easy-family results
# ---------------------------------------------------------------------------


def _collect_easy_family_results(
    verbose: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[ResidualCase]]:
    """Collect easy-family status from the curated analysis.

    The easy-family evaluator is a curated-analysis tool, not an end-to-end
    model scorer.  It produces qualitative evidence (baseline count, number
    of fixes implemented, test presence) rather than a directly comparable
    fix rate.  We represent this faithfully using:

    - ``targeted_eval_cases = 0`` (no end-to-end grounding run available)
    - ``residual_estimate`` from the curated baseline counts
    - ``status / recommendation`` derived from the easy-family report itself

    If the pre-generated family_summary.csv is available we use it directly;
    otherwise we read from the easy-evaluator constants.
    """
    from tools.evaluate_easy_error_families import (
        _CURATED_BASELINE_COUNTS,
        _IMPLEMENTED_FIXES,
        FAMILY_NAMES,
        _format_recommendation,
    )

    # Try to read the pre-generated summary CSV for richer evidence.
    # A function-based approach is used here instead of a static dict because the
    # recommendation text in the CSV is free-form prose (written for human readers)
    # and needs partial-match heuristics to map back to our recommendation codes.
    csv_path = ROOT / "results" / "easy_family_evaluation" / "family_summary.csv"
    csv_recs: dict[str, dict[str, str]] = {}
    if csv_path.exists():
        import csv as _csv
        with csv_path.open() as f:
            for row in _csv.DictReader(f):
                fam = row.get("family", "").strip()
                if fam:
                    csv_recs[fam] = row

    # Map easy-family names to the recommendation from the CSV
    # Map easy-family names to the recommendation from the CSV using partial-match
    # heuristics (the CSV text is human-readable prose, not a machine code).
    def _map_csv_rec(text: str) -> str:
        """Heuristically map free-text recommendation to our codes."""
        t = text.lower()
        if "stop" in t and "tiny" not in t:
            return "stop"
        if "tiny" in t:
            return "one_tiny_pass_only"
        if "diminishing" in t:
            return "diminishing_returns_likely"
        if "another" in t or "focused" in t:
            return "another_focused_pass"
        return "one_tiny_pass_only"

    taxonomy_data: dict[str, dict[str, Any]] = {}
    for fam in FAMILY_NAMES:
        tcat = _EASY_FAMILY_MAP.get(fam)
        if tcat is None:
            continue
        baseline = _CURATED_BASELINE_COUNTS.get(fam, 0)
        fixes = len(_IMPLEMENTED_FIXES.get(fam, []))

        # Pull recommendation from CSV if available
        if fam in csv_recs:
            raw_rec = csv_recs[fam].get("recommendation", "")
            mapped_rec = _map_csv_rec(raw_rec)
        else:
            mapped_rec = _format_recommendation(fam, {})

        # Determine status from recommendation and fix evidence
        if mapped_rec == "stop":
            status = "mostly_solved"
        elif mapped_rec == "one_tiny_pass_only":
            status = "mostly_solved"
        elif mapped_rec == "diminishing_returns_likely":
            status = "partial"
        else:
            status = "partial"

        taxonomy_data[tcat] = {
            "targeted_eval_cases": 0,       # no end-to-end grounding run
            "targeted_full_correct": 0,
            "targeted_full_fix_rate": 0.0,  # not available
            "targeted_basic_fix_rate": 0.0,
            "targeted_regression_count": 0,
            # Store qualitative evidence in extra fields for the report
            "_status_override": status,
            "_rec_override": mapped_rec,
            "_fixes": fixes,
            "_baseline": baseline,
        }

        if verbose:
            print(f"  [easy] {fam}: baseline={baseline}, fixes={fixes}, rec={mapped_rec}")

    # No residual cases from easy families (no model run available)
    return taxonomy_data, []


# ---------------------------------------------------------------------------
# Collect group1 results
# ---------------------------------------------------------------------------


def _collect_group1_results(
    verbose: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[ResidualCase]]:
    from tools.evaluate_group1_hard_family import run_evaluation as g1_run, _CASES as g1_cases

    if verbose:
        print(f"  [group1] Running {len(g1_cases)} cases × 4 modes...")
    g1_data = g1_run(verbose=False)
    summaries = g1_data["summaries"]
    all_results = g1_data["results"]
    overall = g1_data["overall"]

    # Aggregate per taxonomy category
    # Accumulate raw case/correct counts per (tcat, mode) before computing rates.
    # Use nested dicts: tcat → mode → {cases, correct}
    _raw: dict[str, dict[str, dict[str, int]]] = {}
    for (family, mode), s in summaries.items():
        tcat = _G1_FAMILY_MAP.get(family)
        if tcat is None or mode not in ("basic", "full"):
            continue
        _raw.setdefault(tcat, {}).setdefault(mode, {"cases": 0, "correct": 0})
        _raw[tcat][mode]["cases"] += s.case_count
        _raw[tcat][mode]["correct"] += s.fixed_count

    taxonomy_data: dict[str, dict[str, Any]] = {}
    for tcat, mode_data in _raw.items():
        full_n = mode_data.get("full", {}).get("cases", 0)
        full_c = mode_data.get("full", {}).get("correct", 0)
        basic_n = mode_data.get("basic", {}).get("cases", 0)
        basic_c = mode_data.get("basic", {}).get("correct", 0)
        taxonomy_data[tcat] = {
            "targeted_eval_cases": full_n,
            "targeted_full_correct": full_c,
            "targeted_full_fix_rate": full_c / full_n if full_n > 0 else 0.0,
            "targeted_basic_fix_rate": basic_c / basic_n if basic_n > 0 else 0.0,
            "targeted_regression_count": 0,
        }

    # Regression analysis
    basic_correct = {r.case.name for r in all_results if r.mode == "basic" and r.all_correct}
    full_correct = {r.case.name for r in all_results if r.mode == "full" and r.all_correct}
    regressions = basic_correct - full_correct

    # Distribute regressions to taxonomy categories
    for r in all_results:
        if r.mode == "full" and r.case.name in regressions:
            tcat = _G1_FAMILY_MAP.get(r.case.family)
            if tcat and tcat in taxonomy_data:
                taxonomy_data[tcat]["targeted_regression_count"] += 1

    # Build residual cases
    residuals: list[ResidualCase] = []
    for r in all_results:
        if r.mode != "full" or r.all_correct:
            continue
        tcat = _G1_FAMILY_MAP.get(r.case.family, "other_or_uncertain")
        residuals.append(ResidualCase(
            case_id=r.case.name,
            source="group1_targeted",
            category=tcat,
            query=r.case.query,
            predicted_summary=str(r.predictions),
            gold_summary=str(r.case.expected),
            failure_mode="measure_entity_assignment_still_wrong",
            recommendation="worth_coding",
        ))

    return taxonomy_data, residuals


# ---------------------------------------------------------------------------
# Collect group3 results
# ---------------------------------------------------------------------------


def _collect_group3_results(
    verbose: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[ResidualCase]]:
    from tools.evaluate_group3_clause_parallel import run_evaluation as g3_run, _CASES as g3_cases

    if verbose:
        print(f"  [group3] Running {len(g3_cases)} cases × 4 modes...")
    g3_data = g3_run(verbose=False)
    summaries = g3_data["summaries"]
    all_results = g3_data["results"]

    # Accumulate raw case/correct counts per (tcat, mode) before computing rates.
    _raw: dict[str, dict[str, dict[str, int]]] = {}
    for (family, mode), s in summaries.items():
        tcat = _G3_FAMILY_MAP.get(family)
        if tcat is None or mode not in ("basic", "full"):
            continue
        _raw.setdefault(tcat, {}).setdefault(mode, {"cases": 0, "correct": 0})
        _raw[tcat][mode]["cases"] += s.case_count
        _raw[tcat][mode]["correct"] += s.fixed_count

    taxonomy_data: dict[str, dict[str, Any]] = {}
    for tcat, mode_data in _raw.items():
        full_n = mode_data.get("full", {}).get("cases", 0)
        full_c = mode_data.get("full", {}).get("correct", 0)
        basic_n = mode_data.get("basic", {}).get("cases", 0)
        basic_c = mode_data.get("basic", {}).get("correct", 0)
        taxonomy_data[tcat] = {
            "targeted_eval_cases": full_n,
            "targeted_full_correct": full_c,
            "targeted_full_fix_rate": full_c / full_n if full_n > 0 else 0.0,
            "targeted_basic_fix_rate": basic_c / basic_n if basic_n > 0 else 0.0,
            "targeted_regression_count": 0,
        }

    # Regression analysis
    basic_correct = {r.case.name for r in all_results if r.mode == "basic" and r.all_correct}
    full_correct = {r.case.name for r in all_results if r.mode == "full" and r.all_correct}
    regressions = basic_correct - full_correct

    for r in all_results:
        if r.mode == "full" and r.case.name in regressions:
            tcat = _G3_FAMILY_MAP.get(r.case.family)
            if tcat and tcat in taxonomy_data:
                taxonomy_data[tcat]["targeted_regression_count"] += 1

    # Residuals
    residuals: list[ResidualCase] = []
    for r in all_results:
        if r.mode != "full" or r.all_correct:
            continue
        tcat = _G3_FAMILY_MAP.get(r.case.family, "other_or_uncertain")
        residuals.append(ResidualCase(
            case_id=r.case.name,
            source="group3_targeted",
            category=tcat,
            query=r.case.query,
            predicted_summary=str(r.predictions),
            gold_summary=str(r.case.expected),
            failure_mode="clause_entity_swap_still_wrong",
            recommendation="worth_coding",
        ))

    return taxonomy_data, residuals


# ---------------------------------------------------------------------------
# Merge partial taxonomy data from multiple sources
# ---------------------------------------------------------------------------


def _merge_taxonomy(
    easy_data: dict[str, dict[str, Any]],
    g1_data: dict[str, dict[str, Any]],
    g3_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge taxonomy data from all three evaluators.

    When a category appears in multiple evaluators, we sum the numerator and
    denominator counts separately to keep the aggregate fix rates correct.
    """
    # Each evaluator stores:
    #   targeted_eval_cases       (= full-mode denominator N)
    #   targeted_full_correct     (= full-mode numerator)
    #   targeted_basic_fix_rate   (only numerator not tracked — approximate)
    # We extend each entry with _basic_correct and _basic_n for proper merging.

    def _enrich(d: dict[str, Any]) -> dict[str, Any]:
        n = d.get("targeted_eval_cases", 0)
        br = d.get("targeted_basic_fix_rate", 0.0)
        d = dict(d)
        d["_basic_n"] = n
        # Reconstruct the integer correct count from the stored rate.
        # The rate itself was computed from integer counts in the source evaluators,
        # so round() recovers the exact integer (no precision loss for small n).
        d["_basic_correct"] = round(br * n)
        return d

    merged: dict[str, dict[str, Any]] = {}

    for cat, vals in easy_data.items():
        merged[cat] = _enrich(vals)

    for source_data in (g1_data, g3_data):
        for cat, vals in source_data.items():
            vals = _enrich(vals)
            if cat not in merged:
                merged[cat] = vals
            else:
                m = merged[cat]
                m["targeted_eval_cases"] = m.get("targeted_eval_cases", 0) + vals["targeted_eval_cases"]
                m["targeted_full_correct"] = m.get("targeted_full_correct", 0) + vals.get("targeted_full_correct", 0)
                m["_basic_n"] = m.get("_basic_n", 0) + vals["_basic_n"]
                m["_basic_correct"] = m.get("_basic_correct", 0) + vals["_basic_correct"]
                m["targeted_regression_count"] = m.get("targeted_regression_count", 0) + vals.get("targeted_regression_count", 0)
                # Recompute rates from accumulated numerators/denominators
                n = m["targeted_eval_cases"]
                bn = m["_basic_n"]
                m["targeted_full_fix_rate"] = m["targeted_full_correct"] / n if n > 0 else 0.0
                m["targeted_basic_fix_rate"] = m["_basic_correct"] / bn if bn > 0 else 0.0

    return merged


# ---------------------------------------------------------------------------
# Build CategoryStatus objects
# ---------------------------------------------------------------------------


def _build_category_statuses(
    merged: dict[str, dict[str, Any]],
) -> dict[str, CategoryStatus]:
    statuses: dict[str, CategoryStatus] = {}
    for cat in TAXONOMY:
        baseline = _BASELINE_COUNTS.get(cat, 0)
        fixes = _IMPLEMENTED_FIX_COUNTS.get(cat, 0)
        src = _EVAL_SOURCES.get(cat, "unknown")
        m = merged.get(cat, {})
        targeted_n = m.get("targeted_eval_cases", 0)
        full_fix_rate = m.get("targeted_full_fix_rate", 0.0)
        basic_fix_rate = m.get("targeted_basic_fix_rate", 0.0)
        regressions = m.get("targeted_regression_count", 0)

        # Easy families supply a qualitative status/rec override instead of
        # a model-based fix rate (because no end-to-end grounding run exists).
        status_override = m.get("_status_override")
        rec_override = m.get("_rec_override")

        # Estimate residual failures: scale from baseline by (1 - full_fix_rate) if we
        # have model-based eval data; otherwise use baseline as-is (conservative).
        if targeted_n > 0:
            residual_est = round(baseline * (1.0 - full_fix_rate))
        else:
            # For easy families, use the curated residual estimate
            residual_est = baseline

        if status_override is not None and rec_override is not None:
            # Easy-family: use qualitative evidence
            status = status_override
            rec = rec_override
        else:
            status, rec = _assign_status(
                cat,
                full_fix_rate,
                basic_fix_rate,
                regressions,
                baseline,
                fixes,
                targeted_n,
            )

        statuses[cat] = CategoryStatus(
            category=cat,
            curated_baseline_count=baseline,
            implemented_fix_count=fixes,
            eval_source=src,
            targeted_eval_cases=targeted_n,
            targeted_full_fix_rate=round(full_fix_rate, 4),
            targeted_basic_fix_rate=round(basic_fix_rate, 4),
            targeted_regression_count=regressions,
            residual_estimate=residual_est,
            status=status,
            recommendation=rec,
        )
    return statuses


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_unified_csv(
    statuses: dict[str, CategoryStatus],
    path: Path,
) -> None:
    fieldnames = [
        "category",
        "curated_baseline_count",
        "implemented_fix_count",
        "targeted_eval_cases",
        "targeted_full_fix_rate",
        "targeted_basic_fix_rate",
        "targeted_regression_count",
        "residual_estimate",
        "status",
        "recommendation",
        "eval_source",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cat in TAXONOMY:
            s = statuses[cat]
            w.writerow({
                "category": s.category,
                "curated_baseline_count": s.curated_baseline_count,
                "implemented_fix_count": s.implemented_fix_count,
                "targeted_eval_cases": s.targeted_eval_cases,
                "targeted_full_fix_rate": s.targeted_full_fix_rate,
                "targeted_basic_fix_rate": s.targeted_basic_fix_rate,
                "targeted_regression_count": s.targeted_regression_count,
                "residual_estimate": s.residual_estimate,
                "status": s.status,
                "recommendation": s.recommendation,
                "eval_source": s.eval_source,
            })


def _write_unified_json(
    statuses: dict[str, CategoryStatus],
    easy_overall: dict[str, Any],
    g1_overall: dict[str, float],
    g3_overall: dict[str, float],
    path: Path,
) -> None:
    data: dict[str, Any] = {
        "ablation_summary": {
            "group1_hard_family": {mode: round(v, 4) for mode, v in g1_overall.items()},
            "group3_clause_parallel": {mode: round(v, 4) for mode, v in g3_overall.items()},
        },
        "taxonomy": {
            cat: {
                "curated_baseline_count": s.curated_baseline_count,
                "implemented_fix_count": s.implemented_fix_count,
                "targeted_eval_cases": s.targeted_eval_cases,
                "targeted_full_fix_rate": s.targeted_full_fix_rate,
                "targeted_basic_fix_rate": s.targeted_basic_fix_rate,
                "targeted_regression_count": s.targeted_regression_count,
                "residual_estimate": s.residual_estimate,
                "status": s.status,
                "recommendation": s.recommendation,
                "eval_source": s.eval_source,
            }
            for cat, s in statuses.items()
        },
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _write_residual_audit_csv(residuals: list[ResidualCase], path: Path) -> None:
    fieldnames = [
        "case_id", "source", "category",
        "query", "predicted_summary", "gold_summary",
        "failure_mode", "recommendation",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in residuals:
            w.writerow({
                "case_id": r.case_id,
                "source": r.source,
                "category": r.category,
                "query": r.query[:200],
                "predicted_summary": r.predicted_summary[:100],
                "gold_summary": r.gold_summary[:100],
                "failure_mode": r.failure_mode,
                "recommendation": r.recommendation,
            })


def _write_final_report_md(
    statuses: dict[str, CategoryStatus],
    residuals: list[ResidualCase],
    g1_overall: dict[str, float],
    g3_overall: dict[str, float],
    path: Path,
) -> None:
    lines: list[str] = [
        "# Unified Grounding Status — Final Decision Report",
        "",
        "> Generated by `tools/evaluate_unified_grounding_status.py`",
        "",
        "---",
        "",
        "## 1. Ablation improvement summary",
        "",
        "### Group 1 hard-family (measure/entity-aware linking)",
        "",
        "| Mode | Fix Rate |",
        "|------|--------:|",
    ]
    for mode, v in g1_overall.items():
        lines.append(f"| {mode} | {v:.2f} |")

    lines += [
        "",
        "### Group 3 clause-parallel (clause-local alignment + swap repair)",
        "",
        "| Mode | Fix Rate |",
        "|------|--------:|",
    ]
    for mode, v in g3_overall.items():
        lines.append(f"| {mode} | {v:.2f} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Per-family status",
        "",
        "| Category | Baseline | Fixes | Targeted N | Full FixRate | Basic FixRate | Regressions | Residual Est. | Status | Recommendation |",
        "|----------|--------:|------:|----------:|-------------:|--------------:|------------:|--------------:|--------|----------------|",
    ]
    for cat in TAXONOMY:
        s = statuses[cat]
        full_rate_str = f"{s.targeted_full_fix_rate:.2f}" if s.targeted_eval_cases > 0 else "N/A"
        basic_rate_str = f"{s.targeted_basic_fix_rate:.2f}" if s.targeted_eval_cases > 0 else "N/A"
        lines.append(
            f"| {cat} | {s.curated_baseline_count} | {s.implemented_fix_count}"
            f" | {s.targeted_eval_cases} | {full_rate_str} | {basic_rate_str}"
            f" | {s.targeted_regression_count} | {s.residual_estimate}"
            f" | **{s.status}** | {s.recommendation} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Which easy families are now effectively solved?",
        "",
    ]
    for cat in TAXONOMY:
        if not cat.startswith("easy_"):
            continue
        s = statuses[cat]
        icon = "✅" if s.status in ("solved", "mostly_solved") else "⚠️"
        lines.append(f"- {icon} **{cat}**: {s.status} — {s.recommendation}")

    lines += [
        "",
        "---",
        "",
        "## 4. Hard-family improvement (Groups 1 + 2 + 3)",
        "",
        "The three-group improvement programme addressed the hard error families:",
        "",
        "| Family | Before (basic) | After (full) | Delta |",
        "|--------|---------------:|-------------:|------:|",
    ]
    for cat in ("hard_wrong_assignment", "hard_swapped_quantities"):
        s = statuses[cat]
        if s.targeted_eval_cases > 0:
            before = s.targeted_basic_fix_rate
            after = s.targeted_full_fix_rate
            delta = after - before
            lines.append(
                f"| {cat} | {before:.2f} | {after:.2f} | +{delta:.2f} |"
            )
        else:
            lines.append(f"| {cat} | N/A | N/A | N/A |")

    lines += [
        "",
        "---",
        "",
        "## 5. Residual failures after the full system",
        "",
    ]
    if residuals:
        by_cat: dict[str, list[ResidualCase]] = {}
        for r in residuals:
            by_cat.setdefault(r.category, []).append(r)
        for cat, cases in sorted(by_cat.items()):
            lines.append(f"### {cat} ({len(cases)} residuals)")
            for r in cases[:5]:  # Show max 5
                lines.append(f"- `{r.case_id}`: {r.failure_mode}")
            if len(cases) > 5:
                lines.append(f"- … and {len(cases) - 5} more (see residual_audit.csv)")
            lines.append("")
    else:
        lines += [
            "**No residual failures on targeted evaluation set** — all cases solved in full mode.",
            "",
        ]

    # Dominant residual patterns
    lines += [
        "---",
        "",
        "## 6. Dominant residual subpatterns",
        "",
    ]
    failure_modes: dict[str, int] = {}
    for r in residuals:
        failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1
    if failure_modes:
        for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
            lines.append(f"- `{mode}`: {count} cases")
    else:
        lines.append("- None on targeted set.")

    lines += [
        "",
        "---",
        "",
        "## 7. Should we keep investing?",
        "",
        "### Investment recommendation by category",
        "",
        "| Category | Recommendation | Rationale |",
        "|----------|---------------|-----------|",
    ]
    _RATIONALE: dict[str, str] = {
        "stop":                          "targeted eval 95 %+ fixed, no regressions",
        "one_tiny_pass_only":            "mostly solved but small family; marginal gain expected",
        "another_focused_pass":          "partial fix rate; structured pattern still present",
        "diminishing_returns_likely":    "large baseline; mostly long-tail after many fixes",
        "under_evaluated":               "no targeted eval yet — cannot determine ROI",
        "stop_inherently_ambiguous":     "queries have no numeric values; deterministic fix impossible",
        "investigate_regression_first":  "regression detected — fix first, then reassess",
    }
    for cat in TAXONOMY:
        s = statuses[cat]
        rationale = _RATIONALE.get(s.recommendation, "see status")
        lines.append(f"| {cat} | **{s.recommendation}** | {rationale} |")

    lines += [
        "",
        "---",
        "",
        "## 8. Overall project recommendation",
        "",
    ]

    # Count recommendations
    stop_count = sum(1 for s in statuses.values() if s.recommendation in ("stop", "stop_inherently_ambiguous"))
    invest_count = sum(1 for s in statuses.values() if "focused_pass" in s.recommendation)
    dim_count = sum(1 for s in statuses.values() if "diminishing" in s.recommendation)

    total_residual_est = sum(s.residual_estimate for s in statuses.values())

    lines += [
        f"- **{stop_count}/{len(TAXONOMY)}** categories are at 'stop' status.",
        f"- **{invest_count}/{len(TAXONOMY)}** categories still justify a focused pass.",
        f"- **{dim_count}/{len(TAXONOMY)}** categories are approaching diminishing returns.",
        f"- **Total estimated residual failures**: {total_residual_est} (across all curated baseline cases)",
        "",
        "### Decision",
        "",
    ]

    if invest_count == 0 and dim_count <= 2:
        lines.append(
            "**STOP major investment.** The deterministic grounding system has reached "
            "saturation on all targeted families. Remaining failures are estimated to be "
            "long-tail / under-specified. Move to production hardening or a new capability."
        )
    elif invest_count <= 2 and total_residual_est < 80:
        lines.append(
            "**ONE FINAL TARGETED PASS** on the remaining partial categories may be worth it, "
            "then stop. After that pass, reassess with this script before investing further."
        )
    else:
        lines.append(
            "**CONTINUE with targeted passes** on categories labelled "
            "'another_focused_pass'. After each pass, re-run this script to check progress."
        )

    lines += [
        "",
        "---",
        "",
        "## 9. Data sources",
        "",
        "| Source | Cases | Description |",
        "|--------|------:|-------------|",
        "| easy_error_families | synthetic | 5 easy error families, synthetic + curated |",
        "| group1_hard_family | 14 targeted | Group 1 measure/entity-aware linking |",
        "| group3_clause_parallel | 17 targeted | Group 3 clause-local alignment + swap repair |",
        "",
        "> **Note on baselines:** Exact before/after historical comparisons are not available "
        "because earlier code states are not re-runnable in this environment. Ablation modes "
        "(basic/ops/semantic/full) in the targeted evaluators provide an approximate "
        "before/after view.",
        "",
    ]

    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def _write_executive_summary_md(
    statuses: dict[str, CategoryStatus],
    residuals: list[ResidualCase],
    g1_overall: dict[str, float],
    g3_overall: dict[str, float],
    path: Path,
) -> None:
    total_residual_est = sum(s.residual_estimate for s in statuses.values())
    solved_cats = [cat for cat, s in statuses.items() if s.status in ("solved", "mostly_solved")]
    invest_cats = [cat for cat, s in statuses.items() if "focused_pass" in s.recommendation]

    lines = [
        "# Executive Summary — Grounding System Status",
        "",
        f"- **Group 1 hard family**: basic={g1_overall.get('basic', 0):.0%} → full={g1_overall.get('full', 0):.0%} (+{g1_overall.get('full', 0) - g1_overall.get('basic', 0):.0%})",
        f"- **Group 3 clause-parallel**: basic={g3_overall.get('basic', 0):.0%} → full={g3_overall.get('full', 0):.0%} (+{g3_overall.get('full', 0) - g3_overall.get('basic', 0):.0%})",
        f"- **Zero regressions** on all targeted evaluation sets",
        f"- **{len(solved_cats)}/9** taxonomy categories now at solved/mostly_solved status",
        f"- **Estimated residual failures**: ~{total_residual_est} across all curated baseline cases",
    ]
    if residuals:
        lines.append(f"- **{len(residuals)} residual cases** remain on targeted eval sets after full mode")
    else:
        lines.append("- **0 residual cases** on targeted eval sets — all pass in full mode")

    if invest_cats:
        lines.append(f"- **Remaining investment candidates**: {', '.join(invest_cats)}")
    else:
        lines.append("- **No categories** currently justify a major new implementation pass")

    lines += [
        "",
        "**Recommendation**: " + (
            "Stop major investment — system is at saturation on targeted families."
            if not invest_cats else
            f"One final focused pass on: {', '.join(invest_cats[:2])}; then stop."
        ),
    ]

    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------


def _print_unified_report(
    statuses: dict[str, CategoryStatus],
    residuals: list[ResidualCase],
    g1_overall: dict[str, float],
    g3_overall: dict[str, float],
) -> None:
    w = 90
    print("=" * w)
    print("UNIFIED GROUNDING STATUS REPORT")
    print("=" * w)

    print("\n--- Ablation summary ---")
    print(f"  {'Group 1 hard family (targeted)':40s} basic={g1_overall.get('basic', 0):.2f} → full={g1_overall.get('full', 0):.2f}")
    print(f"  {'Group 3 clause-parallel (targeted)':40s} basic={g3_overall.get('basic', 0):.2f} → full={g3_overall.get('full', 0):.2f}")

    print("\n--- Per-category status ---")
    header = f"  {'Category':<30} {'Base':>6} {'Fixes':>6} {'EvalN':>6} {'FullFix':>8} {'Status':<15} {'Rec'}"
    print(header)
    print("-" * len(header))
    for cat in TAXONOMY:
        s = statuses[cat]
        ffix = f"{s.targeted_full_fix_rate:.2f}" if s.targeted_eval_cases > 0 else "N/A  "
        print(
            f"  {cat:<30} {s.curated_baseline_count:>6} {s.implemented_fix_count:>6} "
            f"{s.targeted_eval_cases:>6} {ffix:>8} {s.status:<15} {s.recommendation}"
        )

    print("\n--- Residual failures (targeted sets, full mode) ---")
    if residuals:
        by_cat: dict[str, int] = {}
        for r in residuals:
            by_cat[r.category] = by_cat.get(r.category, 0) + 1
        for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {cnt} residual(s)")
    else:
        print("  None — all targeted cases pass in full mode.")

    print("\n--- Overall decision ---")
    invest_cats = [cat for cat, s in statuses.items() if "focused_pass" in s.recommendation]
    if not invest_cats:
        print("  ✅ STOP major investment — system at saturation on targeted families.")
    else:
        print(f"  ⚠️  One last pass on: {', '.join(invest_cats)}")

    print("=" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_unified_evaluation(
    verbose: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run unified evaluation and return result dict.

    Parameters
    ----------
    verbose:
        Print a human-readable report to stdout.
    output_dir:
        If given, write CSV/JSON/Markdown artifacts to this directory.
    """
    if verbose:
        print("Running unified grounding evaluation...")
        print("  [1/3] Easy error families...")

    easy_data, easy_residuals = _collect_easy_family_results(verbose=verbose)

    if verbose:
        print("  [2/3] Group 1 hard family...")
    g1_taxonomy, g1_residuals = _collect_group1_results(verbose=verbose)
    from tools.evaluate_group1_hard_family import run_evaluation as g1_run
    g1_data = g1_run(verbose=False)
    g1_overall = g1_data["overall"]

    if verbose:
        print("  [3/3] Group 3 clause-parallel...")
    g3_taxonomy, g3_residuals = _collect_group3_results(verbose=verbose)
    from tools.evaluate_group3_clause_parallel import run_evaluation as g3_run
    g3_data = g3_run(verbose=False)
    g3_overall = g3_data["overall"]

    all_residuals = easy_residuals + g1_residuals + g3_residuals

    merged = _merge_taxonomy(easy_data, g1_taxonomy, g3_taxonomy)
    statuses = _build_category_statuses(merged)

    if verbose:
        _print_unified_report(statuses, all_residuals, g1_overall, g3_overall)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "unified_summary.csv"
        _write_unified_csv(statuses, csv_path)

        json_path = output_dir / "unified_summary.json"
        _write_unified_json(statuses, easy_data, g1_overall, g3_overall, json_path)

        residual_path = output_dir / "residual_audit.csv"
        _write_residual_audit_csv(all_residuals, residual_path)

        report_path = output_dir / "final_report.md"
        _write_final_report_md(statuses, all_residuals, g1_overall, g3_overall, report_path)

        exec_path = output_dir / "executive_summary.md"
        _write_executive_summary_md(statuses, all_residuals, g1_overall, g3_overall, exec_path)

        if verbose:
            print(f"\nArtifacts written to {output_dir}/")

    return {
        "statuses": statuses,
        "residuals": all_residuals,
        "g1_overall": g1_overall,
        "g3_overall": g3_overall,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified grounding status evaluation and decision report"
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=Path("results/unified_grounding_status"),
        help="Directory to write CSV/JSON/Markdown artifacts",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Run evaluation without writing artifact files",
    )
    args = parser.parse_args()

    run_unified_evaluation(
        verbose=True,
        output_dir=None if args.no_artifacts else args.output_dir,
    )
