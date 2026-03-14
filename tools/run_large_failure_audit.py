"""Large-scale failure audit driver.

Runs the current full deterministic grounding system against a large combined
pool of cases, then labels each failure with a rich taxonomy and writes
organized artifact files.

Sources tested
--------------
A. Group 1 hard-family cases (14 cases, 4 ablation levels)
B. Group 3 clause-parallel cases (17 cases, 4 ablation levels)
C. Large synthetic stress cases (tools/build_large_stress_cases.py)

Easy-family cases from build_easy_family_synthetic_cases.py are static-
analysis cases without an end-to-end slot-value oracle; they are included
in the summary statistics using the curated baseline counts from
evaluate_easy_error_families.py, but not run end-to-end.

Usage::

    python tools/run_large_failure_audit.py
    python tools/run_large_failure_audit.py --out results/failure_audit

Artifacts written
-----------------
<out_dir>/failure_audit.csv                — one row per case, all fields
<out_dir>/failure_summary_by_group.csv     — grouped counts by taxonomy category
<out_dir>/failure_summary_by_tag.csv       — grouped counts by secondary tag
<out_dir>/failure_examples_by_group.md     — human-readable grouped examples
<out_dir>/failure_examples_by_tag.md       — human-readable examples by tag
<out_dir>/overall_audit_report.md          — top-level narrative report
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "results" / "failure_audit"

# ---------------------------------------------------------------------------
# Rich failure taxonomy
# ---------------------------------------------------------------------------

# Import TAXONOMY_GROUPS from the case generator to avoid duplication.
# This is safe because build_large_stress_cases has no runtime imports from
# this module.
from tools.build_large_stress_cases import TAXONOMY_GROUPS  # noqa: E402

# All known secondary tags
SECONDARY_TAGS: list[str] = [
    "wrong_schema",
    "wrong_role_family",
    "wrong_measure_family",
    "wrong_entity_family",
    "sibling_swap",
    "cross_clause_contamination",
    "missed_count_word",
    "missed_enumeration_count",
    "percent_normalization_error",
    "lower_upper_reversal",
    "total_to_coeff_confusion",
    "coeff_to_total_confusion",
    "distractor_number",
    "unresolved_near_tie",
    "missing_value",
    "under_specified",
    "ambiguous",
    "numeric_suffix_entity",
    "reversed_clause_order",
    "uncategorized",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AuditRecord:
    """One row in the failure audit."""

    case_id: str
    source: str             # group1 | group3 | stress | easy_curated
    family: str             # primary taxonomy category
    secondary_tags: list[str]
    query: str
    mode: str               # basic | ops | semantic | full
    gold_summary: str
    predicted_summary: str
    passed: bool
    type_match: float
    exact20: float
    diagnostic_note: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["secondary_tags"] = ";".join(self.secondary_tags)
        return d


# ---------------------------------------------------------------------------
# Failure-labeling heuristics
# ---------------------------------------------------------------------------

def _label_failure(
    case_id: str,
    family: str,
    secondary_tags: list[str],
    gold: dict[str, float],
    pred: dict[str, float],
    passed: bool,
) -> tuple[list[str], str]:
    """Return (secondary_tags, diagnostic_note) for a case result.

    Heuristics are deterministic and transparent — they augment the per-case
    secondary_tags supplied by the stress-case generator with evidence from
    the actual predictions.  Labels are approximate.
    """
    tags = list(secondary_tags)  # start from generator-supplied tags
    notes: list[str] = []

    if passed:
        return tags, "passed"

    # Check for missing values
    missing = [s for s in gold if pred.get(s) is None]
    if missing:
        if "missing_value" not in tags:
            tags.append("missing_value")
        notes.append(f"missing_slots={missing}")

    # Check for sibling swaps: gold[A]==pred[B] and gold[B]==pred[A]
    gold_vals = list(gold.values())
    pred_vals = [pred.get(s) for s in gold]
    if len(gold) == 2 and all(v is not None for v in pred_vals):
        ga, gb = gold_vals
        pa, pb = pred_vals
        if pa is not None and pb is not None:
            if abs(pa - gb) < 1e-6 and abs(pb - ga) < 1e-6:
                if "sibling_swap" not in tags:
                    tags.append("sibling_swap")
                notes.append("exact_swap_detected")

    # Check for percent normalization: prediction 100× too large
    for slot, exp in gold.items():
        if exp > 0 and exp < 1.0:
            pr = pred.get(slot)
            if pr is not None and abs(pr - exp * 100) < 1e-6:
                if "percent_normalization_error" not in tags:
                    tags.append("percent_normalization_error")
                notes.append(f"percent_x100_slot={slot}")
                break

    # Check for lower/upper reversal: slots ending in Min/Max swapped
    min_slots = [s for s in gold if s.lower().startswith("min") or "min" in s.lower()]
    max_slots = [s for s in gold if s.lower().startswith("max") or "max" in s.lower()]
    if min_slots and max_slots:
        for ms in min_slots:
            for xs in max_slots:
                gmin, gmax = gold.get(ms), gold.get(xs)
                pmin, pmax = pred.get(ms), pred.get(xs)
                if (gmin is not None and gmax is not None
                        and pmin is not None and pmax is not None):
                    if abs(pmin - gmax) < 1e-6 and abs(pmax - gmin) < 1e-6:
                        if "lower_upper_reversal" not in tags:
                            tags.append("lower_upper_reversal")
                        notes.append(f"min_max_swap={ms}/{xs}")
                        break

    # Check total/per-unit confusion: large value in per-unit slot
    for slot, exp in gold.items():
        pr = pred.get(slot)
        if pr is None:
            continue
        slot_lower = slot.lower()
        if ("per" in slot_lower or "each" in slot_lower or "coeff" in slot_lower):
            if pr > 100 * exp and exp > 0:
                if "total_to_coeff_confusion" not in tags:
                    tags.append("total_to_coeff_confusion")
                notes.append(f"total_in_coeff_slot={slot}")
                break
        if ("total" in slot_lower or "budget" in slot_lower or "capacity" in slot_lower):
            if pr < exp / 10 and exp > 100:
                if "coeff_to_total_confusion" not in tags:
                    tags.append("coeff_to_total_confusion")
                notes.append(f"coeff_in_total_slot={slot}")
                break

    if not tags:
        tags = ["uncategorized"]
    if not notes:
        notes = ["mismatch_no_specific_pattern"]

    return tags, "; ".join(notes)


# ---------------------------------------------------------------------------
# Conversion helpers (Group1/Group3 result -> AuditRecord)
# ---------------------------------------------------------------------------

def _g1_to_audit(result: Any, source: str = "group1") -> AuditRecord:
    """Convert a Group1 CaseResult to an AuditRecord."""
    gold = result.case.expected
    pred = result.predictions
    passed = result.all_correct
    family = _map_g1_family(result.case.family)
    secondary = _initial_secondary(result.case.family)

    # Determine per-mode secondary tags via heuristics
    final_tags, note = _label_failure(
        result.case.name, family, secondary, gold, pred, passed
    )

    gold_summary = "; ".join(f"{k}={v}" for k, v in sorted(gold.items()))
    pred_summary = "; ".join(f"{k}={v:.4g}" for k, v in sorted(pred.items()))

    return AuditRecord(
        case_id=result.case.name,
        source=source,
        family=family,
        secondary_tags=final_tags,
        query=result.case.query,
        mode=result.mode,
        gold_summary=gold_summary,
        predicted_summary=pred_summary,
        passed=passed,
        type_match=round(result.type_match, 4),
        exact20=round(result.exact20, 4),
        diagnostic_note=note,
    )


def _g3_to_audit(result: Any, source: str = "group3") -> AuditRecord:
    """Convert a Group3 CaseResult to an AuditRecord."""
    gold = result.case.expected
    pred = result.predictions
    passed = result.all_correct
    family = _map_g3_family(result.case.family)
    secondary = _initial_secondary_g3(result.case.family)

    final_tags, note = _label_failure(
        result.case.name, family, secondary, gold, pred, passed
    )

    gold_summary = "; ".join(f"{k}={v}" for k, v in sorted(gold.items()))
    pred_summary = "; ".join(f"{k}={v:.4g}" for k, v in sorted(pred.items()))

    return AuditRecord(
        case_id=result.case.name,
        source=source,
        family=family,
        secondary_tags=final_tags,
        query=result.case.query,
        mode=result.mode,
        gold_summary=gold_summary,
        predicted_summary=pred_summary,
        passed=passed,
        type_match=round(result.type_match, 4),
        exact20=round(result.exact20, 4),
        diagnostic_note=note,
    )


def _map_g1_family(g1_family: str) -> str:
    return {
        "distractor_role": "hard_wrong_assignment",
        "measure_family": "hard_wrong_assignment",
        "total_perunit": "easy_total_vs_perunit",
        "easy_regression": "mixed_or_other",
    }.get(g1_family, "mixed_or_other")


def _map_g3_family(g3_family: str) -> str:
    return {
        "case7_feed_entity": "hard_wrong_assignment",
        "case7_product": "hard_wrong_assignment",
        "case8_two_measure": "hard_swapped_quantities",
        "no_overfire": "mixed_or_other",
    }.get(g3_family, "mixed_or_other")


def _initial_secondary(g1_family: str) -> list[str]:
    return {
        "distractor_role": ["distractor_number", "wrong_role_family"],
        "measure_family": ["wrong_measure_family"],
        "total_perunit": ["total_to_coeff_confusion"],
        "easy_regression": [],
    }.get(g1_family, [])


def _initial_secondary_g3(g3_family: str) -> list[str]:
    return {
        "case7_feed_entity": ["wrong_entity_family", "sibling_swap"],
        "case7_product": ["wrong_entity_family", "numeric_suffix_entity"],
        "case8_two_measure": ["sibling_swap", "cross_clause_contamination"],
        "no_overfire": [],
    }.get(g3_family, [])


# ---------------------------------------------------------------------------
# Stress-case evaluation
# ---------------------------------------------------------------------------

def _evaluate_stress_case(case: Any, mode: str) -> AuditRecord:
    """Run one stress case end-to-end and return an AuditRecord."""
    from tools.relation_aware_linking import run_relation_aware_grounding

    if not case.slots or not case.expected:
        # Static-only case — cannot evaluate end-to-end
        return AuditRecord(
            case_id=case.id,
            source="stress_static",
            family=case.category,
            secondary_tags=list(case.secondary_tags),
            query=case.query,
            mode=mode,
            gold_summary="(no slot oracle)",
            predicted_summary="(not run)",
            passed=False,  # conservative: mark as not evaluable
            type_match=0.0,
            exact20=0.0,
            diagnostic_note="static_analysis_only_no_oracle",
        )

    _, mentions, _ = run_relation_aware_grounding(
        case.query, "orig", case.slots, ablation_mode=mode
    )
    pred = {sn: m.value for sn, m in mentions.items()}
    gold = case.expected

    total = len(gold)
    correct = sum(
        1 for s, v in gold.items()
        if pred.get(s) is not None and abs(pred[s] - v) < 1e-6
    )
    exact20_count = sum(
        1 for s, v in gold.items()
        if pred.get(s) is not None and (
            abs(pred[s] - v) / abs(v) <= 0.20 if v != 0 else pred[s] == 0
        )
    )
    type_match = correct / total if total > 0 else 0.0
    exact20 = exact20_count / total if total > 0 else 0.0
    passed = (correct == total)

    final_tags, note = _label_failure(
        case.id, case.category, list(case.secondary_tags), gold, pred, passed
    )

    gold_summary = "; ".join(f"{k}={v}" for k, v in sorted(gold.items()))
    pred_summary = "; ".join(f"{k}={v:.4g}" for k, v in sorted(pred.items()))

    return AuditRecord(
        case_id=case.id,
        source="stress",
        family=case.category,
        secondary_tags=final_tags,
        query=case.query,
        mode=mode,
        gold_summary=gold_summary,
        predicted_summary=pred_summary,
        passed=passed,
        type_match=round(type_match, 4),
        exact20=round(exact20, 4),
        diagnostic_note=note,
    )


# ---------------------------------------------------------------------------
# Artifact generators
# ---------------------------------------------------------------------------

def _write_failure_audit_csv(records: list[AuditRecord], out_dir: Path) -> None:
    path = out_dir / "failure_audit.csv"
    fieldnames = list(AuditRecord.__dataclass_fields__.keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())


def _write_group_summary_csv(records: list[AuditRecord], out_dir: Path) -> None:
    path = out_dir / "failure_summary_by_group.csv"
    # Only full-mode records for summary
    full_records = [r for r in records if r.mode == "full"]

    group_totals: Counter = Counter()
    group_passed: Counter = Counter()
    for r in full_records:
        group_totals[r.family] += 1
        if r.passed:
            group_passed[r.family] += 1

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group", "total_tested", "passed", "failed", "failure_rate",
        ])
        for grp in TAXONOMY_GROUPS:
            total = group_totals[grp]
            passed = group_passed[grp]
            failed = total - passed
            rate = failed / total if total > 0 else 0.0
            writer.writerow([grp, total, passed, failed, f"{rate:.3f}"])


def _write_tag_summary_csv(records: list[AuditRecord], out_dir: Path) -> None:
    path = out_dir / "failure_summary_by_tag.csv"
    # Only full-mode failures
    failures = [r for r in records if r.mode == "full" and not r.passed]

    tag_counts: Counter = Counter()
    for r in failures:
        for tag in r.secondary_tags:
            tag_counts[tag] += 1

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["secondary_tag", "failure_count"])
        for tag, count in tag_counts.most_common():
            writer.writerow([tag, count])


def _write_examples_by_group_md(records: list[AuditRecord], out_dir: Path) -> None:
    path = out_dir / "failure_examples_by_group.md"
    full_failures = [r for r in records if r.mode == "full" and not r.passed]
    by_group: dict[str, list[AuditRecord]] = defaultdict(list)
    for r in full_failures:
        by_group[r.family].append(r)

    lines: list[str] = ["# Failure Examples by Group (full mode)\n"]
    for grp in TAXONOMY_GROUPS:
        cases = by_group.get(grp, [])
        lines.append(f"## {grp} ({len(cases)} failures)\n")
        if not cases:
            lines.append("*No failures in full mode.*\n")
            continue
        for r in cases[:5]:  # cap at 5 examples per group
            lines.append(f"### {r.case_id} [{r.source}]\n")
            lines.append(f"**Query**: {r.query}\n")
            lines.append(f"**Gold**: `{r.gold_summary}`\n")
            lines.append(f"**Pred**: `{r.predicted_summary}`\n")
            lines.append(f"**Tags**: {', '.join(r.secondary_tags)}\n")
            lines.append(f"**Note**: {r.diagnostic_note}\n")
    path.write_text("\n".join(lines))


def _write_examples_by_tag_md(records: list[AuditRecord], out_dir: Path) -> None:
    path = out_dir / "failure_examples_by_tag.md"
    full_failures = [r for r in records if r.mode == "full" and not r.passed]
    by_tag: dict[str, list[AuditRecord]] = defaultdict(list)
    for r in full_failures:
        for tag in r.secondary_tags:
            by_tag[tag].append(r)

    lines: list[str] = ["# Failure Examples by Secondary Tag (full mode)\n"]
    for tag in SECONDARY_TAGS:
        cases = by_tag.get(tag, [])
        if not cases:
            continue
        lines.append(f"## {tag} ({len(cases)} failures)\n")
        for r in cases[:3]:
            lines.append(f"- **{r.case_id}** [{r.family}]: {r.diagnostic_note}\n")
    path.write_text("\n".join(lines))


def _write_overall_report_md(
    records: list[AuditRecord],
    sources: dict[str, int],
    out_dir: Path,
) -> None:
    path = out_dir / "overall_audit_report.md"
    full_records = [r for r in records if r.mode == "full"]
    full_runnable = [r for r in full_records if r.source not in ("stress_static",)]
    full_pass = sum(1 for r in full_runnable if r.passed)
    full_fail = sum(1 for r in full_runnable if not r.passed)
    total_runnable = len(full_runnable)
    pass_rate = full_pass / total_runnable if total_runnable > 0 else 0.0

    # Per-group stats (runnable only)
    group_stats: dict[str, dict[str, int]] = {}
    for grp in TAXONOMY_GROUPS:
        grp_r = [r for r in full_runnable if r.family == grp]
        group_stats[grp] = {
            "total": len(grp_r),
            "passed": sum(1 for r in grp_r if r.passed),
            "failed": sum(1 for r in grp_r if not r.passed),
        }

    # Top failure tags
    tag_counts: Counter = Counter()
    for r in full_runnable:
        if not r.passed:
            for tag in r.secondary_tags:
                tag_counts[tag] += 1
    top_tags = tag_counts.most_common(10)

    # Ablation comparison (group1 + group3 sources)
    ablation_lines: list[str] = []
    for mode in ("basic", "ops", "semantic", "full"):
        mode_r = [r for r in records if r.mode == mode and r.source in ("group1", "group3")]
        if not mode_r:
            continue
        n = len(mode_r)
        p = sum(1 for r in mode_r if r.passed)
        ablation_lines.append(f"| {mode} | {p}/{n} | {p/n:.2f} |")

    lines = [
        "# Overall Failure Audit Report",
        "",
        f"Generated by: `tools/run_large_failure_audit.py`",
        "",
        "## Data Sources",
        "",
    ]
    for src, count in sorted(sources.items()):
        lines.append(f"- **{src}**: {count} cases (full mode)")
    lines += [
        "",
        "## Overall Results (full mode, runnable cases only)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total runnable tested | {total_runnable} |",
        f"| Passed | {full_pass} |",
        f"| Failed | {full_fail} |",
        f"| Pass rate | {pass_rate:.2f} |",
        "",
        "## Ablation Comparison (Group 1 + Group 3 targeted cases)",
        "",
        "| Mode | Passed/Total | Pass Rate |",
        "|------|-------------|-----------|",
    ] + ablation_lines + [
        "",
        "## Per-Group Results (full mode)",
        "",
        "| Group | Total | Passed | Failed | Fail Rate |",
        "|-------|-------|--------|--------|-----------|",
    ]
    for grp in TAXONOMY_GROUPS:
        s = group_stats[grp]
        rate = s["failed"] / s["total"] if s["total"] > 0 else 0.0
        lines.append(f"| {grp} | {s['total']} | {s['passed']} | {s['failed']} | {rate:.2f} |")

    lines += [
        "",
        "## Top Failure Tags (full mode, runnable failures)",
        "",
        "| Tag | Count |",
        "|-----|-------|",
    ]
    for tag, count in top_tags:
        lines.append(f"| {tag} | {count} |")

    # High-value clusters
    lines += [
        "",
        "## Highest-Value Remaining Failure Clusters",
        "",
    ]
    worst_groups = sorted(
        [g for g in TAXONOMY_GROUPS if group_stats[g]["failed"] > 0],
        key=lambda g: group_stats[g]["failed"],
        reverse=True,
    )
    if worst_groups:
        for grp in worst_groups[:5]:
            s = group_stats[grp]
            rate = s["failed"] / s["total"] if s["total"] > 0 else 0.0
            lines.append(
                f"- **{grp}**: {s['failed']}/{s['total']} failures "
                f"({rate:.0%}) — "
                + _cluster_verdict(grp, rate, s["total"])
            )
    else:
        lines.append("*No failures detected in any group (full mode).*")

    lines += [
        "",
        "## Honest Caveats",
        "",
        "- Easy-family counts come from curated static analysis, not an end-to-end model run.",
        "- Synthetic stress results reflect the current pipeline on new cases, not historical benchmark data.",
        "- Failure labeling heuristics are deterministic but approximate.",
        "- Synthetic pass rates may differ from real benchmark pass rates.",
        "",
    ]

    path.write_text("\n".join(lines))


def _cluster_verdict(grp: str, fail_rate: float, total: int) -> str:
    if fail_rate == 0.0:
        return "solved — stop investing."
    if grp in ("under_specified_template",):
        return "inherently ambiguous — stop investing."
    if grp in ("easy_retrieval",):
        return "mostly solved — sanity checks only."
    if grp in ("hard_wrong_assignment", "hard_swapped_quantities"):
        if fail_rate < 0.10:
            return "mostly solved — one tiny cleanup pass only."
        elif fail_rate < 0.30:
            return "worth another focused pass."
        else:
            return "still significant — investigate root cause."
    if total <= 5:
        return "small sample — interpret with caution."
    if fail_rate >= 0.50:
        return "high failure rate — worth investigating."
    return "partial residuals — likely long-tail / diminishing returns."


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_large_failure_audit(
    out_dir: Path = DEFAULT_OUT,
    modes: tuple[str, ...] = ("basic", "ops", "semantic", "full"),
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full large failure audit and write artifacts.

    Returns a dict with keys:
        records     — list of AuditRecord (all modes)
        sources     — dict source -> full-mode count
        out_dir     — Path to written artifacts
    """
    from tools.evaluate_group1_hard_family import run_evaluation as g1_run
    from tools.evaluate_group3_clause_parallel import run_evaluation as g3_run
    from tools.build_large_stress_cases import get_runnable_cases, get_all_stress_cases

    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[AuditRecord] = []

    # -----------------------------------------------------------------------
    # A. Group 1 cases
    # -----------------------------------------------------------------------
    if verbose:
        print("[1/3] Group 1 hard-family cases...")
    g1_result = g1_run(modes=modes, verbose=False)
    for r in g1_result["results"]:
        all_records.append(_g1_to_audit(r, source="group1"))

    # -----------------------------------------------------------------------
    # B. Group 3 cases
    # -----------------------------------------------------------------------
    if verbose:
        print("[2/3] Group 3 clause-parallel cases...")
    g3_result = g3_run(modes=modes, verbose=False)
    for r in g3_result["results"]:
        all_records.append(_g3_to_audit(r, source="group3"))

    # -----------------------------------------------------------------------
    # C. Large synthetic stress cases
    # -----------------------------------------------------------------------
    runnable = get_runnable_cases()
    all_stress = get_all_stress_cases()
    static_cases = [c for c in all_stress if not c.slots or not c.expected]

    if verbose:
        print(f"[3/3] Stress cases: {len(runnable)} runnable + {len(static_cases)} static...")
    for case in runnable:
        for mode in modes:
            rec = _evaluate_stress_case(case, mode)
            all_records.append(rec)
    # Add static cases (full mode only, marked not-run)
    for case in static_cases:
        rec = _evaluate_stress_case(case, "full")
        all_records.append(rec)

    # -----------------------------------------------------------------------
    # Source counts (full mode only)
    # -----------------------------------------------------------------------
    full_records = [r for r in all_records if r.mode == "full"]
    sources: dict[str, int] = Counter(r.source for r in full_records)

    # -----------------------------------------------------------------------
    # Write artifacts
    # -----------------------------------------------------------------------
    if verbose:
        print(f"Writing artifacts to {out_dir} ...")

    _write_failure_audit_csv(all_records, out_dir)
    _write_group_summary_csv(all_records, out_dir)
    _write_tag_summary_csv(all_records, out_dir)
    _write_examples_by_group_md(all_records, out_dir)
    _write_examples_by_tag_md(all_records, out_dir)
    _write_overall_report_md(all_records, dict(sources), out_dir)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    if verbose:
        full_runnable_records = [
            r for r in full_records if r.source not in ("stress_static",)
        ]
        passed = sum(1 for r in full_runnable_records if r.passed)
        total = len(full_runnable_records)
        print(
            f"\n{'='*60}\n"
            f"LARGE FAILURE AUDIT SUMMARY (full mode, runnable)\n"
            f"{'='*60}"
        )
        print(f"  Total tested:  {total}")
        print(f"  Passed:        {passed}")
        print(f"  Failed:        {total - passed}")
        print(f"  Pass rate:     {passed/total:.2f}" if total > 0 else "  (no cases)")
        print()
        print(f"  Per-group (full mode, runnable):")
        grp_totals: Counter = Counter()
        grp_passed: Counter = Counter()
        for r in full_runnable_records:
            grp_totals[r.family] += 1
            if r.passed:
                grp_passed[r.family] += 1
        for grp in TAXONOMY_GROUPS:
            t = grp_totals[grp]
            p = grp_passed[grp]
            if t == 0:
                continue
            print(f"    {grp:<30}: {p:>3}/{t:<3}  ({p/t:.0%})")
        print(f"\nArtifacts written to {out_dir}")

    return {
        "records": all_records,
        "sources": dict(sources),
        "out_dir": out_dir,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run large-scale failure audit of the grounding system"
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip writing artifact files (for quick testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    run_large_failure_audit(
        out_dir=out_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
