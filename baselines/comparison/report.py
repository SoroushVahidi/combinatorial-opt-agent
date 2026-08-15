"""Markdown / CSV / JSON report generation. No pandas dependency.

`generate_report` never fabricates a value: a system with zero ingested
rows appears only in the availability/resource sections, never in the
metrics tables with a 0% placeholder.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.comparison.availability import AVAILABILITY
from baselines.comparison.failure_taxonomy import to_top_level
from baselines.comparison.manifests import load_common_manifest, pamop_empirical_manifest_note
from baselines.comparison.metrics import END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY, SHARED_METRICS
from baselines.comparison.resource_profile import RESOURCE_PROFILES
from baselines.comparison.schema import CellState, UnifiedRow, is_measured
from baselines.comparison.statistics import wilson_interval

REPORT_TITLE = "PRELIMINARY_EXTERNAL_BASELINE_STATUS"


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _cell(value: Any) -> str:
    return str(value)


def availability_table() -> tuple[list[str], list[list[Any]]]:
    header = ["system", "status_label", "has_empirical_nlp4lp_rows", "classification", "detail"]
    rows = [[a.system, a.status_label, a.has_empirical_nlp4lp_rows, a.classification, a.detail] for a in AVAILABILITY.values()]
    return header, rows


def resource_profile_table() -> tuple[list[str], list[list[Any]]]:
    header = ["system", "compute", "external_llm_api", "deterministic", "model_size_params", "solver",
               "test_time_learning", "rollouts_per_problem", "training_required_for_faithful_result", "notes"]
    rows = []
    for p in RESOURCE_PROFILES.values():
        rows.append([p.system, p.compute, p.external_llm_api, p.deterministic, p.model_size_params, p.solver,
                     p.test_time_learning, p.rollouts_per_problem, p.training_required_for_faithful_result,
                     "; ".join(p.notes)])
    return header, rows


def native_metrics_table(rows_by_system: dict[str, list[UnifiedRow]]) -> tuple[list[str], list[list[Any]]]:
    header = ["system", "problem_id", "metric_name", "value"]
    out: list[list[Any]] = []
    for system, rows in rows_by_system.items():
        for row in rows:
            for name, value in row.native_metrics.items():
                out.append([system, row.problem_id, name, value])
    return header, out


def shared_metrics_table(rows_by_system: dict[str, list[UnifiedRow]]) -> tuple[list[str], list[list[Any]]]:
    header = ["metric_name", "system", "n_evaluable", "rate_or_state"]
    out: list[list[Any]] = []
    for metric in SHARED_METRICS:
        for system in metric.applicable_systems:
            rows = rows_by_system.get(system, [])
            if not rows:
                out.append([metric.name, system, 0, CellState.PENDING])
                continue
            result = metric.compute(rows)
            out.append([metric.name, system, result["n"], result["rate"]])
    return header, out


def failure_summary_table(rows_by_system: dict[str, list[UnifiedRow]]) -> tuple[list[str], list[list[Any]]]:
    header = ["system", "problem_id", "native_failure_category", "top_level_category"]
    out: list[list[Any]] = []
    for system, rows in rows_by_system.items():
        for row in rows:
            if row.failure_category not in (CellState.NOT_APPLICABLE, None):
                out.append([system, row.problem_id, row.failure_category, to_top_level(row.failure_category)])
    return header, out


def paired_results_table(pairings: list[Any]) -> tuple[list[str], list[list[Any]]]:
    header = ["metric_name", "system_a", "system_b", "n_paired", "both_succeed", "a_only", "b_only",
              "neither", "mcnemar_p_value", "mcnemar_note"]
    out: list[list[Any]] = []
    for p in pairings:
        from baselines.comparison.statistics import mcnemar_exact
        mc = mcnemar_exact(p.table)
        out.append([p.metric_name, p.system_a, p.system_b, len(p.paired_problem_ids), p.table.both_succeed,
                    p.table.a_only, p.table.b_only, p.table.neither, mc.p_value, mc.note])
    return header, out


def wilson_intervals_table(rows_by_system: dict[str, list[UnifiedRow]]) -> tuple[list[str], list[list[Any]]]:
    header = ["metric_name", "system", "n", "point_estimate", "ci_lower_95", "ci_upper_95"]
    out: list[list[Any]] = []
    for metric in SHARED_METRICS:
        for system in metric.applicable_systems:
            rows = rows_by_system.get(system, [])
            if not rows:
                continue
            result = metric.compute(rows)
            if result["n"] == 0 or result["rate"] == CellState.NOT_APPLICABLE:
                continue
            successes = round(result["rate"] * result["n"])
            ci = wilson_interval(successes, result["n"])
            out.append([metric.name, system, result["n"], ci.point_estimate, ci.lower, ci.upper])
    return header, out


def generate_report(
    output_dir: Path | str, rows_by_system: dict[str, list[UnifiedRow]], *, pairings: list[Any] | None = None,
    git_sha: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairings = pairings or []
    common_manifest = load_common_manifest()

    files: dict[str, Path] = {}

    header, rows = availability_table()
    files["availability.csv"] = output_dir / "availability.csv"
    _write_csv(files["availability.csv"], header, rows)

    header, rows = native_metrics_table(rows_by_system)
    files["native_metrics.csv"] = output_dir / "native_metrics.csv"
    _write_csv(files["native_metrics.csv"], header, rows)

    header, rows = shared_metrics_table(rows_by_system)
    files["shared_metrics.csv"] = output_dir / "shared_metrics.csv"
    _write_csv(files["shared_metrics.csv"], header, rows)

    header, rows = resource_profile_table()
    files["resource_profile.csv"] = output_dir / "resource_profile.csv"
    _write_csv(files["resource_profile.csv"], header, rows)

    header, rows = paired_results_table(pairings)
    files["paired_results.csv"] = output_dir / "paired_results.csv"
    _write_csv(files["paired_results.csv"], header, rows)

    header, rows = failure_summary_table(rows_by_system)
    files["failure_summary.csv"] = output_dir / "failure_summary.csv"
    _write_csv(files["failure_summary.csv"], header, rows)

    header, rows = wilson_intervals_table(rows_by_system)
    files["confidence_intervals.csv"] = output_dir / "confidence_intervals.csv"
    _write_csv(files["confidence_intervals.csv"], header, rows)

    comparison_json = {
        "report_title": REPORT_TITLE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "common_manifest": common_manifest,
        "availability": {k: v.to_dict() for k, v in AVAILABILITY.items()},
        "resource_profiles": {k: v.to_dict() for k, v in RESOURCE_PROFILES.items()},
        "end_to_end_objective_success_eligibility": END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY,
        "row_counts_by_system": {k: len(v) for k, v in rows_by_system.items()},
        "rows_by_system": {k: [r.to_dict() for r in v] for k, v in rows_by_system.items()},
        "pairings": [p.to_dict() for p in pairings],
    }
    files["comparison.json"] = output_dir / "comparison.json"
    files["comparison.json"].write_text(json.dumps(comparison_json, indent=2, sort_keys=True, default=str), encoding="utf-8")

    files["comparison.md"] = output_dir / "comparison.md"
    files["comparison.md"].write_text(_render_markdown(rows_by_system, pairings, common_manifest, git_sha), encoding="utf-8")

    files["README.md"] = output_dir / "README.md"
    files["README.md"].write_text(_render_readme(), encoding="utf-8")

    return files


def _render_readme() -> str:
    return (
        "# External baseline comparison — generated report\n\n"
        "This directory is regenerated by `python -m baselines.comparison.cli`. "
        "Do not hand-edit its files; edit the generator in `baselines/comparison/` "
        "and regenerate instead. See `comparison.md` for the narrative report "
        "and `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` for the frozen "
        "protocol (manifest, metric definitions, run-selection rules, "
        "no-cherry-picking policy) this report follows.\n"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(
    rows_by_system: dict[str, list[UnifiedRow]], pairings: list[Any], common_manifest: dict[str, Any], git_sha: str | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# External Baseline Comparison\n")
    lines.append(f"**Status: `{REPORT_TITLE}`** — not a final paper comparison. Generated "
                  f"{datetime.now(timezone.utc).isoformat()}, repository HEAD `{git_sha or 'UNKNOWN'}`.\n")

    lines.append("## Evaluation protocol\n")
    lines.append("See `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` for the frozen protocol this "
                  "report follows (metric definitions, run-selection rules, proxy semantics, "
                  "no-cherry-picking policy).\n")

    lines.append("## Common benchmark manifest\n")
    lines.append(f"- pilot_ids (n={len(common_manifest['pilot_ids'])}): `{common_manifest['pilot_ids']}`")
    lines.append(f"- future_evaluation_ids (n={len(common_manifest['future_evaluation_ids'])}): `{common_manifest['future_evaluation_ids']}`")
    lines.append(f"- source_subset: `{common_manifest['source_subset']}`\n")
    lines.append(f"**Known divergence:** {pamop_empirical_manifest_note(common_manifest)}\n")

    lines.append("## System availability\n")
    lines.append("| System | Status | Has empirical NLP4LP rows | Classification |")
    lines.append("|---|---|---|---|")
    for a in AVAILABILITY.values():
        lines.append(f"| {a.system} | {a.status_label} | {a.has_empirical_nlp4lp_rows} | `{a.classification}` |")
    lines.append("")

    lines.append("## Implementation fidelity\n")
    lines.append("| System | Rows | Fidelity levels seen |")
    lines.append("|---|---|---|")
    for system, rows in rows_by_system.items():
        levels = sorted({r.implementation_fidelity for r in rows}) if rows else []
        lines.append(f"| {system} | {len(rows)} | {', '.join(levels) if levels else CellState.PENDING} |")
    lines.append("")

    lines.append("## Native metrics\n")
    lines.append("Native metrics are NOT comparable numerically across systems -- see "
                  "`docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` §metric taxonomy. "
                  "Full values are in `native_metrics.csv`.\n")

    lines.append("## Shared end-to-end metrics\n")
    lines.append("| Metric | System | n | Rate/state |")
    lines.append("|---|---|---|---|")
    for metric in SHARED_METRICS:
        for system in metric.applicable_systems:
            rows = rows_by_system.get(system, [])
            if not rows:
                lines.append(f"| {metric.name} | {system} | 0 | {CellState.PENDING} |")
                continue
            result = metric.compute(rows)
            lines.append(f"| {metric.name} | {system} | {result['n']} | {_fmt(result['rate'])} |")
    lines.append("")

    lines.append("## Paired comparison results\n")
    if not pairings:
        lines.append(f"{CellState.NOT_APPLICABLE} — no two systems currently share empirical rows on the same problem_ids.\n")
    else:
        lines.append("| Metric | A | B | n paired | both | A only | B only | neither | McNemar p |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        from baselines.comparison.statistics import mcnemar_exact
        for p in pairings:
            mc = mcnemar_exact(p.table)
            lines.append(f"| {p.metric_name} | {p.system_a} | {p.system_b} | {len(p.paired_problem_ids)} | "
                          f"{p.table.both_succeed} | {p.table.a_only} | {p.table.b_only} | {p.table.neither} | {_fmt(mc.p_value)} |")
    lines.append("")

    lines.append("## Resource requirements\n")
    lines.append("| System | Compute | Solver | Test-time learning | Training required for faithful result |")
    lines.append("|---|---|---|---|---|")
    for p in RESOURCE_PROFILES.values():
        lines.append(f"| {p.system} | {p.compute} | {p.solver} | {p.test_time_learning} | {p.training_required_for_faithful_result} |")
    lines.append("")

    lines.append("## Failure analysis\n")
    lines.append("See `failure_summary.csv` for the full per-row native-category -> top-level-bucket mapping.\n")

    lines.append("## Important fairness caveats\n")
    lines.append("- `ours` performs fixed-catalog scalar grounding, not full NL-to-model generation; it is "
                  "excluded from every SharedMetric above (see `END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY` "
                  "in `metrics.py`).")
    lines.append("- PaMOP's objective-value agreement is an exact-match proxy on execution-successful rows "
                  "only, never a structural/semantic correctness judgment.")
    lines.append("- ORLM has 18 real common-18 rows but **no solver execution** (coptpy not installed); its "
                  "executable/feasible/objective-agreement cells are NOT_APPLICABLE, never zero. OptMATH/DeepOR/OR-R1 "
                  "currently have **zero** empirical rows; any non-zero number for them in this report would be "
                  "fabricated and must be treated as a bug.\n")

    lines.append("## OR-R1 transductive-protocol note\n")
    lines.append("The official OR-R1 TGRPO training set is the union of all official evaluation test sets, "
                  "including all 242 official NLP4LP rows (verified by direct file inspection, "
                  "`docs/ORR1_PROVENANCE.md`). Any future OR-R1 empirical row in this report MUST carry "
                  "`transductive_training=True` and must not be compared to an inductively-evaluated system "
                  "without this caveat restated in the same table.\n")

    lines.append("## Missing experiments / blockers\n")
    for a in AVAILABILITY.values():
        if not a.has_empirical_nlp4lp_rows:
            lines.append(f"- **{a.system}**: {a.detail}")
    lines.append("")

    lines.append("## Provenance\n")
    lines.append(f"- Repository HEAD at generation time: `{git_sha or 'UNKNOWN'}`")
    lines.append("- Generator: `baselines/comparison/report.py` via `python -m baselines.comparison.cli`")
    lines.append("- Protocol: `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md`\n")

    return "\n".join(lines) + "\n"
