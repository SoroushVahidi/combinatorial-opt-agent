"""CLI: `python -m baselines.comparison.cli [options]`.

Discovers result rows only from the explicit, known locations in
`ingest.py` -- never a broad filesystem crawl.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from baselines.comparison.ingest import ingest_all
from baselines.comparison.metrics import objective_agreement_rate
from baselines.comparison.pairing import pair_systems
from baselines.comparison.report import generate_report
from baselines.comparison.validation import select_rows

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALL_SYSTEMS = ("ours", "pamop", "orlm", "optmath", "generic", "deepor", "orr1")


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the cross-baseline external comparison report.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "results/external_baseline_comparison"))
    parser.add_argument("--systems", nargs="+", choices=_ALL_SYSTEMS, default=list(_ALL_SYSTEMS))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any ingested row fails provenance validation.")
    parser.add_argument("--include-pilot", action="store_true", help="No-op placeholder: pilot IDs are always included; reserved for a future full-manifest vs. pilot-only toggle.")
    parser.add_argument("--format", choices=["all", "csv", "json", "md"], default="all")
    parser.add_argument("--validate-only", action="store_true", help="Ingest and validate rows; do not write a report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    git_sha = _git_sha()

    save_subset_to = output_dir / "ours_common18_nlp4lp_orig_tfidf_typed_greedy.csv" if "ours" in args.systems else None
    rows_by_system = ingest_all(systems=args.systems, save_ours_subset_to=save_subset_to)

    manifest_ids = None  # `ours` doc_ids and pamop/etc. problem_ids use different id spaces; validated per-system below.
    accepted_by_system: dict[str, list] = {}
    rejection_report: dict[str, dict[str, list[str]]] = {}
    for system, rows in rows_by_system.items():
        accepted, rejected = select_rows(rows, allow_mock=False)
        accepted_by_system[system] = accepted
        if rejected:
            rejection_report[system] = rejected

    if rejection_report:
        print("Rows rejected during provenance validation:", file=sys.stderr)
        for system, reasons in rejection_report.items():
            for key, why in reasons.items():
                print(f"  {key}: {why}", file=sys.stderr)
        if args.strict:
            return 1

    if args.validate_only:
        print(f"Validated {sum(len(v) for v in accepted_by_system.values())} rows across {len(accepted_by_system)} systems.")
        return 0

    pairings = []
    formulation_systems = [s for s in accepted_by_system if s != "ours"]
    for i, a in enumerate(formulation_systems):
        for b in formulation_systems[i + 1:]:
            if accepted_by_system.get(a) and accepted_by_system.get(b):
                pairings.append(pair_systems(accepted_by_system[a], accepted_by_system[b], metric=lambda r: r.objective_match if isinstance(r.objective_match, bool) else None, metric_name="objective_agreement"))

    files = generate_report(output_dir, accepted_by_system, pairings=pairings, git_sha=git_sha)
    print(f"Report written to {output_dir}:")
    for name, path in files.items():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
