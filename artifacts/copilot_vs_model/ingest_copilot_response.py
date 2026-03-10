#!/usr/bin/env python3
"""Helper to ingest a Copilot JSON response for one benchmark case.

Usage:
    # Paste JSON response via stdin:
    echo '{"predicted_problem_type": "lp", ...}' | python ingest_copilot_response.py --case-id nlp4lp_test_0

    # Or from a file:
    python ingest_copilot_response.py --case-id nlp4lp_test_0 --file my_response.json

    # After pasting all responses, run the scorer:
    python score_comparison.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OUTPUTS_FILE = ROOT / "copilot_outputs.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True, help="case_id from benchmark_cases.jsonl")
    ap.add_argument("--file", default=None, help="Path to JSON file with Copilot response (default: stdin)")
    args = ap.parse_args()

    if args.file:
        raw = Path(args.file).read_text()
    else:
        print(f"Paste Copilot raw response for case '{args.case_id}' then press Ctrl-D:", file=sys.stderr)
        raw = sys.stdin.read()

    raw = raw.strip()
    parsed = None
    error = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        error = str(e)

    # Load existing outputs
    outputs = []
    if OUTPUTS_FILE.exists():
        for line in OUTPUTS_FILE.open():
            outputs.append(json.loads(line))

    # Find and update the matching case
    found = False
    for entry in outputs:
        if entry["case_id"] == args.case_id:
            entry["raw_response"] = raw
            entry["parsed"] = parsed
            entry["parse_error"] = error
            found = True
            break

    if not found:
        outputs.append({
            "case_id": args.case_id,
            "model": "github-copilot",
            "raw_response": raw,
            "parsed": parsed,
            "parse_error": error,
        })

    OUTPUTS_FILE.write_text("\n".join(json.dumps(o) for o in outputs) + "\n")

    if error:
        print(f"WARNING: JSON parse error for case {args.case_id}: {error}", file=sys.stderr)
    else:
        n_slots = len((parsed or {}).get("slot_value_assignments", {}))
        print(f"Ingested case {args.case_id}: {n_slots} slot assignments")


if __name__ == "__main__":
    main()
