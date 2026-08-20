#!/usr/bin/env python3
"""Run the non-LLM partitioning stage over a live NLP4LP subset.

Diagnostics only -- prints/writes aggregate numbers (problem count, average
tree depth, average node count, failures, a determinism check), never raw
problem text. See baselines/pamop/README.md "Not implemented yet" for what
this script does NOT do (no LLM calls, no AMPL, no solver).

Usage:
    python -m baselines.pamop.run_partitioning \
        --config baselines/pamop/configs/reconstructed_default.yaml \
        --subset pamop_possible_269 \
        --limit 10 \
        --out /tmp/pamop_smoke_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.pamop import data
from baselines.pamop.config import load_config
from baselines.pamop.partition import build_partition_tree
from baselines.pamop.representations import from_nlp4lp_record


def run(config_path: Path, subset: str, limit: int | None) -> dict:
    config = load_config(config_path)
    ids = data.list_ids_for_subset(subset)
    if limit is not None:
        ids = ids[:limit]

    n_attempted = 0
    n_succeeded = 0
    failures: list[dict] = []
    depths: list[int] = []
    node_counts: list[int] = []
    leaf_counts: list[int] = []
    determinism_mismatches = 0

    for problem_id in ids:
        n_attempted += 1
        try:
            record = data.load_problem_record(problem_id)
            problem = from_nlp4lp_record(str(problem_id), record)
            tree_a = build_partition_tree(problem, config)
            tree_b = build_partition_tree(problem, config)
            if tree_a.to_dict() != tree_b.to_dict():
                determinism_mismatches += 1
        except Exception as exc:  # noqa: BLE001 -- smoke run must not crash on one bad problem
            failures.append({"problem_id": problem_id, "error_type": type(exc).__name__})
            continue

        n_succeeded += 1
        depths.append(tree_a.max_depth())
        node_counts.append(len(tree_a.nodes))
        leaf_counts.append(len(tree_a.leaves()))

    summary = {
        "config_kind": config.config_kind,
        "subset": subset,
        "n_attempted": n_attempted,
        "n_succeeded": n_succeeded,
        "n_failed": len(failures),
        "failure_error_types": sorted({f["error_type"] for f in failures}),
        "determinism_mismatches": determinism_mismatches,
        "avg_tree_depth": (sum(depths) / len(depths)) if depths else None,
        "avg_node_count": (sum(node_counts) / len(node_counts)) if node_counts else None,
        "avg_leaf_count": (sum(leaf_counts) / len(leaf_counts)) if leaf_counts else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset", type=str, default=data.SUBSET_POSSIBLE_269)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write JSON summary here (diagnostics only, no raw text)")
    args = parser.parse_args()

    summary = run(args.config, args.subset, args.limit)
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
