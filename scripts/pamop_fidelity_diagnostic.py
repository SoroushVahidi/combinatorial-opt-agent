#!/usr/bin/env python3
"""Phase 4 PaMOP fidelity diagnostic: C1 (current deployment) vs C3 (stronger deployment).

Reuses tools/pamop_pilot_benchmark.py's own run_problem/write_summary/etc.
directly -- no changes to that file or to any tracked PaMOP config. The
CLI script pins itself to gpt-4.1-mini via a hard RuntimeError guard
(intentional, to protect the "official pilot" numbers from silent
deployment drift); this script builds a modified in-memory config
(dataclasses.replace) for the stronger deployment instead of touching
that guard or any config file.

C1 (current gpt-4.1-mini + current prompts) is NOT rerun here -- it is
already the committed results/pamop/forensics_targeted/summary.json,
same 6 ids, same prompts, temperature 0.2. This script produces C3 only.

Usage (from repo root):
    python3 scripts/pamop_fidelity_diagnostic.py --deployment gpt-5.4 \
        --output-dir results/pamop/fidelity_diagnostic_gpt5
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.pamop_pilot_benchmark import (  # noqa: E402
    load_or_select_slice,
    ensure_artifact_headers,
    run_problem,
    write_summary,
    write_run_metadata,
    write_ours_comparison,
    read_per_problem,
)
from baselines.pamop.config import load_config, reconstructed_default_path  # noqa: E402
from baselines.pamop.llm.registry import get_provider  # noqa: E402

TARGET_IDS = [14, 23, 34, 72, 84, 88]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", required=True, help="Azure deployment name for C3 (e.g. gpt-5.4)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ampl-python", default="/home/soroush/.venvs/gurobi/bin/python")
    p.add_argument("--ampl-timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()
    args.run_gold = True
    args.allow_local = True
    args.max_correction_iterations = 5

    # run_gold_model() (called inside run_problem) resolves its own
    # interpreter via PAMOP_GOLD_PYTHON / PAMOP_AMPLPY_PYTHON / sys.executable
    # -- the CLI script (tools/pamop_pilot_benchmark.py main()) sets this
    # before calling run_problem; this standalone script must do the same,
    # or gold-model comparison silently fails with ModuleNotFoundError:
    # gurobipy under the default interpreter.
    os.environ["PAMOP_AMPLPY_PYTHON"] = args.ampl_python

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select from the same 18-id deterministic pool used by the original
    # pilot, then subset to the same 6 forensics-targeted ids.
    selected_all = load_or_select_slice(out_dir, 18)
    selected_by_id = {m.problem_id: m for m in selected_all}
    missing = [pid for pid in TARGET_IDS if pid not in selected_by_id]
    if missing:
        raise SystemExit(f"Target ids not in the deterministic 18-id pool: {missing}")
    selected = [selected_by_id[pid] for pid in TARGET_IDS]

    ensure_artifact_headers(out_dir)

    base_config = load_config(reconstructed_default_path())
    config = dataclasses.replace(
        base_config,
        llm=dataclasses.replace(base_config.llm, model=args.deployment, temperature=args.temperature),
    )
    provider = get_provider("azure_openai")

    write_run_metadata(out_dir, args, "RUNNING")
    done = {int(r["problem_id"]) for r in read_per_problem(out_dir / "per_problem.csv") if r.get("problem_id")}
    for meta in selected:
        if meta.problem_id in done:
            continue
        print(f"Running problem {meta.problem_id} on deployment={args.deployment} ...")
        row = run_problem(meta, out_dir, args, config, provider)
        print(f"  -> failure_category={row['failure_category']} final_feasible={row['final_feasible']} tokens={row['total_tokens']}")
        write_summary(out_dir, selected, "RUNNING", None)

    all_rows = read_per_problem(out_dir / "per_problem.csv")
    write_ours_comparison(out_dir, selected, all_rows)
    write_run_metadata(out_dir, args, "COMPLETED")
    summary = write_summary(out_dir, selected, "COMPLETED", None)
    print("\n=== C3 SUMMARY ===")
    for k in ("initial_execution_success_rate", "final_execution_success_rate", "semantic_correct_count", "semantic_correct_evaluable_count", "mean_tokens_per_problem"):
        print(f"  {k}: {summary.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
