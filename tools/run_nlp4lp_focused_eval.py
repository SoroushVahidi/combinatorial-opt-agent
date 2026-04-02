#!/usr/bin/env python3
"""Run the focused downstream methods and produce a single comparable summary CSV.

Default (evidence-supported) methods (4):
  - tfidf_acceptance_rerank
  - tfidf_hierarchical_acceptance_rerank
  - tfidf_optimization_role_repair
  - tfidf_optimization_role_relation_repair

With --experimental, also run (archived / not in main pipeline): anchor_linking,
bottomup_beam_repair, entity_semantic_beam_repair.

Outputs:
  - results/paper/nlp4lp_downstream_summary.csv (updated with upserted rows)
  - results/paper/nlp4lp_focused_eval_summary.csv (side-by-side summary for run methods)
  - per-query CSVs and JSONs for each method (from downstream utility)

Usage:
  python tools/run_nlp4lp_focused_eval.py [--variant orig]           # 4 methods
  python tools/run_nlp4lp_focused_eval.py --variant orig --experimental   # 7 methods
  python tools/run_nlp4lp_focused_eval.py --variant orig --safe      # no subprocess, sequential, skip done
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Evidence-supported methods (default pipeline).
FOCUSED_BASELINES_DEFAULT = [
    "tfidf_acceptance_rerank",
    "tfidf_hierarchical_acceptance_rerank",
    "tfidf_optimization_role_repair",
    "tfidf_optimization_role_relation_repair",
    "tfidf_global_consistency_grounding",
    "tfidf_max_weight_matching",
    "tfidf_global_compat_local",
    "tfidf_global_compat_pairwise",
    "tfidf_global_compat_full",
    "tfidf_relation_aware_basic",
    "tfidf_relation_aware_ops",
    "tfidf_relation_aware_semantic",
    "tfidf_relation_aware_full",
    "tfidf_ambiguity_candidate_greedy",
    "tfidf_ambiguity_aware_beam",
    "tfidf_ambiguity_aware_abstain",
    "tfidf_ambiguity_aware_full",
    "tfidf_search_structured_grounding",
    "tfidf_search_structured_grounding_no_global",
]

BASELINE_ASSIGNMENT_DEFAULT = [
    ("tfidf_acceptance_rerank", "typed"),
    ("tfidf_hierarchical_acceptance_rerank", "typed"),
    ("tfidf", "optimization_role_repair"),
    ("tfidf", "optimization_role_relation_repair"),
    ("tfidf", "global_consistency_grounding"),
    ("tfidf", "max_weight_matching"),
    ("tfidf", "global_compat_local"),
    ("tfidf", "global_compat_pairwise"),
    ("tfidf", "global_compat_full"),
    ("tfidf", "relation_aware_basic"),
    ("tfidf", "relation_aware_ops"),
    ("tfidf", "relation_aware_semantic"),
    ("tfidf", "relation_aware_full"),
    ("tfidf", "ambiguity_candidate_greedy"),
    ("tfidf", "ambiguity_aware_beam"),
    ("tfidf", "ambiguity_aware_abstain"),
    ("tfidf", "ambiguity_aware_full"),
    ("tfidf", "search_structured_grounding"),
    ("tfidf", "search_structured_grounding_no_global"),
]

# Experimental/archived methods (only with --experimental).
FOCUSED_BASELINES_EXPERIMENTAL = [
    "tfidf_optimization_role_anchor_linking",
    "tfidf_optimization_role_bottomup_beam_repair",
    "tfidf_optimization_role_entity_semantic_beam_repair",
]

BASELINE_ASSIGNMENT_EXPERIMENTAL = [
    ("tfidf", "optimization_role_anchor_linking"),
    ("tfidf", "optimization_role_bottomup_beam_repair"),
    ("tfidf", "optimization_role_entity_semantic_beam_repair"),
]


def _apply_safe_env() -> None:
    """Set env for low-resource / Wulver-safe execution (no subprocess fork, minimal threads)."""
    env_settings = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    for k, v in env_settings.items():
        if k not in os.environ:
            os.environ[k] = v


def _summary_has_row(summary_path: Path, variant: str, effective_baseline: str) -> bool:
    """True if summary CSV already has (variant, effective_baseline)."""
    if not summary_path.exists():
        return False
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant") == variant and row.get("baseline") == effective_baseline:
                return True
    return False


def _effective_baseline(baseline_arg: str, assignment_mode: str) -> str:
    """Mirror downstream_utility effective_baseline naming."""
    if assignment_mode == "typed":
        return baseline_arg
    if assignment_mode == "optimization_role_repair":
        return f"{baseline_arg}_optimization_role_repair"
    if assignment_mode == "optimization_role_relation_repair":
        return f"{baseline_arg}_optimization_role_relation_repair"
    if assignment_mode == "optimization_role_anchor_linking":
        return f"{baseline_arg}_optimization_role_anchor_linking"
    if assignment_mode == "optimization_role_bottomup_beam_repair":
        return f"{baseline_arg}_optimization_role_bottomup_beam_repair"
    if assignment_mode == "optimization_role_entity_semantic_beam_repair":
        return f"{baseline_arg}_optimization_role_entity_semantic_beam_repair"
    if assignment_mode == "global_consistency_grounding":
        return f"{baseline_arg}_global_consistency_grounding"
    if assignment_mode == "max_weight_matching":
        return f"{baseline_arg}_max_weight_matching"
    if assignment_mode in ("global_compat_local", "global_compat_pairwise", "global_compat_full"):
        return f"{baseline_arg}_{assignment_mode}"
    if assignment_mode in (
        "relation_aware_basic",
        "relation_aware_ops",
        "relation_aware_semantic",
        "relation_aware_full",
    ):
        return f"{baseline_arg}_{assignment_mode}"
    if assignment_mode in (
        "ambiguity_candidate_greedy",
        "ambiguity_aware_beam",
        "ambiguity_aware_abstain",
        "ambiguity_aware_full",
    ):
        return f"{baseline_arg}_{assignment_mode}"
    if assignment_mode in ("search_structured_grounding", "search_structured_grounding_no_global"):
        return f"{baseline_arg}_{assignment_mode}"
    return f"{baseline_arg}_{assignment_mode}"


def run_downstream_subprocess(variant: str, baseline: str, assignment_mode: str, out_dir: Path) -> bool:
    """Run one method via subprocess (original behavior)."""
    import subprocess
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "nlp4lp_downstream_utility.py"),
        "--variant", variant,
        "--baseline", baseline,
        "--assignment-mode", assignment_mode,
    ]
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=600)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Failed: {' '.join(cmd)} -> {e}", file=sys.stderr)
        return False


def run_downstream_safe(
    variant: str,
    out_dir: Path,
    summary_path: Path,
    eval_items: list,
    gold_by_id: dict,
    catalog: list,
    doc_ids: list,
    baseline_assignment: list[tuple[str, str]],
) -> None:
    """Run the given methods in-process, sequentially; skip if summary already has row (resume)."""
    from tools import nlp4lp_downstream_utility as dutil
    dutil._apply_low_resource_env()
    for baseline_arg, assignment_mode in baseline_assignment:
        effective = _effective_baseline(baseline_arg, assignment_mode)
        if _summary_has_row(summary_path, variant, effective):
            print(f"Skipping (already done): {effective}")
            continue
        print(f"Running {effective} ...")
        ok = dutil.run_single_setting(
            variant=variant,
            baseline_arg=baseline_arg,
            assignment_mode=assignment_mode,
            out_dir=out_dir,
            eval_items=eval_items,
            gold_by_id=gold_by_id,
            catalog=catalog,
            doc_ids=doc_ids,
            acceptance_k=10,
        )
        if not ok:
            print(f"Warning: {effective} failed; continuing.", file=sys.stderr)


def build_focused_summary(
    variant: str, summary_path: Path, out_path: Path, focused_baselines: list[str]
) -> None:
    """Read full summary CSV, filter to variant + focused baselines, write side-by-side CSV."""
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}", file=sys.stderr)
        return
    rows: list[dict] = []
    with open(summary_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("variant") == variant and row.get("baseline") in focused_baselines:
                rows.append(row)
    if not rows:
        print(f"No rows for variant={variant} and baselines {focused_baselines}", file=sys.stderr)
        return
    cols = [
        "variant", "baseline", "schema_R1", "param_coverage", "type_match", "key_overlap",
        "exact5_on_hits", "exact20_on_hits", "instantiation_ready", "n",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})
    print(f"Wrote {out_path} ({len(rows)} rows)")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run focused NLP4LP downstream methods (default 4) and build summary CSV")
    p.add_argument("--variant", type=str, default="orig", choices=("orig", "noisy", "short"))
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    p.add_argument("--safe", action="store_true", help="Low-resource mode: no subprocess, sequential, skip done, set thread env")
    p.add_argument("--experimental", action="store_true", help="Include archived methods: anchor_linking, bottomup_beam, entity_semantic_beam (7 methods total)")
    args = p.parse_args()
    out_dir = args.out_dir
    summary_path = out_dir / "nlp4lp_downstream_summary.csv"
    focused_path = out_dir / "nlp4lp_focused_eval_summary.csv"

    focused_baselines = list(FOCUSED_BASELINES_DEFAULT)
    baseline_assignment = list(BASELINE_ASSIGNMENT_DEFAULT)
    if args.experimental:
        focused_baselines.extend(FOCUSED_BASELINES_EXPERIMENTAL)
        baseline_assignment.extend(BASELINE_ASSIGNMENT_EXPERIMENTAL)

    if args.safe:
        _apply_safe_env()
        # Defer heavy imports until after env so they see thread limits
        from tools.nlp4lp_downstream_utility import (
            _load_eval,
            _load_hf_gold,
            _load_catalog_as_problems,
        )
        eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{args.variant}.jsonl"
        catalog_path = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"
        eval_items = _load_eval(eval_path)
        if not eval_items:
            print(f"No eval items from {eval_path}", file=sys.stderr)
            sys.exit(1)
        gold_by_id = _load_hf_gold(split="test")
        catalog, _ = _load_catalog_as_problems(catalog_path)
        doc_ids = [p["id"] for p in catalog if p.get("id")]
        run_downstream_safe(
            variant=args.variant,
            out_dir=out_dir,
            summary_path=summary_path,
            eval_items=eval_items,
            gold_by_id=gold_by_id,
            catalog=catalog,
            doc_ids=doc_ids,
            baseline_assignment=baseline_assignment,
        )
    else:
        for baseline_arg, assignment_mode in baseline_assignment:
            print(f"Running {baseline_arg} + {assignment_mode} ...")
            ok = run_downstream_subprocess(args.variant, baseline_arg, assignment_mode, out_dir)
            if not ok:
                print(f"Warning: {baseline_arg} + {assignment_mode} failed; continuing.", file=sys.stderr)

    build_focused_summary(args.variant, summary_path, focused_path, focused_baselines)


if __name__ == "__main__":
    main()
