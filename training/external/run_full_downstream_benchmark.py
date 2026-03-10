"""Full post-fix downstream benchmark pipeline for ESWA revision.

Runs all 10 methods × 3 variants and writes:
  results/eswa_revision/02_downstream_postfix/   — per-query CSVs + JSON
  results/eswa_revision/03_prefix_vs_postfix/    — pre-fix simulation ablation
  results/eswa_revision/13_tables/               — aggregate CSVs
  results/eswa_revision/14_reports/              — markdown reports
  results/eswa_revision/00_env/hf_access_check_runtime.md
  results/eswa_revision/manifests/commands_run_runtime.md

Usage:
    HF_TOKEN=hf_... python training/external/run_full_downstream_benchmark.py

The script never prints or logs the token value.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"
ESWA_DIR = ROOT / "results" / "eswa_revision"
PAPER_DIR = ROOT / "results" / "paper"

VARIANTS = ["orig", "noisy", "short"]

# Each tuple: (baseline_arg, assignment_mode, display_name)
# These cover all 10 requested methods.
METHODS: list[tuple[str, str, str]] = [
    ("tfidf", "typed", "tfidf_typed_greedy"),
    ("bm25", "typed", "bm25_typed_greedy"),
    ("lsa", "typed", "lsa_typed_greedy"),
    ("oracle", "typed", "oracle_typed_greedy"),
    ("tfidf", "constrained", "tfidf_constrained"),
    ("tfidf", "semantic_ir_repair", "tfidf_semantic_ir_repair"),
    ("tfidf", "optimization_role_repair", "tfidf_optimization_role_repair"),
    ("tfidf_acceptance_rerank", "typed", "tfidf_acceptance_rerank"),
    ("tfidf_hierarchical_acceptance_rerank", "typed", "tfidf_hierarchical_acceptance_rerank"),
]

# Ablation: methods to include in the pre-fix vs post-fix comparison
ABLATION_METHODS: list[tuple[str, str, str]] = [
    ("tfidf", "typed", "tfidf_typed_greedy"),
    ("tfidf", "optimization_role_repair", "tfidf_optimization_role_repair"),
    ("tfidf_hierarchical_acceptance_rerank", "typed", "tfidf_hierarchical_acceptance_rerank"),
    ("oracle", "typed", "oracle_typed_greedy"),
]


# ---------------------------------------------------------------------------
# HF access verification
# ---------------------------------------------------------------------------

def verify_hf_access(token: str) -> tuple[bool, str]:
    """Return (ok, message). Never logs the token."""
    if not token:
        return False, "HF_TOKEN not set in environment"
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False, "huggingface_hub not installed"
    api = HfApi(token=token)
    try:
        info = api.dataset_info("udell-lab/NLP4LP")
        return True, f"Dataset reachable: {info.id}"
    except Exception as e:
        return False, f"Dataset access error: {e}"


def write_hf_runtime_check(ok: bool, message: str, branch: str, commit: str) -> None:
    p = ESWA_DIR / "00_env" / "hf_access_check_runtime.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    status_emoji = "✅" if ok else "❌"
    status_text = "SUCCESS" if ok else "FAILED"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"""# HF Access Check — Runtime Verification

**Date:** {now}
**Branch:** {branch} (commit {commit})
**Environment:** GitHub Actions (authenticated)

## Result: {status_emoji} {status_text}

{message}

## Details

- HF_TOKEN length: {'SET (' + str(len(os.environ.get('HF_TOKEN', ''))) + ' chars)' if ok else 'NOT SET or invalid'}
- Dataset: udell-lab/NLP4LP
- Access method: HuggingFace API (huggingface_hub)

## Next steps

{'Full downstream benchmark is running in this workflow.' if ok else 'Cannot proceed without HF access. Set HF_TOKEN in repository secrets.'}
""")
    print(f"Wrote {p}")


# ---------------------------------------------------------------------------
# Pre-fix simulation
# ---------------------------------------------------------------------------

def _is_type_match_prefix(expected: str, kind: str) -> bool:
    """Pre-fix version: strict equality only (the original bug)."""
    return expected == kind


def _run_setting_prefix(
    variant: str,
    baseline_arg: str,
    assignment_mode: str,
    eval_items: list,
    gold_by_id: dict,
    catalog: list,
    doc_ids: list,
    out_dir: Path,
) -> dict:
    """Run a setting with the pre-fix _is_type_match patched in, then restore."""
    import tools.nlp4lp_downstream_utility as dutil
    orig_fn = dutil._is_type_match
    dutil._is_type_match = _is_type_match_prefix  # type: ignore[assignment]
    try:
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
    finally:
        dutil._is_type_match = orig_fn  # type: ignore[assignment]

    # Read back the summary row
    summary_path = out_dir / "nlp4lp_downstream_summary.csv"
    effective = _effective_name(baseline_arg, assignment_mode)
    return _read_summary_row(summary_path, variant, effective)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_name(baseline_arg: str, assignment_mode: str) -> str:
    if assignment_mode == "typed":
        return baseline_arg
    return f"{baseline_arg}_{assignment_mode}"


def _read_summary_row(summary_path: Path, variant: str, baseline: str) -> dict:
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant") == variant and row.get("baseline") == baseline:
                return dict(row)
    return {}


def _safe_float(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    ).strip()

    branch = os.environ.get("GITHUB_REF_NAME", "unknown-branch")
    commit = os.environ.get("GITHUB_SHA", "unknown-sha")[:8]
    run_id = os.environ.get("GITHUB_RUN_ID", "local")

    print(f"=== NLP4LP Full Downstream Benchmark ===")
    print(f"Branch: {branch}, Commit: {commit}, Run: {run_id}")
    print(f"HF_TOKEN set: {bool(token)} (length {len(token)})")

    # 1. Verify HF access
    ok, msg = verify_hf_access(token)
    write_hf_runtime_check(ok, msg, branch, commit)
    if not ok:
        print(f"FATAL: HF access failed: {msg}")
        sys.exit(1)
    print(f"HF access: {msg}")

    # 2. Set env for low-resource
    _set_env()

    # 3. Cache HF gold to a temp file so we only fetch once
    cache_path = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["NLP4LP_GOLD_CACHE"] = str(cache_path)

    # 4. Load shared resources once
    from tools.nlp4lp_downstream_utility import (
        _load_eval,
        _load_hf_gold,
        _load_catalog_as_problems,
        _apply_low_resource_env,
        run_single_setting,
    )
    _apply_low_resource_env()

    catalog_path = CATALOG_PATH
    catalog, _ = _load_catalog_as_problems(catalog_path)
    doc_ids = [p["id"] for p in catalog if p.get("id")]

    print("Loading gold parameters from HuggingFace…")
    t0 = time.time()
    gold_by_id = _load_hf_gold(split="test", use_cache=True)
    print(f"Loaded {len(gold_by_id)} gold records in {time.time()-t0:.1f}s")

    # 5. Out dirs
    postfix_dir = ESWA_DIR / "02_downstream_postfix"
    postfix_dir.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # We accumulate all results here
    all_results: list[dict] = []

    # 6. Run all methods × variants (post-fix — current code)
    total = len(METHODS) * len(VARIANTS) + len(VARIANTS)  # +random
    done = 0
    for variant in VARIANTS:
        eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
        eval_items = _load_eval(eval_path)
        if not eval_items:
            print(f"WARNING: No eval items for variant={variant}; skipping.", file=sys.stderr)
            continue

        # Random seeded control
        t0 = time.time()
        print(f"\n[{done+1}/{total}] random_seeded / {variant} …")
        ok2 = run_single_setting(
            variant=variant,
            baseline_arg="tfidf",
            assignment_mode="typed",
            out_dir=PAPER_DIR,
            eval_items=eval_items,
            gold_by_id=gold_by_id,
            catalog=catalog,
            doc_ids=doc_ids,
            random_control=True,
        )
        done += 1
        elapsed = time.time() - t0
        summary_path = PAPER_DIR / "nlp4lp_downstream_summary.csv"
        row = _read_summary_row(summary_path, variant, "tfidf_random")
        if row:
            all_results.append({
                "variant": variant,
                "method": "random_seeded",
                "effective_baseline": "tfidf_random",
                "Coverage": row.get("param_coverage", ""),
                "TypeMatch": row.get("type_match", ""),
                "Exact20": row.get("exact20_on_hits", ""),
                "InstReady": row.get("instantiation_ready", ""),
                "n": row.get("n", ""),
                "elapsed_s": f"{elapsed:.0f}",
                "source": "measured",
            })
            print(f"  Done ({elapsed:.0f}s): cov={row.get('param_coverage')} tm={row.get('type_match')} ir={row.get('instantiation_ready')}")

        # All 9 deterministic methods
        for baseline_arg, assignment_mode, display_name in METHODS:
            effective = _effective_name(baseline_arg, assignment_mode)
            done += 1
            t0 = time.time()
            print(f"[{done}/{total}] {display_name} / {variant} …")
            ok2 = run_single_setting(
                variant=variant,
                baseline_arg=baseline_arg,
                assignment_mode=assignment_mode,
                out_dir=PAPER_DIR,
                eval_items=eval_items,
                gold_by_id=gold_by_id,
                catalog=catalog,
                doc_ids=doc_ids,
                acceptance_k=10,
            )
            elapsed = time.time() - t0
            row = _read_summary_row(summary_path, variant, effective)
            if row:
                all_results.append({
                    "variant": variant,
                    "method": display_name,
                    "effective_baseline": effective,
                    "Coverage": row.get("param_coverage", ""),
                    "TypeMatch": row.get("type_match", ""),
                    "Exact20": row.get("exact20_on_hits", ""),
                    "InstReady": row.get("instantiation_ready", ""),
                    "n": row.get("n", ""),
                    "elapsed_s": f"{elapsed:.0f}",
                    "source": "measured",
                })
                print(f"  Done ({elapsed:.0f}s): cov={row.get('param_coverage')} tm={row.get('type_match')} ir={row.get('instantiation_ready')}")
            else:
                print(f"  WARNING: No summary row found for {effective} / {variant}", file=sys.stderr)

    # 7. Pre-fix vs post-fix ablation
    prefix_dir = ESWA_DIR / "03_prefix_vs_postfix"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    preprefix_paper_dir = prefix_dir / "paper_prefix"
    preprefix_paper_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows: list[dict] = []

    for variant in VARIANTS:
        eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
        eval_items = _load_eval(eval_path)
        if not eval_items:
            continue

        for baseline_arg, assignment_mode, display_name in ABLATION_METHODS:
            effective = _effective_name(baseline_arg, assignment_mode)

            # Post-fix: read from all_results
            post_row = next(
                (r for r in all_results if r["variant"] == variant and r["method"] == display_name),
                None,
            )

            # Pre-fix: run with patched function
            print(f"\n[PRE-FIX] {display_name} / {variant} …")
            t0 = time.time()
            pre_row = _run_setting_prefix(
                variant=variant,
                baseline_arg=baseline_arg,
                assignment_mode=assignment_mode,
                eval_items=eval_items,
                gold_by_id=gold_by_id,
                catalog=catalog,
                doc_ids=doc_ids,
                out_dir=preprefix_paper_dir,
            )
            elapsed = time.time() - t0
            print(f"  Pre-fix done ({elapsed:.0f}s): cov={pre_row.get('param_coverage')} tm={pre_row.get('type_match')}")

            ablation_rows.append({
                "variant": variant,
                "method": display_name,
                "Coverage_pre": _safe_float(pre_row.get("param_coverage")),
                "TypeMatch_pre": _safe_float(pre_row.get("type_match")),
                "Exact20_pre": _safe_float(pre_row.get("exact20_on_hits")),
                "InstReady_pre": _safe_float(pre_row.get("instantiation_ready")),
                "Coverage_post": _safe_float(post_row.get("Coverage") if post_row else None),
                "TypeMatch_post": _safe_float(post_row.get("TypeMatch") if post_row else None),
                "Exact20_post": _safe_float(post_row.get("Exact20") if post_row else None),
                "InstReady_post": _safe_float(post_row.get("InstReady") if post_row else None),
                "TypeMatch_delta": "",
                "source": "measured",
            })
            # Compute delta
            if ablation_rows[-1]["TypeMatch_pre"] != "N/A" and ablation_rows[-1]["TypeMatch_post"] != "N/A":
                try:
                    delta = float(ablation_rows[-1]["TypeMatch_post"]) - float(ablation_rows[-1]["TypeMatch_pre"])
                    ablation_rows[-1]["TypeMatch_delta"] = f"{delta:+.4f}"
                except ValueError:
                    pass

    # 8. Write output tables
    _write_tables(all_results, ablation_rows, run_id, commit)

    # 9. Copy per-query files to 02_downstream_postfix
    _copy_per_query_files(PAPER_DIR, postfix_dir)

    # 10. Write manifest
    _write_manifest(run_id, branch, commit, all_results)

    print("\n=== BENCHMARK COMPLETE ===")
    _print_summary(all_results)


def _set_env() -> None:
    for k, v in {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    }.items():
        if k not in os.environ:
            os.environ[k] = v


def _write_tables(all_results: list[dict], ablation_rows: list[dict], run_id: str, commit: str) -> None:
    tables_dir = ESWA_DIR / "13_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ESWA_DIR / "14_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    postfix_dir = ESWA_DIR / "02_downstream_postfix"
    prepostfix_dir = ESWA_DIR / "03_prefix_vs_postfix"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # postfix_main_metrics.csv (replace old manuscript-era table)
    postfix_csv = tables_dir / "postfix_main_metrics.csv"
    cols = ["variant", "method", "Coverage", "TypeMatch", "Exact20_on_hits", "InstReady", "n", "elapsed_s", "source"]
    with open(postfix_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in all_results:
            w.writerow({
                "variant": row["variant"],
                "method": row["method"],
                "Coverage": row.get("Coverage", ""),
                "TypeMatch": row.get("TypeMatch", ""),
                "Exact20_on_hits": row.get("Exact20", ""),
                "InstReady": row.get("InstReady", ""),
                "n": row.get("n", ""),
                "elapsed_s": row.get("elapsed_s", ""),
                "source": row.get("source", "measured"),
            })
    print(f"Wrote {postfix_csv}")

    # Also write the per-variant CSVs for the report
    for variant in VARIANTS:
        variant_rows = [r for r in all_results if r["variant"] == variant]
        if not variant_rows:
            continue
        vpath = postfix_dir / f"postfix_{variant}_summary.csv"
        with open(vpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in variant_rows:
                w.writerow({
                    "variant": row["variant"],
                    "method": row["method"],
                    "Coverage": row.get("Coverage", ""),
                    "TypeMatch": row.get("TypeMatch", ""),
                    "Exact20_on_hits": row.get("Exact20", ""),
                    "InstReady": row.get("InstReady", ""),
                    "n": row.get("n", ""),
                    "elapsed_s": row.get("elapsed_s", ""),
                    "source": row.get("source", "measured"),
                })
        print(f"Wrote {vpath}")

    # prefix_vs_postfix_ablation.csv
    ablation_csv = tables_dir / "prefix_vs_postfix_ablation.csv"
    ablation_cols = [
        "variant", "method",
        "Coverage_pre", "TypeMatch_pre", "Exact20_pre", "InstReady_pre",
        "Coverage_post", "TypeMatch_post", "Exact20_post", "InstReady_post",
        "TypeMatch_delta", "source",
    ]
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ablation_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ablation_rows)
    print(f"Wrote {ablation_csv}")

    # markdown report — postfix
    postfix_md = reports_dir / "postfix_main_metrics.md"
    _write_postfix_report(postfix_md, all_results, now, run_id, commit)
    print(f"Wrote {postfix_md}")

    # markdown report — ablation
    ablation_md = prepostfix_dir / "prefix_vs_postfix_ablation.md"
    _write_ablation_report(ablation_md, ablation_rows, now, run_id, commit)
    print(f"Wrote {ablation_md}")

    # Also write ablation report to reports_dir for consistency
    ablation_md2 = reports_dir / "prefix_vs_postfix_ablation.md"
    _write_ablation_report(ablation_md2, ablation_rows, now, run_id, commit)
    print(f"Wrote {ablation_md2}")


def _write_postfix_report(path: Path, all_results: list[dict], now: str, run_id: str, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Post-Fix Downstream Benchmark Results (Measured)",
        "",
        f"**Date:** {now}  ",
        f"**Source:** GitHub Actions run `{run_id}` (commit `{commit}`)  ",
        "**Status:** NEWLY MEASURED — real authenticated results, NOT manuscript-era estimates",
        "",
        "## Summary by Variant",
        "",
    ]
    for variant in VARIANTS:
        rows = [r for r in all_results if r["variant"] == variant]
        if not rows:
            lines.append(f"### {variant.upper()} — no results\n")
            continue
        lines.append(f"### {variant.upper()}")
        lines.append("")
        lines.append("| Method | Coverage | TypeMatch | Exact20 | InstReady |")
        lines.append("|--------|----------|-----------|---------|-----------|")
        for r in rows:
            cov = _safe_float(r.get("Coverage"))
            tm = _safe_float(r.get("TypeMatch"))
            e20 = _safe_float(r.get("Exact20"))
            ir = _safe_float(r.get("InstReady"))
            lines.append(f"| {r['method']} | {cov} | {tm} | {e20} | {ir} |")
        lines.append("")

    lines += [
        "## Source",
        "",
        "All numbers measured in this GitHub Actions run using the authenticated",
        "`udell-lab/NLP4LP` dataset. No manuscript-era estimates are included.",
        "",
        f"Raw per-query CSVs: `results/eswa_revision/02_downstream_postfix/`  ",
        f"Tables: `results/eswa_revision/13_tables/postfix_main_metrics.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_report(path: Path, ablation_rows: list[dict], now: str, run_id: str, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pre-Fix vs Post-Fix TypeMatch Ablation (Measured)",
        "",
        f"**Date:** {now}  ",
        f"**Source:** GitHub Actions run `{run_id}` (commit `{commit}`)  ",
        "**Status:** NEWLY MEASURED — both pre-fix and post-fix values computed in this run",
        "",
        "## Background",
        "",
        "The pre-fix `_is_type_match` used strict equality (`expected == kind`).",
        "The post-fix version additionally returns `True` for `(float, int)` pairs.",
        "Both were run in this CI job: the pre-fix was simulated by patching the function",
        "in-memory before each pre-fix run and restoring it after.",
        "",
        "## Results by Variant",
        "",
    ]
    for variant in VARIANTS:
        rows = [r for r in ablation_rows if r["variant"] == variant]
        if not rows:
            lines.append(f"### {variant.upper()} — no results\n")
            continue
        lines.append(f"### {variant.upper()}")
        lines.append("")
        lines.append("| Method | TM_pre | TM_post | TM_delta | Coverage_pre | Coverage_post |")
        lines.append("|--------|--------|---------|----------|--------------|---------------|")
        for r in rows:
            lines.append(
                f"| {r['method']} "
                f"| {r['TypeMatch_pre']} "
                f"| {r['TypeMatch_post']} "
                f"| {r['TypeMatch_delta']} "
                f"| {r['Coverage_pre']} "
                f"| {r['Coverage_post']} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "TypeMatch_delta shows the net improvement from the `_is_type_match` fix.",
        "The delta should be positive for all methods since integer tokens are now",
        "correctly counted as matches for float-typed slots.",
        "",
        f"Tables: `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_per_query_files(paper_dir: Path, postfix_dir: Path) -> None:
    """Copy per-query and JSON result files to the eswa revision directory."""
    import shutil
    postfix_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in paper_dir.glob("nlp4lp_downstream_*"):
        dest = postfix_dir / f.name
        shutil.copy2(f, dest)
        count += 1
    print(f"Copied {count} result files to {postfix_dir}")


def _write_manifest(run_id: str, branch: str, commit: str, all_results: list[dict]) -> None:
    manifest_dir = ESWA_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # commands_run_runtime.md
    cmd_path = manifest_dir / "commands_run_runtime.md"
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(f"""# Commands Run — Runtime Benchmark (Authenticated)

**Date:** {now}
**GitHub Actions Run ID:** {run_id}
**Branch:** {branch} (commit {commit})
**Environment:** ubuntu-latest / Python 3.11

## HF Access verification

```bash
python training/external/verify_hf_access.py
```

## Full downstream benchmark

```bash
NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \\
  python training/external/run_full_downstream_benchmark.py
```

## Methods run

The script ran {len(METHODS) + 1} methods (incl. random control) × {len(VARIANTS)} variants
= {(len(METHODS) + 1) * len(VARIANTS)} total settings.

Methods:
""")
        for _, _, name in METHODS:
            f.write(f"- {name}\n")
        f.write("- random_seeded\n")
        f.write(f"""
## Output files

- `results/eswa_revision/02_downstream_postfix/` — per-query CSVs + JSON
- `results/eswa_revision/13_tables/postfix_main_metrics.csv`
- `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`
- `results/eswa_revision/14_reports/postfix_main_metrics.md`
- `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md`
- `results/eswa_revision/00_env/hf_access_check_runtime.md`
- `results/eswa_revision/00_env/nlp4lp_gold_cache.json` (HF gold params cache)
""")
    print(f"Wrote {cmd_path}")

    # JSON manifest
    manifest_json = manifest_dir / "experiment_manifest_runtime.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({
            "date": now,
            "run_id": run_id,
            "branch": branch,
            "commit": commit,
            "n_results": len(all_results),
            "variants": VARIANTS,
            "methods": [name for _, _, name in METHODS] + ["random_seeded"],
            "source": "measured-authenticated",
        }, f, indent=2)
    print(f"Wrote {manifest_json}")


def _print_summary(all_results: list[dict]) -> None:
    print("\n=== ORIG variant results ===")
    orig = [r for r in all_results if r["variant"] == "orig"]
    if not orig:
        print("No orig results found!")
        return
    header = f"{'Method':<45} {'Cov':>6} {'TM':>6} {'E20':>6} {'IR':>6}"
    print(header)
    print("-" * 75)
    for r in orig:
        print(
            f"{r['method']:<45} "
            f"{_safe_float(r.get('Coverage')):>6} "
            f"{_safe_float(r.get('TypeMatch')):>6} "
            f"{_safe_float(r.get('Exact20')):>6} "
            f"{_safe_float(r.get('InstReady')):>6}"
        )


if __name__ == "__main__":
    run_pipeline()
