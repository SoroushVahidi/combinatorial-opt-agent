"""Stage 6: matched same-task grounding baselines under final frozen codebase.

Predeclared methods (mechanism diversity, selected BEFORE seeing scores):
  typed, constrained, max_weight_matching, optimization_role_repair,
  search_structured_grounding, semantic_ir_repair (optional distinct repair).

Fairness: same 331 orig queries, TF-IDF retrieval, fixed 335-schema catalog,
current frozen extractors (typed/constrained share ratio-aware; opt-role and
semantic-IR families retain intrinsic enriched extractors — see fairness_audit.json),
PYTHONHASHSEED=0, gold from committed cache (no HF split drift), no new tuning.
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.comparison.statistics import build_transition_table, mcnemar_exact

OUT_DIR = ROOT / "results" / "stage6_matched_grounding_baselines_2026-08-27"
GOLD_CACHE = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"

# Predeclared set — do not reorder by score after the fact.
METHODS: list[tuple[str, str]] = [
    ("typed", "typed_greedy"),
    ("constrained", "constrained_assignment"),
    ("max_weight_matching", "max_weight_matching"),
    ("optimization_role_repair", "optimization_role_repair"),
    ("search_structured_grounding", "search_structured_grounding"),
    ("semantic_ir_repair", "semantic_ir_repair"),
]

B = 10_000
SEED = 42


def ir_indicator(row: dict) -> float:
    cov = float(row.get("param_coverage") or 0)
    tm = float(row.get("type_match") or 0)
    return 1.0 if cov >= 0.8 and tm >= 0.8 else 0.0


def strict_ir_indicator(row: dict) -> float:
    if int(float(row.get("schema_hit") or 0)) != 1:
        return 0.0
    return ir_indicator(row)


def paired_bootstrap(vals_a: list[float], vals_b: list[float], label_a: str, label_b: str) -> dict:
    rng = random.Random(SEED)
    n = len(vals_a)
    assert n == len(vals_b)
    obs_diff = sum(vals_a) / n - sum(vals_b) / n
    pairs = list(zip(vals_a, vals_b))
    diffs: list[float] = []
    for _ in range(B):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(p[0] for p in sample) / n - sum(p[1] for p in sample) / n)
    diffs.sort()
    p_le0 = sum(1 for d in diffs if d <= 0) / B
    p_gt0 = sum(1 for d in diffs if d > 0) / B
    return {
        "comparison": f"{label_a} vs {label_b}",
        "n": n,
        "diff": obs_diff,
        "ci_95": [diffs[int(B * 0.025)], diffs[int(B * 0.975)]],
        "p_value": max(2 * min(p_le0, p_gt0), 1.0 / B),
        "B": B,
        "seed": SEED,
    }


def baseline_suffix(mode: str) -> str:
    if mode == "typed":
        return "tfidf"
    return f"tfidf_{mode}"


def read_per_query(mode: str) -> list[dict]:
    suffix = baseline_suffix(mode)
    path = OUT_DIR / f"nlp4lp_downstream_per_query_orig_{suffix}.csv"
    return list(csv.DictReader(open(path, encoding="utf-8")))


def read_json_agg(mode: str) -> dict:
    suffix = baseline_suffix(mode)
    path = OUT_DIR / f"nlp4lp_downstream_orig_{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8"))["aggregate"]


def exact20_comparable_stats(rows: list[dict]) -> tuple[float | None, int]:
    vals: list[float] = []
    for r in rows:
        if int(float(r.get("schema_hit") or 0)) != 1:
            continue
        v = r.get("exact20")
        if v is None or v == "" or str(v).lower() == "nan":
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def run_all() -> dict[str, float]:
    import tools.nlp4lp_downstream_utility as u

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer committed gold cache (avoids HF gated-split drift).
    os.environ.setdefault("NLP4LP_GOLD_CACHE", str(GOLD_CACHE))
    eval_items = u._load_eval(ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    gold_by_id = u._load_hf_gold(split="test")
    catalog, _ = u._load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    doc_ids = [p["id"] for p in catalog if p.get("id")]

    timings: dict[str, float] = {}
    for mode, label in METHODS:
        t0 = time.perf_counter()
        ok = u.run_single_setting(
            variant="orig",
            baseline_arg="tfidf",
            assignment_mode=mode,
            out_dir=OUT_DIR,
            eval_items=eval_items,
            gold_by_id=gold_by_id,
            catalog=catalog,
            doc_ids=doc_ids,
        )
        timings[label] = time.perf_counter() - t0
        if not ok:
            raise SystemExit(f"run_single_setting failed for {mode}")
    (OUT_DIR / "runtime.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return timings


def summarize(timings: dict[str, float] | None = None) -> dict:
    ref_rows = read_per_query("typed")
    ref_ids = [r["query_id"] for r in ref_rows]
    results: list[dict] = []
    significance: dict[str, dict] = {}

    for mode, label in METHODS:
        rows = read_per_query(mode)
        ids = [r["query_id"] for r in rows]
        assert ids == ref_ids, f"query order mismatch for {mode}"
        agg = read_json_agg(mode)
        exact20_mean, exact20_n = exact20_comparable_stats(rows)
        strict_rate = sum(strict_ir_indicator(r) for r in rows) / len(rows)
        # Prefer aggregate InstantiationReady; recompute Strict from per-query rows
        # because some aggregate JSON snapshots omit strict_instantiation_ready.
        results.append(
            {
                "method": label,
                "assignment_mode": mode,
                "schema_R1": agg["schema_R1"],
                "Coverage": agg["param_coverage"],
                "TypeMatch": agg["type_match"],
                "Exact20_on_hits": exact20_mean if exact20_mean is not None else agg["exact20_on_hits"],
                "Exact20_n_comparable": exact20_n,
                "InstantiationReady": agg["instantiation_ready"],
                "StrictInstantiationReady": strict_rate,
                "n": agg["n"],
            }
        )
        if mode != "typed":
            ir_a = [ir_indicator(r) for r in ref_rows]
            ir_b = [ir_indicator(r) for r in rows]
            st_a = [strict_ir_indicator(r) for r in ref_rows]
            st_b = [strict_ir_indicator(r) for r in rows]
            significance[f"{label}_instready"] = paired_bootstrap(ir_a, ir_b, "typed_greedy", label)
            pairs = [(bool(a), bool(b)) for a, b in zip(st_a, st_b)]
            mc = mcnemar_exact(build_transition_table(pairs))
            significance[f"{label}_strict_mcnemar"] = mc.to_dict()

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime = timings if timings is not None else {}
    if not runtime and (OUT_DIR / "runtime.json").exists():
        runtime = json.loads((OUT_DIR / "runtime.json").read_text(encoding="utf-8"))
    return {
        "benchmark": "NLP4LP orig, 331 queries",
        "retrieval": "tfidf (fixed)",
        "extraction": "frozen codebase: typed/constrained share ratio-aware; opt-role/semantic-IR use intrinsic extractors",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "gold_cache": str(GOLD_CACHE.relative_to(ROOT)),
        "git_sha": git_sha,
        "methods": results,
        "significance_vs_typed_greedy": significance,
        "runtime_seconds": runtime,
    }


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARNING: PYTHONHASHSEED is not 0; re-run with PYTHONHASHSEED=0", file=sys.stderr)
    timings = run_all()
    summary = summarize(timings)
    out_json = OUT_DIR / "matched_grounding_baselines_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
