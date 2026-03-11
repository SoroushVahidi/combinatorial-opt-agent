"""
1C – More rigorous measured error-analysis breakdowns.

Produces structured per-instance diagnostics, breakdown tables, and example
mining from the existing per-query CSV files.

All inputs are read from:
  results/eswa_revision/02_downstream_postfix/
  data/processed/nlp4lp_eval_orig.jsonl   (for query text)
  data/catalogs/nlp4lp_catalog.jsonl      (for schema text)

Usage (from repo root):
    python tools/run_error_analysis.py

Outputs (written to results/eswa_revision/16_error_analysis/):
  per_instance_diagnostics.csv
  schema_hit_miss_breakdown.csv
  slot_count_breakdown.csv
  ambiguity_breakdown.csv
  ERROR_EXAMPLES.md
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOWNSTREAM_DIR = ROOT / "results" / "eswa_revision" / "02_downstream_postfix"
OUT_DIR = ROOT / "results" / "eswa_revision" / "16_error_analysis"

EVAL_ORIG = ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl"
CATALOG_PATH = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"

# Primary analysis: tfidf, orig variant
PRIMARY_METHOD = "tfidf"
PRIMARY_VARIANT = "orig"

# Methods to include in the cross-method comparison table
COMPARISON_METHODS = [
    "tfidf",
    "bm25",
    "oracle",
    "tfidf_acceptance_rerank",
    "tfidf_hierarchical_acceptance_rerank",
    "tfidf_optimization_role_repair",
    # Method Family 1: Global Compatibility Grounding
    "tfidf_global_compat_local",
    "tfidf_global_compat_pairwise",
    "tfidf_global_compat_full",
    # Method Family 2: Relation-Aware Linking
    "tfidf_relation_aware_basic",
    "tfidf_relation_aware_ops",
    "tfidf_relation_aware_semantic",
    "tfidf_relation_aware_full",
    # Method Family 3: Ambiguity-Aware Grounding
    "tfidf_ambiguity_candidate_greedy",
    "tfidf_ambiguity_aware_beam",
    "tfidf_ambiguity_aware_abstain",
    "tfidf_ambiguity_aware_full",
]

# ── data loading ─────────────────────────────────────────────────────────────

def _load_query_text() -> dict[str, str]:
    """Map query_id -> query text from eval JSONL."""
    result = {}
    if not EVAL_ORIG.exists():
        return result
    with open(EVAL_ORIG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # query_id is the primary key; relevant_doc_id may also serve as ID
            # in older eval file formats where the two fields are identical.
            qid = obj.get("query_id") or obj.get("relevant_doc_id") or ""
            text = obj.get("query") or ""
            if qid:
                result[qid] = text
    return result


def _load_catalog_text() -> dict[str, str]:
    """Map doc_id -> schema text from catalog JSONL."""
    result = {}
    if not CATALOG_PATH.exists():
        return result
    with open(CATALOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("id") or ""
            text = obj.get("text") or obj.get("description") or ""
            if doc_id:
                result[doc_id] = text
    return result


def _load_per_query(variant: str, method: str) -> list[dict]:
    """Load per-query CSV for (variant, method)."""
    fname = f"nlp4lp_downstream_per_query_{variant}_{method}.csv"
    path = DOWNSTREAM_DIR / fname
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── numeric signal from query ─────────────────────────────────────────────────

_NUM_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _count_numeric_mentions(text: str) -> int:
    """Count distinct numeric token occurrences in a query (ambiguity proxy)."""
    return len(_NUM_RE.findall(text or ""))


# ── per-instance diagnostics ─────────────────────────────────────────────────

def _build_diagnostics(
    rows: list[dict],
    query_text: dict[str, str],
    method: str,
    variant: str,
) -> list[dict]:
    """
    Enrich each per-query row with derived diagnostic signals.

    Output columns:
      query_id, variant, method,
      query_text (first 120 chars),
      schema_hit, schema_miss,
      n_expected_scalar, n_filled,
      full_coverage, all_types_correct, inst_ready,
      any_type_mismatch, any_slot_comparable,
      n_numeric_mentions, ambiguity_bucket,
      slot_count_bucket
    """
    out = []
    for r in rows:
        qid = r.get("query_id", "")
        qt = query_text.get(qid, "")
        try:
            schema_hit = int(float(r.get("schema_hit", 0)))
        except (ValueError, TypeError):
            schema_hit = 0
        try:
            n_exp = int(float(r.get("n_expected_scalar", 0)))
        except (ValueError, TypeError):
            n_exp = 0
        try:
            n_filled = int(float(r.get("n_filled", 0)))
        except (ValueError, TypeError):
            n_filled = 0
        try:
            cov = float(r.get("param_coverage", 0))
        except (ValueError, TypeError):
            cov = 0.0
        try:
            tm = float(r.get("type_match", 0))
        except (ValueError, TypeError):
            tm = 0.0
        try:
            exact20 = float(r.get("exact20", "")) if r.get("exact20", "") else float("nan")
        except (ValueError, TypeError):
            exact20 = float("nan")

        inst_ready = 1 if cov >= 0.8 and tm >= 0.8 else 0
        full_coverage = 1 if n_exp > 0 and n_filled >= n_exp else (1 if n_exp == 0 else 0)
        all_types_correct = 1 if tm >= 1.0 and n_filled > 0 else 0
        any_type_mismatch = 1 if tm < 1.0 and n_filled > 0 else 0
        any_slot_comparable = 1 if not math.isnan(exact20) else 0

        n_num = _count_numeric_mentions(qt)
        # Ambiguity bucket: low / medium / high numeric density
        if n_num <= 2:
            amb = "low"
        elif n_num <= 5:
            amb = "medium"
        else:
            amb = "high"

        # Slot count bucket
        if n_exp == 0:
            slot_bucket = "0"
        elif n_exp == 1:
            slot_bucket = "1"
        elif n_exp == 2:
            slot_bucket = "2"
        elif n_exp == 3:
            slot_bucket = "3"
        else:
            slot_bucket = "4+"

        out.append({
            "query_id": qid,
            "variant": variant,
            "method": method,
            "query_text_preview": qt[:120],
            "schema_hit": schema_hit,
            "schema_miss": 1 - schema_hit,
            "n_expected_scalar": n_exp,
            "n_filled": n_filled,
            "full_coverage": full_coverage,
            "all_types_correct": all_types_correct,
            "inst_ready": inst_ready,
            "any_type_mismatch": any_type_mismatch,
            "any_slot_comparable": any_slot_comparable,
            "param_coverage": f"{cov:.4f}",
            "type_match": f"{tm:.4f}",
            "n_numeric_mentions": n_num,
            "ambiguity_bucket": amb,
            "slot_count_bucket": slot_bucket,
            "predicted_doc_id": r.get("predicted_doc_id", ""),
            "gold_doc_id": r.get("gold_doc_id", ""),
        })
    return out


# ── breakdown tables ──────────────────────────────────────────────────────────

def _mean(vals: list) -> str:
    if not vals:
        return "nan"
    return f"{sum(vals) / len(vals):.4f}"


def schema_hit_miss_breakdown(diags: list[dict]) -> list[dict]:
    """Mean Coverage / TypeMatch / InstReady by schema_hit vs schema_miss."""
    groups: dict[str, list] = {"hit": [], "miss": []}
    for d in diags:
        key = "hit" if d["schema_hit"] else "miss"
        groups[key].append(d)
    rows = []
    for label, group in [("hit (schema_hit=1)", groups["hit"]), ("miss (schema_hit=0)", groups["miss"])]:
        if not group:
            continue
        rows.append({
            "group": label,
            "n": len(group),
            "Coverage_mean": _mean([float(g["param_coverage"]) for g in group]),
            "TypeMatch_mean": _mean([float(g["type_match"]) for g in group]),
            "InstReady_rate": _mean([float(g["inst_ready"]) for g in group]),
            "full_coverage_rate": _mean([float(g["full_coverage"]) for g in group]),
        })
    return rows


def slot_count_breakdown(diags: list[dict]) -> list[dict]:
    """Performance by n_expected_scalar bucket."""
    buckets: dict[str, list] = defaultdict(list)
    for d in diags:
        buckets[d["slot_count_bucket"]].append(d)
    rows = []
    for key in ["0", "1", "2", "3", "4+"]:
        group = buckets.get(key, [])
        if not group:
            continue
        rows.append({
            "slot_count_bucket": key,
            "n": len(group),
            "Coverage_mean": _mean([float(g["param_coverage"]) for g in group]),
            "TypeMatch_mean": _mean([float(g["type_match"]) for g in group]),
            "InstReady_rate": _mean([float(g["inst_ready"]) for g in group]),
            "schema_hit_rate": _mean([float(g["schema_hit"]) for g in group]),
        })
    return rows


def ambiguity_breakdown(diags: list[dict]) -> list[dict]:
    """Performance stratified by numeric mention count (ambiguity proxy)."""
    buckets: dict[str, list] = defaultdict(list)
    for d in diags:
        buckets[d["ambiguity_bucket"]].append(d)
    rows = []
    for key in ["low", "medium", "high"]:
        group = buckets.get(key, [])
        if not group:
            continue
        n_num_vals = [g["n_numeric_mentions"] for g in group]
        rows.append({
            "ambiguity_bucket": key,
            "n_mentions_range": f"{min(n_num_vals)}-{max(n_num_vals)}",
            "n": len(group),
            "Coverage_mean": _mean([float(g["param_coverage"]) for g in group]),
            "TypeMatch_mean": _mean([float(g["type_match"]) for g in group]),
            "InstReady_rate": _mean([float(g["inst_ready"]) for g in group]),
            "schema_hit_rate": _mean([float(g["schema_hit"]) for g in group]),
        })
    return rows


# ── example mining ────────────────────────────────────────────────────────────

def _mine_examples(
    diags: list[dict],
    query_text: dict[str, str],
    schema_text: dict[str, str],
    n_per_type: int = 3,
) -> list[dict]:
    """
    Collect representative failure examples. Returns list of annotated dicts.
    """
    examples = []

    # 1. Schema correct but not instantiation-ready
    cand = [
        d for d in diags
        if d["schema_hit"] == 1 and d["inst_ready"] == 0 and d["n_expected_scalar"] > 0
    ]
    for d in cand[:n_per_type]:
        examples.append({
            "type": "schema_hit_but_not_inst_ready",
            "reason": "likely: type mismatch or incomplete coverage despite correct schema",
            **d,
        })

    # 2. Type mismatch failure (schema hit, any_type_mismatch)
    cand = [
        d for d in diags
        if d["schema_hit"] == 1 and d["any_type_mismatch"] == 1
        and float(d["type_match"]) < 0.7
        and d["inst_ready"] == 0
    ]
    for d in cand[:n_per_type]:
        examples.append({
            "type": "type_mismatch",
            "reason": "likely: wrong type assigned to slot (float vs integer, percent vs absolute)",
            **d,
        })

    # 3. Likely slot disambiguation failure (schema hit, coverage >=0.8, type_match <0.5)
    cand = [
        d for d in diags
        if d["schema_hit"] == 1 and float(d["param_coverage"]) >= 0.8
        and float(d["type_match"]) < 0.5
    ]
    for d in cand[:n_per_type]:
        examples.append({
            "type": "likely_slot_disambiguation_failure",
            "reason": "likely: coverage acceptable but most types wrong; probable slot/type confusion",
            **d,
        })

    # 4. Schema miss
    cand = [d for d in diags if d["schema_hit"] == 0]
    for d in cand[:n_per_type]:
        examples.append({
            "type": "schema_miss",
            "reason": "measured: retrieval returned wrong schema; downstream grounding irrelevant",
            **d,
        })

    # Attach query and schema texts
    for ex in examples:
        qid = ex["query_id"]
        ex["query_text"] = query_text.get(qid, "(unavailable)")
        ex["gold_schema_text_preview"] = schema_text.get(ex["gold_doc_id"], "(unavailable)")[:200]
        if ex["schema_hit"] == 0:
            ex["predicted_schema_text_preview"] = schema_text.get(ex["predicted_doc_id"], "(unavailable)")[:200]
        else:
            ex["predicted_schema_text_preview"] = "(same as gold)"

    return examples


def _render_examples_md(examples: list[dict]) -> str:
    lines = [
        "# Representative Failure Examples",
        "",
        "> **Methodology note:** Examples are selected programmatically from per-instance",
        "> diagnostics for the TF-IDF baseline, orig variant.",
        "> Reasons labelled *likely* are inferred from measurable signals;",
        "> reasons labelled *measured* are directly observable from data.",
        "",
    ]
    type_shown: dict[str, int] = defaultdict(int)
    for ex in examples:
        t = ex["type"]
        type_shown[t] += 1
        lines += [
            f"---",
            f"",
            f"## [{t}] Example {type_shown[t]}",
            f"",
            f"**Query ID:** `{ex['query_id']}`  ",
            f"**Reason:** {ex['reason']}  ",
            f"**Schema hit:** {ex['schema_hit']}  ",
            f"**Coverage:** {ex['param_coverage']}  ",
            f"**TypeMatch:** {ex['type_match']}  ",
            f"**InstReady:** {ex['inst_ready']}  ",
            f"**n_expected_scalar:** {ex['n_expected_scalar']}  ",
            f"**n_filled:** {ex['n_filled']}  ",
            f"**Numeric mentions in query:** {ex['n_numeric_mentions']}  ",
            f"",
            f"**Query (full):**",
            f"> {ex['query_text']}",
            f"",
            f"**Gold schema text (preview):**",
            f"> {ex['gold_schema_text_preview']}",
        ]
        if ex["schema_hit"] == 0:
            lines += [
                f"",
                f"**Predicted schema text (preview):**",
                f"> {ex['predicted_schema_text_preview']}",
            ]
        lines.append("")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    query_text = _load_query_text()
    schema_text = _load_catalog_text()

    # ── Per-instance diagnostics (tfidf, orig) ────────────────────────────────
    rows = _load_per_query(PRIMARY_VARIANT, PRIMARY_METHOD)
    if not rows:
        print(f"WARNING: no per-query data found for {PRIMARY_VARIANT}/{PRIMARY_METHOD}")
        return

    diags = _build_diagnostics(rows, query_text, PRIMARY_METHOD, PRIMARY_VARIANT)

    diag_path = OUT_DIR / "per_instance_diagnostics.csv"
    diag_fields = [
        "query_id", "variant", "method", "schema_hit", "schema_miss",
        "n_expected_scalar", "n_filled", "full_coverage", "all_types_correct",
        "inst_ready", "any_type_mismatch", "any_slot_comparable",
        "param_coverage", "type_match", "n_numeric_mentions",
        "ambiguity_bucket", "slot_count_bucket",
        "predicted_doc_id", "gold_doc_id", "query_text_preview",
    ]
    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=diag_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(diags)
    print(f"Wrote {diag_path} ({len(diags)} rows)")

    # ── Schema hit/miss breakdown ─────────────────────────────────────────────
    hm_rows = schema_hit_miss_breakdown(diags)
    hm_path = OUT_DIR / "schema_hit_miss_breakdown.csv"
    with open(hm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n", "Coverage_mean", "TypeMatch_mean", "InstReady_rate", "full_coverage_rate"])
        w.writeheader()
        w.writerows(hm_rows)
    print(f"Wrote {hm_path}")

    # ── Slot count breakdown ──────────────────────────────────────────────────
    sc_rows = slot_count_breakdown(diags)
    sc_path = OUT_DIR / "slot_count_breakdown.csv"
    with open(sc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slot_count_bucket", "n", "Coverage_mean", "TypeMatch_mean", "InstReady_rate", "schema_hit_rate"])
        w.writeheader()
        w.writerows(sc_rows)
    print(f"Wrote {sc_path}")

    # ── Ambiguity breakdown ───────────────────────────────────────────────────
    amb_rows = ambiguity_breakdown(diags)
    amb_path = OUT_DIR / "ambiguity_breakdown.csv"
    with open(amb_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ambiguity_bucket", "n_mentions_range", "n", "Coverage_mean", "TypeMatch_mean", "InstReady_rate", "schema_hit_rate"])
        w.writeheader()
        w.writerows(amb_rows)
    print(f"Wrote {amb_path}")

    # ── Error examples markdown ───────────────────────────────────────────────
    examples = _mine_examples(diags, query_text, schema_text)
    ex_md = _render_examples_md(examples)
    ex_path = OUT_DIR / "ERROR_EXAMPLES.md"
    with open(ex_path, "w", encoding="utf-8") as f:
        f.write(ex_md)
    print(f"Wrote {ex_path} ({len(examples)} examples)")

    # ── Print summary ─────────────────────────────────────────────────────────
    n = len(diags)
    n_hit = sum(d["schema_hit"] for d in diags)
    n_inst = sum(d["inst_ready"] for d in diags)
    print(f"\nSummary ({PRIMARY_METHOD}, {PRIMARY_VARIANT}, n={n}):")
    print(f"  Schema hits: {n_hit} ({n_hit/n:.3f})")
    print(f"  Inst-ready:  {n_inst} ({n_inst/n:.3f})")
    for row in hm_rows:
        print(f"  {row['group']}: cov={row['Coverage_mean']} tm={row['TypeMatch_mean']} ir={row['InstReady_rate']}")

    # ── Cross-method comparison table ─────────────────────────────────────────
    cmp_path = OUT_DIR / "method_comparison_table.csv"
    cmp_fields = ["method", "variant", "n", "schema_R1", "Coverage", "TypeMatch", "InstReady"]
    cmp_rows: list[dict] = []
    for method in COMPARISON_METHODS:
        for variant in ("orig", "noisy", "short"):
            mrows = _load_per_query(variant, method)
            if not mrows:
                continue
            nm = len(mrows)
            s_r1 = sum(int(r.get("schema_hit", 0) or 0) for r in mrows) / nm
            cov = sum(float(r.get("param_coverage", 0) or 0) for r in mrows) / nm
            tm = sum(float(r.get("type_match", 0) or 0) for r in mrows) / nm
            # inst_ready derived from JSON aggregate if per-query lacks the column
            json_path = DOWNSTREAM_DIR / f"nlp4lp_downstream_{variant}_{method}.json"
            ir = 0.0
            if json_path.exists():
                import json as _json
                try:
                    agg = _json.load(open(json_path, encoding="utf-8")).get("aggregate", {})
                    ir = float(agg.get("instantiation_ready", 0))
                except Exception:
                    pass
            cmp_rows.append({"method": method, "variant": variant, "n": nm,
                             "schema_R1": round(s_r1, 4), "Coverage": round(cov, 4),
                             "TypeMatch": round(tm, 4), "InstReady": round(ir, 4)})
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cmp_fields)
        w.writeheader()
        w.writerows(cmp_rows)
    print(f"Wrote {cmp_path} ({len(cmp_rows)} rows)")


if __name__ == "__main__":
    main()
