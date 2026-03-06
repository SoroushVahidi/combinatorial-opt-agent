"""Downstream utility demo for NLP4LP: retrieval enables parameter instantiation from NL.

Deterministic, CPU-only: no LLMs, no torch, no solver dependency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NUM_TOKEN_RE = re.compile(r"[$]?\d[\d,]*(?:\.\d+)?%?")

MONEY_CONTEXT = {"budget", "cost", "price", "profit", "revenue", "dollar", "dollars", "$", "€", "usd", "eur"}
PERCENT_CONTEXT = {"percent", "percentage", "rate", "fraction"}


def _safe_json_loads(s: str | None) -> Any:
    if not s:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def _load_eval(eval_path: Path) -> list[dict]:
    items = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "query_id": obj.get("query_id", ""),
                    "query": (obj.get("query") or "").strip(),
                    "relevant_doc_id": obj.get("relevant_doc_id", ""),
                }
            )
    return items


def _load_catalog_as_problems(catalog_path: Path) -> tuple[list[dict], dict[str, str]]:
    """Load catalog JSONL and return list[problem] for baselines + id->text for snippets."""
    catalog: list[dict] = []
    id_to_text: dict[str, str] = {}
    with open(catalog_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("id")
            text = (obj.get("text") or obj.get("description") or "").strip()
            if not doc_id:
                continue
            catalog.append({"id": doc_id, "name": doc_id, "description": text, "aliases": []})
            id_to_text[doc_id] = text
    return catalog, id_to_text


def _load_hf_gold(split: str = "test") -> dict[str, dict]:
    """Load NLP4LP HF split and return doc_id -> parsed fields."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(f"datasets not available: {e}")

    raw = (
        (os.environ.get("HF_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_TOKEN") or "")
    ).strip()
    kwargs = {"token": raw} if raw else {}
    ds = load_dataset("udell-lab/NLP4LP", split=split, **kwargs)

    gold: dict[str, dict] = {}
    for i, ex in enumerate(ds):
        doc_id = f"nlp4lp_{split}_{i}"
        params = _safe_json_loads(ex.get("parameters"))
        pinfo = _safe_json_loads(ex.get("problem_info"))
        gold[doc_id] = {
            "parameters": params if isinstance(params, dict) else {},
            "problem_info": pinfo if isinstance(pinfo, dict) else {},
            "optimus_code": ex.get("optimus_code") or "",
            "solution": _safe_json_loads(ex.get("solution")),
        }
    return gold


def _tokens_lower(text: str) -> list[str]:
    return re.findall(r"\w+|<num>|[$]?\d[\d,]*(?:\.\d+)?%?", text.lower())


@dataclass(frozen=True)
class NumTok:
    raw: str
    value: float | None
    kind: str  # percent|currency|int|float|unknown


def _parse_num_token(tok: str, context_words: set[str]) -> NumTok:
    t = tok.strip()
    if t == "<num>":
        return NumTok(raw=t, value=None, kind="unknown")
    has_dollar = "$" in t
    is_pct = t.endswith("%")
    num_str = t.replace("$", "").replace("%", "").replace(",", "")
    try:
        val = float(num_str)
    except Exception:
        return NumTok(raw=t, value=None, kind="unknown")

    if is_pct:
        return NumTok(raw=t, value=val / 100.0, kind="percent")

    # Percent context without %: treat e.g. "20 percent" as 0.20.
    if ("percent" in context_words or "percentage" in context_words) and val > 1.0:
        return NumTok(raw=t, value=val / 100.0, kind="percent")

    if 0.0 < val <= 1.0 and (context_words & PERCENT_CONTEXT):
        return NumTok(raw=t, value=val, kind="percent")

    if has_dollar or (context_words & MONEY_CONTEXT) or abs(val) >= 1000:
        # Heuristic: treat large numbers as amounts in many optimization word problems.
        return NumTok(raw=t, value=val, kind="currency")

    # Integer vs float
    if float(int(val)) == val:
        return NumTok(raw=t, value=float(int(val)), kind="int")
    return NumTok(raw=t, value=val, kind="float")


def _extract_num_tokens(query: str, variant: str) -> list[NumTok]:
    toks = query.split()
    out: list[NumTok] = []
    for i, w in enumerate(toks):
        if w == "<num>" and variant in ("noisy", "nonum"):
            out.append(NumTok(raw=w, value=None, kind="unknown"))
            continue
        m = NUM_TOKEN_RE.fullmatch(w.strip())
        if not m:
            continue
        # local context window
        ctx = set(x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 3) : i + 4])
        out.append(_parse_num_token(w, ctx))
    return out


def _expected_type(param_name: str) -> str:
    n = (param_name or "").lower()
    if any(s in n for s in ("percent", "percentage", "rate", "fraction")):
        return "percent"
    if any(s in n for s in ("num", "count", "types", "items", "ingredients", "nodes", "edges")):
        return "int"
    if any(
        s in n
        for s in (
            "budget",
            "cost",
            "price",
            "revenue",
            "profit",
            "penalty",
            "investment",
            "demand",
            "capacity",
            "minimum",
            "maximum",
            "limit",
        )
    ):
        return "currency"
    return "float"


def _choose_token(expected: str, candidates: list[NumTok]) -> tuple[int | None, NumTok | None]:
    """Return (index, token) to use, or (None, None) if no candidates."""
    if not candidates:
        return None, None

    def score(tok: NumTok) -> tuple:
        # higher is better; deterministic tie-break via raw
        val = tok.value if tok.value is not None else 0.0
        absval = abs(val)
        has_decimal = (tok.value is not None) and (float(int(val)) != val)
        if expected == "percent":
            pref = 2 if tok.kind == "percent" else (1 if (tok.value is not None and 0.0 < tok.value <= 1.0) else 0)
            return (pref, absval, tok.raw)
        if expected == "int":
            pref = 2 if tok.kind == "int" else (1 if tok.value is not None and float(int(val)) == val else 0)
            return (pref, absval, tok.raw)
        if expected == "currency":
            pref = 2 if tok.kind == "currency" else 0
            return (pref, absval, tok.raw)
        # float
        pref = 2 if tok.kind == "float" and has_decimal else (1 if tok.kind in ("float", "int", "currency") else 0)
        return (pref, absval, tok.raw)

    best_i = 0
    best_s = score(candidates[0])
    for i in range(1, len(candidates)):
        s = score(candidates[i])
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, candidates[best_i]


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _rel_err(pred: float, gold: float) -> float:
    return abs(pred - gold) / max(1.0, abs(gold))


def _md5_seed(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _upsert_summary_row(summary_path: Path, row: dict) -> None:
    cols = [
        "variant",
        "baseline",
        "schema_R1",
        "param_coverage",
        "type_match",
        "exact5_on_hits",
        "exact20_on_hits",
        "param_coverage_hits",
        "param_coverage_miss",
        "type_match_hits",
        "type_match_miss",
        "key_overlap",
        "key_overlap_hits",
        "key_overlap_miss",
        "instantiation_ready",
        "n",
    ]
    rows: list[dict] = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    d = {(r.get("variant"), r.get("baseline")): r for r in rows}
    d[(row["variant"], row["baseline"])] = {k: row.get(k, "") for k in cols}

    # deterministic order
    ordered = sorted(d.values(), key=lambda x: (x.get("variant", ""), x.get("baseline", "")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ordered)


def _upsert_types_rows(types_path: Path, types_agg: dict[str, dict]) -> None:
    cols = [
        "variant",
        "baseline",
        "param_type",
        "n_expected",
        "n_filled",
        "param_coverage",
        "type_match",
        "exact5_on_hits",
        "exact20_on_hits",
        "n_queries",
    ]
    rows: list[dict] = []
    if types_path.exists():
        with open(types_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    d = {(r.get("variant"), r.get("baseline"), r.get("param_type")): r for r in rows}
    for t, info in types_agg.items():
        key = (info["variant"], info["baseline"], info["param_type"])
        d[key] = {k: info.get(k, "") for k in cols}

    ordered = sorted(d.values(), key=lambda x: (x.get("variant", ""), x.get("baseline", ""), x.get("param_type", "")))
    types_path.parent.mkdir(parents=True, exist_ok=True)
    with open(types_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ordered)


def run_setting(
    variant: str,
    baseline_name: str,
    eval_items: list[dict],
    gold_by_id: dict[str, dict],
    rank_fn,
    doc_ids: list[str],
    random_control: bool,
    assignment_mode: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_path = out_dir / f"nlp4lp_downstream_per_query_{variant}_{baseline_name}.csv"
    json_path = out_dir / f"nlp4lp_downstream_{variant}_{baseline_name}.json"
    summary_path = out_dir / "nlp4lp_downstream_summary.csv"

    def run_one(label: str, mode: str) -> tuple[list[dict], dict, dict]:
        rows: list[dict] = []
        hit_flags: list[int] = []
        cov_vals: list[float] = []
        type_vals: list[float] = []
        exact5_vals: list[float] = []
        exact20_vals: list[float] = []
        inst_ready_flags: list[int] = []
        cov_hits: list[float] = []
        cov_miss: list[float] = []
        type_hits: list[float] = []
        type_miss: list[float] = []
        ko_all: list[float] = []
        ko_hits: list[float] = []
        ko_miss: list[float] = []

        type_names = ["percent", "integer", "currency", "float"]
        type_expected_total = {t: 0 for t in type_names}
        type_filled_total = {t: 0 for t in type_names}
        type_correct_total = {t: 0 for t in type_names}
        type_exact5_num = {t: 0 for t in type_names}
        type_exact5_den = {t: 0 for t in type_names}
        type_exact20_num = {t: 0 for t in type_names}
        type_exact20_den = {t: 0 for t in type_names}

        for ex in eval_items:
            qid = ex["query_id"]
            query = ex["query"]
            gold_id = ex["relevant_doc_id"]
            if mode == "oracle":
                pred_id = gold_id
            elif mode == "random":
                rng = random.Random(_md5_seed(qid))
                pred_id = doc_ids[rng.randrange(len(doc_ids))] if doc_ids else ""
            else:  # retrieval
                ranked = rank_fn(query, top_k=1)
                pred_id = ranked[0][0] if ranked else ""
            schema_hit = 1 if pred_id == gold_id else 0

            # Gold problem (for evaluation) and predicted problem (for schema)
            gold = gold_by_id.get(gold_id) or {}
            gold_params = gold.get("parameters") or {}
            pred = gold_by_id.get(pred_id) or {}
            pred_params = pred.get("parameters") or {}
            pred_info = pred.get("problem_info") or {}

            expected_params: list[str] = []
            if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
                expected_params = list(pred_info["parameters"].keys())
            elif isinstance(pred_params, dict):
                expected_params = list(pred_params.keys())

            # scalar keys based on gold values
            gold_scalar_keys = {p for p, v in (gold_params or {}).items() if _is_scalar(v)}

            def _bucket_type(pname: str) -> str:
                et = _expected_type(pname)
                if et == "percent":
                    return "percent"
                if et == "int":
                    return "integer"
                if et == "currency":
                    return "currency"
                return "float"

            # count expected scalar params per type from gold schema
            for p in gold_scalar_keys:
                t = _bucket_type(p)
                type_expected_total[t] += 1

            pred_scalar_keys = {
                p for p in expected_params if _is_scalar(gold_params.get(p))
            } if isinstance(gold_params, dict) else set()
            # expected scalar params (for filling) are intersection of predicted keys with scalar gold
            expected_scalar = list(pred_scalar_keys)
            n_expected_scalar = len(expected_scalar)
            # schema key overlap (relative to gold scalar keys)
            if gold_scalar_keys:
                key_overlap = len(pred_scalar_keys & gold_scalar_keys) / float(len(gold_scalar_keys))
            else:
                key_overlap = 0.0

            num_toks = _extract_num_tokens(query, variant)
            candidates = list(num_toks)

            filled = {}
            type_matches = 0
            n_filled = 0
            comparable_errs = []

            # per-query per-type counts (optional, used for per-query CSV enrichment if desired later)
            type_expected_q = {t: 0 for t in type_names}
            type_filled_q = {t: 0 for t in type_names}
            type_correct_q = {t: 0 for t in type_names}

            for p in gold_scalar_keys:
                t = _bucket_type(p)
                type_expected_q[t] += 1

            for p in expected_scalar:
                et = _expected_type(p)
                if assignment_mode == "untyped":
                    idx, tok = (0, candidates[0]) if candidates else (None, None)
                else:
                    idx, tok = _choose_token(et, candidates)
                if tok is None:
                    continue
                # remove used token
                if idx is not None:
                    candidates.pop(idx)
                n_filled += 1
                filled[p] = tok.value if tok.value is not None else tok.raw
                btype = _bucket_type(p)
                type_filled_total[btype] += 1
                type_filled_q[btype] += 1
                if tok.kind == et:
                    type_matches += 1
                    type_correct_total[btype] += 1
                    type_correct_q[btype] += 1
                # error only if numeric and schema hit
                if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                    gold_val = float(gold_params[p])
                    err = _rel_err(float(tok.value), gold_val)
                    comparable_errs.append(err)
                    if btype in type_names:
                        type_exact5_den[btype] += 1
                        type_exact20_den[btype] += 1
                        if err <= 0.05:
                            type_exact5_num[btype] += 1
                        if err <= 0.20:
                            type_exact20_num[btype] += 1

            param_coverage = (n_filled / max(1, n_expected_scalar)) if n_expected_scalar else 0.0
            type_match = (type_matches / max(1, n_filled)) if n_filled else 0.0
            ko_all.append(key_overlap)

            exact5 = ""
            exact20 = ""
            if schema_hit:
                if comparable_errs:
                    exact5 = sum(1 for e in comparable_errs if e <= 0.05) / len(comparable_errs)
                    exact20 = sum(1 for e in comparable_errs if e <= 0.20) / len(comparable_errs)
                else:
                    exact5 = ""
                    exact20 = ""

            rows.append(
                {
                    "query_id": qid,
                    "variant": variant,
                    "baseline": label,
                    "predicted_doc_id": pred_id,
                    "gold_doc_id": gold_id,
                    "schema_hit": schema_hit,
                    "n_expected_scalar": n_expected_scalar,
                    "n_filled": n_filled,
                    "param_coverage": param_coverage,
                    "type_match": type_match,
                    "exact5": exact5,
                    "exact20": exact20,
                    "key_overlap": key_overlap,
                }
            )

            hit_flags.append(schema_hit)
            cov_vals.append(param_coverage)
            type_vals.append(type_match)
            if schema_hit:
                cov_hits.append(param_coverage)
                type_hits.append(type_match)
                ko_hits.append(key_overlap)
            else:
                cov_miss.append(param_coverage)
                type_miss.append(type_match)
                ko_miss.append(key_overlap)
            if isinstance(exact5, float):
                exact5_vals.append(exact5)
            if isinstance(exact20, float):
                exact20_vals.append(exact20)
            inst_ready_flags.append(1 if (param_coverage >= 0.8 and type_match >= 0.8) else 0)

        n = len(rows)
        def _mean(xs: list[float]) -> float | str:
            return (sum(xs) / len(xs)) if xs else ""

        agg = {
            "variant": variant,
            "baseline": label,
            "schema_R1": sum(hit_flags) / n if n else 0.0,
            "param_coverage": sum(cov_vals) / n if n else 0.0,
            "type_match": sum(type_vals) / n if n else 0.0,
            "exact5_on_hits": (sum(exact5_vals) / len(exact5_vals)) if exact5_vals else "",
            "exact20_on_hits": (sum(exact20_vals) / len(exact20_vals)) if exact20_vals else "",
            "param_coverage_hits": _mean(cov_hits),
            "param_coverage_miss": _mean(cov_miss),
            "type_match_hits": _mean(type_hits),
            "type_match_miss": _mean(type_miss),
            "key_overlap": _mean(ko_all),
            "key_overlap_hits": _mean(ko_hits),
            "key_overlap_miss": _mean(ko_miss),
            "instantiation_ready": sum(inst_ready_flags) / n if n else 0.0,
            "n": n,
        }
        # per-type aggregate summary for this (variant, baseline)
        types_agg: dict[str, dict] = {}
        n_queries = len(rows)
        for t in type_names:
            n_exp_t = type_expected_total[t]
            n_fill_t = type_filled_total[t]
            cov_t = (n_fill_t / max(1, n_exp_t)) if n_exp_t else 0.0
            tm_t = (
                (type_correct_total[t] / n_fill_t) if n_fill_t else ""
            )
            e5 = (
                (type_exact5_num[t] / type_exact5_den[t]) if type_exact5_den[t] else ""
            )
            e20 = (
                (type_exact20_num[t] / type_exact20_den[t]) if type_exact20_den[t] else ""
            )
            types_agg[t] = {
                "variant": variant,
                "baseline": label,
                "param_type": t,
                "n_expected": n_exp_t,
                "n_filled": n_fill_t,
                "param_coverage": cov_t,
                "type_match": tm_t,
                "exact5_on_hits": e5,
                "exact20_on_hits": e20,
                "n_queries": n_queries,
            }

        return rows, agg, types_agg

    rows_main, agg_main, types_main = run_one(baseline_name, "oracle" if baseline_name.startswith("oracle") else "retrieval")

    cols = [
        "query_id",
        "variant",
        "baseline",
        "predicted_doc_id",
        "gold_doc_id",
        "schema_hit",
        "n_expected_scalar",
        "n_filled",
        "param_coverage",
        "type_match",
        "exact5",
        "exact20",
        "key_overlap",
    ]
    with open(per_query_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_main)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"variant": variant, "baseline": baseline_name, "k": 1, "random_control": random_control},
                "aggregate": agg_main,
            },
            f,
            indent=2,
        )

    _upsert_summary_row(summary_path, agg_main)

    # Update per-type summary
    types_summary_path = out_dir / "nlp4lp_downstream_types_summary.csv"
    _upsert_types_rows(types_summary_path, types_main)

    if random_control:
        rand_label = "random_untyped" if baseline_name.endswith("_untyped") else "random"
        _, agg_rand, types_rand = run_one(rand_label, "random")
        _upsert_summary_row(summary_path, agg_rand)
        _upsert_types_rows(types_summary_path, types_rand)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="NLP4LP downstream utility demo (retrieval -> parameter instantiation)")
    p.add_argument("--variant", type=str, default="orig", choices=("orig", "noisy", "short"))
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    p.add_argument("--eval", type=Path, default=None)
    p.add_argument("--baseline", type=str, default="tfidf", choices=("bm25", "tfidf", "lsa", "oracle"))
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--random-control", action="store_true")
    p.add_argument("--assignment-mode", type=str, default="typed", choices=("typed", "untyped"))
    args = p.parse_args()

    if args.k != 1:
        raise SystemExit("This demo currently supports --k 1 only (top-1 schema selection).")

    eval_path = args.eval or (ROOT / "data" / "processed" / f"nlp4lp_eval_{args.variant}.jsonl")
    eval_items = _load_eval(Path(eval_path))
    if not eval_items:
        raise SystemExit(f"No eval instances loaded from {eval_path}")

    # Gold schema/parameters from HF
    gold_by_id = _load_hf_gold(split="test")

    catalog, _id_to_text = _load_catalog_as_problems(args.catalog)
    doc_ids = [p["id"] for p in catalog if p.get("id")]

    rank_fn = None
    if args.baseline != "oracle":
        from retrieval.baselines import get_baseline

        baseline = get_baseline(args.baseline)
        baseline.fit(catalog)
        rank_fn = baseline.rank

    effective_baseline = args.baseline
    if args.assignment_mode == "untyped":
        effective_baseline = f"{args.baseline}_untyped"

    out_dir = ROOT / "results" / "paper"
    run_setting(
        variant=args.variant,
        baseline_name=effective_baseline,
        eval_items=eval_items,
        gold_by_id=gold_by_id,
        rank_fn=rank_fn,
        doc_ids=doc_ids,
        random_control=bool(args.random_control),
        assignment_mode=args.assignment_mode,
        out_dir=out_dir,
    )

    print(f"Wrote {out_dir / f'nlp4lp_downstream_{args.variant}_{effective_baseline}.json'}")
    print(f"Wrote {out_dir / f'nlp4lp_downstream_per_query_{args.variant}_{effective_baseline}.csv'}")
    print(f"Updated {out_dir / 'nlp4lp_downstream_summary.csv'}")


if __name__ == "__main__":
    main()

