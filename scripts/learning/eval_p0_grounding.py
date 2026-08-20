#!/usr/bin/env python3
"""Evaluate M2 (P0 greedy), M3 (P0 + max-weight assignment), M4 (M3 + validate/repair)
on the P0 test split, using the same Coverage/TypeMatch/InstantiationReady
definitions as the canonical downstream pipeline
(``tools/nlp4lp_downstream_utility.py``): param_coverage = n_filled /
n_expected_scalar; type_match = type_matches / n_filled; instantiation_ready
= 1 iff param_coverage >= 0.8 AND type_match >= 0.8.

M0 (canonical oracle + typed greedy) and M1 (NR10, historical) are NOT
recomputed here -- M0 is produced by directly invoking the unmodified
canonical CLI (see scripts/learning/run_p0_pipeline.sh) on the same 50-query
test subset; M1 is the already-committed NR10 result
(docs/NEGATIVE_RESULTS.md NR10), reused, not rerun.

Usage:
    python3 scripts/learning/eval_p0_grounding.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from tools.learned_local_scorer import ALL_FEATURE_NAMES, feature_dict_to_vector, validate_assignment  # noqa: E402
from tools.nlp4lp_downstream_utility import (  # noqa: E402
    MentionOptIR,
    SlotOptIR,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _is_type_match,
)

DATA_DIR = ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp_p0"
MODEL_DIR = ROOT / "artifacts" / "learning_runs" / "p0"
EVAL_ORIG_PATH = ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl"
OUT_DIR = ROOT / "results" / "learned_grounding_p0"


def load_split(split: str) -> list[dict]:
    rows = []
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_vector(row: dict) -> list[float]:
    feats = dict(row["engineered_features"])
    feats["embedding_similarity"] = row.get("embedding_similarity", 0.0)
    return feature_dict_to_vector(feats, ALL_FEATURE_NAMES)


def per_instance_metrics(
    instance_ids: list[str],
    n_expected_by_instance: dict[str, int],
    filled_by_instance: dict[str, dict[str, tuple[str, bool]]],
    # filled_by_instance[instance_id][slot_name] = (mention_id_str, type_matched)
) -> list[dict]:
    rows = []
    for iid in instance_ids:
        n_expected = n_expected_by_instance.get(iid, 0)
        filled = filled_by_instance.get(iid, {})
        n_filled = len(filled)
        type_matches = sum(1 for _, tm in filled.values() if tm)
        coverage = (n_filled / n_expected) if n_expected else 0.0
        type_match = (type_matches / n_filled) if n_filled else 0.0
        inst_ready = 1 if (coverage >= 0.8 and type_match >= 0.8) else 0
        rows.append(
            {
                "instance_id": iid,
                "n_expected_scalar": n_expected,
                "n_filled": n_filled,
                "param_coverage": coverage,
                "type_match": type_match,
                "instantiation_ready": inst_ready,
            }
        )
    return rows


def aggregate(per_query: list[dict]) -> dict:
    n = len(per_query)
    if n == 0:
        return {}
    return {
        "n": n,
        "param_coverage": sum(r["param_coverage"] for r in per_query) / n,
        "type_match": sum(r["type_match"] for r in per_query) / n,
        "instantiation_ready": sum(r["instantiation_ready"] for r in per_query) / n,
    }


def decode_greedy(scores_by_group: dict[str, list[tuple[str, float, str, str]]]) -> dict[str, tuple[str, str]]:
    """M2: independent per-slot argmax. Returns group_id -> (mention_id, expected_type)."""
    out = {}
    for gid, candidates in scores_by_group.items():
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c[1])
        out[gid] = (best[0], best[3])
    return out


def decode_max_weight(
    slot_names: list[str],
    mention_ids: list[str],
    score_matrix: np.ndarray,
    expected_type_by_slot: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """M3: exact Hungarian assignment (scipy.optimize.linear_sum_assignment), same
    algorithm the canonical `_run_max_weight_matching_grounding` already uses."""
    if not slot_names or not mention_ids:
        return {}
    cost = -score_matrix  # linear_sum_assignment minimizes; we want to maximize score
    row_ind, col_ind = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(row_ind, col_ind):
        slot = slot_names[r]
        mid = mention_ids[c]
        out[slot] = (mid, expected_type_by_slot[slot])
    return out


def decode_validate_repair(
    m3_assignment: dict[str, tuple[str, str]],
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
    score_lookup: dict[tuple[str, str], float],
) -> dict[str, tuple[str, str]]:
    """M4: M3 + reuse the canonical single-assignment plausibility check
    (`_opt_role_validate_one`) to drop implausible fills and reassign from
    the next-best remaining candidate for that slot."""
    mention_by_id = {m.mention_id: m for m in mentions}
    slot_by_name = {s.name: s for s in slots}
    filled = dict(m3_assignment)
    used = {mid for mid, _ in filled.values()}

    for slot_name in list(filled.keys()):
        mid, _ = filled[slot_name]
        m = mention_by_id.get(mid)
        s = slot_by_name.get(slot_name)
        if m is None or s is None:
            continue
        score = score_lookup.get((slot_name, mid), 0.0)
        valid, _ = validate_assignment(slot_name, m, s, score)
        if not valid:
            del filled[slot_name]
            used.discard(mid)

    unfilled = [s for s in slots if s.name not in filled]
    for s in unfilled:
        candidates = sorted(
            ((mid, score_lookup.get((s.name, mid), 0.0)) for mid in mention_by_id if mid not in used),
            key=lambda x: -x[1],
        )
        for mid, sc in candidates:
            m = mention_by_id[mid]
            valid, _ = validate_assignment(s.name, m, s, sc)
            if valid:
                filled[s.name] = (mid, s.expected_type)
                used.add(mid)
                break
    return filled


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = load_split("test")
    model = joblib.load(MODEL_DIR / "p0_model.joblib")
    with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    X_test = [row_vector(r) for r in test_rows]
    scores = [float(p[1]) for p in model.predict_proba(X_test)] if X_test else []

    # Regroup by instance and slot for decoding; also recompute canonical
    # MentionOptIR/SlotOptIR per instance for the validate/repair stage,
    # since decode_validate_repair needs the actual IR objects, not just rows.
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for row, sc in zip(test_rows, scores):
        row = dict(row)
        row["p0_score"] = sc
        by_instance[row["instance_id"]].append(row)

    # Determine the full set of 50 test instance_ids (including any skipped
    # for zero extracted mentions), so all methods share the same denominator.
    test_ids_all = set()
    with open(ROOT / "artifacts" / "learning_corpus" / "nlp4lp_test.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_ids_all.add(json.loads(line)["instance_id"])
    eval_items = {}
    with open(EVAL_ORIG_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                if it["query_id"] in test_ids_all:
                    eval_items[it["query_id"]] = it

    n_expected_by_instance: dict[str, int] = {}
    for iid, rows in by_instance.items():
        n_expected_by_instance[iid] = rows[0]["n_expected_scalar"]
    # For skipped instances (0 mentions extracted, absent from by_instance),
    # n_expected_scalar must be filled in from the gold cache directly.
    import os

    os.environ.setdefault("NLP4LP_GOLD_CACHE", str(ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"))
    with open(os.environ["NLP4LP_GOLD_CACHE"], encoding="utf-8") as f:
        gold_by_id = json.load(f)["gold_by_id"]

    def _is_scalar(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    for iid in test_ids_all:
        if iid not in n_expected_by_instance:
            doc_id = eval_items[iid]["relevant_doc_id"]
            gold_entry = gold_by_id.get(doc_id) or {}
            params = gold_entry.get("parameters") or {}
            pinfo = gold_entry.get("problem_info") or {}
            if isinstance(pinfo, str):
                try:
                    pinfo = json.loads(pinfo)
                except Exception:
                    pinfo = {}
            expected_params = list((pinfo.get("parameters") or params or {}).keys())
            scalar_params = [p for p in expected_params if _is_scalar(params.get(p))]
            n_expected_by_instance[iid] = len(scalar_params)

    filled_m2: dict[str, dict[str, tuple[str, bool]]] = {}
    filled_m3: dict[str, dict[str, tuple[str, bool]]] = {}
    filled_m4: dict[str, dict[str, tuple[str, bool]]] = {}

    for iid, rows in by_instance.items():
        query = eval_items[iid]["query"]
        scalar_params = sorted({r["slot_name"] for r in rows})
        mentions = _extract_opt_role_mentions(query, variant="orig")
        slots = _build_slot_opt_irs(scalar_params)
        expected_type_by_slot = {s.name: s.expected_type for s in slots}

        score_lookup: dict[tuple[str, str], float] = {}
        by_group: dict[str, list[tuple[str, float, str, str]]] = defaultdict(list)
        for r in rows:
            mid = str(r["mention_id"])
            key = (r["slot_name"], mid)
            score_lookup[key] = r["p0_score"]
            by_group[r["group_id"]].append((mid, r["p0_score"], r["mention_type_bucket"], r["expected_type"]))

        # M2: greedy per-slot argmax
        m2_raw = decode_greedy(by_group)
        filled_m2[iid] = {}
        for gid, (mid, exp_type) in m2_raw.items():
            slot_name = gid.split("::", 1)[1]
            type_bucket = next(tb for (m, s, tb, et) in by_group[gid] if m == mid)
            filled_m2[iid][slot_name] = (mid, _is_type_match(exp_type, type_bucket))

        # M3: exact max-weight bipartite matching (Hungarian, via scipy)
        mention_ids_sorted = sorted({str(m.mention_id) for m in mentions})
        slot_names_sorted = [s.name for s in slots]
        mat = np.zeros((len(slot_names_sorted), len(mention_ids_sorted)))
        for si, sname in enumerate(slot_names_sorted):
            for mi_, mid in enumerate(mention_ids_sorted):
                mat[si, mi_] = score_lookup.get((sname, mid), -1e6)
        m3_raw = decode_max_weight(slot_names_sorted, mention_ids_sorted, mat, expected_type_by_slot)
        filled_m3[iid] = {}
        for slot_name, (mid, exp_type) in m3_raw.items():
            if score_lookup.get((slot_name, mid), -1e6) <= -1e5:
                continue  # no real candidate scored for this slot (padding assignment)
            mention_obj = next((m for m in mentions if str(m.mention_id) == mid), None)
            type_bucket = mention_obj.type_bucket if mention_obj else "unknown"
            filled_m3[iid][slot_name] = (mid, _is_type_match(exp_type, type_bucket))

        # M4: M3 + validate/repair
        m3_for_repair = {s: (mid, "") for s, (mid, tb) in filled_m3[iid].items()}
        m4_raw = decode_validate_repair(m3_for_repair, mentions, slots, score_lookup)
        filled_m4[iid] = {}
        for slot_name, (mid, _unused) in m4_raw.items():
            mention_obj = next((m for m in mentions if str(m.mention_id) == mid), None)
            type_bucket = mention_obj.type_bucket if mention_obj else "unknown"
            filled_m4[iid][slot_name] = (mid, _is_type_match(expected_type_by_slot[slot_name], type_bucket))

    instance_ids_all = sorted(test_ids_all)
    results = {}
    for name, filled in [("M2_p0_greedy", filled_m2), ("M3_p0_max_weight", filled_m3), ("M4_p0_max_weight_repair", filled_m4)]:
        per_query = per_instance_metrics(instance_ids_all, n_expected_by_instance, filled)
        agg = aggregate(per_query)
        results[name] = {"aggregate": agg, "per_query": per_query}
        print(f"{name}: n={agg['n']} coverage={agg['param_coverage']:.4f} type_match={agg['type_match']:.4f} inst_ready={agg['instantiation_ready']:.4f}")

    with open(OUT_DIR / "test_results.csv", "w", encoding="utf-8") as f:
        f.write("method,n,param_coverage,type_match,instantiation_ready\n")
        for name, r in results.items():
            a = r["aggregate"]
            f.write(f"{name},{a['n']},{a['param_coverage']:.6f},{a['type_match']:.6f},{a['instantiation_ready']:.6f}\n")

    with open(OUT_DIR / "per_query_test.csv", "w", encoding="utf-8") as f:
        f.write("method,instance_id,n_expected_scalar,n_filled,param_coverage,type_match,instantiation_ready\n")
        for name, r in results.items():
            for row in r["per_query"]:
                f.write(
                    f"{name},{row['instance_id']},{row['n_expected_scalar']},{row['n_filled']},"
                    f"{row['param_coverage']:.6f},{row['type_match']:.6f},{row['instantiation_ready']}\n"
                )

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"p0_model_config": config, "methods": list(results.keys()), "test_n": len(instance_ids_all)}, f, indent=2)

    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
