#!/usr/bin/env python3
"""Score both systems on the 30-case benchmark.

Reads:
    artifacts/copilot_vs_model/benchmark_cases.jsonl
    artifacts/copilot_vs_model/our_model_outputs.jsonl
    artifacts/copilot_vs_model/copilot_outputs.jsonl

Writes:
    artifacts/copilot_vs_model/comparison_summary.csv

Usage:
    python artifacts/copilot_vs_model/score_comparison.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH_FILE  = ROOT / "benchmark_cases.jsonl"
OUR_FILE    = ROOT / "our_model_outputs.jsonl"
COPILOT_FILE= ROOT / "copilot_outputs.jsonl"
OUT_CSV     = ROOT / "comparison_summary.csv"

# Weights from evaluation_rubric.md
W_SCHEMA   = 0.30
W_COVERAGE = 0.35
W_TYPE     = 0.20
W_OBJ      = 0.10
W_HALLUC   = 0.05

PENDING_SENTINEL = "PENDING"


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_numbers_in_text(text: str) -> set[float]:
    """All numeric values appearing in the input text."""
    raw = re.findall(r'\$([\d,]+(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*%|([\d,]+(?:\.\d+)?)', text)
    out: set[float] = set()
    for g1, g2, g3 in raw:
        v = g1 or g2 or g3
        v = v.replace(',', '')
        try:
            out.add(float(v))
        except ValueError:
            pass
    return out


def _infer_gold_objective(text: str) -> str:
    q = text.lower()
    if 'maximize' in q or 'maximum' in q or 'max profit' in q or 'max revenue' in q:
        return 'maximize'
    if 'minimize' in q or 'minimum' in q or 'min cost' in q:
        return 'minimize'
    return 'unknown'


def _schema_correct_our(out: dict, gold_schema_id: str) -> float:
    if out.get('schema_correct') == 1:
        return 1.0
    top3 = [pid for pid, _ in (out.get('retrieval_top3') or [])]
    if gold_schema_id in top3:
        return 0.5
    return 0.0


def _schema_correct_copilot(parsed: dict | None, gold_schema_text: str) -> float:
    if parsed is None:
        return 0.0
    pt = (parsed.get('predicted_problem_type') or '').lower().strip()
    gt = gold_schema_text.lower()
    if not pt:
        return 0.0
    # Check for family keywords
    keywords = set(pt.replace('_', ' ').split())
    hits = sum(1 for kw in keywords if kw in gt and len(kw) > 3)
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.5
    # fallback: any overlap
    for kw in ['lp', 'linear', 'transport', 'knapsack', 'diet', 'invest',
               'product', 'mix', 'schedule', 'assignment']:
        if kw in pt:
            return 0.5
    return 0.0


def _grounding_coverage(assigned: dict, gold_slots: list[str]) -> float:
    if not gold_slots:
        return 0.0
    filled = {k.lower() for k, v in assigned.items() if v is not None}
    gold   = {s.lower() for s in gold_slots}
    return len(filled & gold) / len(gold)


def _type_correctness(assigned: dict) -> float:
    """Fraction of assignments that pass a basic type check."""
    if not assigned:
        return 0.0
    ok = 0
    for slot, val in assigned.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        slot_lo = slot.lower()
        is_pct_slot = any(kw in slot_lo for kw in ('percent', 'fraction', 'rate', 'ratio'))
        if is_pct_slot:
            ok += 1 if 0 <= v <= 1.0 else 0
        else:
            ok += 1 if v >= 0 else 0
    return ok / len(assigned)


def _objective_dir_score(predicted: str, gold_text: str) -> float:
    gold = _infer_gold_objective(gold_text)
    pred = (predicted or '').lower().strip()
    if gold == 'unknown':
        return 1.0  # can't penalise
    if pred == gold:
        return 1.0
    if pred == 'unknown':
        return 0.5
    return 0.0


def _hallucination_score(assigned: dict, input_text: str) -> float:
    """Penalty: any assigned value not present in the input text."""
    numbers_in_text = _extract_numbers_in_text(input_text)
    penalty = 0
    for val in assigned.values():
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        # Check if value or percentage-converted value appears in text
        if v not in numbers_in_text and (v * 100) not in numbers_in_text and (v / 100) not in numbers_in_text:
            penalty += 1
    return max(0.0, 1.0 - 0.1 * penalty)


def _value_exact_match(assigned: dict, gold_params: dict | None) -> float | None:
    """Exact match rate against gold param values (hand-crafted cases only)."""
    if not gold_params:
        return None
    correct = 0
    total   = 0
    for slot, gold_val in gold_params.items():
        total += 1
        pred_val = assigned.get(slot)
        if pred_val is None:
            # try case-insensitive key lookup
            for k, v in assigned.items():
                if k.lower() == slot.lower():
                    pred_val = v
                    break
        if pred_val is None:
            continue
        try:
            pv = float(pred_val)
            gv = float(gold_val)
            denom = max(1.0, abs(gv))
            if abs(pv - gv) / denom <= 0.01:
                correct += 1
        except (TypeError, ValueError):
            pass
    return correct / total if total else None


def _overall(schema: float, coverage: float, type_c: float,
             obj_dir: float, halluc: float) -> float:
    return (W_SCHEMA * schema + W_COVERAGE * coverage +
            W_TYPE * type_c + W_OBJ * obj_dir + W_HALLUC * halluc)


def _winner(our: float, cop: float) -> str:
    if our > cop + 0.05:
        return 'our_model'
    if cop > our + 0.05:
        return 'copilot'
    return 'tie'


# ── main ─────────────────────────────────────────────────────────────────────

def score() -> None:
    cases   = {json.loads(l)['case_id']: json.loads(l) for l in BENCH_FILE.open()}
    our_outs= {json.loads(l)['case_id']: json.loads(l) for l in OUR_FILE.open()}
    cop_outs: dict[str, dict] = {}
    if COPILOT_FILE.exists():
        for l in COPILOT_FILE.open():
            o = json.loads(l)
            cop_outs[o['case_id']] = o

    rows = []
    for case_id, case in cases.items():
        gold_sid    = case.get('gold_schema_id', '')
        gold_text   = case.get('gold_schema_text', '')
        gold_slots  = case.get('gold_scalar_slots', [])
        gold_params = case.get('gold_param_values') or {}
        input_text  = case.get('input_text', '')
        cats        = ','.join(case.get('difficulty_tags', []))
        gold_obj    = _infer_gold_objective(input_text)
        has_gold_vals = bool(gold_params)

        # ── Our model ──
        our = our_outs.get(case_id, {})
        our_assigned = our.get('slot_value_assignments', {}) or {}
        our_schema   = _schema_correct_our(our, gold_sid)
        our_coverage = _grounding_coverage(our_assigned, gold_slots)
        our_type     = _type_correctness(our_assigned)
        our_obj      = _objective_dir_score(our.get('predicted_objective_direction', ''), input_text)
        our_halluc   = _hallucination_score(our_assigned, input_text)
        our_overall  = _overall(our_schema, our_coverage, our_type, our_obj, our_halluc)
        our_exact    = _value_exact_match(our_assigned, gold_params if has_gold_vals else None)

        # ── Copilot ──
        cop_raw = cop_outs.get(case_id, {})
        cop_parsed = cop_raw.get('parsed') or {}
        cop_pending = cop_raw.get('parse_error') == PENDING_SENTINEL or cop_raw.get('raw_response') == f"PENDING — paste Copilot output here"
        cop_assigned = (cop_parsed.get('slot_value_assignments') or {}) if cop_parsed else {}
        cop_schema   = _schema_correct_copilot(cop_parsed if cop_parsed else None, gold_text)
        cop_coverage = _grounding_coverage(cop_assigned, gold_slots)
        cop_type     = _type_correctness(cop_assigned)
        cop_obj      = _objective_dir_score((cop_parsed or {}).get('objective_direction', ''), input_text)
        cop_halluc   = _hallucination_score(cop_assigned, input_text)
        cop_overall  = _overall(cop_schema, cop_coverage, cop_type, cop_obj, cop_halluc)
        cop_exact    = _value_exact_match(cop_assigned, gold_params if has_gold_vals else None)

        winner = _winner(our_overall, cop_overall) if not cop_pending else 'pending'

        row = {
            'case_id': case_id,
            'category': cats,
            'has_gold_values': int(has_gold_vals),
            'copilot_pending': int(cop_pending),
            # our model
            'our_model_schema_correct': round(our_schema, 3),
            'our_model_grounding_coverage': round(our_coverage, 3),
            'our_model_type_correct': round(our_type, 3),
            'our_model_objective_dir': round(our_obj, 3),
            'our_model_no_hallucination': round(our_halluc, 3),
            'our_model_overall': round(our_overall, 3),
            'our_model_value_exact': round(our_exact, 3) if our_exact is not None else '',
            # copilot
            'copilot_schema_correct': round(cop_schema, 3),
            'copilot_grounding_coverage': round(cop_coverage, 3),
            'copilot_type_correct': round(cop_type, 3),
            'copilot_objective_dir': round(cop_obj, 3),
            'copilot_no_hallucination': round(cop_halluc, 3),
            'copilot_overall': round(cop_overall, 3),
            'copilot_value_exact': round(cop_exact, 3) if cop_exact is not None else '',
            'winner': winner,
            'notes': 'copilot_manual_pending' if cop_pending else '',
        }
        rows.append(row)

    # Write CSV
    fieldnames = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Print summary
    n = len(rows)
    n_pending = sum(r['copilot_pending'] for r in rows)
    n_scored  = n - n_pending

    our_avg   = sum(r['our_model_overall'] for r in rows) / n
    cop_avg   = sum(r['copilot_overall'] for r in rows) / n

    wins_our  = sum(1 for r in rows if r['winner'] == 'our_model')
    wins_cop  = sum(1 for r in rows if r['winner'] == 'copilot')
    ties      = sum(1 for r in rows if r['winner'] == 'tie')
    pending   = sum(1 for r in rows if r['winner'] == 'pending')

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total cases:       {n}")
    print(f"Copilot pending:   {n_pending} (fill in copilot_outputs.jsonl)")
    print(f"Scored cases:      {n_scored}")
    print()
    print(f"{'Metric':<35} {'Our Model':>12} {'Copilot':>12}")
    print("-" * 60)
    for metric, col_our, col_cop in [
        ('Schema correctness',           'our_model_schema_correct',     'copilot_schema_correct'),
        ('Grounding coverage',           'our_model_grounding_coverage', 'copilot_grounding_coverage'),
        ('Type correctness',             'our_model_type_correct',        'copilot_type_correct'),
        ('Objective direction',          'our_model_objective_dir',       'copilot_objective_dir'),
        ('No hallucination',             'our_model_no_hallucination',    'copilot_no_hallucination'),
        ('OVERALL (weighted)',           'our_model_overall',             'copilot_overall'),
    ]:
        ours = sum(r[col_our] for r in rows) / n
        cops = sum(r[col_cop] for r in rows) / n
        print(f"  {metric:<33} {ours:>12.3f} {cops:>12.3f}")

    print()
    # Value exact match on hand-crafted only
    hc_rows = [r for r in rows if r['has_gold_values']]
    if hc_rows:
        hc_our = [r['our_model_value_exact'] for r in hc_rows if r['our_model_value_exact'] != '']
        hc_cop = [r['copilot_value_exact']   for r in hc_rows if r['copilot_value_exact']   != '']
        if hc_our:
            print(f"  {'Value exact match (handcrafted)':<33} {sum(hc_our)/len(hc_our):>12.3f}", end='')
        if hc_cop:
            print(f" {sum(hc_cop)/len(hc_cop):>12.3f}", end='')
        print()

    print()
    print(f"Win/loss/tie counts (scored cases):")
    print(f"  Our model wins: {wins_our}")
    print(f"  Copilot wins:   {wins_cop}")
    print(f"  Ties:           {ties}")
    print(f"  Pending:        {pending}")
    print()

    # Category breakdown
    all_cats = set()
    for r in rows:
        for c in r['category'].split(','):
            all_cats.add(c.strip())
    print(f"{'Category':<22} {'n':>4} {'Our':>8} {'Copilot':>8}")
    print("-" * 44)
    for cat in sorted(all_cats):
        cat_rows = [r for r in rows if cat in r['category']]
        if not cat_rows:
            continue
        o = sum(r['our_model_overall'] for r in cat_rows) / len(cat_rows)
        c = sum(r['copilot_overall']   for r in cat_rows) / len(cat_rows)
        print(f"  {cat:<20} {len(cat_rows):>4} {o:>8.3f} {c:>8.3f}")

    print()
    if n_pending > 0:
        print(f"NOTE: {n_pending} Copilot responses are still PENDING.")
        print("Fill them in via: python artifacts/copilot_vs_model/ingest_copilot_response.py --case-id <id>")
        print("Then re-run this scorer.")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    score()
