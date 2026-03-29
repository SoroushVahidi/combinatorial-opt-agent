#!/usr/bin/env python3
"""Summarize common learning corpus and write corpus_summary.md + corpus_summary.json. Optional spot-checks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.common_corpus_schema import validate_record


def _load_corpus(corpus_dir: Path) -> list[tuple[str, dict]]:
    """Yield (filename, record) for all JSONL in corpus_dir."""
    out = []
    for path in sorted(corpus_dir.glob("*.jsonl")):
        if path.name in ("corpus_summary.md", "corpus_summary.json") or path.name.startswith("."):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out.append((path.name, rec))
                except Exception:
                    pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", type=Path, default=ROOT / "artifacts" / "learning_corpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spot_check_n", type=int, default=2, help="N random examples per dataset to print")
    ap.add_argument("--multi_float_n", type=int, default=2, help="N examples with 3+ float-like mentions")
    ap.add_argument("--bound_cues_n", type=int, default=2, help="N examples with min/max cues")
    ap.add_argument("--multi_entity_n", type=int, default=2, help="N examples with multiple entities/vars")
    args = ap.parse_args()
    corpus_dir = args.corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records = _load_corpus(corpus_dir)
    rng = random.Random(args.seed)

    # Per-file and per-dataset/split counts
    by_file: dict[str, list[dict]] = {}
    for fname, rec in records:
        by_file.setdefault(fname, []).append(rec)
    by_dataset_split: dict[str, dict[str, int]] = {}
    has_slot_supervision = 0
    has_entity_labels = 0
    has_bound_labels = 0
    has_role_labels = 0
    missing_schema_name = 0
    missing_slots = 0
    missing_mentions = 0
    for _fname, rec in records:
        ds = rec.get("dataset", "?")
        sp = rec.get("split", "?")
        by_dataset_split.setdefault(ds, {})
        by_dataset_split[ds][sp] = by_dataset_split[ds].get(sp, 0) + 1
        if rec.get("gold_slot_assignments") and rec.get("slots"):
            has_slot_supervision += 1
        if rec.get("entity_labels"):
            has_entity_labels += 1
        if rec.get("bound_labels"):
            has_bound_labels += 1
        if rec.get("role_labels"):
            has_role_labels += 1
        if not rec.get("schema_name"):
            missing_schema_name += 1
        if not rec.get("slots"):
            missing_slots += 1
        if not rec.get("numeric_mentions"):
            missing_mentions += 1

    summary = {
        "total_records": len(records),
        "by_file": {f: len(recs) for f, recs in by_file.items()},
        "by_dataset_split": by_dataset_split,
        "with_slot_supervision": has_slot_supervision,
        "with_entity_labels": has_entity_labels,
        "with_bound_labels": has_bound_labels,
        "with_role_labels": has_role_labels,
        "missing_schema_name": missing_schema_name,
        "missing_slots": missing_slots,
        "missing_numeric_mentions": missing_mentions,
    }
    out_json = corpus_dir / "corpus_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    # Markdown report
    md_lines = [
        "# Learning corpus summary",
        "",
        f"**Total records:** {len(records)}",
        "",
        "## By file",
        "| File | Count |",
        "|------|-------|",
    ]
    for f in sorted(summary["by_file"].keys()):
        md_lines.append(f"| {f} | {summary['by_file'][f]} |")
    md_lines.extend([
        "",
        "## By dataset / split",
        "| Dataset | train | dev | test |",
        "|---------|-------|-----|------|",
    ])
    for ds in sorted(summary["by_dataset_split"].keys()):
        row = summary["by_dataset_split"][ds]
        md_lines.append(f"| {ds} | {row.get('train', 0)} | {row.get('dev', 0)} | {row.get('test', 0)} |")
    md_lines.extend([
        "",
        "## Supervision",
        f"- With usable slot supervision (slots + gold_slot_assignments): **{has_slot_supervision}**",
        f"- With entity_labels: **{has_entity_labels}**",
        f"- With bound_labels: **{has_bound_labels}**",
        f"- With role_labels: **{has_role_labels}**",
        "",
        "## Nullable / missing",
        f"- Missing schema_name: **{missing_schema_name}**",
        f"- Missing slots (empty list): **{missing_slots}**",
        f"- Missing numeric_mentions: **{missing_mentions}**",
        "",
    ])
    out_md = corpus_dir / "corpus_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {out_md}")

    # Spot-checks
    by_ds: dict[str, list[tuple[str, dict]]] = {}
    for fname, rec in records:
        ds = rec.get("dataset", "?")
        by_ds.setdefault(ds, []).append((fname, rec))
    print("\n--- Spot check: random examples per dataset ---")
    for ds in sorted(by_ds.keys()):
        pool = by_ds[ds]
        n = min(args.spot_check_n, len(pool))
        for fname, rec in rng.sample(pool, n):
            print(f"\n[{ds}] {rec.get('instance_id')} (from {fname})")
            print(f"  problem_text: { (rec.get('problem_text') or '')[:200]}...")
            print(f"  slots: {len(rec.get('slots') or [])}, mentions: {len(rec.get('numeric_mentions') or [])}")

    multi_float = [r for _, r in records if len(r.get("numeric_mentions") or []) >= 3]
    if multi_float:
        print("\n--- Examples with 3+ numeric mentions ---")
        for rec in rng.sample(multi_float, min(args.multi_float_n, len(multi_float))):
            print(f"\n  {rec.get('dataset')} {rec.get('instance_id')}: {len(rec.get('numeric_mentions', []))} mentions")

    bound_cues = []
    for _, rec in records:
        mentions = rec.get("numeric_mentions") or []
        if any(m.get("operator_cues") for m in mentions):
            bound_cues.append(rec)
    if bound_cues:
        print("\n--- Examples with operator (min/max) cues ---")
        for rec in rng.sample(bound_cues, min(args.bound_cues_n, len(bound_cues))):
            print(f"\n  {rec.get('dataset')} {rec.get('instance_id')}")
            for m in (rec.get("numeric_mentions") or [])[:5]:
                if m.get("operator_cues"):
                    print(f"    mention {m.get('mention_id')}: {m.get('surface')} cues={m.get('operator_cues')}")

    multi_entity = [r for _, r in records if len(r.get("slots") or []) >= 2 and any(s.get("variable_entity") for s in (r.get("slots") or []))]
    if multi_entity:
        print("\n--- Examples with multiple entities/variables ---")
        for rec in rng.sample(multi_entity, min(args.multi_entity_n, len(multi_entity))):
            print(f"\n  {rec.get('dataset')} {rec.get('instance_id')}: {len(rec.get('slots', []))} slots")
            for s in (rec.get("slots") or [])[:6]:
                if s.get("variable_entity"):
                    print(f"    slot {s.get('slot_id')} -> {s.get('variable_entity')}")


if __name__ == "__main__":
    main()
