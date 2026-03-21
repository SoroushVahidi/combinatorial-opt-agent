"""
check_nlp4lp_pairwise_data_quality.py — Data quality checker for NLP4LP pairwise ranker data.

Inspects JSONL files under artifacts/learning_ranker_data/nlp4lp/ and reports:
  - number of instances / groups / positives / negatives
  - groups with no positive candidate
  - groups with multiple positive candidates
  - duplicate rows
  - missing slot_name / mention_surface / group_id
  - feature sparsity / constant features
  - label balance
  - average candidates per slot
  - examples of suspicious rows

Writes:
  artifacts/learning_audit/pairwise_data_quality.json
  artifacts/learning_audit/pairwise_data_quality.md

Usage:
    python src/learning/check_nlp4lp_pairwise_data_quality.py [options]

Options:
    --ranker-data DIR  Path to ranker data dir (default: artifacts/learning_ranker_data/nlp4lp)
    --out DIR          Output dir (default: artifacts/learning_audit)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Features expected in pairwise rows (if present)
EXPECTED_NUMERIC_FEATURES = [
    "type_match",
    "operator_cue_match",
    "lower_cue_present",
    "upper_cue_present",
    "slot_mention_overlap",
    "entity_match",
    "sentence_proximity",
]


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def _check_split(split: str, rows: list[dict]) -> dict:
    """Run quality checks on a single split's rows."""
    n = len(rows)
    if n == 0:
        return {"split": split, "n_rows": 0, "empty": True}

    # Label stats
    labels = [int(r.get("label", -1)) for r in rows]
    label_counts = Counter(labels)
    n_pos = label_counts.get(1, 0)
    n_neg = label_counts.get(0, 0)
    n_missing_label = label_counts.get(-1, 0)

    # Group analysis
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        gid = r.get("group_id") or r.get("query_id") or r.get("slot_name") or "__no_group__"
        groups[gid].append(r)

    n_groups = len(groups)
    no_positive_groups = []
    multi_positive_groups = []
    for gid, grp in groups.items():
        pos = sum(1 for r in grp if int(r.get("label", -1)) == 1)
        if pos == 0:
            no_positive_groups.append(gid)
        elif pos > 1:
            multi_positive_groups.append(gid)

    avg_candidates = n / max(n_groups, 1)

    # Missing fields
    required_fields = ["slot_name", "mention_surface", "group_id"]
    missing: dict[str, int] = {}
    for field in required_fields:
        missing[field] = sum(1 for r in rows if not r.get(field))

    # Duplicate rows (by slot_name + mention_surface + label)
    seen: set[tuple] = set()
    n_duplicates = 0
    for r in rows:
        key = (
            r.get("slot_name", ""),
            r.get("mention_surface", ""),
            r.get("label", ""),
            r.get("group_id", ""),
        )
        if key in seen:
            n_duplicates += 1
        seen.add(key)

    # Feature sparsity / constant features
    feature_stats: dict[str, dict] = {}
    for feat in EXPECTED_NUMERIC_FEATURES:
        feature_rows = [r for r in rows if feat in r]
        if not feature_rows:
            feature_stats[feat] = {"present": False}
            continue

        non_null_values = [r[feat] for r in feature_rows if r[feat] is not None]
        pos_values = [r[feat] for r in feature_rows if int(r.get("label", -1)) == 1 and r[feat] is not None]
        neg_values = [r[feat] for r in feature_rows if int(r.get("label", -1)) == 0 and r[feat] is not None]
        mean_pos = sum(pos_values) / max(len(pos_values), 1)
        mean_neg = sum(neg_values) / max(len(neg_values), 1)
        unique_vals = set(non_null_values)
        feature_stats[feat] = {
            "present": True,
            "coverage": round(len(non_null_values) / max(n, 1), 4),
            "is_constant": len(unique_vals) <= 1,
            "mean_on_positive": round(mean_pos, 4),
            "mean_on_negative": round(mean_neg, 4),
        }

    # Suspicious rows: label != 0 and != 1, or slot_name is numeric, or mention_surface is slot_name
    suspicious: list[dict] = []
    for r in rows:
        label = r.get("label")
        slot = r.get("slot_name", "")
        mention = r.get("mention_surface", "")
        if label not in (0, 1):
            suspicious.append({**r, "_issue": f"invalid label: {label}"})
        elif slot and slot.isdigit():
            suspicious.append({**r, "_issue": "slot_name is purely numeric"})
        elif slot and mention and slot.strip().lower() == mention.strip().lower():
            suspicious.append({**r, "_issue": "slot_name == mention_surface"})
        if len(suspicious) >= 20:
            break

    return {
        "split": split,
        "n_rows": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_missing_label": n_missing_label,
        "label_balance": round(n_pos / max(n_pos + n_neg, 1), 4),
        "n_groups": n_groups,
        "avg_candidates_per_group": round(avg_candidates, 2),
        "no_positive_groups": len(no_positive_groups),
        "no_positive_group_examples": no_positive_groups[:5],
        "multi_positive_groups": len(multi_positive_groups),
        "multi_positive_group_examples": multi_positive_groups[:5],
        "n_duplicates": n_duplicates,
        "missing_fields": missing,
        "feature_stats": feature_stats,
        "n_suspicious_rows": len(suspicious),
        "suspicious_row_examples": suspicious[:5],
    }


def run_quality_check(ranker_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ranker_dir.exists():
        print(f"WARNING: Ranker data directory not found: {ranker_dir}")
        print("  No pairwise data to check. Writing empty report.")
        report = {
            "ranker_data_dir": str(ranker_dir),
            "data_available": False,
            "message": "Ranker data directory not found. Run data preparation first.",
            "splits": {},
        }
    else:
        splits_found = []
        split_results: dict[str, dict] = {}
        for split in ("train", "dev", "test"):
            p = ranker_dir / f"{split}.jsonl"
            if p.exists():
                rows = _load_jsonl(p)
                split_results[split] = _check_split(split, rows)
                splits_found.append(split)
                print(f"  Checked split={split}: {len(rows)} rows")
            else:
                print(f"  Split not found: {p}")

        if not splits_found:
            print(f"WARNING: No JSONL splits found in {ranker_dir}")

        report = {
            "ranker_data_dir": str(ranker_dir),
            "data_available": bool(splits_found),
            "splits_found": splits_found,
            "splits": split_results,
        }

    # Write JSON
    json_path = out_dir / "pairwise_data_quality.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Written → {json_path}")

    # Write Markdown
    _write_md(report, out_dir)


def _write_md(report: dict, out_dir: Path) -> None:
    lines = [
        "# NLP4LP Pairwise Data Quality Report",
        "",
        f"**Ranker data dir**: `{report['ranker_data_dir']}`",
        f"**Data available**: {report['data_available']}",
        "",
    ]

    if not report.get("data_available"):
        lines.append(f"> {report.get('message', 'No data available.')}")
    else:
        for split, result in report.get("splits", {}).items():
            if result.get("empty"):
                lines.append(f"## Split: {split}  (empty)")
                continue
            lines += [
                f"## Split: {split}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Rows | {result['n_rows']} |",
                f"| Positives | {result['n_positive']} |",
                f"| Negatives | {result['n_negative']} |",
                f"| Label balance (pos rate) | {result['label_balance']:.1%} |",
                f"| Groups | {result['n_groups']} |",
                f"| Avg candidates/group | {result['avg_candidates_per_group']:.1f} |",
                f"| Groups with no positive | {result['no_positive_groups']} |",
                f"| Groups with multi positive | {result['multi_positive_groups']} |",
                f"| Duplicate rows | {result['n_duplicates']} |",
                f"| Missing label | {result['n_missing_label']} |",
                "",
            ]
            # Missing fields
            mf = result.get("missing_fields", {})
            if any(v > 0 for v in mf.values()):
                lines += ["**Missing fields:**", ""]
                for field, cnt in mf.items():
                    if cnt > 0:
                        lines.append(f"- `{field}`: {cnt} missing")
                lines.append("")

            # Feature stats
            fs = result.get("feature_stats", {})
            present_feats = {k: v for k, v in fs.items() if v.get("present")}
            if present_feats:
                lines += [
                    "**Feature statistics:**",
                    "",
                    "| Feature | Coverage | Constant? | Mean (pos) | Mean (neg) |",
                    "|---------|----------|-----------|------------|------------|",
                ]
                for feat, info in present_feats.items():
                    lines.append(
                        f"| {feat} | {info['coverage']:.1%} | {info['is_constant']} "
                        f"| {info['mean_on_positive']:.3f} | {info['mean_on_negative']:.3f} |"
                    )
                lines.append("")

            # Suspicious examples
            sus = result.get("suspicious_row_examples", [])
            if sus:
                lines += ["**Suspicious row examples:**", ""]
                for row in sus[:3]:
                    issue = row.get("_issue", "?")
                    slot = row.get("slot_name", "?")
                    mention = row.get("mention_surface", "?")
                    label = row.get("label", "?")
                    lines.append(
                        f"- issue=`{issue}` | slot=`{slot}` | mention=`{mention}` | label={label}"
                    )
                lines.append("")

    md_path = out_dir / "pairwise_data_quality.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written → {md_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Data quality check for NLP4LP pairwise ranker data."
    )
    p.add_argument(
        "--ranker-data",
        type=Path,
        default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp",
        help="Ranker data directory (default: artifacts/learning_ranker_data/nlp4lp)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "learning_audit",
        help="Output directory (default: artifacts/learning_audit)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    run_quality_check(ranker_dir=args.ranker_data, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
