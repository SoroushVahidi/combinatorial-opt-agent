"""
audit_nlp4lp_bottlenecks.py — CPU-only bottleneck audit for NLP4LP grounding.

Reads corpus data (nlp4lp_eval_orig.jsonl) and optionally pairwise ranker data
(artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl) and applies
transparent heuristics to flag examples that are likely hard for downstream
number-to-slot grounding.

Bottleneck slices identified:
  1. entity_association_risk  — query mentions entities/names alongside numbers,
                                raising wrong variable/entity association risk.
  2. lower_upper_risk         — query contains cues for both lower and upper bounds
                                (at most / at least / minimum / maximum together).
  3. multi_numeric_confusion  — query contains >=3 distinct numeric values,
                                raising confusable multi-float grounding risk.
  4. total_vs_per_unit_risk   — query uses "total" and "per" / "each" language
                                alongside numbers, risking total-vs-per-unit swap.
  5. percent_vs_absolute_risk — query mixes percentage cues and absolute numeric
                                mentions, risking % vs absolute confusion.

Each flagged example is written to a per-slice JSONL and included in a
consolidated summary JSON + Markdown.

Usage:
    python src/learning/audit_nlp4lp_bottlenecks.py [options]

Options:
    --eval PATH       Path to eval JSONL (default: data/processed/nlp4lp_eval_orig.jsonl)
    --ranker-data DIR Path to pairwise ranker data dir (default: artifacts/learning_ranker_data/nlp4lp)
    --out DIR         Output directory (default: artifacts/learning_audit)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Numbers: integers, floats, percentages, currency amounts
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

# Entity cues: proper names, named persons, companies.
# Require title prefix OR two or more consecutive capitalized words (named entity),
# to avoid matching ordinary sentence-initial capitals like "Each", "Both", "The".
_ENTITY_CUES = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s*[A-Z][a-z]+"
    r"|\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b",
    re.UNICODE,
)

# Lower-bound cues
_LOWER_CUES = re.compile(
    r"\b(?:at least|minimum|min|no less than|greater than or equal|>=|≥|must be at least)\b",
    re.IGNORECASE,
)

# Upper-bound cues
_UPPER_CUES = re.compile(
    r"\b(?:at most|maximum|max|no more than|less than or equal|<=|≤|must not exceed|cannot exceed|up to)\b",
    re.IGNORECASE,
)

# Total / aggregate cues
_TOTAL_CUES = re.compile(r"\b(?:total|overall|aggregate|combined|sum)\b", re.IGNORECASE)

# Per-unit cues
_PER_UNIT_CUES = re.compile(r"\b(?:per|each|every|unit|piece|item)\b", re.IGNORECASE)

# Percentage cues
_PERCENT_CUES = re.compile(r"\b\d+(?:\.\d+)?%|\bpercent(?:age)?\b", re.IGNORECASE)

# Absolute value cues (dollar amounts or plain large integers not followed by %)
_ABSOLUTE_CUES = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\b\d{4,}(?!\s*%)\b")


# ---------------------------------------------------------------------------
# Heuristic functions
# ---------------------------------------------------------------------------

def _get_numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text)


def _flag_entity_association(query: str) -> tuple[bool, str]:
    """Flag if entity names appear near numeric mentions."""
    nums = _get_numbers(query)
    if not nums:
        return False, ""
    entities = _ENTITY_CUES.findall(query)
    if entities:
        reason = f"entity cues ({entities[:3]}) co-occur with {len(nums)} numeric mentions"
        return True, reason
    return False, ""


def _flag_lower_upper(query: str) -> tuple[bool, str]:
    """Flag if both lower-bound and upper-bound cues are present."""
    has_lower = bool(_LOWER_CUES.search(query))
    has_upper = bool(_UPPER_CUES.search(query))
    if has_lower and has_upper:
        lowers = _LOWER_CUES.findall(query)[:2]
        uppers = _UPPER_CUES.findall(query)[:2]
        reason = f"lower cues {lowers} AND upper cues {uppers} both present"
        return True, reason
    return False, ""


def _flag_multi_numeric(query: str, threshold: int = 3) -> tuple[bool, str]:
    """Flag if >=threshold distinct numeric values appear in the query."""
    nums = _get_numbers(query)
    # Normalise: strip $,% and deduplicate
    unique = set()
    for n in nums:
        clean = n.replace("$", "").replace(",", "").replace("%", "")
        unique.add(clean)
    if len(unique) >= threshold:
        reason = f"{len(unique)} distinct numeric values: {sorted(unique)[:6]}"
        return True, reason
    return False, ""


def _flag_total_vs_per_unit(query: str) -> tuple[bool, str]:
    """Flag if both 'total' aggregate and 'per/each' per-unit language appear with numbers."""
    nums = _get_numbers(query)
    if not nums:
        return False, ""
    has_total = bool(_TOTAL_CUES.search(query))
    has_per = bool(_PER_UNIT_CUES.search(query))
    if has_total and has_per:
        totals = _TOTAL_CUES.findall(query)[:2]
        pers = _PER_UNIT_CUES.findall(query)[:2]
        reason = f"total cues {totals} AND per-unit cues {pers} both present with {len(nums)} numbers"
        return True, reason
    return False, ""


def _flag_percent_vs_absolute(query: str) -> tuple[bool, str]:
    """Flag if both percentage and absolute large-value mentions appear together."""
    has_pct = bool(_PERCENT_CUES.search(query))
    has_abs = bool(_ABSOLUTE_CUES.search(query))
    if has_pct and has_abs:
        pcts = _PERCENT_CUES.findall(query)[:2]
        abs_vals = _ABSOLUTE_CUES.findall(query)[:2]
        reason = f"percent cues {pcts} AND absolute value cues {abs_vals} co-present"
        return True, reason
    return False, ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def _load_eval_corpus(eval_path: Path) -> list[dict]:
    return _load_jsonl(eval_path)


def _load_ranker_data(ranker_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "dev", "test"):
        p = ranker_dir / f"{split}.jsonl"
        if p.exists():
            rows.extend(_load_jsonl(p))
    return rows


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit(
    eval_path: Path,
    ranker_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval corpus from: {eval_path}")
    eval_items = _load_eval_corpus(eval_path)
    print(f"  Loaded {len(eval_items)} eval examples")

    ranker_items: list[dict] = []
    if ranker_dir.exists():
        ranker_items = _load_ranker_data(ranker_dir)
        print(f"  Loaded {len(ranker_items)} pairwise ranker rows from {ranker_dir}")
    else:
        print(f"  Pairwise ranker data dir not found ({ranker_dir}); using eval corpus only")

    # Build id→text map from ranker data for enrichment
    ranker_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in ranker_items:
        qid = row.get("query_id") or row.get("instance_id") or ""
        ranker_by_qid[qid].append(row)

    # Heuristic slices
    slice_names = [
        "entity_association_risk",
        "lower_upper_risk",
        "multi_numeric_confusion",
        "total_vs_per_unit_risk",
        "percent_vs_absolute_risk",
    ]
    slice_funcs = {
        "entity_association_risk": _flag_entity_association,
        "lower_upper_risk": _flag_lower_upper,
        "multi_numeric_confusion": _flag_multi_numeric,
        "total_vs_per_unit_risk": _flag_total_vs_per_unit,
        "percent_vs_absolute_risk": _flag_percent_vs_absolute,
    }

    slices: dict[str, list[dict]] = {s: [] for s in slice_names}

    for item in eval_items:
        qid = item.get("query_id") or item.get("doc_id") or ""
        query = item.get("query") or item.get("text") or ""
        nums = _get_numbers(query)

        for slice_name in slice_names:
            flagged, reason = slice_funcs[slice_name](query)
            if flagged:
                entry: dict = {
                    "instance_id": qid,
                    "query": query,
                    "numeric_mentions": nums[:10],
                    "heuristic_reason": reason,
                    "slice": slice_name,
                }
                # Enrich with any ranker metadata if available
                if qid in ranker_by_qid:
                    entry["num_ranker_rows"] = len(ranker_by_qid[qid])
                slices[slice_name].append(entry)

    # Write per-slice JSONL files
    slice_file_map = {
        "entity_association_risk": "entity_association_risk_examples.jsonl",
        "lower_upper_risk": "lower_upper_risk_examples.jsonl",
        "multi_numeric_confusion": "multi_numeric_confusion_examples.jsonl",
        "total_vs_per_unit_risk": "total_vs_per_unit_risk_examples.jsonl",
        "percent_vs_absolute_risk": "percent_vs_absolute_risk_examples.jsonl",
    }
    for slice_name, fname in slice_file_map.items():
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            for ex in slices[slice_name]:
                fh.write(json.dumps(ex) + "\n")
        print(f"  Written {len(slices[slice_name])} examples → {out_path}")

    # Compute per-slice counts and overlap stats
    flagged_ids: dict[str, set] = {
        s: {ex["instance_id"] for ex in slices[s]} for s in slice_names
    }
    all_flagged = set().union(*flagged_ids.values())
    multi_flagged = {
        qid for qid in all_flagged
        if sum(1 for s in slice_names if qid in flagged_ids[s]) >= 2
    }

    summary: dict = {
        "total_eval_examples": len(eval_items),
        "total_ranker_rows": len(ranker_items),
        "ranker_data_available": ranker_dir.exists(),
        "slices": {
            s: {
                "count": len(slices[s]),
                "fraction": round(len(slices[s]) / max(len(eval_items), 1), 4),
            }
            for s in slice_names
        },
        "total_flagged_any_slice": len(all_flagged),
        "flagged_multiple_slices": len(multi_flagged),
        "heuristic_descriptions": {
            "entity_association_risk": (
                "Query mentions entity/person names alongside numbers, "
                "raising risk of wrong variable-entity association."
            ),
            "lower_upper_risk": (
                "Query contains both lower-bound and upper-bound cues "
                "(e.g., 'at least' and 'at most'), risking lower/upper confusion."
            ),
            "multi_numeric_confusion": (
                "Query contains >=3 distinct numeric values, "
                "raising multi-float confusable grounding risk."
            ),
            "total_vs_per_unit_risk": (
                "Query uses both 'total/aggregate' and 'per/each/unit' language "
                "alongside numbers, risking total-vs-per-unit confusion."
            ),
            "percent_vs_absolute_risk": (
                "Query mixes percentage mentions and large absolute numeric values, "
                "risking percent vs absolute value confusion."
            ),
        },
    }

    # Write JSON summary
    summary_json = out_dir / "bottleneck_audit_summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Written summary → {summary_json}")

    # Write Markdown summary
    _write_markdown_summary(summary, slices, out_dir)

    print("\nBottleneck audit complete.")
    print(f"  Total examples: {len(eval_items)}")
    print(f"  Flagged in any slice: {len(all_flagged)}")
    print(f"  Flagged in >=2 slices: {len(multi_flagged)}")


def _write_markdown_summary(
    summary: dict,
    slices: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    lines = [
        "# NLP4LP Bottleneck Audit Summary",
        "",
        "## Overview",
        "",
        f"- **Total eval examples**: {summary['total_eval_examples']}",
        f"- **Pairwise ranker data available**: {summary['ranker_data_available']}",
        f"- **Total ranker rows**: {summary['total_ranker_rows']}",
        f"- **Flagged in any slice**: {summary['total_flagged_any_slice']}",
        f"- **Flagged in ≥2 slices**: {summary['flagged_multiple_slices']}",
        "",
        "## Slice Counts",
        "",
        "| Slice | Count | Fraction |",
        "|-------|-------|----------|",
    ]
    for s, info in summary["slices"].items():
        lines.append(f"| {s} | {info['count']} | {info['fraction']:.1%} |")

    lines += [
        "",
        "## Heuristic Definitions",
        "",
    ]
    for s, desc in summary["heuristic_descriptions"].items():
        lines.append(f"### {s}")
        lines.append("")
        lines.append(desc)
        lines.append("")

    lines += ["## Example Flagged Cases", ""]
    for s, items in slices.items():
        lines.append(f"### {s} (first 3 examples)")
        lines.append("")
        for ex in items[:3]:
            lines.append(f"- **ID**: `{ex['instance_id']}`")
            lines.append(f"  - **Reason**: {ex['heuristic_reason']}")
            snippet = ex["query"][:180].replace("\n", " ")
            lines.append(f"  - **Query snippet**: *{snippet}*")
            lines.append("")

    md_path = out_dir / "bottleneck_audit_summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written markdown → {md_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPU-only bottleneck audit for NLP4LP grounding examples."
    )
    p.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl",
        help="Path to eval JSONL (default: data/processed/nlp4lp_eval_orig.jsonl)",
    )
    p.add_argument(
        "--ranker-data",
        type=Path,
        default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp",
        help="Path to pairwise ranker data dir (optional)",
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

    if not args.eval.exists():
        print(f"ERROR: eval file not found: {args.eval}", file=sys.stderr)
        return 1

    audit(
        eval_path=args.eval,
        ranker_dir=args.ranker_data,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
