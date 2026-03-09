"""
export_manual_inspection_cases.py — Export compact sets of hard cases for manual inspection.

Exports:
  - 25 entity-association-heavy examples
  - 25 lower/upper-heavy examples
  - 25 multi-number confusion examples
  - 25 mixed hard cases (flagged in >=2 bottleneck slices)

Writes:
  artifacts/learning_audit/manual_inspection_cases.md
  artifacts/learning_audit/manual_inspection_cases.jsonl

The markdown is formatted for easy reading with:
  - problem text snippets
  - numeric mentions
  - slot context (from ranker data if available)
  - reason the case is hard

Usage:
    python src/learning/export_manual_inspection_cases.py [options]

Options:
    --eval PATH        Eval JSONL path (default: data/processed/nlp4lp_eval_orig.jsonl)
    --ranker-data DIR  Pairwise ranker data dir (optional enrichment)
    --out DIR          Output dir (default: artifacts/learning_audit)
    --n N              Max examples per category (default: 25)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# --- Reuse heuristic patterns ---
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_ENTITY_CUES = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s*[A-Z][a-z]+"
    r"|\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b",
    re.UNICODE,
)
_LOWER_CUES = re.compile(
    r"\b(?:at least|minimum|min|no less than|greater than or equal|>=|≥|must be at least)\b",
    re.IGNORECASE,
)
_UPPER_CUES = re.compile(
    r"\b(?:at most|maximum|max|no more than|less than or equal|<=|≤|must not exceed|cannot exceed|up to)\b",
    re.IGNORECASE,
)
_TOTAL_CUES = re.compile(r"\b(?:total|overall|aggregate|combined|sum)\b", re.IGNORECASE)
_PER_UNIT_CUES = re.compile(r"\b(?:per|each|every|unit|piece|item)\b", re.IGNORECASE)
_PERCENT_CUES = re.compile(r"\b\d+(?:\.\d+)?%|\bpercent(?:age)?\b", re.IGNORECASE)
_ABSOLUTE_CUES = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\b\d{4,}(?!\s*%)\b")


def _get_numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text)


def _get_unique_numbers(text: str) -> list[str]:
    nums = _get_numbers(text)
    seen: set[str] = set()
    unique = []
    for n in nums:
        clean = n.replace("$", "").replace(",", "").replace("%", "")
        if clean not in seen:
            seen.add(clean)
            unique.append(n)
    return unique


def _flag_entity(q: str) -> tuple[bool, str]:
    nums = _get_numbers(q)
    entities = _ENTITY_CUES.findall(q)
    if nums and entities:
        return True, f"entity cues {entities[:3]} with {len(nums)} numbers"
    return False, ""


def _flag_lower_upper(q: str) -> tuple[bool, str]:
    has_lower = bool(_LOWER_CUES.search(q))
    has_upper = bool(_UPPER_CUES.search(q))
    if has_lower and has_upper:
        lowers = _LOWER_CUES.findall(q)[:2]
        uppers = _UPPER_CUES.findall(q)[:2]
        return True, f"lower cues {lowers} AND upper cues {uppers}"
    return False, ""


def _flag_multi_numeric(q: str) -> tuple[bool, str]:
    unique = _get_unique_numbers(q)
    if len(unique) >= 3:
        return True, f"{len(unique)} distinct numbers: {unique[:6]}"
    return False, ""


def _flag_total_per_unit(q: str) -> tuple[bool, str]:
    nums = _get_numbers(q)
    if nums and _TOTAL_CUES.search(q) and _PER_UNIT_CUES.search(q):
        return True, "total+per-unit language with numbers"
    return False, ""


def _flag_pct_vs_abs(q: str) -> tuple[bool, str]:
    if _PERCENT_CUES.search(q) and _ABSOLUTE_CUES.search(q):
        pcts = _PERCENT_CUES.findall(q)[:2]
        abs_vals = _ABSOLUTE_CUES.findall(q)[:2]
        return True, f"percent cues {pcts} + absolute values {abs_vals}"
    return False, ""


SLICE_FUNCS = {
    "entity_association": _flag_entity,
    "lower_upper": _flag_lower_upper,
    "multi_numeric": _flag_multi_numeric,
    "total_vs_per_unit": _flag_total_per_unit,
    "percent_vs_absolute": _flag_pct_vs_abs,
}


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


def _get_first_sentence(text: str, max_chars: int = 200) -> str:
    """Return approximately the first sentence of text."""
    m = re.search(r"[.!?]\s", text)
    if m:
        end = m.start() + 1
        return text[:end].strip()
    return text[:max_chars].strip()


def _build_entry(
    item: dict,
    category: str,
    reason: str,
    ranker_rows: list[dict],
) -> dict:
    qid = item.get("query_id") or item.get("doc_id") or ""
    query = item.get("query") or item.get("text") or ""
    nums = _get_unique_numbers(query)
    first_sent = _get_first_sentence(query)

    entry: dict = {
        "instance_id": qid,
        "category": category,
        "heuristic_reason": reason,
        "query_snippet": query[:400],
        "first_sentence": first_sent,
        "numeric_mentions": nums[:10],
        "n_numeric": len(nums),
    }

    if ranker_rows:
        slot_names = list({r.get("slot_name", "") for r in ranker_rows if r.get("slot_name")})
        mention_surfaces = list({r.get("mention_surface", "") for r in ranker_rows if r.get("mention_surface")})
        entry["slot_names"] = slot_names[:8]
        entry["mention_surfaces"] = mention_surfaces[:8]
        entry["n_ranker_rows"] = len(ranker_rows)

    return entry


def _select_diverse(items: list[dict], n: int) -> list[dict]:
    """Select up to n items, preferring diverse numeric counts."""
    if len(items) <= n:
        return items
    # Sort by number of numeric mentions descending for diversity
    items_sorted = sorted(items, key=lambda x: len(_get_numbers(x.get("query") or x.get("text") or "")), reverse=True)
    # Take evenly spaced
    step = max(1, len(items_sorted) // n)
    selected = items_sorted[::step][:n]
    return selected


def export_cases(
    eval_path: Path,
    ranker_dir: Path,
    out_dir: Path,
    n_per_category: int = 25,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval data from: {eval_path}")
    eval_items = _load_jsonl(eval_path) if eval_path.exists() else []
    print(f"  {len(eval_items)} eval items loaded")

    # Load ranker data for enrichment
    ranker_by_qid: dict[str, list[dict]] = defaultdict(list)
    if ranker_dir.exists():
        for split in ("train", "dev", "test"):
            p = ranker_dir / f"{split}.jsonl"
            if p.exists():
                for row in _load_jsonl(p):
                    qid = row.get("query_id") or row.get("instance_id") or ""
                    ranker_by_qid[qid].append(row)

    # Categorize each eval item
    categorized: dict[str, list[dict]] = {
        "entity_association": [],
        "lower_upper": [],
        "multi_numeric": [],
        "mixed_hard": [],
    }

    for item in eval_items:
        query = item.get("query") or item.get("text") or ""
        flags: dict[str, tuple[bool, str]] = {}
        for slice_name, fn in SLICE_FUNCS.items():
            ok, reason = fn(query)
            flags[slice_name] = (ok, reason)

        n_flagged = sum(1 for (ok, _) in flags.values() if ok)
        qid = item.get("query_id") or item.get("doc_id") or ""
        ranker_rows = ranker_by_qid.get(qid, [])

        if flags["entity_association"][0]:
            reason = flags["entity_association"][1]
            categorized["entity_association"].append(
                _build_entry(item, "entity_association", reason, ranker_rows)
            )

        if flags["lower_upper"][0]:
            reason = flags["lower_upper"][1]
            categorized["lower_upper"].append(
                _build_entry(item, "lower_upper", reason, ranker_rows)
            )

        if flags["multi_numeric"][0]:
            reason = flags["multi_numeric"][1]
            categorized["multi_numeric"].append(
                _build_entry(item, "multi_numeric", reason, ranker_rows)
            )

        if n_flagged >= 2:
            all_reasons = "; ".join(r for (ok, r) in flags.values() if ok)
            categorized["mixed_hard"].append(
                _build_entry(item, "mixed_hard", all_reasons, ranker_rows)
            )

    # Select diverse subsets
    selected: dict[str, list[dict]] = {}
    for cat in ("entity_association", "lower_upper", "multi_numeric", "mixed_hard"):
        pool = categorized[cat]
        sel = _select_diverse(pool, n_per_category)
        selected[cat] = sel
        print(f"  {cat}: {len(pool)} total → {len(sel)} selected for export")

    # Write JSONL
    jsonl_path = out_dir / "manual_inspection_cases.jsonl"
    total = 0
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for cat, cases in selected.items():
            for entry in cases:
                fh.write(json.dumps(entry) + "\n")
                total += 1
    print(f"  Written {total} cases → {jsonl_path}")

    # Write Markdown
    _write_md(selected, out_dir)


def _write_md(selected: dict[str, list[dict]], out_dir: Path) -> None:
    CATEGORY_LABELS = {
        "entity_association": "Entity-Association-Heavy Cases",
        "lower_upper": "Lower/Upper-Bound-Heavy Cases",
        "multi_numeric": "Multi-Number Confusion Cases",
        "mixed_hard": "Mixed Hard Cases (≥2 Slices)",
    }

    lines = [
        "# NLP4LP Manual Inspection Cases",
        "",
        "These cases were selected by heuristic rules as likely hard for downstream",
        "number-to-slot grounding. Each entry shows the problem text, numeric mentions,",
        "and the reason the case was flagged.",
        "",
        "---",
        "",
    ]

    for cat, label in CATEGORY_LABELS.items():
        cases = selected.get(cat, [])
        lines += [
            f"## {label}  ({len(cases)} cases)",
            "",
        ]
        if not cases:
            lines += ["> No cases found for this category.", "", "---", ""]
            continue

        for i, entry in enumerate(cases, 1):
            qid = entry.get("instance_id", "?")
            reason = entry.get("heuristic_reason", "")
            snippet = entry.get("query_snippet", "")[:350].replace("\n", " ")
            nums = entry.get("numeric_mentions", [])
            slots = entry.get("slot_names", [])
            mentions = entry.get("mention_surfaces", [])

            lines += [
                f"### {i}. `{qid}`",
                "",
                f"**Why hard**: {reason}",
                "",
                f"**Text**: {snippet}",
                "",
            ]
            if nums:
                lines.append(f"**Numeric mentions**: {', '.join(f'`{n}`' for n in nums)}")
                lines.append("")
            if slots:
                lines.append(f"**Slot names** (from ranker data): {', '.join(f'`{s}`' for s in slots)}")
                lines.append("")
            if mentions:
                lines.append(f"**Mention surfaces**: {', '.join(f'`{m}`' for m in mentions)}")
                lines.append("")
            lines.append("---")
            lines.append("")

    md_path = out_dir / "manual_inspection_cases.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written → {md_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export NLP4LP hard cases for manual inspection."
    )
    p.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl",
        help="Eval JSONL path (default: data/processed/nlp4lp_eval_orig.jsonl)",
    )
    p.add_argument(
        "--ranker-data",
        type=Path,
        default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp",
        help="Pairwise ranker data dir (optional enrichment)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "learning_audit",
        help="Output directory (default: artifacts/learning_audit)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=25,
        help="Max examples per category (default: 25)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    export_cases(
        eval_path=args.eval,
        ranker_dir=args.ranker_data,
        out_dir=args.out,
        n_per_category=args.n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
