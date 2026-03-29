"""
analyze_pairwise_features.py — Feature usefulness analysis for NLP4LP pairwise ranker data.

Computes simple descriptive stats for structured features in the pairwise ranker data:
  - frequency / coverage
  - mean on positive pairs vs mean on negative pairs
  - simple separation signal (mean_pos - mean_neg)

Falls back gracefully to corpus-level feature analysis if no pairwise data is available.

Writes:
  artifacts/learning_audit/pairwise_feature_analysis.json
  artifacts/learning_audit/pairwise_feature_analysis.md

Usage:
    python src/learning/analyze_pairwise_features.py [options]

Options:
    --ranker-data DIR  Pairwise ranker data dir (default: artifacts/learning_ranker_data/nlp4lp)
    --eval PATH        Fallback eval JSONL for corpus-level analysis
    --out DIR          Output dir (default: artifacts/learning_audit)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# All features we attempt to measure. When pairwise data is absent we derive
# proxy measurements from the eval corpus.
FEATURE_DEFINITIONS = {
    "type_match": "Whether the mention's inferred type matches the slot's expected type (1/0).",
    "operator_cue_match": "Whether operator cues (≤/≥/=) near the mention match the slot role.",
    "lower_cue_present": "Whether lower-bound language ('at least', 'minimum') precedes the mention.",
    "upper_cue_present": "Whether upper-bound language ('at most', 'maximum') precedes the mention.",
    "slot_mention_overlap": "Lexical overlap (Jaccard) between slot name tokens and mention context.",
    "entity_match": "Whether an entity/name near the mention matches entity cues in the slot name.",
    "sentence_proximity": "How close (in sentences) the mention is to the slot's description sentence.",
}

# Regex-based proxy feature extractors for corpus-level analysis
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_LOWER_RE = re.compile(r"\b(?:at least|minimum|no less than|>=|≥)\b", re.IGNORECASE)
_UPPER_RE = re.compile(r"\b(?:at most|maximum|no more than|<=|≤|not exceed)\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\b\.?\s*[A-Z][a-z]+|[A-Z][a-z]{2,}", re.UNICODE)
_OPERATOR_RE = re.compile(r"[<>]=?|[≤≥]|\b(?:equal|equals)\b", re.IGNORECASE)


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


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Analysis from pairwise data
# ---------------------------------------------------------------------------

def _analyze_pairwise(rows: list[dict]) -> dict:
    """Compute per-feature stats from pairwise ranker rows."""
    pos_rows = [r for r in rows if int(r.get("label", -1)) == 1]
    neg_rows = [r for r in rows if int(r.get("label", -1)) == 0]
    n = len(rows)
    n_pos = len(pos_rows)
    n_neg = len(neg_rows)

    features_found = {
        feat for row in rows for feat in FEATURE_DEFINITIONS if feat in row
    }

    stats: dict[str, dict] = {}
    for feat, desc in FEATURE_DEFINITIONS.items():
        if feat not in features_found:
            stats[feat] = {
                "definition": desc,
                "data_source": "pairwise",
                "present_in_data": False,
            }
            continue

        all_vals = [r[feat] for r in rows if feat in r and r[feat] is not None]
        pos_vals = [r[feat] for r in pos_rows if feat in r and r[feat] is not None]
        neg_vals = [r[feat] for r in neg_rows if feat in r and r[feat] is not None]

        mean_all = sum(all_vals) / max(len(all_vals), 1)
        mean_pos = sum(pos_vals) / max(len(pos_vals), 1)
        mean_neg = sum(neg_vals) / max(len(neg_vals), 1)
        sep = mean_pos - mean_neg

        stats[feat] = {
            "definition": desc,
            "data_source": "pairwise",
            "present_in_data": True,
            "coverage": round(len(all_vals) / max(n, 1), 4),
            "mean_overall": round(mean_all, 4),
            "mean_on_positive": round(mean_pos, 4),
            "mean_on_negative": round(mean_neg, 4),
            "separation_signal": round(sep, 4),
            "n_pos_values": len(pos_vals),
            "n_neg_values": len(neg_vals),
        }

    return {
        "n_rows": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "features": stats,
    }


# ---------------------------------------------------------------------------
# Corpus-level proxy analysis (fallback)
# ---------------------------------------------------------------------------

def _corpus_proxy_stats(eval_items: list[dict]) -> dict:
    """Derive proxy feature measurements from eval corpus when no pairwise data."""
    n = len(eval_items)
    if n == 0:
        return {"n_items": 0, "features": {}}

    # Per-item proxy measurements
    lower_cue_counts = []
    upper_cue_counts = []
    entity_counts = []
    operator_counts = []
    num_counts = []
    multi_numeric = []

    for item in eval_items:
        q = item.get("query") or item.get("text") or ""
        nums = _NUM_RE.findall(q)
        lower_cue_counts.append(1 if _LOWER_RE.search(q) else 0)
        upper_cue_counts.append(1 if _UPPER_RE.search(q) else 0)
        entity_counts.append(1 if _ENTITY_RE.search(q) else 0)
        operator_counts.append(1 if _OPERATOR_RE.search(q) else 0)
        num_counts.append(len(nums))
        unique_nums = set(n.replace("$", "").replace(",", "").replace("%", "") for n in nums)
        multi_numeric.append(1 if len(unique_nums) >= 3 else 0)

    def _mean(lst: list) -> float:
        return round(sum(lst) / max(len(lst), 1), 4)

    stats: dict[str, dict] = {}
    for feat, desc in FEATURE_DEFINITIONS.items():
        if feat == "lower_cue_present":
            freq = _mean(lower_cue_counts)
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "corpus_frequency": freq,
                "note": "Fraction of eval queries containing lower-bound cues.",
            }
        elif feat == "upper_cue_present":
            freq = _mean(upper_cue_counts)
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "corpus_frequency": freq,
                "note": "Fraction of eval queries containing upper-bound cues.",
            }
        elif feat == "entity_match":
            freq = _mean(entity_counts)
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "corpus_frequency": freq,
                "note": "Fraction of eval queries containing entity/name cues.",
            }
        elif feat == "operator_cue_match":
            freq = _mean(operator_counts)
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "corpus_frequency": freq,
                "note": "Fraction of eval queries containing operator cues.",
            }
        elif feat == "slot_mention_overlap":
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "note": "Requires slot names; not computable from eval corpus alone.",
            }
        elif feat == "type_match":
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "note": "Requires slot type annotations; not computable from eval corpus alone.",
            }
        elif feat == "sentence_proximity":
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "note": "Requires slot-sentence alignment; not computable from eval corpus alone.",
            }
        else:
            stats[feat] = {
                "definition": desc,
                "data_source": "corpus_proxy",
                "note": "Not computable without pairwise annotations.",
            }

    # Extra corpus-level stats
    corpus_stats = {
        "avg_numeric_mentions_per_query": _mean(num_counts),
        "fraction_with_lower_cue": _mean(lower_cue_counts),
        "fraction_with_upper_cue": _mean(upper_cue_counts),
        "fraction_with_both_cues": _mean(
            [1 if (l and u) else 0 for l, u in zip(lower_cue_counts, upper_cue_counts)]
        ),
        "fraction_with_entity_cue": _mean(entity_counts),
        "fraction_with_multi_numeric": _mean(multi_numeric),
    }

    return {
        "n_items": n,
        "features": stats,
        "corpus_stats": corpus_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(ranker_dir: Path, eval_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try pairwise data first
    rows: list[dict] = []
    if ranker_dir.exists():
        for split in ("train", "dev", "test"):
            p = ranker_dir / f"{split}.jsonl"
            if p.exists():
                rows.extend(_load_jsonl(p))
        print(f"  Loaded {len(rows)} pairwise rows from {ranker_dir}")

    if rows:
        result = _analyze_pairwise(rows)
        result["analysis_mode"] = "pairwise"
        print(f"  Mode: pairwise ({len(rows)} rows)")
    else:
        print(f"  No pairwise data; using corpus proxy analysis from {eval_path}")
        if eval_path.exists():
            eval_items = _load_jsonl(eval_path)
        else:
            eval_items = []
            print(f"  WARNING: eval file not found: {eval_path}")
        result = _corpus_proxy_stats(eval_items)
        result["analysis_mode"] = "corpus_proxy"

    # Write JSON
    json_path = out_dir / "pairwise_feature_analysis.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Written → {json_path}")

    # Write Markdown
    _write_md(result, out_dir)


def _write_md(result: dict, out_dir: Path) -> None:
    mode = result.get("analysis_mode", "unknown")
    lines = [
        "# NLP4LP Pairwise Feature Analysis",
        "",
        f"**Analysis mode**: {mode}",
        "",
    ]

    if mode == "pairwise":
        lines += [
            f"**Total rows**: {result.get('n_rows', 0)}",
            f"**Positives**: {result.get('n_positive', 0)}",
            f"**Negatives**: {result.get('n_negative', 0)}",
            "",
            "## Feature Statistics",
            "",
            "| Feature | Present | Coverage | Mean (pos) | Mean (neg) | Separation |",
            "|---------|---------|----------|------------|------------|------------|",
        ]
        for feat, info in result.get("features", {}).items():
            if not info.get("present_in_data"):
                lines.append(f"| {feat} | ✗ | — | — | — | — |")
            else:
                lines.append(
                    f"| {feat} | ✓ | {info['coverage']:.1%} "
                    f"| {info['mean_on_positive']:.3f} | {info['mean_on_negative']:.3f} "
                    f"| {info['separation_signal']:+.3f} |"
                )
    else:
        lines += [
            f"**Eval items**: {result.get('n_items', 0)}",
            "",
            "> Pairwise data not available. Showing corpus-level proxy statistics.",
            "",
        ]
        cs = result.get("corpus_stats", {})
        if cs:
            lines += [
                "## Corpus-Level Proxy Stats",
                "",
                "| Statistic | Value |",
                "|-----------|-------|",
            ]
            for k, v in cs.items():
                lines.append(f"| {k} | {v:.3f} |")
            lines.append("")

        lines += [
            "## Feature Notes",
            "",
        ]
        for feat, info in result.get("features", {}).items():
            note = info.get("note", "")
            freq = info.get("corpus_frequency")
            freq_str = f" (corpus freq: {freq:.1%})" if freq is not None else ""
            lines.append(f"### {feat}{freq_str}")
            lines.append("")
            lines.append(info.get("definition", ""))
            if note:
                lines.append(f"> {note}")
            lines.append("")

    md_path = out_dir / "pairwise_feature_analysis.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written → {md_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Feature usefulness analysis for NLP4LP pairwise ranker data."
    )
    p.add_argument(
        "--ranker-data",
        type=Path,
        default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp",
        help="Pairwise ranker data dir (default: artifacts/learning_ranker_data/nlp4lp)",
    )
    p.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl",
        help="Fallback eval JSONL (default: data/processed/nlp4lp_eval_orig.jsonl)",
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
    run_analysis(
        ranker_dir=args.ranker_data,
        eval_path=args.eval,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
