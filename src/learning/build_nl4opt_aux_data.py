#!/usr/bin/env python3
"""Build NL4Opt auxiliary training data: entity association, bound direction, number-role classification."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Paths to NL4Opt generation_data
NL4OPT_PATHS = {
    "train": ROOT / "data_external" / "raw" / "nl4opt_competition" / "generation_data" / "train.jsonl",
    "dev": ROOT / "data_external" / "raw" / "nl4opt_competition" / "generation_data" / "dev.jsonl",
    "test": ROOT / "data_external" / "raw" / "nl4opt_competition" / "generation_data" / "test.jsonl",
}

# Normalize value string for matching (strip $ % , and collapse whitespace)
def _norm_val(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return s.replace("$", "").replace("%", "").replace(",", "").strip()


def _direction_to_bound(direction: str, operator: str) -> str:
    d = (direction or "").lower()
    o = (operator or "").upper()
    if "GREATER" in o or "at least" in d or "minimum" in d or "least" in d:
        return "lower"
    if "LESS" in o or "at most" in d or "below" in d or "up to" in d or "more than" in d or "exceed" in d:
        return "upper"
    return "other"


def _const_type_to_role(ctype: str) -> str:
    c = (ctype or "").lower()
    if c in ("lowerbound", "upperbound"):
        return "limit"
    if c in ("sum", "linear"):
        return "rhs_total"
    if c in ("ratio", "xby", "xy"):
        return "ratio"
    return "other"


def _load_nl4opt_split(path: Path) -> list[tuple[str, dict]]:
    """Yield (hash_key, record) for each line."""
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k, v in obj.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, dict) and v.get("document"):
                    out.append((k, v))
    return out


def build_entity_task(records: list[tuple[str, dict]], split: str, output_dir: Path) -> int:
    """Entity/variable association: which variable does this numeric mention belong to?"""
    out_path = output_dir / f"entity_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    count = 0
    for hash_key, rec in records:
        doc = (rec.get("document") or "").strip()
        if not doc:
            continue
        vars_list = rec.get("vars") or []
        var_mention_to_first = rec.get("var_mention_to_first_var") or {}
        first_to_mentions = rec.get("first_var_to_mentions") or {}
        obj_decl = rec.get("obj_declaration") or {}
        const_decls = rec.get("const_declarations") or []
        terms_obj = obj_decl.get("terms") or {}
        spans = rec.get("spans") or []
        for s in spans:
            if s.get("label") not in ("PARAM", "LIMIT"):
                continue
            start = s.get("start", 0)
            end = s.get("end", start)
            text = (s.get("text") or "").strip()
            if not text:
                continue
            mention_id = f"{hash_key}_{start}_{end}"
            local_ctx = doc[max(0, start - 60) : end + 60]
            gold_var = None
            if s.get("label") == "PARAM":
                norm = _norm_val(text)
                for var_mention, val in terms_obj.items():
                    if _norm_val(str(val)) == norm:
                        gold_var = var_mention_to_first.get(var_mention, var_mention)
                        break
            else:
                norm = _norm_val(text)
                for c in const_decls:
                    lim = c.get("limit")
                    if lim is not None and _norm_val(str(lim)) == norm:
                        var_mention = c.get("var")
                        if var_mention:
                            gold_var = var_mention_to_first.get(var_mention, var_mention)
                        break
            if gold_var is None and s.get("label") == "PARAM":
                for var_mention, val in terms_obj.items():
                    if _norm_val(str(val)) == norm:
                        gold_var = var_mention_to_first.get(var_mention, var_mention)
                        break
            candidate_ids = list(vars_list)
            candidate_texts = [first_to_mentions.get(v, [v]) for v in vars_list]
            if not candidate_ids:
                continue
            row = {
                "dataset": "nl4opt",
                "split": split,
                "instance_id": hash_key,
                "problem_text": doc,
                "mention_id": mention_id,
                "mention_surface": text,
                "mention_span": {"start": start, "end": end},
                "candidate_variable_ids": candidate_ids,
                "candidate_variable_texts": candidate_texts,
                "gold_variable_id": gold_var,
                "local_context": local_ctx,
                "metadata": {"span_label": s.get("label")},
            }
            if gold_var is None:
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_bound_task(records: list[tuple[str, dict]], split: str, output_dir: Path) -> int:
    """Bound direction: lower / upper / equality / other for LIMIT spans."""
    out_path = output_dir / f"bound_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    count = 0
    for hash_key, rec in records:
        doc = (rec.get("document") or "").strip()
        if not doc:
            continue
        const_decls = rec.get("const_declarations") or []
        spans = rec.get("spans") or []
        for s in spans:
            if s.get("label") != "LIMIT":
                continue
            start = s.get("start", 0)
            end = s.get("end", start)
            text = (s.get("text") or "").strip()
            if not text:
                continue
            norm = _norm_val(text)
            gold_bound = "other"
            for c in const_decls:
                lim = c.get("limit")
                if lim is not None and _norm_val(str(lim)) == norm:
                    gold_bound = _direction_to_bound(
                        c.get("direction"),
                        c.get("operator"),
                    )
                    break
            mention_id = f"{hash_key}_{start}_{end}"
            local_ctx = doc[max(0, start - 60) : end + 60]
            row = {
                "dataset": "nl4opt",
                "split": split,
                "instance_id": hash_key,
                "problem_text": doc,
                "mention_id": mention_id,
                "surface": text,
                "span": {"start": start, "end": end},
                "local_context": local_ctx,
                "gold_bound_label": gold_bound,
                "metadata": {},
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_role_task(records: list[tuple[str, dict]], split: str, output_dir: Path) -> int:
    """Number-role: objective_coeff / limit / rhs_total / ratio / other."""
    out_path = output_dir / f"role_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    count = 0
    for hash_key, rec in records:
        doc = (rec.get("document") or "").strip()
        if not doc:
            continue
        obj_decl = rec.get("obj_declaration") or {}
        const_decls = rec.get("const_declarations") or []
        terms_obj = obj_decl.get("terms") or {}
        spans = rec.get("spans") or []
        for s in spans:
            if s.get("label") not in ("PARAM", "LIMIT"):
                continue
            start = s.get("start", 0)
            end = s.get("end", start)
            text = (s.get("text") or "").strip()
            if not text:
                continue
            mention_id = f"{hash_key}_{start}_{end}"
            local_ctx = doc[max(0, start - 60) : end + 60]
            gold_role = "other"
            if s.get("label") == "PARAM":
                norm = _norm_val(text)
                if any(_norm_val(str(v)) == norm for v in terms_obj.values()):
                    gold_role = "objective_coeff"
                else:
                    for c in const_decls:
                        if c.get("param") and _norm_val(str(c.get("param"))) == norm:
                            gold_role = "ratio"
                            break
            else:
                norm = _norm_val(text)
                for c in const_decls:
                    lim = c.get("limit")
                    if lim is not None and _norm_val(str(lim)) == norm:
                        gold_role = _const_type_to_role(c.get("type"))
                        break
            row = {
                "dataset": "nl4opt",
                "split": split,
                "instance_id": hash_key,
                "problem_text": doc,
                "mention_id": mention_id,
                "surface": text,
                "span": {"start": start, "end": end},
                "local_context": local_ctx,
                "gold_role_label": gold_role,
                "metadata": {"span_label": s.get("label")},
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=ROOT / "artifacts" / "learning_aux_data" / "nl4opt")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = __import__("random").Random(args.seed)
    summary = {"by_split": {}, "by_task": {"entity": {}, "bound": {}, "role": {}}}
    for split in ("train", "dev", "test"):
        path = NL4OPT_PATHS.get(split)
        if not path or not path.exists():
            print(f"Skip {split}: {path} not found", file=sys.stderr)
            continue
        records = _load_nl4opt_split(path)
        if args.max_examples and len(records) > args.max_examples:
            records = rng.sample(records, args.max_examples)
        summary["by_split"][split] = len(records)
        n_entity = build_entity_task(records, split, output_dir)
        n_bound = build_bound_task(records, split, output_dir)
        n_role = build_role_task(records, split, output_dir)
        summary["by_task"]["entity"][split] = n_entity
        summary["by_task"]["bound"][split] = n_bound
        summary["by_task"]["role"][split] = n_role
        print(f"{split}: records={len(records)} entity={n_entity} bound={n_bound} role={n_role}")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lines = [
        "# NL4Opt auxiliary data summary",
        "",
        f"**Instances per split:** {summary.get('by_split', {})}",
        "",
        "## Examples per task and split",
        "| Split | entity | bound | role |",
        "|-------|--------|-------|------|",
    ]
    for sp in ("train", "dev", "test"):
        e = summary["by_task"]["entity"].get(sp, 0)
        b = summary["by_task"]["bound"].get(sp, 0)
        r = summary["by_task"]["role"].get(sp, 0)
        lines.append(f"| {sp} | {e} | {b} | {r} |")
    lines.extend([
        "",
        "**Files:** `entity_{split}.jsonl`, `bound_{split}.jsonl`, `role_{split}.jsonl`",
        "",
        "**Label sets:**",
        "- Entity: gold_variable_id = canonical var from var_mention_to_first_var",
        "- Bound: gold_bound_label in {lower, upper, equality, other}",
        "- Role: gold_role_label in {objective_coeff, limit, rhs_total, ratio, other}",
    ])
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_dir / 'summary.json'}, {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
