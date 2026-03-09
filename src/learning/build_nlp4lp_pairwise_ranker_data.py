#!/usr/bin/env python3
"""Build pairwise (slot, mention) ranker training data from NLP4LP common corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _type_match(slot_type: str | None, mention_type: str) -> int:
    if not slot_type or mention_type == "unknown":
        return 0
    if slot_type == mention_type:
        return 1
    if slot_type == "currency" and mention_type in ("int", "float"):
        return 0
    if slot_type == "percent" and mention_type != "percent":
        return -1
    if slot_type in ("int", "float") and mention_type in ("int", "float"):
        return 1
    return 0


def _operator_cue_match(slot_role: str | None, mention_cues: list) -> int:
    if not mention_cues:
        return 0
    if not slot_role:
        return 0
    slot_lower = "lower" if slot_role and "lower" in (slot_role or "").lower() else ("upper" if slot_role and "upper" in (slot_role or "").lower() else None)
    if slot_lower == "lower" and "min" in (mention_cues or []):
        return 1
    if slot_lower == "upper" and "max" in (mention_cues or []):
        return 1
    if slot_lower == "lower" and "max" in (mention_cues or []):
        return -1
    if slot_lower == "upper" and "min" in (mention_cues or []):
        return -1
    return 0


def _total_per_cues(slot_name: str | None, local_context: str | None) -> tuple[int, int]:
    """Return (total_like, per_unit_like) from slot name and mention context."""
    total_like = 0
    per_like = 0
    ctx = (local_context or "").lower() + " " + (slot_name or "").lower()
    if re.search(r"\b(total|budget|available|capacity|limit)\b", ctx):
        total_like = 1
    if re.search(r"\b(per|each|per unit|per item)\b", ctx):
        per_like = 1
    return total_like, per_like


def _token_overlap(slot_name: str | None, local_context: str | None) -> int:
    if not slot_name or not local_context:
        return 0
    slot_tokens = set(re.findall(r"\w+", slot_name.lower()))
    ctx_tokens = set(re.findall(r"\w+", local_context.lower()))
    return min(5, len(slot_tokens & ctx_tokens))


def build_row(
    instance_id: str,
    schema_name: str | None,
    slot: dict,
    mention: dict,
    gold_mention_id: str | None,
    problem_text: str,
) -> dict:
    slot_id = slot.get("slot_id", "")
    slot_name = slot.get("slot_name", "")
    slot_role = slot.get("slot_role")
    expected_type = slot.get("expected_type")
    mention_id = mention.get("mention_id", "")
    surface = mention.get("surface", "")
    normalized_value = mention.get("normalized_value")
    type_bucket = mention.get("type_bucket", "unknown")
    local_context = mention.get("local_context")
    operator_cues = mention.get("operator_cues") or []
    label = 1 if (gold_mention_id and mention_id == gold_mention_id) else 0
    type_match = _type_match(expected_type, type_bucket)
    op_match = _operator_cue_match(slot_role, operator_cues)
    total_like, per_like = _total_per_cues(slot_name, local_context)
    overlap = _token_overlap(slot_name, local_context)
    group_id = f"{instance_id}::{slot_id}"
    return {
        "group_id": group_id,
        "instance_id": instance_id,
        "schema_name": schema_name,
        "slot_id": slot_id,
        "slot_name": slot_name,
        "slot_role": slot_role,
        "expected_type": expected_type,
        "mention_id": mention_id,
        "mention_surface": surface,
        "mention_value": normalized_value,
        "mention_type_bucket": type_bucket,
        "sentence_or_context": local_context,
        "label": label,
        "gold_mention_id_for_slot": gold_mention_id,
        "feat_type_match": type_match,
        "feat_operator_cue_match": op_match,
        "feat_total_like": total_like,
        "feat_per_unit_like": per_like,
        "feat_slot_mention_overlap": overlap,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", type=Path, default=ROOT / "artifacts" / "learning_corpus")
    ap.add_argument("--output_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    corpus_dir = args.corpus_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        path = corpus_dir / f"nlp4lp_{split}.jsonl"
        if not path.exists():
            if args.verbose:
                print(f"Skip {split}: {path} not found", file=sys.stderr)
            continue
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("dataset") != "nlp4lp":
                    continue
                instance_id = rec.get("instance_id", "")
                schema_name = rec.get("schema_name")
                slots = rec.get("slots") or []
                mentions = rec.get("numeric_mentions") or []
                gold = rec.get("gold_slot_assignments") or {}
                problem_text = rec.get("problem_text") or ""
                if not slots or not mentions:
                    continue
                for slot in slots:
                    slot_id = slot.get("slot_id", "")
                    gold_mid = gold.get(slot_id)
                    for mention in mentions:
                        row = build_row(
                            instance_id=instance_id,
                            schema_name=schema_name,
                            slot=slot,
                            mention=mention,
                            gold_mention_id=gold_mid,
                            problem_text=problem_text,
                        )
                        rows.append(row)
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
