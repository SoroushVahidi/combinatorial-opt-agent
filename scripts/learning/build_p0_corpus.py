#!/usr/bin/env python3
"""Build the P0 feature-augmented pairwise (mention, slot) corpus.

Reuses the EXISTING, already-verified instance-level train/dev/test
partition (built by ``src.learning.split_nlp4lp_corpus_for_benchmark``,
seed 42, 230/50/50 instances -- see docs/LEARNED_GROUNDING_P0.md "Data
Split") rather than constructing a new split. For each instance in that
partition, this script re-extracts mentions/slots using the CANONICAL
downstream pipeline's own extraction (``tools/learned_local_scorer.py``,
which wraps ``tools/nlp4lp_downstream_utility.py``) instead of the
simpler, independent regex-based extractor used to build the original
5-feature NR10 corpus. This gives every (mention, slot) pair the same
~24 hand-engineered features the canonical `optimization_role_repair` /
`typed_greedy` pipeline itself uses, plus one new frozen sentence-embedding
similarity feature.

Usage:
    export NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json
    python3 scripts/learning/build_p0_corpus.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.learned_local_scorer import (  # noqa: E402
    ALL_FEATURE_NAMES,
    FrozenEncoder,
    build_engineered_rows,
)

EVAL_PATH = ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl"
EXISTING_SPLIT_DIR = ROOT / "artifacts" / "learning_corpus"
OUT_DIR = ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp_p0"
GOLD_MATCH_TOL = 1e-6


def _is_scalar(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_gold_cache() -> dict:
    cache_path = os.environ.get("NLP4LP_GOLD_CACHE") or str(ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json")
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("gold_by_id", {})


def load_eval_items() -> list[dict]:
    items = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def scalar_params_for(gold_entry: dict) -> list[str]:
    params = gold_entry.get("parameters") or {}
    pinfo = gold_entry.get("problem_info") or {}
    if isinstance(pinfo, str):
        try:
            pinfo = json.loads(pinfo)
        except Exception:
            pinfo = {}
    expected_params = list((pinfo.get("parameters") or params or {}).keys())
    return [p for p in expected_params if _is_scalar(params.get(p))]


def instance_ids_for_split(split: str) -> set[str]:
    path = EXISTING_SPLIT_DIR / f"nlp4lp_{split}.jsonl"
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ids.add(rec["instance_id"])
    return ids


def main() -> None:
    gold_by_id = load_gold_cache()
    eval_items = load_eval_items()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_ids = {split: instance_ids_for_split(split) for split in ("train", "dev", "test")}
    print(f"Reusing existing split partition: train={len(split_ids['train'])} dev={len(split_ids['dev'])} test={len(split_ids['test'])} instances")

    encoder = FrozenEncoder()
    stats = {}

    for split, ids in split_ids.items():
        all_rows: list[dict] = []
        mention_texts: list[str] = []
        slot_texts: list[str] = []
        pending: list[tuple[dict, int]] = []  # (row, index into text lists)

        n_instances_processed = 0
        n_skipped_no_slots = 0
        n_skipped_no_mentions = 0

        for item in eval_items:
            query_id = item.get("query_id", "")
            if query_id not in ids:
                continue
            doc_id = item.get("relevant_doc_id", "")
            gold_entry = gold_by_id.get(doc_id) or {}
            params = gold_entry.get("parameters") or {}
            scalar_params = scalar_params_for(gold_entry)
            if not scalar_params:
                n_skipped_no_slots += 1
                continue
            query = (item.get("query") or "").strip()
            mentions, slots, rows = build_engineered_rows(query, scalar_params)
            if not mentions:
                n_skipped_no_mentions += 1
                continue
            n_instances_processed += 1

            # Gold matching: mention.tok.value vs gold param value, per slot.
            mention_by_id = {m.mention_id: m for m in mentions}
            gold_mention_for_slot: dict[str, int | None] = {}
            for slot_name in scalar_params:
                gval = params.get(slot_name)
                match_id = None
                if _is_scalar(gval):
                    for m in mentions:
                        mv = m.tok.value
                        if mv is not None and abs(float(mv) - float(gval)) <= GOLD_MATCH_TOL * max(1.0, abs(float(gval))):
                            match_id = m.mention_id
                            break
                gold_mention_for_slot[slot_name] = match_id

            for row in rows:
                slot_name = row["slot_name"]
                mid = row["mention_id"]
                gold_mid = gold_mention_for_slot.get(slot_name)
                label = 1 if (gold_mid is not None and mid == gold_mid) else 0
                m = mention_by_id[mid]
                full_row = {
                    "instance_id": query_id,
                    "schema_name": doc_id,
                    "group_id": f"{query_id}::{slot_name}",
                    "slot_name": slot_name,
                    "mention_id": mid,
                    "mention_value": m.tok.value,
                    "mention_type_bucket": m.type_bucket,
                    "expected_type": next((s.expected_type for s in slots if s.name == slot_name), None),
                    "label": label,
                    "gold_mention_id_for_slot": gold_mid,
                    "n_expected_scalar": len(scalar_params),
                    "engineered_features": row["engineered_features"],
                }
                pending.append((full_row, len(mention_texts)))
                mention_texts.append(row["mention_text"])
                slot_texts.append(row["slot_text"])

        # Batch-compute the frozen embedding similarity feature for all rows in this split.
        sims = encoder.pair_similarities(mention_texts, slot_texts) if mention_texts else []
        for full_row, idx in pending:
            full_row["embedding_similarity"] = float(sims[idx]) if len(sims) else 0.0
            all_rows.append(full_row)

        out_path = OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_groups = len(set(r["group_id"] for r in all_rows))
        n_pos_groups = len(set(r["group_id"] for r in all_rows if r["label"] == 1))
        n_trivial_incompat = sum(1 for r in all_rows if r["engineered_features"]["type_incompatible"] and r["label"] == 0)
        stats[split] = {
            "instances_processed": n_instances_processed,
            "instances_skipped_no_scalar_slots": n_skipped_no_slots,
            "instances_skipped_no_mentions_extracted": n_skipped_no_mentions,
            "pairs": len(all_rows),
            "groups_slots": n_groups,
            "groups_with_positive_label": n_pos_groups,
            "groups_with_positive_label_rate": round(n_pos_groups / n_groups, 4) if n_groups else 0.0,
            "trivial_type_incompatible_negatives": n_trivial_incompat,
            "trivial_type_incompatible_negative_rate": round(n_trivial_incompat / len(all_rows), 4) if all_rows else 0.0,
        }
        print(f"{split}: {json.dumps(stats[split])}")

    with open(OUT_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": list(ALL_FEATURE_NAMES),
                "encoder": encoder.model_name,
                "gold_match_tolerance": GOLD_MATCH_TOL,
                "reused_split_source": str(EXISTING_SPLIT_DIR),
                "splits": stats,
            },
            f,
            indent=2,
        )
    print(f"Wrote corpus to {OUT_DIR}")


if __name__ == "__main__":
    main()
