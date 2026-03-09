"""
Generate (mention_context, slot_name, label) training pairs from NLP4LP eval + HF gold.
Writes JSONL for train_mention_slot_scorer.py. Run from repo root.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM_RE = re.compile(r"^[$]?\d[\d,]*(?:\.\d+)?%?$")

# Digit → word lookup (int values 0-19 and round tens)
_NUM_TO_WORD: dict[int, str] = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    100: "hundred", 1000: "thousand",
}


def _int_to_word(n: int) -> str | None:
    """Return English spelling of integer n, or None if not in the lookup."""
    if n in _NUM_TO_WORD:
        return _NUM_TO_WORD[n]
    if 20 < n < 100:
        tens, ones = divmod(n, 10)
        t = _NUM_TO_WORD.get(tens * 10)
        o = _NUM_TO_WORD.get(ones)
        if t and o:
            return f"{t}-{o}"
    return None


def _word_paraphrase(context: str, digit_str: str, word: str) -> str:
    """Replace the first whole-word occurrence of digit_str with word in context."""
    pattern = re.compile(r"(?<!\w)" + re.escape(digit_str) + r"(?!\w)")
    return pattern.sub(word, context, count=1)


def _parse_value(tok: str) -> float | None:
    t = tok.strip().replace("$", "").replace("%", "").replace(",", "")
    if t.endswith("%"):
        t = t[:-1]
    try:
        v = float(t)
        if "%" in tok:
            v /= 100.0
        return v
    except Exception:
        return None


def _extract_mentions_with_context(query: str, window: int = 8) -> list[tuple[str, float | None]]:
    toks = query.split()
    out: list[tuple[str, float | None]] = []
    for i, w in enumerate(toks):
        if not NUM_RE.match(w.strip()):
            continue
        ctx_tokens = toks[max(0, i - window) : i + window + 1]
        context = " ".join(ctx_tokens)
        val = _parse_value(w)
        out.append((context, val))
    return out


def _augment_with_word_paraphrases(
    pairs: list[tuple[dict, int]],
    query: str,
    scalar_slots: list[tuple[str, float]],
    item_idx: int,
) -> list[tuple[dict, int]]:
    """For each digit mention in query that matches an int value, add a written-word variant."""
    toks = query.split()
    extra: list[tuple[dict, int]] = []
    for i, w in enumerate(toks):
        if not NUM_RE.match(w.strip()):
            continue
        val = _parse_value(w)
        if val is None:
            continue
        try:
            int_val = int(val)
        except (OverflowError, ValueError):
            continue
        if float(int_val) != val:
            continue
        word = _int_to_word(int_val)
        if word is None:
            continue
        ctx_tokens = toks[max(0, i - 8) : i + 9]
        context = " ".join(ctx_tokens)
        word_context = _word_paraphrase(context, w, word)
        for slot_name, gold_val in scalar_slots:
            if gold_val is not None:
                rel_ok = abs(val - gold_val) <= 1e-5 or (
                    abs(gold_val) >= 1e-6 and abs(val - gold_val) / abs(gold_val) <= 0.01
                )
            else:
                rel_ok = False
            label = 1 if rel_ok else 0
            extra.append(({"sentence1": word_context, "sentence2": slot_name, "label": label}, item_idx))
    return extra


def _is_scalar(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate mention-slot pairs from NLP4LP eval + gold")
    p.add_argument("--eval", type=Path, default=ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "mention_slot_pairs.jsonl")
    p.add_argument("--val-ratio", type=float, default=0.12, help="Fraction for dev split (by query)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true", help="Skip written-word paraphrase augmentation")
    args = p.parse_args()

    eval_path = args.eval
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    eval_items: list[dict] = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eval_items.append(json.loads(line))

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets required: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        ds = load_dataset("udell-lab/NLP4LP", split="test")
    except Exception as e:
        print(f"Failed to load HF dataset: {e}", file=sys.stderr)
        sys.exit(1)

    def gold_params(doc_id: str) -> dict | None:
        m = re.match(r"nlp4lp_test_(\d+)$", doc_id)
        if not m:
            return None
        idx = int(m.group(1))
        if idx < 0 or idx >= len(ds):
            return None
        raw = ds[idx].get("parameters")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return None

    pairs_with_idx: list[tuple[dict, int]] = []
    for idx, item in enumerate(eval_items):
        query = (item.get("query") or "").strip()
        doc_id = item.get("relevant_doc_id") or ""
        params = gold_params(doc_id)
        if not params:
            continue
        scalar_slots = [(k, float(v)) for k, v in params.items() if _is_scalar(v)]
        if not scalar_slots:
            continue
        mentions = _extract_mentions_with_context(query)
        if not mentions:
            continue
        for ctx, mention_val in mentions:
            for slot_name, gold_val in scalar_slots:
                if mention_val is not None and gold_val is not None:
                    rel_ok = abs(mention_val - gold_val) <= 1e-5 or (
                        abs(gold_val) >= 1e-6 and abs(mention_val - gold_val) / abs(gold_val) <= 0.01
                    )
                else:
                    rel_ok = False
                label = 1 if rel_ok else 0
                pairs_with_idx.append(({"sentence1": ctx, "sentence2": slot_name, "label": label}, idx))

        if not args.no_augment:
            extra = _augment_with_word_paraphrases(pairs_with_idx, query, scalar_slots, idx)
            pairs_with_idx.extend(extra)

    if not pairs_with_idx:
        print("No pairs generated.", file=sys.stderr)
        sys.exit(1)

    import random
    rng = random.Random(args.seed)
    indices = list(range(len(eval_items)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_ratio))
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    train_pairs = [p for p, i in pairs_with_idx if i in train_indices]
    val_pairs = [p for p, i in pairs_with_idx if i in val_indices]
    if not val_pairs:
        val_pairs = train_pairs[-len(train_pairs) // 10 :]
        train_pairs = train_pairs[: -len(train_pairs) // 10]

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    out_val = out_path.parent / (out_path.stem + "_dev.jsonl")
    with open(out_val, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_pairs)} train, {len(val_pairs)} dev to {out_path} / {out_val}")


if __name__ == "__main__":
    main()
