"""Build an end-to-end NLP4LP retrieval benchmark.

Outputs:
- Catalog JSONL: one document per NLP4LP instance with doc_id, text, meta.
- Eval JSONL: one per variant (query_id, query, relevant_doc_id).
- Stats JSON: per-variant stats (total, avg lengths, avg_num_count for orig/nonum/noisy).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent.parent

VARIANT_CHOICES = ("orig", "nonum", "short", "noentity", "noisy")

NOISY_STOPWORDS = frozenset(
    "the a an of to and or in on with for by at from".split()
)


def _load_split(split: str):
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - defensive
        raise SystemExit(f"datasets not available: {e}")

    raw = (
        (os.environ.get("HF_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_TOKEN") or "")
    ).strip()
    kwargs = {"token": raw} if raw else {}
    try:
        ds = load_dataset("udell-lab/NLP4LP", split=split, **kwargs)
    except Exception as e:  # pragma: no cover - HF gating / network
        raise SystemExit(
            "Could not load udell-lab/NLP4LP. If this dataset is gated, request access "
            "on HuggingFace and set HF_TOKEN / HUGGINGFACE_HUB_TOKEN in your shell.\n"
            f"Loader error was: {e}"
        )
    return ds


def _count_numbers(text: str) -> int:
    """Count number-like tokens (integers, floats, $, %, comma-separated numbers)."""
    if not text:
        return 0
    # Match integers, floats, currency like $1.5, 1,000, 50%
    pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+"
    return len(re.findall(pattern, text))


def _query_variant(description: str, variant: str, query_id: str | None = None) -> str:
    """Produce query text for the given variant. description is the raw ex['description'].
    query_id is required for variant 'noisy' (deterministic RNG seed)."""
    text = (description or "").strip()
    if not text:
        return ""
    if variant == "orig":
        return text
    if variant == "nonum":
        # Replace numbers (integers, floats, $, %, commas in numbers) with <num>
        text = re.sub(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+", "<num>", text)
        return re.sub(r"\s+", " ", text).strip()
    if variant == "short":
        # First sentence only: first [.!?] followed by space that is not part of Mr./Mrs./Ms./Dr.
        for m in re.finditer(r"[.!?]\s+", text):
            prefix = text[: m.start()]  # text before the [.!?]
            # Skip abbreviation periods (check Mrs before Mr)
            if not (
                prefix.endswith("Mrs") or prefix.endswith("Mr") or prefix.endswith("Ms") or prefix.endswith("Dr")
            ):
                first = (text[: m.end()]).strip()
                return first if first else text
        return text
    if variant == "noentity":
        # Remove Mr./Mrs./Ms./Dr. plus following capitalized word
        text = re.sub(r"\b(?:Mr|Mrs|Ms|Dr)\.\s*[A-Z][a-z]*\b", "", text, flags=re.IGNORECASE)
        # Drop standalone capitalized word at start of sentence (simple: after ^ or . ? !)
        text = re.sub(r"(^|[.!?]\s+)([A-Z][a-z]+)\b", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()
    if variant == "noisy":
        # Deterministic noise: lowercase, numbers-><num>, drop stopwords, delete 10% tokens (seeded by query_id).
        text = text.lower()
        text = re.sub(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+", "<num>", text)
        tokens = [t for t in text.split() if t not in NOISY_STOPWORDS]
        if not tokens:
            return re.sub(r"\s+", " ", text).strip()
        seed = int(hashlib.md5((query_id or "").encode()).hexdigest()[:8], 16)
        rng = __import__("random").Random(seed)
        n_drop = max(0, int(len(tokens) * 0.10))
        drop_indices = set(rng.sample(range(len(tokens)), min(n_drop, len(tokens))))
        tokens = [t for i, t in enumerate(tokens) if i not in drop_indices]
        return " ".join(tokens) if tokens else text.strip()
    return text


def _extract_passage_and_meta(ex) -> Tuple[str, dict]:
    """Prefer parametrized_description/description from problem_info, else fallback to description."""
    raw_desc = (ex.get("description") or "").strip()
    problem_info_raw = ex.get("problem_info")
    meta: dict = {}
    passage = raw_desc

    pi = None
    if problem_info_raw:
        try:
            pi = json.loads(problem_info_raw)
        except Exception:
            pi = None

    if isinstance(pi, dict):
        for key in ("parametrized_description", "description"):
            val = pi.get(key)
            if isinstance(val, str) and val.strip():
                passage = val.strip()
                break
        kws = pi.get("keywords")
        if kws and kws != ["N.A."]:
            meta["keywords"] = kws

    # Ensure passage != query when possible by appending a short deterministic signature.
    if passage.strip() == raw_desc:
        suffix_parts = []
        if isinstance(pi, dict):
            params = sorted((pi.get("parameters") or {}).keys())
            if params:
                suffix_parts.append("params=" + ",".join(params[:5]))
            vars_ = sorted((pi.get("variables") or {}).keys())
            if vars_:
                suffix_parts.append("vars=" + ",".join(vars_[:5]))
            if not suffix_parts:
                kws = pi.get("keywords") or []
                if isinstance(kws, list) and kws:
                    suffix_parts.append("keywords=" + ",".join(kws[:5]))
        if suffix_parts:
            passage = passage + " | " + "; ".join(suffix_parts)

    return passage, meta


def build_nlp4lp_benchmark(
    split: str,
    out_catalog: Path,
    variants: list[str],
    out_eval_dir: Path,
    out_stats_dir: Path,
) -> None:
    ds = _load_split(split)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows once: (doc_id, passage, meta, raw_description)
    rows: list[Tuple[str, str, dict, str]] = []
    for i, ex in enumerate(ds):
        raw_desc = (ex.get("description") or "").strip()
        if not raw_desc:
            continue
        doc_id = f"nlp4lp_{split}_{i}"
        passage, meta = _extract_passage_and_meta(ex)
        meta.update({"split": split, "index": i})
        rows.append((doc_id, passage, meta, raw_desc))

    # Write catalog once (unchanged)
    with out_catalog.open("w", encoding="utf-8") as f_cat:
        for doc_id, passage, meta, _ in rows:
            f_cat.write(
                json.dumps(
                    {"doc_id": doc_id, "text": passage, "meta": meta},
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_docs = len(rows)
    sum_passage_tokens = sum(len(p.split()) for _, p, _, _ in rows)

    for variant in variants:
        if variant not in VARIANT_CHOICES:
            continue
        eval_path = out_eval_dir / f"nlp4lp_eval_{variant}.jsonl"
        stats_path = out_stats_dir / f"nlp4lp_stats_{variant}.json"

        sum_query_tokens = 0
        sum_num_count = 0
        with eval_path.open("w", encoding="utf-8") as f_eval:
            for doc_id, passage, _meta, raw_desc in rows:
                query = _query_variant(raw_desc, variant, query_id=doc_id if variant == "noisy" else None)
                if not query:
                    query = raw_desc
                f_eval.write(
                    json.dumps(
                        {
                            "query_id": doc_id,
                            "query": query,
                            "relevant_doc_id": doc_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sum_query_tokens += len(query.split())
                if variant in ("orig", "nonum", "noisy"):
                    sum_num_count += _count_numbers(raw_desc)

        total_queries = total_docs
        stats = {
            "split": split,
            "variant": variant,
            "total_docs": total_docs,
            "total_queries": total_queries,
            "avg_query_length_tokens": (sum_query_tokens / total_queries) if total_queries else 0.0,
            "avg_passage_length_tokens": (sum_passage_tokens / total_docs) if total_docs else 0.0,
        }
        if variant in ("orig", "nonum", "noisy"):
            stats["avg_num_count_per_query"] = (sum_num_count / total_queries) if total_queries else 0.0
        with stats_path.open("w", encoding="utf-8") as f_stats:
            json.dump(stats, f_stats, indent=2)
        print(f"  Wrote {eval_path} and {stats_path}")

    print(
        f"NLP4LP benchmark built for split='{split}': {total_docs} docs, "
        f"variants {variants} -> {out_eval_dir}"
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build NLP4LP retrieval benchmark files.")
    p.add_argument("--split", type=str, default="test", help="NLP4LP split (default: test)")
    p.add_argument(
        "--out-catalog",
        type=Path,
        default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="orig",
        help="Comma-separated variants: orig,nonum,short,noentity (default: orig)",
    )
    p.add_argument(
        "--out-eval-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory for nlp4lp_eval_<variant>.jsonl (default: data/processed/)",
    )
    p.add_argument(
        "--out-stats-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for nlp4lp_stats_<variant>.json (default: results/)",
    )
    args = p.parse_args()
    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]

    build_nlp4lp_benchmark(
        split=args.split,
        out_catalog=args.out_catalog,
        variants=variants,
        out_eval_dir=args.out_eval_dir,
        out_stats_dir=args.out_stats_dir,
    )


if __name__ == "__main__":
    main()

