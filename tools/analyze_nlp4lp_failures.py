"""Failure analysis for NLP4LP retrieval: per-query outcomes, stratified metrics, qualitative examples."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Length buckets (tokens): [0-25], [26-50], [51-100], [101-150], [151+]
LENGTH_BUCKETS = [(0, 25), (26, 50), (51, 100), (101, 150), (151, 10**9)]

# Number-count buckets: [0], [1-2], [3-5], [6-10], [11+]
NUM_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)]

FAMILY_KEYWORDS = {
    "maximize", "minimize", "minimise", "maximise",
    "budget", "capacity", "constraint", "constraints",
    "assignment", "routing", "schedule", "allocation",
    "profit", "cost", "investment", "production",
    "linear", "integer", "optimization", "optimisation",
}


def _load_catalog_list(catalog_path: Path) -> list[dict]:
    """Load catalog as list of problem dicts for baselines.fit()."""
    with open(catalog_path, encoding="utf-8") as f:
        items = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "doc_id" in obj or "text" in obj:
                doc_id = obj.get("doc_id") or obj.get("id")
                text = obj.get("text") or obj.get("description") or ""
                meta = obj.get("meta") or {}
                items.append({
                    "id": doc_id,
                    "name": doc_id,
                    "description": text,
                    "aliases": [],
                    "meta": meta,
                })
            else:
                items.append(obj)
        return items


def _catalog_id_to_text(catalog_list: list[dict]) -> dict[str, str]:
    """Build doc_id -> text for snippets."""
    return {p.get("id", ""): (p.get("description") or p.get("name") or "") for p in catalog_list if p.get("id")}


def _num_count_in_query(query: str, variant: str) -> int:
    """Count numbers in query; for nonum also count <num> tokens."""
    if not query:
        return 0
    count = len(re.findall(r"[$]?\d[\d,]*(?:\.\d+)?%?", query))
    if variant == "nonum":
        count += query.split().count("<num>")
    return count


def _bucket_length(tokens: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= tokens <= hi:
            if hi >= 10**9:
                return "151+"
            return f"{lo}-{hi}"
    return "0-25"


def _bucket_num_count(n: int) -> str:
    for lo, hi in NUM_BUCKETS:
        if lo <= n <= hi:
            if lo == hi:
                return str(lo)
            if hi >= 10**9:
                return "11+"
            return f"{lo}-{hi}"
    return "0"


def _same_family_heuristic(text_a: str, text_b: str) -> bool:
    """True if both texts share at least one family keyword (lowercased)."""
    a = set(re.findall(r"\w+", (text_a or "").lower()))
    b = set(re.findall(r"\w+", (text_b or "").lower()))
    return bool((a & FAMILY_KEYWORDS) & (b & FAMILY_KEYWORDS))


def _load_eval_per_variant(eval_dir: Path, variant: str) -> list[dict]:
    """Load eval JSONL for variant; each item has query_id, query, relevant_doc_id."""
    path = eval_dir / f"nlp4lp_eval_{variant}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append({
                "query_id": obj.get("query_id", ""),
                "query": (obj.get("query") or "").strip(),
                "relevant_doc_id": obj.get("relevant_doc_id", ""),
            })
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="NLP4LP retrieval failure analysis")
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    p.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--variants", type=str, default="orig,nonum,short,noentity")
    p.add_argument("--baselines", type=str, nargs="+", default=["bm25", "tfidf"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    catalog_list = _load_catalog_list(args.catalog)
    id_to_text = _catalog_id_to_text(catalog_list)

    from retrieval.baselines import get_baseline

    # Per-variant, per-baseline: run ranking and collect rows for failure CSVs + stratified
    all_rows: dict[tuple[str, str], list[dict]] = {}  # (variant, baseline) -> list of row dicts
    stratified_rows: list[dict] = []

    for variant in variants:
        eval_items = _load_eval_per_variant(args.eval_dir, variant)
        if not eval_items:
            print(f"No eval for variant {variant}, skipping.")
            continue

        for bl_name in args.baselines:
            try:
                baseline = get_baseline(bl_name)
                baseline.fit(catalog_list)
            except Exception as e:
                print(f"Skipping {bl_name} for {variant}: {e}")
                continue

            key = (variant, bl_name)
            rows = []
            for item in eval_items:
                q = item["query"]
                gold_id = item["relevant_doc_id"]
                query_id = item["query_id"]

                ranked = baseline.rank(q, top_k=args.k)
                ranked_ids = [doc_id for doc_id, _ in ranked]
                ranked_with_scores = ranked  # [(doc_id, score), ...]

                hit_1 = 1 if gold_id in ranked_ids[:1] else 0
                hit_5 = 1 if gold_id in ranked_ids[:5] else 0
                hit_10 = 1 if gold_id in ranked_ids[:10] else 0
                try:
                    rank_of_gold = ranked_ids.index(gold_id) + 1 if gold_id in ranked_ids else -1
                except ValueError:
                    rank_of_gold = -1

                query_len = len(q.split())
                num_count = _num_count_in_query(q, variant)

                top1_id = ranked_ids[0] if ranked_ids else ""
                top1_score = ranked_with_scores[0][1] if ranked_with_scores else 0.0
                top1_snippet = (id_to_text.get(top1_id, "")[:80] or "").replace("\n", " ")
                top5_ids_str = ";".join(ranked_ids[:5]) if ranked_ids else ""

                rows.append({
                    "query_id": query_id,
                    "relevant_doc_id": gold_id,
                    "hit_at_1": hit_1,
                    "hit_at_5": hit_5,
                    "hit_at_10": hit_10,
                    "rank_of_gold": rank_of_gold,
                    "query_len_tokens": query_len,
                    "num_count_in_query": num_count,
                    "top1_doc_id": top1_id,
                    "top1_score": top1_score,
                    "top1_text_snippet": top1_snippet,
                    "top5_doc_ids": top5_ids_str,
                })
            all_rows[key] = rows

            # Stratified by length
            by_len: dict[str, list[dict]] = {}
            for r in rows:
                b = _bucket_length(r["query_len_tokens"])
                by_len.setdefault(b, []).append(r)
            for bucket_name, bucket_rows in by_len.items():
                n = len(bucket_rows)
                rec1 = sum(r["hit_at_1"] for r in bucket_rows) / n if n else 0.0
                rec10 = sum(r["hit_at_10"] for r in bucket_rows) / n if n else 0.0
                mrr = 0.0
                for r in bucket_rows:
                    if r["rank_of_gold"] > 0:
                        mrr += 1.0 / r["rank_of_gold"]
                mrr = mrr / n if n else 0.0
                stratified_rows.append({
                    "variant": variant,
                    "baseline": bl_name,
                    "bucket_type": "length_tokens",
                    "bucket_name": bucket_name,
                    "n": n,
                    "Recall@1": rec1,
                    "Recall@10": rec10,
                    "MRR@10": mrr,
                })

            # Stratified by num_count
            by_num: dict[str, list[dict]] = {}
            for r in rows:
                b = _bucket_num_count(r["num_count_in_query"])
                by_num.setdefault(b, []).append(r)
            for bucket_name, bucket_rows in by_num.items():
                n = len(bucket_rows)
                rec1 = sum(r["hit_at_1"] for r in bucket_rows) / n if n else 0.0
                rec10 = sum(r["hit_at_10"] for r in bucket_rows) / n if n else 0.0
                mrr = 0.0
                for r in bucket_rows:
                    if r["rank_of_gold"] > 0:
                        mrr += 1.0 / r["rank_of_gold"]
                mrr = mrr / n if n else 0.0
                stratified_rows.append({
                    "variant": variant,
                    "baseline": bl_name,
                    "bucket_type": "num_count",
                    "bucket_name": bucket_name,
                    "n": n,
                    "Recall@1": rec1,
                    "Recall@10": rec10,
                    "MRR@10": mrr,
                })

            # Write failure CSV for this (variant, baseline)
            out_csv = results_dir / f"nlp4lp_failures_{variant}_{bl_name}.csv"
            cols = [
                "query_id", "relevant_doc_id", "hit_at_1", "hit_at_5", "hit_at_10",
                "rank_of_gold", "query_len_tokens", "num_count_in_query",
                "top1_doc_id", "top1_score", "top1_text_snippet", "top5_doc_ids",
            ]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {out_csv} ({len(rows)} rows)")

    # Stratified metrics CSV
    strat_path = results_dir / "nlp4lp_stratified_metrics.csv"
    strat_cols = ["variant", "baseline", "bucket_type", "bucket_name", "n", "Recall@1", "Recall@10", "MRR@10"]
    with open(strat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=strat_cols)
        w.writeheader()
        w.writerows(stratified_rows)
    print(f"Wrote {strat_path}")

    # Failure examples (SHORT variant, 10 per baseline)
    examples_path = results_dir / "nlp4lp_failure_examples.md"
    short_bm25 = all_rows.get(("short", "bm25"), [])
    short_tfidf = all_rows.get(("short", "tfidf"), [])

    failures_bm25 = [r for r in short_bm25 if r["hit_at_1"] == 0]
    failures_tfidf = [r for r in short_tfidf if r["hit_at_1"] == 0]

    def take_representative(failures: list[dict], n: int) -> list[dict]:
        """Take up to n failures; prefer rank_of_gold > 5 (harder)."""
        by_rank = sorted(failures, key=lambda r: (-1 if r["rank_of_gold"] < 0 else r["rank_of_gold"]), reverse=True)
        return by_rank[:n]

    sel_bm25 = take_representative(failures_bm25, 10)
    sel_tfidf = take_representative(failures_tfidf, 10)

    lines = [
        "# NLP4LP retrieval: qualitative failure examples (SHORT variant)",
        "",
        "Representative failures (miss at rank 1) with top-5 predictions.",
        "",
    ]

    for bl_name, sel in [("bm25", sel_bm25), ("tfidf", sel_tfidf)]:
        lines.append(f"## Baseline: {bl_name.upper()}")
        lines.append("")
        for i, r in enumerate(sel, 1):
            query_id = r["query_id"]
            gold_id = r["relevant_doc_id"]
            # Reload query text from eval for display
            eval_items = _load_eval_per_variant(args.eval_dir, "short")
            query_text = ""
            for it in eval_items:
                if it["query_id"] == query_id:
                    query_text = (it["query"] or "")[:300]
                    if len(it.get("query", "")) > 300:
                        query_text += "..."
                    break
            top5_ids = [x.strip() for x in r["top5_doc_ids"].split(";") if x.strip()]
            top1_id = r["top1_doc_id"]
            gold_text = id_to_text.get(gold_id, "")
            top1_text = id_to_text.get(top1_id, "")

            lines.append(f"### Example {i}: `{query_id}`")
            lines.append("")
            lines.append(f"- **Query** (trimmed): {query_text}")
            lines.append(f"- **Gold doc_id**: `{gold_id}`")
            lines.append("- **Top-5 retrieved:**")
            for j, doc_id in enumerate(top5_ids[:5], 1):
                snip = (id_to_text.get(doc_id, "").replace("\n", " "))[:80]
                mark = " (gold)" if doc_id == gold_id else ""
                lines.append(f"  {j}. `{doc_id}`: {snip}...{mark}")
            same_fam = _same_family_heuristic(gold_text, top1_text)
            note = "Same family (shared keywords)." if same_fam else "Different family."
            lines.append(f"- **Note**: Wrong top-1 is `{top1_id}`. {note}")
            lines.append("")

    with open(examples_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {examples_path}")


if __name__ == "__main__":
    main()
