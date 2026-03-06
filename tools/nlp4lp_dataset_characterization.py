"""Dataset characterization for NLP4LP: query/passage length and number-count stats."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Same regex as in analyze_nlp4lp_failures.py
NUM_PATTERN = re.compile(r"[$]?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+")


def _num_count(text: str) -> int:
    return len(NUM_PATTERN.findall(text or ""))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=Path, default=ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "paper" / "nlp4lp_dataset_characterization.csv")
    args = p.parse_args()

    query_lens: list[int] = []
    num_counts: list[int] = []
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            query_lens.append(len(q.split()))
            num_counts.append(_num_count(q))

    passage_lens: list[int] = []
    with open(args.catalog, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("description") or "").strip()
            passage_lens.append(len(text.split()))

    def stats(vals: list[int]):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "p10": 0, "p50": 0, "p90": 0, "max": 0}
        s = sorted(vals)
        n = len(s)
        mean = sum(s) / n
        variance = sum((x - mean) ** 2 for x in s) / n if n else 0
        std = variance ** 0.5
        p10 = s[int(0.10 * n)] if n else 0
        p50 = s[int(0.50 * n)] if n else 0
        p90 = s[int(0.90 * n)] if n else 0
        return {"mean": round(mean, 2), "std": round(std, 2), "p10": p10, "p50": p50, "p90": p90, "max": max(s)}

    qs = stats(query_lens)
    ps = stats(passage_lens)
    ncs = stats(num_counts)

    row = {
        "n_queries": len(query_lens),
        "n_docs": len(passage_lens),
        "query_len_tokens_mean": qs["mean"],
        "query_len_tokens_std": qs["std"],
        "query_len_tokens_p10": qs["p10"],
        "query_len_tokens_p50": qs["p50"],
        "query_len_tokens_p90": qs["p90"],
        "query_len_tokens_max": qs["max"],
        "passage_len_tokens_mean": ps["mean"],
        "passage_len_tokens_std": ps["std"],
        "passage_len_tokens_p10": ps["p10"],
        "passage_len_tokens_p50": ps["p50"],
        "passage_len_tokens_p90": ps["p90"],
        "passage_len_tokens_max": ps["max"],
        "num_count_per_query_mean": round(ncs["mean"], 2),
        "num_count_per_query_p50": ncs["p50"],
        "num_count_per_query_p90": ncs["p90"],
        "num_count_per_query_max": ncs["max"],
    }
    cols = list(row.keys())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
