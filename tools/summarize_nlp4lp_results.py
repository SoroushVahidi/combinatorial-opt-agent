"""Read per-variant NLP4LP retrieval metrics JSONs and write one summary CSV + JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    results_dir = ROOT / "results"
    variants = ("orig", "nonum", "short", "noentity", "noisy")
    rows: list[dict] = []

    for variant in variants:
        path = results_dir / f"nlp4lp_retrieval_metrics_{variant}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        baselines = data.get("baselines") or {}
        for bl_name, bl_data in baselines.items():
            rows.append({
                "variant": variant,
                "baseline": bl_name,
                "Recall@1": bl_data.get("Recall@1", 0.0),
                "Recall@5": bl_data.get("Recall@5", 0.0),
                "Recall@10": bl_data.get("Recall@10", 0.0),
                "MRR@10": bl_data.get("MRR@10", 0.0),
                "nDCG@10": bl_data.get("nDCG@10", 0.0),
                "runtime_sec": bl_data.get("runtime_sec", 0.0),
            })

    out_csv = results_dir / "nlp4lp_retrieval_summary.csv"
    out_json = results_dir / "nlp4lp_retrieval_summary.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    cols = ["variant", "baseline", "Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "runtime_sec"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    summary = {"rows": rows, "columns": cols}
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
