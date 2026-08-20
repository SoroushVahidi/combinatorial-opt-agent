#!/usr/bin/env python3
"""Deterministic recomputation of the frozen oracle (upper-bound) control.

The oracle control uses gold (relevant-doc) retrieval for every query, i.e.
`pred_id = gold_id` for all 331 NLP4LP `orig` queries, then runs the frozen
patched grounding / extraction / typed-greedy slot assignment on the gold
problem. This is NOT an experiment: it is a deterministic verification that
reproduces the manuscript's oracle column with the frozen method.

Outputs (results/oracle_recomputation_2026-08-15/):
  nlp4lp_downstream_per_query_orig_oracle.csv  -- per-query oracle row
  nlp4lp_downstream_orig_oracle.json           -- canonical aggregate sidecar
  oracle_frozen_verification.json              -- this run's verification record
  oracle_frozen_summary.csv                    -- manuscript-ready oracle metrics

Run from the repo root with the frozen method (current HEAD is the frozen SHA):
    PYTHONHASHSEED=0 \
    NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \
    python3 tools/recompute_frozen_oracle.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import nlp4lp_downstream_utility as u  # noqa: E402

OUT_DIR = ROOT / "results" / "oracle_recomputation_2026-08-15"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    command = (
        "PYTHONHASHSEED=0 "
        "NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json "
        "python3 tools/recompute_frozen_oracle.py"
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    ok = u.run_single_setting(
        variant="orig",
        baseline_arg="oracle",
        assignment_mode="typed",
        out_dir=OUT_DIR,
    )
    if not ok:
        raise SystemExit("oracle run_single_setting failed")

    rows = list(
        csv.DictReader(
            (OUT_DIR / "nlp4lp_downstream_per_query_orig_oracle.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    n = len(rows)

    schema_r1 = sum(1 for r in rows if r.get("schema_hit") == "1") / n
    coverage = sum(float(r["param_coverage"]) for r in rows) / n
    type_match = sum(float(r["type_match"]) for r in rows) / n
    key_overlap = sum(float(r["key_overlap"]) for r in rows) / n

    def _full_denominator(key: str) -> float:
        vals = [
            float(r[key]) if r.get(key) not in ("", "None") else 0.0 for r in rows
        ]
        return sum(vals) / n

    def _ready(require_schema: bool) -> tuple[int, float]:
        count = 0
        for r in rows:
            hit = float(r["schema_hit"]) == 1.0
            ok_ = (
                float(r["param_coverage"]) >= 0.8
                and float(r["type_match"]) >= 0.8
            )
            if ok_ and (not require_schema or hit):
                count += 1
        return count, count / n

    inst_ready_count, inst_ready = _ready(require_schema=False)
    strict_count, strict_ready = _ready(require_schema=True)

    summary = {
        "oracle_n": n,
        "Schema_R1": round(schema_r1, 6),
        "Coverage": round(coverage, 6),
        "TypeMatch": round(type_match, 6),
        "KeyOverlap": round(key_overlap, 6),
        "InstantiationReady": round(inst_ready, 6),
        "InstantiationReady_count": inst_ready_count,
        "StrictInstantiationReady": round(strict_ready, 6),
        "StrictInstantiationReady_count": strict_count,
        "Exact5": round(_full_denominator("exact5"), 6),
        "Exact20": round(_full_denominator("exact20"), 6),
        "Exact5_on_hits": round(
            sum(
                float(r["exact5"])
                for r in rows
                if r.get("exact5") not in ("", "None")
            )
            / sum(1 for r in rows if r.get("exact5") not in ("", "None")),
            6,
        ),
        "Exact20_on_hits": round(
            sum(
                float(r["exact20"])
                for r in rows
                if r.get("exact20") not in ("", "None")
            )
            / sum(1 for r in rows if r.get("exact20") not in ("", "None")),
            6,
        ),
    }

    verification = {
        "kind": "oracle_frozen_recomputation",
        "method": "FROZEN_FOR_RESUBMISSION (tfidf_typed_greedy_ratio_extraction, gold retrieval)",
        "benchmark": "NLP4LP orig, 331 queries",
        "assignment_mode": "typed",
        "predicted_doc_id_rule": "pred_id = gold_id for every query (upper-bound control)",
        "extraction": "frozen multiplicative ratio-word numeric extraction",
        "metric_definitions": {
            "StrictInstantiationReady": (
                "1[SchemaHit AND param_coverage >= 0.8 AND type_match >= 0.8]"
            ),
            "InstantiationReady": "1[param_coverage >= 0.8 AND type_match >= 0.8]",
            "Exact5/Exact20": "full-denominator mean over all 331 queries",
            "Exact5_on_hits/Exact20_on_hits": (
                "mean over queries with at least one comparable scalar value"
            ),
        },
        "command": command,
        "git_sha": git_sha,
        "pythonhashseed": "0",
        "gold_cache": "results/eswa_revision/00_env/nlp4lp_gold_cache.json",
        "inputs": {
            "eval": "data/processed/nlp4lp_eval_orig.jsonl",
            "catalog": "data/catalogs/nlp4lp_catalog.jsonl",
        },
        "outputs": {
            "per_query": "results/oracle_recomputation_2026-08-15/nlp4lp_downstream_per_query_orig_oracle.csv",
            "aggregate_sidecar": "results/oracle_recomputation_2026-08-15/nlp4lp_downstream_orig_oracle.json",
            "summary": "results/oracle_recomputation_2026-08-15/oracle_frozen_summary.csv",
        },
        "summary": summary,
        "note": (
            "Deterministic verification of the oracle (gold-schema retrieval) control "
            "with the frozen patched grounding/extraction; not a new experiment."
        ),
    }

    with open(OUT_DIR / "oracle_frozen_verification.json", "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    with open(OUT_DIR / "oracle_frozen_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()