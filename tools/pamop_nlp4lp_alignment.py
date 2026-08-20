#!/usr/bin/env python3
"""Build the PaMOP <-> NLP4LP catalog alignment manifest.

Determines, for each of our 331 canonical NLP4LP `test`-split catalog
entries (``data/catalogs/nlp4lp_catalog.jsonl``), whether it could possibly
be one of the 67 problems (54 LP + 13 MILP) that PaMOP (IJCAI 2025) cites
from OptiMUS v2 (AhmadiTeshnizi et al. 2024, arXiv:2402.10172) — the
"NLP4LP... 54 LP and 13 MILP problems (67 instances in total)" benchmark,
originally released 2024-05-13.

The mapping from catalog index to the underlying `udell-lab/NLP4LP`
Hugging Face problem id was established empirically (see
docs/PAMOP_REPRODUCTION_PLAN.md, "NLP4LP Subset Alignment" section) by
matching catalog `text` against HF `problem_info.json:parametrized_description`
at multiple points:

    catalog index i, 0 <= i <= 268   -> HF problem id (i + 1)      [ids 1-269]
    catalog index i, 269 <= i <= 330 -> HF problem id (i + 24)     [ids 293-354]

HF ids 1-269 are the snapshot that existed continuously from the dataset's
initial HF upload (2024-11-02) through at least 2026-04-20 (i.e. spanning
PaMOP's entire research/publication window, since IJCAI 2025 camera-ready
predates that range). HF ids 270-292 (a "dev" split) and 293-354 (new test
instances) were added 2026-02-12, and ids 355-361 (case study) on
2026-02-27 -- all **after** PaMOP (IJCAI 2025, main track) was published,
so none of those can be part of PaMOP's evaluation set.

This script does NOT attempt to identify PaMOP's exact 67 problems (the
original 2024-05-13 archived release is only available via a JS-rendered,
reCAPTCHA-gated page and a bot-walled OpenReview supplementary link -- both
require interactive/manual access, out of scope for this script). It only
computes the deterministic, defensible bound: which of our 331 catalog
entries COULD possibly be one of PaMOP's 67 (because the underlying problem
already existed pre-publication) versus which PROVABLY CANNOT (because the
problem did not exist yet when PaMOP was written).

No gated NLP4LP problem text is written to the output manifest -- only
catalog ids/indices and structural metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"
OUTPUT_PATH = ROOT / "data" / "baselines" / "pamop" / "nlp4lp_pamop_subset.jsonl"

# HF ids confirmed present before PaMOP's IJCAI 2025 publication window.
PRE_PAMOP_HF_ID_MAX = 269
# HF ids 270-292 (dev split) and 355-361 (case study) are excluded from the
# 331-entry `test` catalog already; only two contiguous test blocks remain.
NEW_TEST_BLOCK_START_INDEX = 269  # catalog index where the post-2026 block begins
NEW_TEST_BLOCK_HF_ID_START = 293


def catalog_index_to_hf_id(index: int) -> int:
    if index <= PRE_PAMOP_HF_ID_MAX - 1:
        return index + 1
    return (index - NEW_TEST_BLOCK_START_INDEX) + NEW_TEST_BLOCK_HF_ID_START


def main() -> None:
    rows = [json.loads(line) for line in CATALOG_PATH.open(encoding="utf-8")]
    test_rows = [r for r in rows if r.get("meta", {}).get("split") == "test"]
    test_rows.sort(key=lambda r: r["meta"]["index"])
    assert len(test_rows) == 331, f"expected 331 test rows, got {len(test_rows)}"

    out_records = []
    n_possible = 0
    n_no_match = 0
    for r in test_rows:
        idx = r["meta"]["index"]
        hf_id = catalog_index_to_hf_id(idx)
        pre_pamop = hf_id <= PRE_PAMOP_HF_ID_MAX
        if pre_pamop:
            n_possible += 1
            status = "POSSIBLE_MATCH"
            evidence = (
                "HF id <= 269: present in the NLP4LP snapshot continuously "
                "available since 2024-11-02, spanning PaMOP's IJCAI 2025 "
                "publication window. Could be one of PaMOP's cited 67 "
                "problems, but exact membership is unresolved (original "
                "2024-05-13 archived release not programmatically retrievable)."
            )
        else:
            n_no_match += 1
            status = "NO_MATCH"
            evidence = (
                "HF id >= 293: added to the dataset 2026-02-12, six months "
                "after PaMOP (IJCAI 2025, Aug 2025) was published. Cannot "
                "have been part of PaMOP's evaluation set."
            )

        out_records.append(
            {
                "pamop_problem_identifier": None,
                "current_nlp4lp_catalog_doc_id": r["doc_id"],
                "current_nlp4lp_catalog_index": idx,
                "current_nlp4lp_hf_problem_id": hf_id,
                "lp_or_milp": None,
                "text_match_status": "NOT_ATTEMPTED",
                "schema_match_status": "NOT_ATTEMPTED",
                "optimus_code_available": True,
                "mapping_confidence": status,
                "evidence": evidence,
                "notes": (
                    "PaMOP problem identifier is unknown; PaMOP does not "
                    "publish per-problem IDs, and no official code/data "
                    "release exists (see docs/PAMOP_REPRODUCTION_PLAN.md)."
                ),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec) + "\n")

    print(f"wrote {len(out_records)} rows to {OUTPUT_PATH}")
    print(f"POSSIBLE_MATCH (pre-PaMOP HF ids <=269): {n_possible}")
    print(f"NO_MATCH (post-PaMOP HF ids >=293): {n_no_match}")
    print("EXACT_MATCH: 0 (no archived 2024-05-13 snapshot available to confirm identity)")
    print("HIGH_CONFIDENCE_MATCH: 0")
    print(f"UNRESOLVED: which specific 67 of the {n_possible} POSSIBLE_MATCH rows PaMOP used")


if __name__ == "__main__":
    main()
