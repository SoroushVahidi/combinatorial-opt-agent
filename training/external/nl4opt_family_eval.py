"""
Build an NL4Opt family benchmark from the local NL4Opt test split.

Families are defined by normalized description "templates":
  - lowercase
  - numbers -> <NUM>
  - currency amounts -> <MONEY>
  - percentages -> <PCT>
  - collapsed whitespace

We keep only families with at least MIN_FAMILY_SIZE members.

Outputs:
  - data/processed/nl4opt_family_eval_test.jsonl
      {"query": <original_description>, "problem_id": <family_id>}
  - data/processed/nl4opt_family_eval_test_masked.jsonl
      {"query": <masked_description>, "problem_id": <family_id>}
  - data/processed/nl4opt_family_catalog.json
      list of {"id": family_id, "name": "NL4Opt family <id>", "description": <rep text>}
  - results/nl4opt_family_coverage.json
      {"total": ..., "kept": ..., "num_families": ..., "avg_family_size": ...}
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MIN_FAMILY_SIZE = 5


def _load_nl4opt_test() -> list[dict]:
    path = ROOT / "data" / "raw" / "nl4opt" / "test.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_template(text: str) -> str:
    """Normalize description into a template key."""
    t = (text or "").lower()
    # currency amounts like $1000 or €500
    t = re.sub(r"[$£€]\\s*\\d[\\d.,]*", "<MONEY>", t)
    # percentages like 30% or 30 %
    t = re.sub(r"\\d+(?:\\.\\d+)?\\s*%", "<PCT>", t)
    # plain numbers
    t = re.sub(r"\\d+(?:\\.\\d+)?", "<NUM>", t)
    # collapse whitespace
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def _mask_description(text: str) -> str:
    """Harder masked version: remove numbers/currencies and anonymize entities."""
    s = text or ""
    # remove currency tokens and numbers
    s = re.sub(r"[$£€]\\s*\\d[\\d.,]*", " ", s)
    s = re.sub(r"\\d+(?:\\.\\d+)?\\s*%", " ", s)
    s = re.sub(r"\\d+(?:\\.\\d+)?", " ", s)
    # anonymize capitalized words (simple entity proxy)
    s = re.sub(r"\\b[A-Z][a-zA-Z]+\\b", "<ENT>", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def main() -> None:
    items = _load_nl4opt_test()
    eval_path = ROOT / "data" / "processed" / "nl4opt_family_eval_test.jsonl"
    masked_path = ROOT / "data" / "processed" / "nl4opt_family_eval_test_masked.jsonl"
    catalog_path = ROOT / "data" / "processed" / "nl4opt_family_catalog.json"
    cov_path = ROOT / "results" / "nl4opt_family_coverage.json"

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    cov_path.parent.mkdir(parents=True, exist_ok=True)

    # Group descriptions by template key
    groups: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for obj in items:
        desc = (obj.get("description") or "").strip()
        if not desc:
            continue
        total += 1
        key = _normalize_template(desc)
        groups[key].append({"description": desc})

    # Filter families by size
    kept_families: dict[str, list[dict]] = {
        key: members for key, members in groups.items() if len(members) >= MIN_FAMILY_SIZE
    }

    # Assign stable family IDs
    key_to_family_id: dict[str, str] = {}
    for key in sorted(kept_families.keys()):
        fam_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        key_to_family_id[key] = fam_id

    # Build catalog entries
    catalog = []
    for key, members in kept_families.items():
        fam_id = key_to_family_id[key]
        # Representative description: first up to 3 examples concatenated
        reps = [m["description"] for m in members[:3]]
        rep_text = " ".join(reps)
        catalog.append(
            {
                "id": fam_id,
                "name": f"NL4Opt family {fam_id}",
                "aliases": [],
                "description": rep_text,
            }
        )

    # Write eval JSONL (normal + masked)
    kept_instances = 0
    with open(eval_path, "w", encoding="utf-8") as f_eval, open(
        masked_path, "w", encoding="utf-8"
    ) as f_masked:
        for key, members in kept_families.items():
            fam_id = key_to_family_id[key]
            for m in members:
                desc = m["description"]
                kept_instances += 1
                f_eval.write(
                    json.dumps(
                        {"query": desc, "problem_id": fam_id}, ensure_ascii=False
                    )
                    + "\n"
                )
                masked = _mask_description(desc)
                f_masked.write(
                    json.dumps(
                        {"query": masked, "problem_id": fam_id}, ensure_ascii=False
                    )
                    + "\n"
                )

    with open(catalog_path, "w", encoding="utf-8") as f_cat:
        json.dump(catalog, f_cat, indent=2, ensure_ascii=False)

    num_families = len(kept_families)
    avg_size = kept_instances / num_families if num_families else 0.0
    cov = {
        "total": total,
        "kept": kept_instances,
        "num_families": num_families,
        "avg_family_size": avg_size,
        "min_family_size": MIN_FAMILY_SIZE,
    }
    with open(cov_path, "w", encoding="utf-8") as f_cov:
        json.dump(cov, f_cov, indent=2)

    print(
        f"NL4Opt family eval: total={total}, kept={kept_instances}, "
        f"num_families={num_families}, avg_family_size={avg_size:.2f}"
    )
    print(f"Wrote {eval_path}, {masked_path}, and {catalog_path}")


if __name__ == "__main__":
    main()

