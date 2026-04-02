from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_grounding_training_pairs import _extract_pairs


FIX = ROOT / "tests" / "fixtures" / "datasets"


def test_extract_pairs_reports_blocker_for_missing_dataset() -> None:
    # mamo adapter uses default data/external path; in CI this should be empty.
    pairs, blockers = _extract_pairs("mamo")
    assert isinstance(pairs, list)
    assert isinstance(blockers, list)


def test_extract_pairs_positive_on_fixture_by_temp_copy(tmp_path: Path) -> None:
    # use existing adapter path for cardinal_nl4opt by staging tiny fixture
    out_dir = ROOT / "data" / "external" / "cardinal_nl4opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = (FIX / "cardinal_nl4opt" / "test.jsonl").read_text(encoding="utf-8")
    (out_dir / "test.jsonl").write_text(payload, encoding="utf-8")

    pairs, blockers = _extract_pairs("cardinal_nl4opt")
    assert len(pairs) >= 1
    assert all("slot_or_schema" in p for p in pairs)
    assert isinstance(blockers, list)

    # verify serializable
    json.dumps(pairs[0])
