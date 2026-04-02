from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_adapters.nl4opt import NL4OptAdapter
from data_adapters.nlp4lp import NLP4LPAdapter
from data_adapters.registry import create_adapter, list_datasets
from data_adapters.text2zinc import Text2ZincAdapter
from tools.run_dataset_benchmarks import evaluate_dataset

FIX = ROOT / "tests" / "fixtures" / "datasets"


def test_registry_contains_expected_datasets() -> None:
    names = list_datasets()
    assert "nlp4lp" in names
    assert "nl4opt" in names
    assert "text2zinc" in names
    assert "optmath" in names
    assert "complexor" in names
    assert create_adapter("nl4opt").name == "nl4opt"


def test_nlp4lp_adapter_loads_local_temp_split(tmp_path: Path) -> None:
    (tmp_path / "processed").mkdir(parents=True)
    (tmp_path / "catalogs").mkdir(parents=True)
    with open(tmp_path / "processed" / "nlp4lp_eval_orig.jsonl", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"query_id": "q1", "query": "minimize cost", "relevant_doc_id": "facility_location"}
            )
            + "\n"
        )
    with open(tmp_path / "catalogs" / "nlp4lp_catalog.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"doc_id": "facility_location", "text": "Facility location model"}) + "\n")

    ad = NLP4LPAdapter(data_root=tmp_path)
    assert ad.list_splits() == ["orig"]
    row = ad.to_internal_example(ad.load_split("orig")[0], "orig")
    assert row.schema_id == "facility_location"
    assert row.source_dataset == "nlp4lp"


def test_nl4opt_adapter_conversion_from_fixture() -> None:
    ad = NL4OptAdapter(data_root=FIX / "nl4opt")
    ex = ad.load_split("test")[0]
    internal = ad.to_internal_example(ex, "test")
    assert internal.source_dataset == "nl4opt"
    assert internal.schema_id == "transportation"
    assert internal.scalar_gold_params is not None
    assert "num_plants" in internal.scalar_gold_params


def test_text2zinc_adapter_capability_and_conversion() -> None:
    ad = Text2ZincAdapter(data_root=FIX / "text2zinc")
    assert ad.capabilities.supports_scalar_instantiation is True
    assert ad.capabilities.supports_schema_retrieval is False
    ex = ad.load_split("test")[0]
    internal = ad.to_internal_example(ex, "test")
    assert internal.id == "t2z_1"
    assert internal.formulation_text == "var int: x;"


def test_runner_na_for_unsupported_metrics() -> None:
    # Use fixture-backed text2zinc adapter through registry by temporarily writing a local snapshot
    # into expected location.
    out_root = ROOT / "data" / "external" / "text2zinc"
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "test.jsonl"
    payload = (FIX / "text2zinc" / "test.jsonl").read_text(encoding="utf-8")
    target.write_text(payload, encoding="utf-8")
    rows = evaluate_dataset("text2zinc")
    assert rows
    row = next(r for r in rows if r["split"] == "test")
    assert row["schema_target_coverage"] == "N/A"
    assert row["scalar_target_coverage"] != "N/A"

