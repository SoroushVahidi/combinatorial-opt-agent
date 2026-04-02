from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_adapters.gams_models import GAMSModelsAdapter
from data_adapters.gurobi_modeling_examples import GurobiModelingExamplesAdapter
from data_adapters.gurobi_optimods import GurobiOptimodsAdapter
from data_adapters.cardinal_nl4opt import CardinalNL4OPTAdapter
from data_adapters.industryor import IndustryORAdapter
from data_adapters.mamo import MAMOAdapter
from data_adapters.miplib import MIPLIBAdapter
from data_adapters.nl4opt import NL4OptAdapter
from data_adapters.nlp4lp import NLP4LPAdapter
from data_adapters.optimus import OptiMUSAdapter
from data_adapters.structuredor import StructuredORAdapter
from data_adapters.or_library import ORLibraryAdapter
from data_adapters.pyomo_examples import PyomoExamplesAdapter
from data_adapters.registry import create_adapter, list_datasets
from data_adapters.text2zinc import Text2ZincAdapter
from scripts.build_expanded_schema_catalog import build_catalog, collect_schema_entries
from tools.run_dataset_benchmarks import evaluate_dataset

FIX = ROOT / "tests" / "fixtures" / "datasets"


def test_registry_contains_expected_datasets() -> None:
    names = list_datasets()
    assert "nlp4lp" in names
    assert "nl4opt" in names
    assert "text2zinc" in names
    assert "optmath" in names
    assert "complexor" in names
    assert "mamo" in names
    assert "structuredor" in names
    assert "cardinal_nl4opt" in names
    assert "industryor" in names
    # newly added adapters
    assert "gurobi_modeling_examples" in names
    assert "gurobi_optimods" in names
    assert "gams_models" in names
    assert "miplib" in names
    assert "or_library" in names
    assert "pyomo_examples" in names
    assert "optimus" in names
    assert create_adapter("nl4opt").name == "nl4opt"
    assert create_adapter("gurobi_modeling_examples").name == "gurobi_modeling_examples"


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



def test_mamo_adapter_conversion_from_fixture() -> None:
    ad = MAMOAdapter(data_root=FIX / "mamo")
    ex = ad.load_split("test")[0]
    ie = ad.to_internal_example(ex, "test")
    assert ie.source_dataset == "mamo"
    assert ie.schema_id == "production_planning"
    assert ie.scalar_gold_params is not None


def test_structuredor_adapter_conversion_from_fixture() -> None:
    ad = StructuredORAdapter(data_root=FIX / "structuredor")
    ex = ad.load_split("test")[0]
    ie = ad.to_internal_example(ex, "test")
    assert ie.source_dataset == "structuredor"
    assert ie.schema_id == "assignment"


def test_cardinal_nl4opt_adapter_conversion_from_fixture() -> None:
    ad = CardinalNL4OPTAdapter(data_root=FIX / "cardinal_nl4opt")
    ex = ad.load_split("test")[0]
    ie = ad.to_internal_example(ex, "test")
    assert ie.source_dataset == "cardinal_nl4opt"
    assert ie.schema_id == "transportation"


def test_industryor_adapter_conversion_from_fixture() -> None:
    ad = IndustryORAdapter(data_root=FIX / "industryor")
    ex = ad.load_split("test")[0]
    ie = ad.to_internal_example(ex, "test")
    assert ie.source_dataset == "industryor"
    assert ie.schema_id == "workforce_scheduling"



# ---------------------------------------------------------------------------
# Catalog-only adapter tests
# ---------------------------------------------------------------------------


def test_gurobi_modeling_examples_catalog_entries() -> None:
    ad = GurobiModelingExamplesAdapter()
    assert ad.list_splits() == ["catalog"]
    exs = ad.load_split("catalog")
    assert len(exs) > 0
    ie = ad.to_internal_example(exs[0], "catalog")
    assert ie.source_dataset == "gurobi_modeling_examples"
    assert ie.split == "catalog"
    assert ie.schema_id is not None
    assert ie.scalar_gold_params is None
    assert ie.formulation_text is None
    assert ie.metadata.get("catalog_only") is True


def test_gurobi_optimods_catalog_entries() -> None:
    ad = GurobiOptimodsAdapter()
    exs = ad.load_split("catalog")
    assert len(exs) > 0
    ie = ad.to_internal_example(exs[0], "catalog")
    assert ie.source_dataset == "gurobi_optimods"
    assert ie.metadata.get("catalog_only") is True


def test_gams_models_catalog_entries() -> None:
    ad = GAMSModelsAdapter()
    exs = ad.load_split("catalog")
    assert len(exs) > 0
    ie = ad.to_internal_example(exs[0], "catalog")
    assert ie.source_dataset == "gams_models"
    assert ie.metadata.get("catalog_only") is True


def test_miplib_catalog_entries() -> None:
    ad = MIPLIBAdapter()
    exs = ad.load_split("catalog")
    assert len(exs) > 0
    ie = ad.to_internal_example(exs[0], "catalog")
    assert ie.source_dataset == "miplib"
    assert ie.schema_id == "miplib_2017"
    assert ie.metadata.get("catalog_only") is True


def test_or_library_catalog_entries() -> None:
    ad = ORLibraryAdapter()
    exs = ad.load_split("catalog")
    assert len(exs) > 0
    ie = ad.to_internal_example(exs[0], "catalog")
    assert ie.source_dataset == "or_library"
    assert ie.metadata.get("catalog_only") is True


def test_pyomo_examples_catalog_entries() -> None:
    ad = PyomoExamplesAdapter()
    exs = ad.load_split("catalog")
    assert len(exs) > 0
    ie = ad.to_internal_example(exs[0], "catalog")
    assert ie.source_dataset == "pyomo_examples"
    assert ie.metadata.get("catalog_only") is True


def test_all_catalog_adapters_have_false_capabilities() -> None:
    catalog_adapters = [
        "gurobi_modeling_examples",
        "gurobi_optimods",
        "gams_models",
        "miplib",
        "or_library",
        "pyomo_examples",
    ]
    for name in catalog_adapters:
        ad = create_adapter(name)
        caps = ad.capabilities
        assert caps.supports_schema_retrieval is False, name
        assert caps.supports_scalar_instantiation is False, name
        assert caps.supports_solver_eval is False, name
        assert caps.supports_full_formulation is False, name


# ---------------------------------------------------------------------------
# OptiMUS adapter tests
# ---------------------------------------------------------------------------


def test_optimus_adapter_empty_when_no_data() -> None:
    ad = OptiMUSAdapter(data_root=ROOT / "data" / "external" / "optimus_nonexistent")
    assert ad.list_splits() == []


def test_optimus_adapter_loads_fixture(tmp_path: Path) -> None:
    src = FIX / "optimus" / "test.jsonl"
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "test.jsonl").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    ad = OptiMUSAdapter(data_root=tmp_path)
    assert ad.list_splits() == ["test"]
    exs = ad.load_split("test")
    assert len(exs) == 1
    ie = ad.to_internal_example(exs[0], "test")
    assert ie.source_dataset == "optimus"
    assert ie.schema_id == "transportation"
    assert ie.scalar_gold_params is not None
    assert ie.formulation_text is not None


def test_optimus_capabilities() -> None:
    ad = OptiMUSAdapter()
    assert ad.capabilities.supports_schema_retrieval is True
    assert ad.capabilities.supports_full_formulation is True


# ---------------------------------------------------------------------------
# Benchmark runner: catalog-only adapters produce N/A for all metrics
# ---------------------------------------------------------------------------


def test_runner_catalog_only_produces_na_metrics() -> None:
    rows = evaluate_dataset("gurobi_modeling_examples")
    assert rows
    row = rows[0]
    assert row["schema_target_coverage"] == "N/A"
    assert row["scalar_target_coverage"] == "N/A"
    assert row["formulation_target_coverage"] == "N/A"


# ---------------------------------------------------------------------------
# Expanded schema catalog build
# ---------------------------------------------------------------------------


def test_build_expanded_schema_catalog(tmp_path: Path) -> None:
    out = tmp_path / "catalog.jsonl"
    n = build_catalog(out)
    assert n > 0
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == n
    # Every row must have required fields
    for row in lines:
        assert "id" in row
        assert "source_dataset" in row
        assert "source_metadata" in row
        assert "entry_status" in row
        assert "benchmark_labeled" in row


def test_collect_schema_entries_gurobi() -> None:
    entries = collect_schema_entries("gurobi_modeling_examples")
    assert len(entries) > 50
    for e in entries:
        assert e["entry_status"] == "catalog-only"
        assert e["benchmark_labeled"] is False


def test_build_expanded_schema_catalog_contains_source_only_rows(tmp_path: Path) -> None:
    out = tmp_path / "catalog_source_only.jsonl"
    build_catalog(out)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert any(r.get("entry_status") == "source-only" for r in rows)
