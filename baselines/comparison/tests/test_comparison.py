"""Lightweight cross-baseline comparison-harness tests. No GPU/solver/network."""
from __future__ import annotations

import json

from baselines.comparison.adapters import adapt_deepor, adapt_optmath, adapt_orlm, adapt_orr1, adapt_ours, adapt_pamop
from baselines.comparison.availability import AVAILABILITY
from baselines.comparison.failure_taxonomy import to_top_level
from baselines.comparison.manifests import load_common_manifest, pamop_empirical_manifest_note, verify_baseline_manifests
from baselines.comparison.metrics import END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY, SHARED_METRICS
from baselines.comparison.pairing import pair_systems
from baselines.comparison.report import generate_report
from baselines.comparison.resource_profile import RESOURCE_PROFILES
from baselines.comparison.schema import CellState, UnifiedRow, is_measured, is_state
from baselines.comparison.statistics import build_transition_table, mcnemar_exact, wilson_interval
from baselines.comparison.validation import detect_ambiguous_runs, is_mock_evidence, select_rows, validate_row

OURS_ROW = {"query_id": "nlp4lp_test_13", "variant": "orig", "baseline": "tfidf", "predicted_doc_id": "nlp4lp_test_13",
            "gold_doc_id": "nlp4lp_test_13", "schema_hit": "1", "n_expected_scalar": "13", "n_filled": "12",
            "param_coverage": "0.9230769230769231", "type_match": "1.0", "exact5": "0.08333333333333333",
            "exact20": "0.08333333333333333", "key_overlap": "1.0"}

PAMOP_ROW = {"problem_id": "14", "lp_or_milp": "LP", "initial_generation_valid": "YES",
             "initial_ampl_parse_success": "True", "final_ampl_parse_success": "True",
             "correction_invoked": "False", "correction_iterations": "0", "final_solver_status": "solved",
             "final_feasible": "True", "objective_produced": "True", "objective_value": "480.0",
             "failure_category": "A. SUCCESS_NO_CORRECTION", "prompt_tokens": "2862", "completion_tokens": "1248",
             "total_tokens": "4110", "latency_seconds": "19.369827", "underlying_model": "gpt-5.4-2026-03-05",
             "deployment": "gpt-5.4", "gold_objective": "486.0", "objective_match_with_gold": "False"}

PAMOP_RUN_METADATA = {"git_commit": "0f0b24ed33e19eff1d43bc385802e6be97532987", "deployment": "gpt-5.4"}


def _orlm_style_record(*, mock: bool = False) -> dict:
    return {
        "problem_id": "14", "dataset": "nlp4lp", "raw_problem_text_sha256": "a" * 64,
        "prompt_version": "v1", "prompt_sha256": "b" * 64,
        "generation": {"raw_output": "x", "status": "COMPLETED", "model_id": "MOCK_PROXY" if mock else "CardinalOperations/ORLM-LLaMA-3-8B",
                       "model_revision": None, "prompt_sha256": "b" * 64, "runtime_seconds": 1.0,
                       "token_counts": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        "parsed": {"raw_output": "x", "model_description": "d", "coptpy_code": "import coptpy\nmodel=1\nmodel.solve()", "code_block_found": True},
        "static_validation": {"status": "STATIC_VALID"},
        "execution_attempted": True, "execution": {"status": "COMPLETED"},
        "gold_objective": 40.0, "objective_value": 40.0, "objective_proxy_status": "PASS",
        "error_category": None, "git_sha": None if mock else "deadbeef", "timestamp_utc": "2026-08-13T00:00:00Z",
    }


# --- schema -------------------------------------------------------------------

def test_cell_state_helpers():
    assert is_state(CellState.PENDING) and not is_state(1.0) and not is_state(True)
    assert is_measured(1.0) and is_measured(False) and not is_measured(None) and not is_measured(CellState.PENDING)


def test_unified_row_default_states_are_never_none():
    row = UnifiedRow(system="x", method_variant="y", problem_id="1", dataset="d", input_hash="h")
    for key, value in row.to_dict().items():
        if key in {"system", "method_variant", "problem_id", "dataset", "input_hash", "native_record", "native_metrics"}:
            continue
        if key.startswith(("full_formulation", "fixed_schema", "scalar_grounding_only", "generative", "test_time_learning", "transductive_training")):
            continue
        assert value is not None, key


# --- adapters -------------------------------------------------------------------

def test_adapt_ours_never_populates_shared_metric_fields():
    row = adapt_ours(OURS_ROW)
    assert row.system == "ours" and row.scalar_grounding_only is True and row.full_formulation is False
    assert row.execution_attempted == CellState.NOT_APPLICABLE
    assert row.native_metrics["schema_hit"] is True
    assert row.native_metrics["instantiation_ready"] is True  # current definition is Coverage/TypeMatch >= 0.8
    assert row.native_metrics["strict_instantiation_ready"] is True


def test_adapt_pamop_objective_proxy_not_semantic_correctness():
    row = adapt_pamop(PAMOP_ROW, run_metadata=PAMOP_RUN_METADATA)
    assert row.system == "pamop" and row.implementation_fidelity == "INDEPENDENT_RECONSTRUCTION"
    assert row.semantic_correct == CellState.PROXY  # never a bare True/False here
    assert row.objective_match is False  # 480 vs gold 486, exact-match proxy
    assert row.local_git_sha == "0f0b24ed33e19eff1d43bc385802e6be97532987"


def test_adapt_pamop_handles_missing_objective_gracefully():
    row = dict(PAMOP_ROW)
    row["objective_produced"] = "False"
    row["objective_value"] = ""
    row["objective_match_with_gold"] = ""
    unified = adapt_pamop(row, run_metadata=PAMOP_RUN_METADATA)
    assert unified.objective_available is False
    assert unified.objective_predicted == CellState.UNAVAILABLE
    assert unified.objective_match == CellState.NOT_APPLICABLE


def test_adapt_orlm_optmath_deepor_orr1_shared_shape():
    orlm = adapt_orlm(_orlm_style_record())
    optmath = adapt_optmath(_orlm_style_record())
    deepor = adapt_deepor(_orlm_style_record())
    orr1 = adapt_orr1({**_orlm_style_record(), "checkpoint_stage": "MERGED", "rollout_count": 8, "tgrpo_steps_applied": 0})
    assert orlm.system == "orlm" and orlm.official_checkpoint_used is True
    assert optmath.system == "optmath" and optmath.source_repo.endswith("OptMATH")
    assert deepor.system == "deepor" and deepor.official_code_used is False and deepor.test_time_learning is True
    assert orr1.system == "orr1" and orr1.transductive_training is True and orr1.rollout_count == 8
    assert orr1.official_code_used is True and orr1.official_checkpoint_used is False


def test_adapters_handle_missing_and_unknown_fields():
    sparse = {"problem_id": "1", "generation": {"status": "FAILED"}}
    row = adapt_orlm(sparse)
    assert row.parse_success is False
    assert row.execution_attempted is False  # not attempted is a measured fact, not NOT_APPLICABLE, when the field is simply absent/False
    assert row.objective_available is False
    assert row.checkpoint_model == CellState.UNAVAILABLE  # no model_id anywhere in a genuinely sparse record


# --- manifest ---------------------------------------------------------------

def test_common_manifest_loads_and_baseline_manifests_have_no_drift():
    common = load_common_manifest()
    assert common["pilot_ids"] == [14, 23, 34, 59, 69, 72]
    drift = verify_baseline_manifests(common)
    assert all(issues == [] for issues in drift.values()), drift


def test_pamop_manifest_divergence_is_documented_not_hidden():
    note = pamop_empirical_manifest_note()
    assert "14, 23, 34, 72, 84, 88" in note.replace("[", "").replace("]", "")
    assert "NOT" in note


# --- metrics / taxonomy -------------------------------------------------------

def test_ours_is_ineligible_for_end_to_end_objective_success():
    assert "INELIGIBLE" in END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY["ours"]


def test_orr1_eligibility_carries_transductive_caveat():
    assert "transductive" in END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY["orr1"].lower()


def test_shared_metrics_return_pending_state_on_no_rows():
    for metric in SHARED_METRICS:
        result = metric.compute([])
        assert result["n"] == 0 and result["rate"] == CellState.NOT_APPLICABLE


def test_shared_metrics_compute_on_real_pamop_rows():
    rows = [adapt_pamop(PAMOP_ROW, run_metadata=PAMOP_RUN_METADATA)]
    for metric in SHARED_METRICS:
        result = metric.compute(rows)
        assert result["n"] == 1


def test_failure_taxonomy_maps_known_categories_without_losing_native_detail():
    assert to_top_level("E. AMPL_PARSE_FAILURE") == "parse_failure"
    assert to_top_level("checkpoint_unavailable") == "unavailable_artifact"
    assert to_top_level(None) == "none"
    assert to_top_level("some_totally_unknown_category") == "evaluation_ambiguous"


# --- validation / mock exclusion / duplicate runs -----------------------------

def test_mock_evidence_is_detected_and_excluded():
    mock_row = adapt_orlm(_orlm_style_record(mock=True))
    assert is_mock_evidence(mock_row)
    accepted, rejected = select_rows([mock_row], allow_mock=False)
    assert accepted == [] and "mock_evidence" in list(rejected.values())[0]


def test_real_row_passes_validation():
    real_row = adapt_orlm(_orlm_style_record(mock=False))
    assert not is_mock_evidence(real_row)
    accepted, rejected = select_rows([real_row], allow_mock=False)
    assert len(accepted) == 1 and rejected == {}


def test_duplicate_run_ambiguity_is_rejected_not_resolved():
    row_a = adapt_orlm(_orlm_style_record(mock=False))
    row_b = adapt_orlm(_orlm_style_record(mock=False))
    row_b.checkpoint_revision = "different-revision"
    ambiguous = detect_ambiguous_runs([row_a, row_b])
    assert (row_a.system, row_a.problem_id) in ambiguous
    accepted, rejected = select_rows([row_a, row_b], allow_mock=False)
    assert accepted == []
    assert all("ambiguous_run_selection_required" in reasons for reasons in rejected.values())


def test_validate_row_flags_unknown_system():
    bad = UnifiedRow(system="not_a_real_system", method_variant="x", problem_id="1", dataset="d", input_hash="h", local_git_sha="abc")
    problems = validate_row(bad)
    assert any(p.startswith("unknown_system") for p in problems)


# --- pairing / statistics -----------------------------------------------------

def test_wilson_interval_bounds_and_edges():
    ci = wilson_interval(4, 5)
    assert 0.0 <= ci.lower <= ci.point_estimate <= ci.upper <= 1.0
    all_success = wilson_interval(5, 5)
    assert all_success.upper == 1.0
    all_fail = wilson_interval(0, 5)
    assert all_fail.lower == 0.0


def test_mcnemar_zero_discordant_pairs_is_undefined_not_fabricated():
    table = build_transition_table([(True, True), (False, False)])
    result = mcnemar_exact(table)
    assert result.p_value is None
    assert "undefined" in result.note


def test_mcnemar_small_sample_note():
    table = build_transition_table([(True, False), (False, True), (True, False)])
    result = mcnemar_exact(table)
    assert result.p_value is not None
    assert "small-sample" in result.note


def test_pair_systems_only_pairs_measured_booleans():
    a = [UnifiedRow(system="a", method_variant="v", problem_id="1", dataset="d", input_hash="h", execution_success=True),
         UnifiedRow(system="a", method_variant="v", problem_id="2", dataset="d", input_hash="h", execution_success=CellState.NOT_APPLICABLE)]
    b = [UnifiedRow(system="b", method_variant="v", problem_id="1", dataset="d", input_hash="h", execution_success=False),
         UnifiedRow(system="b", method_variant="v", problem_id="3", dataset="d", input_hash="h", execution_success=True)]
    result = pair_systems(a, b, metric=lambda r: r.execution_success if isinstance(r.execution_success, bool) else None, metric_name="execution_success")
    assert result.paired_problem_ids == ("1",)
    assert result.table.a_only == 1 and result.table.b_only == 0
    assert result.unpaired_b_only == ("3",)


# --- OR-R1 transductive flag ---------------------------------------------------

def test_orr1_row_always_carries_transductive_training_flag():
    row = adapt_orr1({**_orlm_style_record(), "checkpoint_stage": "MERGED"})
    assert row.transductive_training is True


# --- report generation: CSV/JSON/Markdown consistency --------------------------

def test_generate_report_never_fabricates_a_value_for_empty_systems(tmp_path):
    rows_by_system = {"ours": [adapt_ours(OURS_ROW)], "pamop": [adapt_pamop(PAMOP_ROW, run_metadata=PAMOP_RUN_METADATA)],
                       "orlm": [], "optmath": [], "deepor": [], "orr1": []}
    files = generate_report(tmp_path, rows_by_system, git_sha="testsha")
    shared_csv = files["shared_metrics.csv"].read_text(encoding="utf-8")
    assert "PENDING" in shared_csv
    assert "0.0000" not in shared_csv  # PENDING must never render as a numeric zero
    comparison = json.loads(files["comparison.json"].read_text(encoding="utf-8"))
    assert comparison["row_counts_by_system"]["orr1"] == 0
    assert "transductive" in files["comparison.md"].read_text(encoding="utf-8").lower()
    for system in ("orlm", "optmath", "deepor", "orr1"):
        assert system not in shared_csv.split("\n")[0]  # header sanity: file is well-formed, not asserting absence of the label itself


def test_availability_and_resource_profile_cover_all_six_systems():
    assert set(AVAILABILITY) == {"ours", "pamop", "orlm", "optmath", "deepor", "orr1"}
    assert set(RESOURCE_PROFILES) == {"ours", "pamop", "orlm", "optmath", "deepor", "orr1"}


# --- CLI smoke test -------------------------------------------------------------

def test_cli_validate_only_smoke(capsys, monkeypatch):
    import baselines.comparison.cli as cli_module
    monkeypatch.setattr(cli_module, "ingest_all", lambda systems=None, save_ours_subset_to=None: {
        "pamop": [adapt_pamop(PAMOP_ROW, run_metadata=PAMOP_RUN_METADATA)],
    })
    rc = cli_module.main(["--validate-only", "--systems", "pamop"])
    assert rc == 0
    assert "Validated 1 rows" in capsys.readouterr().out
