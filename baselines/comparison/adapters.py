"""Per-baseline native-record -> UnifiedRow adapters.

Each adapter accepts one native record (the same dict shape each baseline's
own `result_schema.py`/`to_dict()` -- or, for PaMOP, its `per_problem.csv`
row -- already produces) and returns one `UnifiedRow`. No native field is
discarded: everything lands in `native_record`; system-specific metrics not
covered by `SharedMetric` land in `native_metrics`. No metric is
reinterpreted as something stronger than it is (an objective-value proxy is
never promoted to `semantic_correct`).
"""
from __future__ import annotations

from typing import Any

from baselines.comparison.schema import CellState, UnifiedRow


def _bool_or(value: Any, default: Any = CellState.UNKNOWN) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "1"}:
            return True
        if value.strip().lower() in {"false", "no", "0"}:
            return False
        if value.strip() == "":
            return default
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return default
    return default


def _num_or(value: Any, default: Any = CellState.UNAVAILABLE) -> Any:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --- ours -------------------------------------------------------------------

def adapt_ours(row: dict[str, Any], *, method_variant: str = "tfidf_typed_greedy", dataset: str = "nlp4lp_orig") -> UnifiedRow:
    """From a `nlp4lp_downstream_per_query_*.csv` row (schema_hit/param_coverage/type_match/...)."""
    problem_id = str(row["query_id"])
    schema_hit = _bool_or(row.get("schema_hit"))
    instantiation_ready = (
        _num_or(row.get("param_coverage"), 0.0) >= 0.8
        and _num_or(row.get("type_match"), 0.0) >= 0.8
    )
    strict_instantiation_ready = (
        schema_hit is True
        and instantiation_ready
    )
    return UnifiedRow(
        system="ours", method_variant=method_variant, problem_id=problem_id, dataset=dataset,
        input_hash=CellState.UNKNOWN,
        implementation_fidelity="NATIVE_METHOD",
        official_code_used=CellState.NOT_APPLICABLE, official_checkpoint_used=CellState.NOT_APPLICABLE,
        checkpoint_model=CellState.NOT_APPLICABLE, source_repo=CellState.NOT_APPLICABLE,
        generation_attempted=CellState.NOT_APPLICABLE, generation_completed=CellState.NOT_APPLICABLE,
        parse_success=CellState.NOT_APPLICABLE, static_valid=CellState.NOT_APPLICABLE,
        execution_attempted=CellState.NOT_APPLICABLE, execution_success=CellState.NOT_APPLICABLE,
        feasible=CellState.NOT_APPLICABLE, bounded=CellState.NOT_APPLICABLE, solver_status=CellState.NOT_APPLICABLE,
        objective_available=CellState.NOT_APPLICABLE, objective_predicted=CellState.NOT_APPLICABLE,
        objective_gold=CellState.NOT_APPLICABLE, objective_match=CellState.NOT_APPLICABLE,
        objective_tolerance=CellState.NOT_APPLICABLE,
        semantic_correct=CellState.NOT_APPLICABLE, semantic_metric_available=False,
        correctness_metric_name="instantiation_ready",
        runtime_seconds=CellState.NOT_APPLICABLE, prompt_tokens=CellState.NOT_APPLICABLE,
        generated_tokens=CellState.NOT_APPLICABLE, total_tokens=CellState.NOT_APPLICABLE,
        rollout_count=CellState.NOT_APPLICABLE, correction_iterations=CellState.NOT_APPLICABLE,
        test_time_training_steps=CellState.NOT_APPLICABLE, estimated_cost=0.0,
        failure_category=CellState.NOT_APPLICABLE, failure_detail=CellState.NOT_APPLICABLE,
        full_formulation=False, fixed_schema=True, scalar_grounding_only=True,
        generative=False, test_time_learning=False, transductive_training=False,
        native_record=dict(row),
        native_metrics={
            "schema_hit": schema_hit,
            "param_coverage": _num_or(row.get("param_coverage")),
            "type_match": _num_or(row.get("type_match")),
            "exact5": _num_or(row.get("exact5"), CellState.NOT_APPLICABLE),
            "exact20": _num_or(row.get("exact20"), CellState.NOT_APPLICABLE),
            "key_overlap": _num_or(row.get("key_overlap"), CellState.NOT_APPLICABLE),
            "instantiation_ready": instantiation_ready,
            "strict_instantiation_ready": strict_instantiation_ready,
        },
    )


# --- PaMOP --------------------------------------------------------------------

def adapt_pamop(row: dict[str, Any], *, run_metadata: dict[str, Any] | None = None, dataset: str = "nlp4lp") -> UnifiedRow:
    """From a `results/pamop/*/per_problem.csv` row plus its sibling `run_metadata.json`."""
    run_metadata = run_metadata or {}
    objective_available = _bool_or(row.get("objective_produced"), False)
    objective_match_raw = row.get("objective_match_with_gold")
    objective_match = _bool_or(objective_match_raw, CellState.NOT_APPLICABLE) if objective_match_raw not in (None, "") else CellState.NOT_APPLICABLE
    execution_success = _bool_or(row.get("final_ampl_parse_success"), False)
    feasible = _bool_or(row.get("final_feasible"), CellState.NOT_APPLICABLE)
    return UnifiedRow(
        system="pamop", method_variant=f"pamop_reconstruction_{run_metadata.get('deployment', 'unknown')}",
        problem_id=str(row["problem_id"]), dataset=dataset,
        input_hash=CellState.UNKNOWN,  # per_problem.csv records prompt/AMPL hashes, not a raw-input hash.
        implementation_fidelity="INDEPENDENT_RECONSTRUCTION",
        official_code_used=False, official_checkpoint_used=CellState.NOT_APPLICABLE,
        checkpoint_model=row.get("underlying_model", CellState.UNKNOWN),
        checkpoint_revision=row.get("deployment", CellState.UNKNOWN),
        source_repo=CellState.NOT_APPLICABLE, source_repo_revision=CellState.NOT_APPLICABLE,
        local_git_sha=run_metadata.get("git_commit", CellState.UNKNOWN),
        generation_attempted=True, generation_completed=_bool_or(row.get("initial_generation_valid"), False),
        parse_success=execution_success, static_valid=CellState.NOT_APPLICABLE,
        execution_attempted=True, execution_success=execution_success,
        feasible=feasible, bounded=CellState.UNKNOWN,
        solver_status=row.get("final_solver_status") or CellState.UNAVAILABLE,
        objective_available=objective_available,
        objective_predicted=_num_or(row.get("objective_value")) if objective_available else CellState.UNAVAILABLE,
        objective_gold=_num_or(row.get("gold_objective"), CellState.UNAVAILABLE),
        objective_match=objective_match, objective_tolerance="exact_match",
        semantic_correct=CellState.PROXY,  # PaMOP's own "semantic_correctness_status" is an exact-objective-match proxy, not a structural judgment.
        semantic_metric_available=False,
        correctness_metric_name="objective_value_exact_match_proxy",
        runtime_seconds=_num_or(row.get("latency_seconds")),
        prompt_tokens=_num_or(row.get("prompt_tokens")), generated_tokens=_num_or(row.get("completion_tokens")),
        total_tokens=_num_or(row.get("total_tokens")), rollout_count=1,
        correction_iterations=_num_or(row.get("correction_iterations"), 0),
        test_time_training_steps=CellState.NOT_APPLICABLE, estimated_cost=CellState.UNKNOWN,
        failure_category=row.get("failure_category") or CellState.NOT_APPLICABLE,
        failure_detail=CellState.NOT_APPLICABLE,
        full_formulation=True, fixed_schema=False, scalar_grounding_only=False,
        generative=True, test_time_learning=False, transductive_training=False,
        native_record=dict(row),
        native_metrics={"correction_invoked": _bool_or(row.get("correction_invoked"), False), "lp_or_milp": row.get("lp_or_milp")},
    )


# --- ORLM / OptMATH / DeepOR / OR-R1 (shared shape family) --------------------

def _adapt_coptpy_gurobi_family(
    record: dict[str, Any], *, system: str, method_variant: str, fidelity: str,
    official_code_used: bool, official_checkpoint_used: Any, source_repo: Any, source_repo_revision: Any,
    scope_kwargs: dict[str, Any],
) -> UnifiedRow:
    generation = record.get("generation", {}) or {}
    parsed = record.get("parsed") or {}
    static = record.get("static_validation") or {}
    execution = record.get("execution") or {}
    execution_attempted = bool(record.get("execution_attempted", False))
    exec_status = str(execution.get("status", ""))
    code_key = next((k for k in parsed if "code" in k.lower()), None)
    parse_success = bool(parsed.get(code_key)) if code_key else False
    return UnifiedRow(
        system=system, method_variant=method_variant, problem_id=str(record.get("problem_id", CellState.UNKNOWN)),
        dataset=str(record.get("dataset", "nlp4lp")),
        input_hash=record.get("input_sha256") or record.get("raw_problem_text_sha256") or CellState.UNKNOWN,
        implementation_fidelity=fidelity,
        official_code_used=official_code_used, official_checkpoint_used=official_checkpoint_used,
        checkpoint_model=record.get("model_id") or record.get("checkpoint") or generation.get("model_id") or CellState.UNAVAILABLE,
        checkpoint_revision=record.get("model_revision") or record.get("checkpoint_revision") or generation.get("model_revision") or CellState.NOT_APPLICABLE,
        source_repo=source_repo, source_repo_revision=source_repo_revision,
        local_git_sha=record.get("git_sha", CellState.UNKNOWN),
        generation_attempted=True, generation_completed=str(generation.get("status")) == "COMPLETED",
        parse_success=parse_success,
        static_valid=(str(static.get("status")) == "STATIC_VALID") if static else CellState.NOT_APPLICABLE,
        execution_attempted=execution_attempted,
        execution_success=(exec_status.startswith("COMPLETED")) if execution_attempted else CellState.NOT_APPLICABLE,
        feasible=("INFEASIBLE" not in exec_status and "UNBOUNDED" not in exec_status) if execution_attempted and exec_status else CellState.NOT_APPLICABLE,
        bounded=("UNBOUNDED" not in exec_status) if execution_attempted and exec_status else CellState.NOT_APPLICABLE,
        solver_status=exec_status or CellState.NOT_APPLICABLE,
        objective_available=record.get("objective") is not None or record.get("objective_value") is not None,
        objective_predicted=record.get("objective", record.get("objective_value")) if (record.get("objective") is not None or record.get("objective_value") is not None) else CellState.UNAVAILABLE,
        objective_gold=record.get("gold_objective", CellState.UNAVAILABLE),
        objective_match=(record.get("objective_proxy_status") == "PASS") if record.get("objective_proxy_status") in ("PASS", "FAIL") else CellState.NOT_APPLICABLE,
        objective_tolerance=0.05,
        semantic_correct=CellState.NOT_APPLICABLE, semantic_metric_available=False,
        correctness_metric_name="objective_value_tolerance_proxy",
        runtime_seconds=generation.get("runtime_seconds", CellState.NOT_APPLICABLE),
        prompt_tokens=(generation.get("token_counts") or {}).get("prompt_tokens", CellState.NOT_APPLICABLE),
        generated_tokens=(generation.get("token_counts") or {}).get("generated_tokens") or (generation.get("token_counts") or {}).get("completion_tokens", CellState.NOT_APPLICABLE),
        total_tokens=(generation.get("token_counts") or {}).get("total_tokens", CellState.NOT_APPLICABLE),
        rollout_count=record.get("rollout_count", 1),
        correction_iterations=CellState.NOT_APPLICABLE,
        test_time_training_steps=record.get("tgrpo_steps_applied", CellState.NOT_APPLICABLE),
        estimated_cost=CellState.UNKNOWN,
        failure_category=record.get("failure_category") or record.get("error_category") or CellState.NOT_APPLICABLE,
        failure_detail=CellState.NOT_APPLICABLE,
        native_record=dict(record),
        native_metrics={"objective_proxy_status": record.get("objective_proxy_status")},
        **scope_kwargs,
    )


def adapt_orlm(record: dict[str, Any]) -> UnifiedRow:
    return _adapt_coptpy_gurobi_family(
        record, system="orlm", method_variant="orlm_llama3_8b", fidelity="ADAPTED_OFFICIAL",
        official_code_used=True, official_checkpoint_used=True,
        source_repo="https://github.com/Cardinal-Operations/ORLM",
        source_repo_revision="33bc47d0a1d1710d24ab839118bdf4cb89b9e31b",
        scope_kwargs=dict(full_formulation=True, fixed_schema=False, scalar_grounding_only=False,
                           generative=True, test_time_learning=False, transductive_training=False),
    )


def adapt_optmath(record: dict[str, Any]) -> UnifiedRow:
    return _adapt_coptpy_gurobi_family(
        record, system="optmath", method_variant="optmath_qwen25_7b", fidelity="ADAPTED_OFFICIAL",
        official_code_used=True, official_checkpoint_used=True,
        source_repo="https://github.com/optsuite/OptMATH",
        source_repo_revision="f15bbc4477c70db85ad148df8bcc1b780bca0f8c",
        scope_kwargs=dict(full_formulation=True, fixed_schema=False, scalar_grounding_only=False,
                           generative=True, test_time_learning=False, transductive_training=False),
    )


def adapt_deepor(record: dict[str, Any]) -> UnifiedRow:
    return _adapt_coptpy_gurobi_family(
        record, system="deepor", method_variant="deepor_paper_reconstruction", fidelity="PAPER_RECONSTRUCTED",
        official_code_used=False, official_checkpoint_used=False,
        source_repo=CellState.UNAVAILABLE, source_repo_revision=CellState.NOT_APPLICABLE,
        scope_kwargs=dict(full_formulation=True, fixed_schema=False, scalar_grounding_only=False,
                           generative=True, test_time_learning=True, transductive_training=False),
    )


def adapt_orr1(record: dict[str, Any]) -> UnifiedRow:
    row = _adapt_coptpy_gurobi_family(
        record, system="orr1", method_variant=f"orr1_{record.get('checkpoint_stage', 'unknown')}", fidelity="ADAPTED_OFFICIAL",
        official_code_used=True, official_checkpoint_used=False,
        source_repo="https://github.com/SCUTE-ZZ/OR-R1",
        source_repo_revision="9de48e3b22555e729ec032e7efd00ebaaa8e78d5",
        scope_kwargs=dict(full_formulation=True, fixed_schema=False, scalar_grounding_only=False,
                           generative=True, test_time_learning=True, transductive_training=True),
    )
    row.rollout_count = record.get("rollout_count", CellState.NOT_APPLICABLE)
    row.correctness_metric_name = "pass_at_k_or_mj_at_k"
    return row
