"""Run the controlled PaMOP pilot benchmark slice.

This script intentionally writes only non-gated metadata and aggregate
execution traces. NLP4LP problem text is loaded only in memory for prompts.
By default it refuses to execute the benchmark outside Slurm; use
``--select-only`` to construct the deterministic slice without running LLMs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.pamop.ampl.executor import AmplExecutor
from baselines.pamop.ampl.renderer import render_merged_model
from baselines.pamop.config import load_config, reconstructed_default_path
from baselines.pamop.correction import CorrectionTrace, run_correction_loop
from baselines.pamop.data import (
    DATASET_ID,
    SUBSET_POSSIBLE_269,
    MissingStructuredDataError,
    _get_hf_token,
    list_ids_for_subset,
    load_alignment_manifest,
    load_problem_record,
)
from baselines.pamop.extraction import ExtractionValidationError, extract_structured_problem
from baselines.pamop.llm.registry import get_provider
from baselines.pamop.llm.types import LLMResponse, ProviderAuthError, ProviderCallError
from baselines.pamop.modeling import ModelingValidationError, build_merged_model
from baselines.pamop.partition import build_partition_tree

DEFAULT_OUT = ROOT / "results" / "pamop" / "pilot"
KNOWN_MISSING_STRUCTURED = {28, 51, 57, 123, 126, 135}
FAILURE_CATEGORIES = [
    "A. SUCCESS_NO_CORRECTION",
    "B. SUCCESS_AFTER_CORRECTION",
    "C. MODEL_PARSE_FAILURE",
    "D. AMPL_RENDER_FAILURE",
    "E. AMPL_PARSE_FAILURE",
    "F. SOLVER_INFEASIBLE",
    "G. SOLVER_UNBOUNDED",
    "H. SOLVER_RUNTIME_ERROR",
    "I. CORRECTION_EXHAUSTED",
    "J. DATA_FAILURE",
    "K. ENVIRONMENT_FAILURE",
    "L. OTHER_MODEL_FAILURE",
]


@dataclass(frozen=True)
class SliceMeta:
    problem_id: int
    doc_id: str
    lp_or_milp: str
    objective_sense: str
    variable_count: int
    parameter_count: int
    constraint_count: int
    numeric_mentions: int | None
    partition_node_count: int
    partition_depth: int
    family_proxy: str
    optimus_code_available: bool
    selection_bucket: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()


def ensure_artifact_headers(out_dir: Path) -> None:
    ensure_csv_header(out_dir / "per_problem.csv", PER_FIELDS)
    ensure_csv_header(out_dir / "failure_analysis.csv", FAIL_FIELDS)
    ensure_csv_header(out_dir / "correction_analysis.csv", CORR_FIELDS)


def read_numeric_mentions() -> dict[int, int]:
    path = ROOT / "results" / "eswa_revision" / "16_error_analysis" / "per_instance_diagnostics.csv"
    out: dict[int, int] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("variant") == "orig" and row.get("method") == "tfidf":
                try:
                    idx = int(row["query_id"].split("_")[-1])
                    out[idx + 1] = int(row["n_numeric_mentions"])
                except (KeyError, ValueError):
                    continue
    return out


def manifest_by_problem_id() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in load_alignment_manifest():
        pid = row.get("current_nlp4lp_hf_problem_id")
        if isinstance(pid, int):
            out[pid] = row
    return out


def download_hf_file(problem_id: int, filename: str) -> Path | None:
    token = _get_hf_token() or None
    try:
        return Path(
            hf_hub_download(
                DATASET_ID,
                f"data/{problem_id}/{filename}",
                repo_type="dataset",
                token=token,
            )
        )
    except Exception:
        return None


def read_problem_type(problem_id: int, record: dict[str, Any]) -> str:
    path = download_hf_file(problem_id, "optimus-code.py")
    if path and path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")[:1000]
        match = re.search(r"Problem type:\s*([A-Za-z0-9_ -]+)", text)
        if match:
            raw = match.group(1).strip().upper()
            if raw in {"MIP", "MILP", "IP"}:
                return "MILP"
            if raw == "LP":
                return "LP"
            return raw
    types = {str((v or {}).get("type") or "").lower() for v in (record.get("variables") or {}).values()}
    return "MILP" if types & {"integer", "binary", "int"} else "LP"


def objective_sense(record: dict[str, Any]) -> str:
    text = " ".join(
        str((record.get("objective") or {}).get(k, ""))
        for k in ("description", "formulation", "code")
    ).lower()
    if "min" in text:
        return "minimize"
    if "max" in text:
        return "maximize"
    return "unknown"


def tree_shape(problem_id: int, record: dict[str, Any], config: Any) -> tuple[int, int]:
    from baselines.pamop.representations import from_nlp4lp_record

    structured = from_nlp4lp_record(str(problem_id), record)
    tree = build_partition_tree(structured, config)
    return len(tree.nodes), tree.max_depth()


def collect_slice_metadata() -> list[SliceMeta]:
    config = load_config(reconstructed_default_path())
    numeric_mentions = read_numeric_mentions()
    manifest = manifest_by_problem_id()
    rows: list[SliceMeta] = []
    for problem_id in list_ids_for_subset(SUBSET_POSSIBLE_269):
        if problem_id in KNOWN_MISSING_STRUCTURED:
            continue
        record = load_problem_record(problem_id)
        lp_or_milp = read_problem_type(problem_id, record)
        sense = objective_sense(record)
        variables = record.get("variables") or {}
        parameters = record.get("parameters") or {}
        constraints = record.get("constraints") or []
        node_count, depth = tree_shape(problem_id, record, config)
        n_mentions = numeric_mentions.get(problem_id)
        family_proxy = (
            f"{lp_or_milp}|{sense}|v{len(variables)}|p{len(parameters)}|"
            f"c{len(constraints)}|tree{node_count}x{depth}"
        )
        rows.append(
            SliceMeta(
                problem_id=problem_id,
                doc_id=manifest.get(problem_id, {}).get("current_nlp4lp_catalog_doc_id", f"nlp4lp_test_{problem_id - 1}"),
                lp_or_milp=lp_or_milp,
                objective_sense=sense,
                variable_count=len(variables),
                parameter_count=len(parameters),
                constraint_count=len(constraints),
                numeric_mentions=n_mentions,
                partition_node_count=node_count,
                partition_depth=depth,
                family_proxy=family_proxy,
                optimus_code_available=bool(download_hf_file(problem_id, "optimus-code.py")),
                selection_bucket="",
            )
        )
    return rows


def stable_score(meta: SliceMeta) -> str:
    return sha256_text(f"pamop-pilot-v1:{meta.problem_id}:{meta.family_proxy}")


def with_bucket(meta: SliceMeta, all_rows: list[SliceMeta]) -> SliceMeta:
    c_values = sorted(r.constraint_count for r in all_rows)
    n_values = sorted(r.numeric_mentions or 0 for r in all_rows)
    node_values = sorted(r.partition_node_count for r in all_rows)
    c_mid = c_values[len(c_values) // 2]
    n_mid = n_values[len(n_values) // 2]
    node_mid = node_values[len(node_values) // 2]
    bucket = "|".join(
        [
            meta.lp_or_milp,
            "multi_constraint" if meta.constraint_count > c_mid else "simple",
            "high_numeric" if (meta.numeric_mentions or 0) > n_mid else "low_numeric",
            "large_tree" if meta.partition_node_count > node_mid else "small_tree",
            meta.objective_sense,
        ]
    )
    return SliceMeta(**{**asdict(meta), "selection_bucket": bucket})


def select_slice(target: int = 18) -> list[SliceMeta]:
    raw_rows = collect_slice_metadata()
    rows = [with_bucket(r, raw_rows) for r in raw_rows]

    by_bucket: dict[str, list[SliceMeta]] = {}
    for row in rows:
        by_bucket.setdefault(row.selection_bucket, []).append(row)
    selected: list[SliceMeta] = []
    used: set[int] = set()
    for bucket in sorted(by_bucket):
        candidate = sorted(by_bucket[bucket], key=lambda r: (stable_score(r), r.problem_id))[0]
        selected.append(candidate)
        used.add(candidate.problem_id)
        if len(selected) >= target:
            break

    def add_extreme(candidates: list[SliceMeta]) -> None:
        for candidate in candidates:
            if len(selected) >= target:
                return
            if candidate.problem_id not in used:
                selected.append(candidate)
                used.add(candidate.problem_id)

    # Force explicit coverage of high-count/tree and both LP/MILP, without
    # cherry-picking by execution outcome (none has been run yet).
    add_extreme(sorted(rows, key=lambda r: (-(r.numeric_mentions or 0), stable_score(r))))
    add_extreme(sorted(rows, key=lambda r: (-r.partition_node_count, -r.partition_depth, stable_score(r))))
    add_extreme(sorted([r for r in rows if r.lp_or_milp == "LP"], key=lambda r: stable_score(r)))
    add_extreme(sorted([r for r in rows if r.lp_or_milp == "MILP"], key=lambda r: stable_score(r)))
    add_extreme(sorted(rows, key=lambda r: stable_score(r)))
    return sorted(selected[:target], key=lambda r: r.problem_id)


def write_selected_ids(out_dir: Path, selected: list[SliceMeta]) -> None:
    payload = {
        "subset": SUBSET_POSSIBLE_269,
        "excluded_missing_structured_data_ids": sorted(KNOWN_MISSING_STRUCTURED),
        "selection_algorithm": (
            "Deterministic pamop-pilot-v1 stratified bucket pass over accessible "
            "ids 1-269, using LP/MILP, simple/multi-constraint, low/high numeric "
            "mentions, small/large partition tree, objective sense, then stable "
            "SHA-256 tie-breaking. No execution outcomes are used."
        ),
        "selected_count": len(selected),
        "selected_ids": [m.problem_id for m in selected],
        "problems": [asdict(m) for m in selected],
    }
    safe_write_json(out_dir / "selected_ids.json", payload)


def load_or_select_slice(out_dir: Path, target: int) -> list[SliceMeta]:
    selected_path = out_dir / "selected_ids.json"
    if not selected_path.exists():
        selected = select_slice(target)
        write_selected_ids(out_dir, selected)
        return selected
    data = json.loads(selected_path.read_text(encoding="utf-8"))
    problems = data.get("problems") or []
    if not problems:
        raise RuntimeError(f"{selected_path} exists but contains no selected problem metadata")
    selected = [SliceMeta(**item) for item in problems]
    ids = [m.problem_id for m in selected]
    recorded_ids = data.get("selected_ids")
    if recorded_ids and ids != recorded_ids:
        raise RuntimeError(f"{selected_path} selected_ids do not match problems metadata")
    return selected


def llm_usage(responses: list[LLMResponse]) -> dict[str, Any]:
    def total(attr: str) -> int:
        return sum(int(getattr(r, attr) or 0) for r in responses)

    return {
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "latency_seconds": sum(float(r.latency_seconds or 0.0) for r in responses),
        "prompt_hashes": sorted({r.prompt_hash for r in responses if r.prompt_hash}),
        "provider": responses[0].provider if responses else "",
        "deployment": responses[0].model if responses else "",
        "underlying_model": responses[0].underlying_model if responses else "",
    }


def correction_responses(trace: CorrectionTrace) -> list[LLMResponse]:
    responses: list[LLMResponse] = []
    for item in trace.iterations:
        for obj in (item.review, item.comparison, item.remodel):
            if obj is not None:
                responses.append(obj.llm_response)
    return responses


def prompt_hashes_for_problem(extraction: LLMResponse, model_responses: list[LLMResponse], trace: CorrectionTrace) -> list[str]:
    hashes = [extraction.prompt_hash]
    hashes.extend(r.prompt_hash for r in model_responses)
    hashes.extend(r.prompt_hash for r in correction_responses(trace))
    return sorted({h for h in hashes if h})


def map_failure_category(trace: CorrectionTrace | None, exc: Exception | None, render_valid: bool | None = None) -> str:
    if exc is not None:
        if isinstance(exc, MissingStructuredDataError):
            return "J. DATA_FAILURE"
        if isinstance(exc, ProviderAuthError):
            return "K. ENVIRONMENT_FAILURE"
        if isinstance(exc, ProviderCallError):
            return "K. ENVIRONMENT_FAILURE"
        if isinstance(exc, ExtractionValidationError):
            return "C. MODEL_PARSE_FAILURE"
        if isinstance(exc, ModelingValidationError):
            return "L. OTHER_MODEL_FAILURE"
        return "L. OTHER_MODEL_FAILURE"
    if render_valid is False:
        return "D. AMPL_RENDER_FAILURE"
    if trace is None:
        return "L. OTHER_MODEL_FAILURE"
    initial = trace.iterations[0].execution if trace.iterations else None
    final = trace.iterations[-1].execution if trace.iterations else None
    if trace.final_success:
        return "A. SUCCESS_NO_CORRECTION" if trace.correction_iterations_observed == 0 else "B. SUCCESS_AFTER_CORRECTION"
    if final is None:
        return "L. OTHER_MODEL_FAILURE"
    status = (final.solver_status or "").lower()
    if final.error_category.value == "environment_error":
        return "K. ENVIRONMENT_FAILURE"
    if final.error_category.value == "data_error":
        return "J. DATA_FAILURE"
    if not final.parse_success:
        return "E. AMPL_PARSE_FAILURE"
    if "infeasible" in status:
        return "F. SOLVER_INFEASIBLE"
    if "unbounded" in status:
        return "G. SOLVER_UNBOUNDED"
    if final.solver_invocation_success is False and initial is not None:
        return "H. SOLVER_RUNTIME_ERROR"
    if trace.stopped_reason == "max_correction_iterations":
        return "I. CORRECTION_EXHAUSTED"
    return "L. OTHER_MODEL_FAILURE"


def run_gold_model(problem_id: int, timeout_seconds: int = 60) -> dict[str, Any]:
    code = download_hf_file(problem_id, "optimus-code.py")
    params = download_hf_file(problem_id, "parameters.json")
    if not code or not params:
        return {"gold_comparison_eligible": False, "gold_reason": "missing optimus-code.py or parameters.json"}
    python_exe = os.environ.get("PAMOP_GOLD_PYTHON") or os.environ.get("PAMOP_AMPLPY_PYTHON") or sys.executable
    with tempfile.TemporaryDirectory(prefix=f"pamop_gold_{problem_id}_") as tmp:
        tmpdir = Path(tmp)
        shutil.copy2(code, tmpdir / "optimus-code.py")
        shutil.copy2(params, tmpdir / "parameters.json")
        start = time.monotonic()
        proc = subprocess.run(
            [python_exe, "optimus-code.py"],
            cwd=tmpdir,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        latency = time.monotonic() - start
        solution_path = tmpdir / "solution.json"
        objective = None
        if solution_path.exists():
            try:
                objective = json.loads(solution_path.read_text(encoding="utf-8")).get("objective")
            except json.JSONDecodeError:
                objective = None
        return {
            "gold_comparison_eligible": True,
            "gold_returncode": proc.returncode,
            "gold_solver_success": proc.returncode == 0 and objective is not None,
            "gold_feasible": proc.returncode == 0 and objective is not None,
            "gold_objective": objective,
            "gold_latency_seconds": round(latency, 6),
            "gold_reason": "" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-500:],
        }


PER_FIELDS = [
    "problem_id",
    "lp_or_milp",
    "partition_node_count",
    "partition_depth",
    "initial_generation_valid",
    "initial_ampl_parse_success",
    "initial_solver_success",
    "initial_feasible_status",
    "correction_invoked",
    "correction_iterations",
    "final_ampl_parse_success",
    "final_solver_status",
    "final_feasible",
    "objective_produced",
    "objective_value",
    "failure_category",
    "total_llm_calls",
    "g_extr_calls",
    "g_mod_calls",
    "correction_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_seconds",
    "estimated_cost_usd",
    "cost_status",
    "provider",
    "deployment",
    "underlying_model",
    "temperature",
    "prompt_hashes",
    "final_ampl_hash",
    "gold_comparison_eligible",
    "gold_solver_success",
    "gold_feasible",
    "gold_objective",
    "objective_match_with_gold",
    "semantic_correctness_status",
    "semantically_wrong_but_feasible",
]

FAIL_FIELDS = ["problem_id", "failure_category", "primary_reason", "issue_tags", "diagnostic_codes"]
CORR_FIELDS = [
    "problem_id",
    "initial_success",
    "final_success",
    "rescued_by_correction",
    "harmed_by_correction",
    "correction_iterations",
    "correction_calls",
    "correction_tokens",
    "correction_latency_seconds",
]
OURS_FIELDS = [
    "problem_id",
    "doc_id",
    "our_method",
    "our_external_llm_calls",
    "our_llm_tokens",
    "our_api_cost",
    "our_deterministic",
    "our_inst_ready",
    "our_param_coverage",
    "our_type_match",
    "pamop_failure_category",
    "pamop_total_llm_calls",
    "pamop_total_tokens",
    "pamop_final_feasible",
    "common_metric_status",
]


def diagnostic_tags(trace: CorrectionTrace | None) -> tuple[str, str, str]:
    if trace is None or not trace.iterations:
        return "", "", ""
    diagnostics = trace.iterations[-1].execution.diagnostics
    codes = sorted({d.code for d in diagnostics})
    messages = " ".join(d.message.lower() for d in diagnostics)
    tags = []
    for marker, tag in [
        ("not defined", "hallucinated_or_missing_symbol"),
        ("already defined", "duplicated_symbol"),
        ("syntax", "incorrect_ampl_syntax"),
        ("infeasible", "solver_infeasibility"),
        ("unbounded", "solver_unbounded"),
        ("timeout", "solver_runtime_error"),
        ("license", "environment_solver_license"),
    ]:
        if marker in messages:
            tags.append(tag)
    return ";".join(tags), ";".join(codes), (diagnostics[-1].message[:500] if diagnostics else "")


def load_ours_rows() -> dict[int, dict[str, str]]:
    path = ROOT / "results" / "eswa_revision" / "16_error_analysis" / "per_instance_diagnostics.csv"
    out: dict[int, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("variant") == "orig" and row.get("method") == "tfidf":
                try:
                    out[int(row["query_id"].split("_")[-1]) + 1] = row
                except ValueError:
                    pass
    return out


def run_problem(meta: SliceMeta, out_dir: Path, args: argparse.Namespace, config: Any, provider: Any) -> dict[str, Any]:
    start = time.monotonic()
    trace = None
    exc = None
    render_valid = None
    responses: list[LLMResponse] = []
    g_mod_calls = 0
    gold = run_gold_model(meta.problem_id) if args.run_gold else {"gold_comparison_eligible": False, "gold_reason": "disabled"}
    try:
        record = load_problem_record(meta.problem_id)
        raw_text = record.get("parametrized_description") or ""
        extraction = extract_structured_problem(str(meta.problem_id), raw_text, provider, config)
        responses.append(extraction.llm_response)
        tree = build_partition_tree(extraction.structured_problem, config)
        merged = build_merged_model(tree, extraction.structured_problem, provider, config)
        model_responses = [r.llm_response for r in merged.leaf_results] + [merged.root_llm_response]
        responses.extend(model_responses)
        g_mod_calls = len(model_responses)
        render = render_merged_model(merged)
        render_valid = render.valid
        executor = AmplExecutor(
            solver="gurobi",
            python_executable=args.ampl_python,
            timeout_seconds=args.ampl_timeout,
        )
        trace = run_correction_loop(
            merged_model=merged,
            structured_problem=extraction.structured_problem,
            provider=provider,
            config=config,
            executor=executor,
        )
        responses.extend(correction_responses(trace))
        prompt_hashes = prompt_hashes_for_problem(extraction.llm_response, model_responses, trace)
    except Exception as error:  # noqa: BLE001 - per-problem isolation is required.
        exc = error
        prompt_hashes = sorted({r.prompt_hash for r in responses if r.prompt_hash})

    usage = llm_usage(responses)
    category = map_failure_category(trace, exc, render_valid)
    initial = trace.iterations[0].execution if trace and trace.iterations else None
    final = trace.iterations[-1].execution if trace and trace.iterations else None
    correction_calls = len(correction_responses(trace)) if trace else 0
    objective_match = ""
    semantic_status = "NOT_EVALUABLE"
    wrong_but_feasible = ""
    if final and final.objective_value is not None and gold.get("gold_objective") is not None:
        try:
            objective_match = abs(float(final.objective_value) - float(gold["gold_objective"])) <= 1.0e-6
            semantic_status = "PARTIAL_GOLD_OBJECTIVE_STATUS_ONLY"
            wrong_but_feasible = bool(final.success and not objective_match)
        except (TypeError, ValueError):
            objective_match = ""
    elif final and final.success:
        semantic_status = "FEASIBLE_NOT_SEMANTICALLY_EVALUATED"
        wrong_but_feasible = ""

    row = {
        "problem_id": meta.problem_id,
        "lp_or_milp": meta.lp_or_milp,
        "partition_node_count": meta.partition_node_count,
        "partition_depth": meta.partition_depth,
        "initial_generation_valid": "YES" if exc is None and render_valid else "NO",
        "initial_ampl_parse_success": initial.parse_success if initial else "",
        "initial_solver_success": initial.success if initial else "",
        "initial_feasible_status": initial.success if initial else "",
        "correction_invoked": bool(trace and trace.correction_iterations_observed > 0),
        "correction_iterations": trace.correction_iterations_observed if trace else 0,
        "final_ampl_parse_success": final.parse_success if final else "",
        "final_solver_status": final.solver_status if final else "",
        "final_feasible": final.success if final else False,
        "objective_produced": bool(final and final.objective_value is not None),
        "objective_value": final.objective_value if final else "",
        "failure_category": category,
        "total_llm_calls": len(responses),
        "g_extr_calls": 1 if responses else 0,
        "g_mod_calls": g_mod_calls,
        "correction_calls": correction_calls,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_seconds": round(time.monotonic() - start, 6),
        "estimated_cost_usd": "",
        "cost_status": "UNAVAILABLE_AZURE_ACCOUNT_PRICING_NOT_DETERMINED",
        "provider": usage["provider"] or "azure_openai",
        "deployment": usage["deployment"] or config.llm.model,
        "underlying_model": usage["underlying_model"],
        "temperature": config.llm.temperature,
        "prompt_hashes": ";".join(prompt_hashes),
        "final_ampl_hash": trace.final_ampl_hash if trace else "",
        "gold_comparison_eligible": gold.get("gold_comparison_eligible", False),
        "gold_solver_success": gold.get("gold_solver_success", ""),
        "gold_feasible": gold.get("gold_feasible", ""),
        "gold_objective": gold.get("gold_objective", ""),
        "objective_match_with_gold": objective_match,
        "semantic_correctness_status": semantic_status,
        "semantically_wrong_but_feasible": wrong_but_feasible,
    }
    append_csv(out_dir / "per_problem.csv", PER_FIELDS, row)

    tags, codes, reason = diagnostic_tags(trace)
    if category not in {"A. SUCCESS_NO_CORRECTION", "B. SUCCESS_AFTER_CORRECTION"}:
        append_csv(
            out_dir / "failure_analysis.csv",
            FAIL_FIELDS,
            {
                "problem_id": meta.problem_id,
                "failure_category": category,
                "primary_reason": reason or (type(exc).__name__ if exc else trace.stopped_reason if trace else ""),
                "issue_tags": tags,
                "diagnostic_codes": codes,
            },
        )

    correction_tokens = sum(int(r.total_tokens or 0) for r in correction_responses(trace)) if trace else 0
    correction_latency = sum(float(r.latency_seconds or 0.0) for r in correction_responses(trace)) if trace else 0.0
    append_csv(
        out_dir / "correction_analysis.csv",
        CORR_FIELDS,
        {
            "problem_id": meta.problem_id,
            "initial_success": initial.success if initial else False,
            "final_success": final.success if final else False,
            "rescued_by_correction": bool(initial and final and (not initial.success) and final.success),
            "harmed_by_correction": bool(initial and final and initial.success and not final.success),
            "correction_iterations": trace.correction_iterations_observed if trace else 0,
            "correction_calls": correction_calls,
            "correction_tokens": correction_tokens,
            "correction_latency_seconds": round(correction_latency, 6),
        },
    )

    safe_write_json(
        out_dir / "incremental" / f"problem_{meta.problem_id}.json",
        {
            "problem_id": meta.problem_id,
            "metadata": asdict(meta),
            "per_problem_row": row,
            "gold": gold,
            "trace": trace.to_dict() if trace else None,
            "exception_type": type(exc).__name__ if exc else "",
            "exception_message": str(exc)[:500] if exc else "",
        },
    )
    return row


def write_ours_comparison(out_dir: Path, selected: list[SliceMeta], pamop_rows: list[dict[str, Any]]) -> None:
    comparison_path = out_dir / "comparison_with_ours.csv"
    if comparison_path.exists():
        comparison_path.unlink()
    ours = load_ours_rows()
    pamop = {int(r["problem_id"]): r for r in pamop_rows if r.get("problem_id") not in ("", None)}
    for meta in selected:
        row = ours.get(meta.problem_id, {})
        p = pamop.get(meta.problem_id, {})
        append_csv(
            comparison_path,
            OURS_FIELDS,
            {
                "problem_id": meta.problem_id,
                "doc_id": meta.doc_id,
                "our_method": "tfidf deterministic grounding",
                "our_external_llm_calls": 0,
                "our_llm_tokens": 0,
                "our_api_cost": 0,
                "our_deterministic": True,
                "our_inst_ready": row.get("inst_ready", ""),
                "our_param_coverage": row.get("param_coverage", ""),
                "our_type_match": row.get("type_match", ""),
                "pamop_failure_category": p.get("failure_category", ""),
                "pamop_total_llm_calls": p.get("total_llm_calls", ""),
                "pamop_total_tokens": p.get("total_tokens", ""),
                "pamop_final_feasible": p.get("final_feasible", ""),
                "common_metric_status": "NO_COMMON_ACCURACY_METRIC; operational metrics only",
            },
        )


def read_per_problem(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def write_summary(out_dir: Path, selected: list[SliceMeta], status: str, slurm_job_id: str | None) -> dict[str, Any]:
    rows = read_per_problem(out_dir / "per_problem.csv")
    n = len(rows)
    initial_success = sum(truthy(r.get("initial_solver_success")) for r in rows)
    final_success = sum(r.get("failure_category") in {"A. SUCCESS_NO_CORRECTION", "B. SUCCESS_AFTER_CORRECTION"} for r in rows)
    no_corr = sum(r.get("failure_category") == "A. SUCCESS_NO_CORRECTION" for r in rows)
    after_corr = sum(r.get("failure_category") == "B. SUCCESS_AFTER_CORRECTION" for r in rows)
    feasible = sum(truthy(r.get("final_feasible")) for r in rows)
    objective = sum(truthy(r.get("objective_produced")) for r in rows)
    correction_iters = [int(r.get("correction_iterations") or 0) for r in rows if int(r.get("correction_iterations") or 0) > 0]
    tokens = sum(int(r.get("total_tokens") or 0) for r in rows)
    latencies = [float(r.get("latency_seconds") or 0.0) for r in rows]
    failures = Counter(r.get("failure_category") for r in rows)
    rescue = 0
    harm = 0
    corr_path = out_dir / "correction_analysis.csv"
    if corr_path.exists():
        with corr_path.open(encoding="utf-8") as fh:
            correction_rows = list(csv.DictReader(fh))
            rescue = sum(truthy(r.get("rescued_by_correction")) for r in correction_rows)
            harm = sum(truthy(r.get("harmed_by_correction")) for r in correction_rows)
    semantic_evaluable = [
        r for r in rows
        if r.get("objective_match_with_gold") in {"True", "true", "False", "false"}
    ]
    semantic_correct = sum(r.get("objective_match_with_gold") in {"True", "true"} for r in semantic_evaluable)
    decision_gate = "B. FIX SYSTEMATIC ISSUE FIRST"
    if status == "COMPLETED" and n and (final_success / n) >= 0.7:
        decision_gate = "A. PROCEED TO LARGER RUN"
    summary = {
        "label": "PILOT / SMALL-SLICE RESULTS",
        "status": status,
        "slurm_job_id": slurm_job_id,
        "failure_categories_defined_before_run": FAILURE_CATEGORIES,
        "selected_count": len(selected),
        "evaluated_count": n,
        "initial_execution_success_count": initial_success,
        "initial_execution_success_rate": initial_success / n if n else 0.0,
        "final_execution_success_count": final_success,
        "final_execution_success_rate": final_success / n if n else 0.0,
        "success_without_correction": no_corr,
        "success_after_correction": after_corr,
        "correction_rescue_count": rescue,
        "correction_harm_count": harm,
        "mean_correction_iterations_among_corrected": (sum(correction_iters) / len(correction_iters)) if correction_iters else 0.0,
        "solver_feasible_count": feasible,
        "solver_feasible_rate": feasible / n if n else 0.0,
        "objective_produced_count": objective,
        "objective_produced_rate": objective / n if n else 0.0,
        "semantic_evaluable_count": len(semantic_evaluable),
        "semantic_correct_count": semantic_correct,
        "semantic_correct_rate": semantic_correct / len(semantic_evaluable) if semantic_evaluable else 0.0,
        "semantic_correct_evaluable_count": semantic_correct,
        "semantically_wrong_but_feasible_count": sum(r.get("semantically_wrong_but_feasible") in {"True", "true"} for r in rows),
        "failure_categories": dict(failures),
        "total_tokens": tokens,
        "mean_tokens_per_problem": tokens / n if n else 0.0,
        "latency_seconds": {
            "min": min(latencies) if latencies else 0.0,
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "cost_status": "UNAVAILABLE_AZURE_ACCOUNT_PRICING_NOT_DETERMINED",
        "decision_gate": decision_gate,
    }
    safe_write_json(out_dir / "summary.json", summary)
    return summary


def write_run_metadata(out_dir: Path, args: argparse.Namespace, status: str, message: str = "") -> None:
    safe_write_json(
        out_dir / "run_metadata.json",
        {
            "status": status,
            "message": message,
            "run_id": os.environ.get("PAMOP_PILOT_RUN_ID", ""),
            "execution_mode": "local" if args.allow_local else "slurm_required",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cwd": str(ROOT),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "required_slurm": not args.allow_local,
            "provider": "azure_openai",
            "deployment": args.deployment,
            "temperature": args.temperature,
            "max_correction_iterations": 5,
            "ampl_python": args.ampl_python,
            "cost_status": "UNAVAILABLE_AZURE_ACCOUNT_PRICING_NOT_DETERMINED",
            "secrets_policy": "No API keys, HF tokens, AMPL/Gurobi licenses, or gated problem text are written.",
        },
    )


def write_checkpoint_state(
    out_dir: Path,
    *,
    run_id: str,
    selected: list[SliceMeta],
    status: str,
    execution_mode: str,
    session_name: str | None = None,
) -> None:
    rows = read_per_problem(out_dir / "per_problem.csv")
    completed = [int(r["problem_id"]) for r in rows if r.get("problem_id")]
    selected_ids = [m.problem_id for m in selected]
    safe_write_json(
        out_dir / "local_run_state.json",
        {
            "run_id": run_id,
            "status": status,
            "execution_mode": execution_mode,
            "pid": os.getpid(),
            "tmux_session": session_name or os.environ.get("PAMOP_PILOT_TMUX_SESSION", ""),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_ids": completed,
            "remaining_ids": [pid for pid in selected_ids if pid not in set(completed)],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-count", type=int, default=18)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--allow-local", action="store_true", help="Permit benchmark execution outside Slurm.")
    parser.add_argument("--deployment", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--ampl-python", default="/home/soroush/.venvs/gurobi/bin/python")
    parser.add_argument("--ampl-timeout", type=int, default=60)
    parser.add_argument("--run-gold", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = load_or_select_slice(out_dir, args.target_count)
    ensure_artifact_headers(out_dir)
    run_id = os.environ.get("PAMOP_PILOT_RUN_ID") or f"pamop-pilot-local-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    os.environ["PAMOP_PILOT_RUN_ID"] = run_id

    if args.select_only:
        write_run_metadata(out_dir, args, "SELECTED_ONLY")
        write_checkpoint_state(out_dir, run_id=run_id, selected=selected, status="SELECTED_ONLY", execution_mode="select_only")
        write_ours_comparison(out_dir, selected, read_per_problem(out_dir / "per_problem.csv"))
        write_summary(out_dir, selected, "SELECTED_ONLY", os.environ.get("SLURM_JOB_ID"))
        return 0

    if not args.allow_local and not os.environ.get("SLURM_JOB_ID"):
        write_run_metadata(out_dir, args, "ENVIRONMENT_FAILURE", "Refusing to run PaMOP pilot outside Slurm; submit with sbatch.")
        write_checkpoint_state(out_dir, run_id=run_id, selected=selected, status="ENVIRONMENT_FAILURE", execution_mode="slurm_required")
        write_ours_comparison(out_dir, selected, read_per_problem(out_dir / "per_problem.csv"))
        write_summary(out_dir, selected, "ENVIRONMENT_FAILURE", None)
        return 2

    os.environ["PAMOP_AMPLPY_PYTHON"] = args.ampl_python
    config = load_config(reconstructed_default_path())
    if config.llm.provider != "azure_openai" or config.llm.model != args.deployment or config.llm.temperature != args.temperature:
        raise RuntimeError("This pilot is pinned to Azure OpenAI gpt-4.1-mini at temperature 0.2; config mismatch.")
    provider = get_provider("azure_openai")
    write_run_metadata(out_dir, args, "RUNNING")
    write_checkpoint_state(out_dir, run_id=run_id, selected=selected, status="RUNNING", execution_mode="local")
    rows = []
    done = {int(r["problem_id"]) for r in read_per_problem(out_dir / "per_problem.csv") if r.get("problem_id")}
    for meta in selected:
        if meta.problem_id in done:
            continue
        rows.append(run_problem(meta, out_dir, args, config, provider))
        write_summary(out_dir, selected, "RUNNING", os.environ.get("SLURM_JOB_ID"))
        write_checkpoint_state(out_dir, run_id=run_id, selected=selected, status="RUNNING", execution_mode="local")
    all_rows = read_per_problem(out_dir / "per_problem.csv")
    write_ours_comparison(out_dir, selected, all_rows)
    write_run_metadata(out_dir, args, "COMPLETED")
    write_summary(out_dir, selected, "COMPLETED", os.environ.get("SLURM_JOB_ID"))
    write_checkpoint_state(out_dir, run_id=run_id, selected=selected, status="COMPLETED", execution_mode="local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
