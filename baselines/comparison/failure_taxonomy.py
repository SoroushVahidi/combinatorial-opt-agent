"""Cross-baseline failure-category mapping.

Maps each system's native failure-category strings to one shared top-level
bucket, without deleting the native detail (native value is preserved
alongside the mapped bucket, never replaced).
"""
from __future__ import annotations

TOP_LEVEL_CATEGORIES = (
    "unsupported_input", "generation_failure", "parse_failure", "static_invalid",
    "execution_failure", "infeasible", "unbounded", "objective_mismatch",
    "semantic_mismatch", "timeout", "evaluation_ambiguous", "unavailable_artifact",
    "none",
)

# Native category string (case-sensitive, as emitted by each system) -> top-level bucket.
_MAPPING: dict[str, str] = {
    # PaMOP (results/pamop/*/summary.json failure_categories_defined_before_run)
    "A. SUCCESS_NO_CORRECTION": "none",
    "B. SUCCESS_AFTER_CORRECTION": "none",
    "C. MODEL_PARSE_FAILURE": "parse_failure",
    "D. AMPL_RENDER_FAILURE": "generation_failure",
    "E. AMPL_PARSE_FAILURE": "parse_failure",
    "F. SOLVER_INFEASIBLE": "infeasible",
    "G. SOLVER_UNBOUNDED": "unbounded",
    "H. SOLVER_RUNTIME_ERROR": "execution_failure",
    "I. CORRECTION_EXHAUSTED": "execution_failure",
    "J. DATA_FAILURE": "unsupported_input",
    "K. ENVIRONMENT_FAILURE": "unavailable_artifact",
    "L. OTHER_MODEL_FAILURE": "generation_failure",
    # ORLM / OptMATH / OR-R1 shared runner/harness categories
    "empty_prompt": "unsupported_input",
    "generation_timeout": "timeout",
    "generation_error": "generation_failure",
    "transformers_backend_unavailable": "unavailable_artifact",
    "vllm_backend_unavailable": "unavailable_artifact",
    "checkpoint_unavailable": "unavailable_artifact",
    "python_syntax_failure": "static_invalid",
    "static_validation_failure": "static_invalid",
    "execution_failure": "execution_failure",
    "execution_timeout": "timeout",
    "copt_api_failure": "execution_failure",
    "infeasible_model": "infeasible",
    "unbounded_model": "unbounded",
    "output_out_of_expectation": "evaluation_ambiguous",
    # Adapter-level
    "missing_problem_id": "unsupported_input",
    "missing_or_empty_problem_text": "unsupported_input",
    "invalid_json": "unsupported_input",
    "blank_line": "unsupported_input",
    "record_not_object": "unsupported_input",
}


def to_top_level(native_category: object) -> str:
    if native_category is None or native_category == "":
        return "none"
    text = str(native_category)
    if text in _MAPPING:
        return _MAPPING[text]
    for prefix, bucket in _MAPPING.items():
        if text.startswith(prefix.split(":")[0]):
            return bucket
    return "evaluation_ambiguous"
