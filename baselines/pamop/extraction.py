"""LLM-based structured extraction (PaMOP's ``G_extr``, paper section 3.2).

"Before partitioning, we derive a structured representation of the problem,
extracting textual descriptions for the objective functions t_o,
constraints t_c, and parameters and variables t_v. To avoid modeling bias
from missing the global context, we also generate a concise problem summary
g. This extraction process, guided by G_extr, prompts the LLM to produce
the structured elements... We prompt the LLM to assign a vagueness score to
each constraint."

This module calls an LLM (via ``baselines.pamop.llm``) with the
reconstructed prompt in ``prompts/extraction_v1.txt`` (see
``prompts/PROVENANCE.md`` -- the prompt wording is our own; the four
required output fields and the vagueness score are paper-specified), parses
the JSON response, validates it strictly, and retries (asking again, never
silently patching content) up to ``config.llm.extraction_max_retries``
times on validation failure.

This is the first stage that touches an external LLM. It produces a
``StructuredProblem`` (representations.py) that feeds directly into the
already-implemented, non-LLM ``partition.build_partition_tree`` -- the two
stages remain independently inspectable and serializable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import PamopConfig
from .llm.base import LLMProvider
from .llm.types import LLMResponse, ModelConfig
from .prompts import PromptTemplate, load_prompt
from .representations import StructuredProblem, from_llm_extraction

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_VAR_TYPES = {"continuous", "integer", "binary", "parameter", None}


class ExtractionValidationError(ValueError):
    """Raised when an LLM's extraction response fails schema validation.

    Carries the specific reason so callers/tests can assert on *why* it
    failed, not just that it did.
    """


@dataclass(frozen=True)
class ExtractionResult:
    structured_problem: StructuredProblem
    llm_response: LLMResponse
    prompt_template: PromptTemplate
    validation_attempts: int


def _extract_json_object(text: str) -> Any:
    """Parse ``text`` as JSON, tolerating a leading/trailing prose wrapper
    (e.g. a stray ```json fence) but not attempting to repair malformed
    JSON structure itself -- that is a validation failure, not something to
    silently fix."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)  # raises json.JSONDecodeError on malformed JSON


def validate_extraction(raw: Any) -> dict[str, Any]:
    """Strictly validate a parsed extraction response.

    Returns ``raw`` unchanged if valid. Raises ``ExtractionValidationError``
    with a specific reason otherwise. Never mutates or fills in missing
    content -- PaMOP's own repair mechanisms (basic inspection, solver-debug,
    reverse translation) are a separate, later stage (paper section 3.3) and
    are not implemented here.
    """
    if not isinstance(raw, dict):
        raise ExtractionValidationError(f"top-level response must be a JSON object, got {type(raw).__name__}")

    for field in ("global_summary", "objective_text", "constraints", "variables"):
        if field not in raw:
            raise ExtractionValidationError(f"missing required field: {field!r}")

    if not isinstance(raw["global_summary"], str) or not raw["global_summary"].strip():
        raise ExtractionValidationError("global_summary must be a non-empty string")
    if not isinstance(raw["objective_text"], str) or not raw["objective_text"].strip():
        raise ExtractionValidationError("objective_text must be a non-empty string")

    constraints = raw["constraints"]
    if not isinstance(constraints, list) or not constraints:
        raise ExtractionValidationError("constraints must be a non-empty JSON array")
    for i, c in enumerate(constraints):
        if not isinstance(c, dict):
            raise ExtractionValidationError(f"constraints[{i}] must be a JSON object")
        if not isinstance(c.get("description"), str) or not c["description"].strip():
            raise ExtractionValidationError(f"constraints[{i}].description must be a non-empty string")
        score = c.get("vagueness_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not (0.0 <= float(score) <= 1.0):
            raise ExtractionValidationError(
                f"constraints[{i}].vagueness_score must be a number in [0, 1], got {score!r}"
            )

    variables = raw["variables"]
    if not isinstance(variables, list):
        raise ExtractionValidationError("variables must be a JSON array")
    seen_names: set[str] = set()
    for i, v in enumerate(variables):
        if not isinstance(v, dict):
            raise ExtractionValidationError(f"variables[{i}] must be a JSON object")
        name = v.get("name")
        if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
            raise ExtractionValidationError(f"variables[{i}].name must be a valid identifier, got {name!r}")
        if name in seen_names:
            raise ExtractionValidationError(f"variables[{i}].name {name!r} is a duplicate")
        seen_names.add(name)
        if not isinstance(v.get("description"), str) or not v["description"].strip():
            raise ExtractionValidationError(f"variables[{i}].description must be a non-empty string")
        if v.get("type") not in _VALID_VAR_TYPES:
            raise ExtractionValidationError(
                f"variables[{i}].type must be one of {sorted(t for t in _VALID_VAR_TYPES if t)} or null, "
                f"got {v.get('type')!r}"
            )

    return raw


def extract_structured_problem(
    problem_id: str,
    raw_problem_text: str,
    provider: LLMProvider,
    config: PamopConfig,
) -> ExtractionResult:
    """Run G_extr: call ``provider`` with the extraction prompt, validate the
    JSON response, and return a ``StructuredProblem`` on success.

    Raises ``ExtractionValidationError`` (chained to the last attempt's
    failure) if every retry within ``config.llm.extraction_max_retries``
    fails validation, and whatever the provider itself raises
    (``ProviderAuthError`` / ``ProviderCallError``) unmodified on a call
    failure.
    """
    model_config = ModelConfig(
        provider=config.require("llm", "provider"),
        model=config.require("llm", "model"),
        temperature=config.require("llm", "temperature"),
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
    )
    max_retries = config.require("llm", "extraction_max_retries")
    template = load_prompt("extraction_v1.txt")
    prompt = template.render(problem_text=raw_problem_text)

    attempt = 0
    last_error: Exception | None = None
    last_response: LLMResponse | None = None
    while attempt <= max_retries:
        attempt += 1
        response = provider.generate(prompt, model_config)
        last_response = response
        try:
            raw = _extract_json_object(response.text)
            validated = validate_extraction(raw)
        except (json.JSONDecodeError, ExtractionValidationError) as exc:
            last_error = exc
            continue

        structured = from_llm_extraction(
            problem_id, validated, model_tag=f"{provider.name}:{model_config.model}"
        )
        return ExtractionResult(
            structured_problem=structured,
            llm_response=response,
            prompt_template=template,
            validation_attempts=attempt,
        )

    raise ExtractionValidationError(
        f"extraction for problem {problem_id!r} failed validation after "
        f"{attempt} attempt(s); last response prompt_hash="
        f"{last_response.prompt_hash if last_response else 'n/a'}"
    ) from last_error
