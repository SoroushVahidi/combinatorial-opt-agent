"""The unified cross-baseline analysis view.

This is an ANALYSIS VIEW that sits above each baseline's own native
`result_schema.py`, never a replacement for it. Adapters (`adapters.py`)
populate a `UnifiedRow` from native records without discarding native
fields (native records remain retrievable via `native_record`).

Every metric-shaped field's value is either a real measurement (bool,
float, int, or str payload) or one of the `CellState` sentinel strings
below. A field is never silently blank: absence of a measurement is always
one specific, explained state, never `None`/`0`/`""` doing double duty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CellState:
    """Standardized non-measurement states. See PHASE 17 cell-semantics rules."""

    PENDING = "PENDING"  # Baseline is implemented but has not been run.
    NOT_APPLICABLE = "NOT_APPLICABLE"  # This concept does not apply to this system.
    UNAVAILABLE = "UNAVAILABLE"  # Blocked by a missing artifact (checkpoint, code, license).
    MOCK_ONLY = "MOCK_ONLY"  # Only synthetic/mock evidence exists; never a comparison row.
    PROXY = "PROXY"  # A value exists but is explicitly a proxy, not the named quantity.
    UNSUPPORTED = "UNSUPPORTED"  # The input/record was excluded/unsupported, not a failure.
    UNKNOWN = "UNKNOWN"  # State genuinely not determined from available evidence.


FIDELITY_LEVELS = (
    "EXACT_OFFICIAL", "ADAPTED_OFFICIAL", "PAPER_SPECIFIED", "PAPER_RECONSTRUCTED",
    "LOCAL_ENGINEERING", "INDEPENDENT_RECONSTRUCTION", "NATIVE_METHOD", "UNKNOWN",
)  # Documentation only -- not runtime-enforced; see each baseline's own provenance doc.

ALL_STATES = frozenset({
    CellState.PENDING, CellState.NOT_APPLICABLE, CellState.UNAVAILABLE,
    CellState.MOCK_ONLY, CellState.PROXY, CellState.UNSUPPORTED, CellState.UNKNOWN,
})


def is_state(value: Any) -> bool:
    return isinstance(value, str) and value in ALL_STATES


def is_measured(value: Any) -> bool:
    return value is not None and not is_state(value)


@dataclass
class UnifiedRow:
    # --- IDENTITY ---
    system: str  # ours | pamop | orlm | optmath | deepor | orr1
    method_variant: str
    problem_id: str
    dataset: str
    input_hash: str

    # --- PROVENANCE ---
    implementation_fidelity: str = CellState.UNKNOWN  # EXACT_OFFICIAL | ADAPTED_OFFICIAL | PAPER_SPECIFIED | PAPER_RECONSTRUCTED | LOCAL_ENGINEERING | INDEPENDENT_RECONSTRUCTION | UNKNOWN
    official_code_used: Any = CellState.UNKNOWN  # bool | CellState
    official_checkpoint_used: Any = CellState.UNKNOWN  # bool | CellState
    checkpoint_model: Any = CellState.NOT_APPLICABLE
    checkpoint_revision: Any = CellState.NOT_APPLICABLE
    source_repo: Any = CellState.NOT_APPLICABLE
    source_repo_revision: Any = CellState.NOT_APPLICABLE
    local_git_sha: Any = CellState.UNKNOWN

    # --- EXECUTION ---
    generation_attempted: Any = CellState.PENDING  # bool | CellState
    generation_completed: Any = CellState.PENDING
    parse_success: Any = CellState.NOT_APPLICABLE
    static_valid: Any = CellState.NOT_APPLICABLE
    execution_attempted: Any = CellState.NOT_APPLICABLE
    execution_success: Any = CellState.NOT_APPLICABLE
    feasible: Any = CellState.NOT_APPLICABLE
    bounded: Any = CellState.NOT_APPLICABLE
    solver_status: Any = CellState.NOT_APPLICABLE

    # --- CORRECTNESS ---
    objective_available: Any = CellState.NOT_APPLICABLE
    objective_predicted: Any = CellState.NOT_APPLICABLE
    objective_gold: Any = CellState.NOT_APPLICABLE
    objective_match: Any = CellState.NOT_APPLICABLE  # bool | CellState -- a tolerance-based proxy, see objective_tolerance
    objective_tolerance: Any = CellState.NOT_APPLICABLE
    semantic_correct: Any = CellState.NOT_APPLICABLE  # bool | CellState -- reserved for a genuine solver-verified semantic judgment
    semantic_metric_available: Any = False
    correctness_metric_name: Any = CellState.NOT_APPLICABLE  # e.g. "InstantiationReady", "objective_value_proxy", "pass@8"

    # --- COMPUTE ---
    runtime_seconds: Any = CellState.NOT_APPLICABLE
    prompt_tokens: Any = CellState.NOT_APPLICABLE
    generated_tokens: Any = CellState.NOT_APPLICABLE
    total_tokens: Any = CellState.NOT_APPLICABLE
    rollout_count: Any = CellState.NOT_APPLICABLE
    correction_iterations: Any = CellState.NOT_APPLICABLE
    test_time_training_steps: Any = CellState.NOT_APPLICABLE
    estimated_cost: Any = CellState.UNKNOWN

    # --- FAILURE ---
    failure_category: Any = CellState.NOT_APPLICABLE
    failure_detail: Any = CellState.NOT_APPLICABLE

    # --- SCOPE (static, method-level facts, not per-row measurements) ---
    full_formulation: bool = False
    fixed_schema: bool = False
    scalar_grounding_only: bool = False
    generative: bool = False
    test_time_learning: bool = False
    transductive_training: bool = False

    native_record: dict[str, Any] = field(default_factory=dict)
    native_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def cell_states(self) -> dict[str, str]:
        """Map of field -> state label ('MEASURED' for any real value)."""
        out = {}
        for k, v in self.__dict__.items():
            if k in {"native_record", "native_metrics"}:
                continue
            out[k] = v if is_state(v) else ("MEASURED" if v is not None else CellState.UNKNOWN)
        return out
