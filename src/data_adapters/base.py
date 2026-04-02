"""Base interfaces and normalized schema for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class DatasetCapabilities:
    """Feature flags describing which evaluations are meaningful."""

    supports_schema_retrieval: bool
    supports_scalar_instantiation: bool
    supports_solver_eval: bool
    supports_full_formulation: bool


@dataclass
class InternalExample:
    """Capability-aware normalized record used by the generic runner."""

    id: str
    source_dataset: str
    split: str
    nl_query: str
    schema_id: str | None = None
    schema_text: str | None = None
    candidate_schemas: list[str] | None = None
    scalar_gold_params: dict[str, Any] | None = None
    structured_gold_params: dict[str, Any] | None = None
    formulation_text: str | None = None
    solver_artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    """Protocol implemented by every dataset adapter."""

    name: str
    capabilities: DatasetCapabilities
    data_root: Path

    def list_splits(self) -> list[str]:
        """Return available split names."""

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        """Load raw examples for one split."""

    def iter_examples(self, split_name: str) -> Iterable[dict[str, Any]]:
        """Iterate raw examples for one split."""

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        """Map one raw record to InternalExample."""

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        """Return schema/template candidates when available."""

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        """Return gold targets keyed by example id when available."""

