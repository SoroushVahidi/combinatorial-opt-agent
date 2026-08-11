"""Configuration schema for the PaMOP reproduction.

Every parameter the PaMOP paper leaves unspecified (see
``docs/PAMOP_REPRODUCTION_PLAN.md``, sections 2 and 9) is a named, typed field
here rather than a hard-coded constant. ``configs/paper_faithful.yaml`` sets
only what the paper actually states and leaves everything else ``null``;
``configs/reconstructed_default.yaml`` fills those gaps with documented
reproduction choices.

Loading ``paper_faithful.yaml`` and then asking for a partitioning-stage
value that the paper never specified raises ``UnspecifiedPaperDetailError``
rather than silently substituting a default -- see ``PamopConfig.require``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


class UnspecifiedPaperDetailError(RuntimeError):
    """Raised when a run needs a value the PaMOP paper never specifies.

    Only ever raised for ``paper_faithful`` configs -- ``reconstructed_default``
    fills every such field with a documented choice, so this should never fire
    for it. If it does, that reconstructed config is incomplete and should be
    fixed, not worked around by catching this exception.
    """


@dataclass(frozen=True)
class LlmConfig:
    model: str | None = None
    temperature: float | None = None
    max_correction_iterations: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class PartitioningConfig:
    # "vector similarity" signal source. Paper: GloVe trained on Wikipedia 2014.
    embedding_source: str | None = None  # "glove" | "tfidf_fallback"
    glove_variant: str | None = None  # e.g. "6B.300d" -- paper doesn't say which
    tfidf_top_k: int | None = None
    epsilon: float | None = None  # eq. (2): small fixed value, not given
    # Per-layer weights for combining {adjacency, keyword, vector} similarity.
    # Paper: "weighted averages ... different [weights] to different layers",
    # no numbers given. Keys are layer names ("root", "default") -> weight dict.
    similarity_weights_by_layer: dict[str, dict[str, float]] | None = None
    clustering_algorithm: str | None = None  # paper gives the distance metric only
    independent_set_algorithm: str | None = None  # paper: "graph search algorithms"
    bipartite_edge_confidence_threshold: int | None = None
    leaf_stop_min_constraints: int | None = None  # paper: "a small number"
    leaf_stop_similarity_threshold: float | None = None  # paper: "highly similar"
    deterministic_seed: int | None = None


@dataclass(frozen=True)
class CorrectionConfig:
    correctness_tolerance: float | None = None
    iteration_scope: str | None = None  # "global" | "per_node" -- paper is ambiguous
    reverse_translation_scope: str | None = None  # "all_constraints" | "vague_only"


@dataclass(frozen=True)
class ExecutionConfig:
    generation_target: str | None = None  # paper: "AMPL"
    solver_backend: str | None = None  # paper: Gurobi, invoked via AMPL


@dataclass(frozen=True)
class DatasetConfig:
    subset: str | None = None


@dataclass(frozen=True)
class PamopConfig:
    config_kind: str
    citation: str
    llm: LlmConfig = field(default_factory=LlmConfig)
    partitioning: PartitioningConfig = field(default_factory=PartitioningConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def require(self, section: str, name: str) -> Any:
        """Return ``getattr(self.<section>, name)``, or raise if it is ``None``.

        Use this (never bare ``config.partitioning.epsilon``) anywhere a
        concrete value is actually needed to run partitioning/correction/etc.
        so that an incomplete ``paper_faithful`` config fails loudly instead
        of silently running with an invented number.
        """
        section_obj = getattr(self, section)
        value = getattr(section_obj, name)
        if value is None:
            raise UnspecifiedPaperDetailError(
                f"{self.config_kind}.{section}.{name} is not set. The PaMOP "
                f"paper does not specify this value (see "
                f"docs/PAMOP_REPRODUCTION_PLAN.md sections 2 and 9). Use "
                f"configs/reconstructed_default.yaml, or set this field "
                f"explicitly with a justified, documented choice."
            )
        return value


_SECTION_TYPES: dict[str, type] = {
    "llm": LlmConfig,
    "partitioning": PartitioningConfig,
    "correction": CorrectionConfig,
    "execution": ExecutionConfig,
    "dataset": DatasetConfig,
}


def _build_section(section_cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {f.name for f in fields(section_cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown fields for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**raw)


def load_config(path: str | Path) -> PamopConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    sections = {
        name: _build_section(cls, raw.get(name))
        for name, cls in _SECTION_TYPES.items()
    }
    return PamopConfig(
        config_kind=raw.get("config_kind", path.stem),
        citation=raw.get("citation", ""),
        **sections,
    )


def paper_faithful_path() -> Path:
    return CONFIGS_DIR / "paper_faithful.yaml"


def reconstructed_default_path() -> Path:
    return CONFIGS_DIR / "reconstructed_default.yaml"
