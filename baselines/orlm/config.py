"""Configuration for the ORLM baseline scaffold.

Every field the upstream ORLM repo leaves as a script constant is named
here instead, matching this repo's baselines/pamop/config.py convention:
document reproduction choices explicitly rather than hard-coding them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrlmConfig:
    # HuggingFace repo id of the checkpoint to use. Only this one is
    # confirmed publicly retrievable as of 2026-08-12 -- see README.md.
    model_id: str = "CardinalOperations/ORLM-LLaMA-3-8B"

    # Upstream official prompt template. RECONSTRUCTED from public
    # documentation, not yet verified byte-for-byte against the upstream
    # repo's own prompt file -- do this before first real use (README.md
    # "Exact first practical smoke-test milestone", step 4).
    prompt_template: str = (
        "Below is an operations research question. Build a mathematical "
        "model and corresponding python code using `coptpy` that "
        "appropriately addresses the question.\n"
        "# Question:\n{question}\n"
        "# Response:\n"
    )

    max_new_tokens: int = 2048
    temperature: float = 0.0  # deterministic-as-possible; upstream default not confirmed
    max_seq_len: int = 8192

    # Solver the generated code targets -- NOT Gurobi/Pyomo/plain LP.
    solver: str = "coptpy"

    # Minimum GPU memory class this checkpoint plausibly needs for
    # inference (not training). Documented estimate, not measured on this
    # workstation (no GPU provisioned here as of 2026-08-12).
    min_gpu_memory_gb: int = 24

    # Whether the upstream repo requires fine-tuning before use.
    finetuning_required: bool = False

    # Whether an external API is required (it is not -- fully local once
    # weights + a COPT license are obtained).
    requires_external_api: bool = False
