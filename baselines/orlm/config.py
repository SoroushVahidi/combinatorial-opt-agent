"""Configuration and provenance for lightweight ORLM inference preparation."""
from __future__ import annotations

from dataclasses import dataclass


ORLM_UPSTREAM_REPOSITORY = "https://github.com/Cardinal-Operations/ORLM"
ORLM_UPSTREAM_REVISION = "33bc47d0a1d1710d24ab839118bdf4cb89b9e31b"
ORLM_CHECKPOINT_REVISION = "94fdc3c5738c6536d4880dc19a78f215529181c5"
ORLM_PROMPT_VERSION = "upstream-eval-generate-TEMPLATE_q2mc_en-v1"
ORLM_PROMPT_TEMPLATE = (
    "Below is an operations research question. Build a mathematical model "
    "and corresponding python code using `coptpy` that appropriately "
    "addresses the question.\n\n"
    "# Question:\n"
    "{Question}\n"
    "# Response:\n"
)


@dataclass(frozen=True)
class OrlmConfig:
    model_id: str = "CardinalOperations/ORLM-LLaMA-3-8B"
    model_revision: str | None = ORLM_CHECKPOINT_REVISION
    prompt_template: str = ORLM_PROMPT_TEMPLATE
    prompt_version: str = ORLM_PROMPT_VERSION
    # Upstream vLLM uses max_tokens=None, which resolves to this checkpoint's
    # max_model_len (8192).
    max_new_tokens: int = 8192
    temperature: float = 0.0  # Official eval/generate.py greedy path.
    top_p: float = 1.0
    top_k: int = 1
    decoding_method: str = "greedy"
    stop_tokens: tuple[str, ...] = ("</s>",)
    max_seq_len: int = 8192
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    device_map: str = "auto"
    seed: int = 0
    solver: str = "coptpy"
    min_gpu_memory_gb: int = 24
    finetuning_required: bool = False
    requires_external_api: bool = False

    def generation_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "decoding_method": self.decoding_method,
            "stop_tokens": list(self.stop_tokens),
            "max_seq_len": self.max_seq_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "seed": self.seed,
        }
