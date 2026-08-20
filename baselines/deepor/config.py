"""DeepOR provenance and inference configuration.

The AAAI paper specifies Qwen3-8B and greedy decoding, but does not release a
DeepOR checkpoint or an inference repository.  ``model_id`` is consequently
unset by default: a proxy must never be mistaken for DeepOR.
"""
from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://ojs.aaai.org/index.php/AAAI/article/view/40699"
PAPER_PDF_URL = "https://ojs.aaai.org/index.php/AAAI/article/download/40699/44660"
DOI = "10.1609/aaai.v40i40.40699"
PAPER_REVISION = "AAAI-26 proceedings, published 2026-03-14"
PROMPT_VERSION = "deepor-paper-reconstruction-v1"
DEFAULT_REASONING_TAGS = ("<think>", "</think>")

@dataclass(frozen=True)
class DeepORConfig:
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str = PROMPT_VERSION
    paper_revision: str = PAPER_REVISION
    solver: str = "pyomo"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    do_sample: bool = False
    repetition_penalty: float = 1.0
    max_new_tokens: int = 8192
    reasoning_budget: int | None = None
    seed: int = 0
    timeout_seconds: int = 100
    rollouts: int = 1
    model_path: str | None = None

    def generation_dict(self) -> dict[str, object]:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "do_sample": self.do_sample,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_new_tokens, "seed": self.seed,
                "rollouts": self.rollouts}
