"""Fixed zero-shot prompt for the GENERAL_PURPOSE_LLM_BASELINE.

No gold schema, no gold objective, no demonstrations, no tool calls, no
iterative correction: the model simply receives the problem text and is
asked to produce `gurobipy` code. The prompt is intentionally aligned with
the OptMATH protocol so the ONLY difference between this baseline and the
OptMATH baseline is the model itself.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from baselines.generic_llm.config import GENERIC_LLM_SYSTEM_PROMPT, GENERIC_LLM_USER_TEMPLATE, GenericLLMConfig


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    version: str
    user_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"system": self.system, "user": self.user, "version": self.version, "user_sha256": self.user_sha256}


def build_prompt(question: str, config: GenericLLMConfig | None = None) -> PromptBundle:
    config = config or GenericLLMConfig()
    if not isinstance(question, str) or not question.strip():
        raise ValueError("generic LLM question must be a non-empty string")
    user = GENERIC_LLM_USER_TEMPLATE.format(question=question.strip())
    return PromptBundle(GENERIC_LLM_SYSTEM_PROMPT, user, config.prompt_version, hashlib.sha256(user.encode()).hexdigest())