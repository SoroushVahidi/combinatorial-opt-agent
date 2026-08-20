"""Official OptMATH evaluator prompt construction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from baselines.optmath.config import OPTMATH_SYSTEM_PROMPT, OPTMATH_USER_TEMPLATE, OptmathConfig


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    version: str
    user_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"system": self.system, "user": self.user, "version": self.version, "user_sha256": self.user_sha256}


def build_prompt(question: str, config: OptmathConfig | None = None) -> PromptBundle:
    config = config or OptmathConfig()
    if not isinstance(question, str) or not question.strip():
        raise ValueError("OptMATH question must be a non-empty string")
    user = OPTMATH_USER_TEMPLATE.format(question=question.strip())
    return PromptBundle(OPTMATH_SYSTEM_PROMPT, user, config.prompt_version, hashlib.sha256(user.encode()).hexdigest())
