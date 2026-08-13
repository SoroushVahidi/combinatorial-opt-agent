"""Paper-level prompt reconstruction.

The proceedings describe the reasoning stages and generated Pyomo program,
but do not publish a literal inference prompt.  This template is therefore
explicitly PAPER_RECONSTRUCTED, not an official prompt claim.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from .config import DeepORConfig

@dataclass(frozen=True)
class PromptBundle:
    user: str
    version: str
    sha256: str
    def to_dict(self): return {"user": self.user, "version": self.version, "sha256": self.sha256}

def build_prompt(problem_text: str, config: DeepORConfig | None = None) -> PromptBundle:
    config = config or DeepORConfig()
    if not isinstance(problem_text, str) or not problem_text.strip(): raise ValueError("problem text must be non-empty")
    user = ("You are an expert in operations research optimization modeling.\n"
            "Reason step by step about the problem, then provide a complete "
            "Pyomo optimization model. Preserve the reasoning and put the "
            "final executable Python model in a ```python block.\n\n"
            "Problem description:\n" + problem_text)
    return PromptBundle(user, config.prompt_version, hashlib.sha256(user.encode()).hexdigest())
