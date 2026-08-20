"""Versioned, content-hashed prompt templates.

Every template here is a reconstruction, never PaMOP's own wording -- see
PROVENANCE.md in this directory. ``load_prompt`` returns the raw template
text plus its content hash so callers (``extraction.py``) can stamp every
``LLMResponse`` with exactly which prompt version produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..llm.base import prompt_hash

PROMPTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    text: str
    content_hash: str

    def render(self, **kwargs: str) -> str:
        return self.text.format(**kwargs)


def load_prompt(filename: str) -> PromptTemplate:
    """``filename`` like ``"extraction_v1.txt"`` -> ``PromptTemplate``."""
    path = PROMPTS_DIR / filename
    text = path.read_text(encoding="utf-8")
    stem = path.stem  # e.g. "extraction_v1"
    if "_v" in stem:
        name, version = stem.rsplit("_v", 1)
        version = f"v{version}"
    else:
        name, version = stem, "v0"
    return PromptTemplate(name=name, version=version, text=text, content_hash=prompt_hash(text))
