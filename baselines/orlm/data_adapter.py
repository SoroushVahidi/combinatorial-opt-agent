"""Adapt an NLP4LP query into ORLM's expected prompt format.

ORLM was not evaluated on NLP4LP by its authors (see README.md "Does ORLM
cover NLP4LP?"). This module only wraps a query in the prompt template --
it does not attempt to align ORLM's expected input granularity with
NLP4LP's schema-conditioned setup, since ORLM expects an open-ended NL
problem description and produces its own model from scratch, unlike this
repository's schema-conditioned grounding pipeline.
"""
from __future__ import annotations

from baselines.orlm.config import OrlmConfig


def build_orlm_prompt(nlp4lp_query: str, config: OrlmConfig | None = None) -> str:
    """Wrap a raw NLP4LP query string in ORLM's official prompt template.

    Does not call the model -- see runner.py for that, which is not
    implemented (no GPU/weights available in this environment).
    """
    config = config or OrlmConfig()
    return config.prompt_template.format(question=nlp4lp_query.strip())
