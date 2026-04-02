"""Dataset adapter registry."""

from __future__ import annotations

from .complexor import ComplexORAdapter
from .nl4opt import NL4OptAdapter
from .nlp4lp import NLP4LPAdapter
from .optmath import OptMATHAdapter
from .text2zinc import Text2ZincAdapter


ADAPTERS = {
    "nlp4lp": NLP4LPAdapter,
    "nl4opt": NL4OptAdapter,
    "text2zinc": Text2ZincAdapter,
    "optmath": OptMATHAdapter,
    "complexor": ComplexORAdapter,
}


def create_adapter(name: str):
    key = name.strip().lower()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown dataset adapter: {name}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[key]()


def list_datasets() -> list[str]:
    return sorted(ADAPTERS.keys())

