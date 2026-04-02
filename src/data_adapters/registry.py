"""Dataset adapter registry."""

from __future__ import annotations

from .complexor import ComplexORAdapter
from .gams_models import GAMSModelsAdapter
from .gurobi_modeling_examples import GurobiModelingExamplesAdapter
from .gurobi_optimods import GurobiOptimodsAdapter
from .miplib import MIPLIBAdapter
from .nl4opt import NL4OptAdapter
from .nlp4lp import NLP4LPAdapter
from .optimus import OptiMUSAdapter
from .optmath import OptMATHAdapter
from .or_library import ORLibraryAdapter
from .pyomo_examples import PyomoExamplesAdapter
from .text2zinc import Text2ZincAdapter


ADAPTERS = {
    "nlp4lp": NLP4LPAdapter,
    "nl4opt": NL4OptAdapter,
    "text2zinc": Text2ZincAdapter,
    "optmath": OptMATHAdapter,
    "complexor": ComplexORAdapter,
    "gurobi_modeling_examples": GurobiModelingExamplesAdapter,
    "gurobi_optimods": GurobiOptimodsAdapter,
    "gams_models": GAMSModelsAdapter,
    "miplib": MIPLIBAdapter,
    "or_library": ORLibraryAdapter,
    "pyomo_examples": PyomoExamplesAdapter,
    "optimus": OptiMUSAdapter,
}


def create_adapter(name: str):
    key = name.strip().lower()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown dataset adapter: {name}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[key]()


def list_datasets() -> list[str]:
    return sorted(ADAPTERS.keys())

