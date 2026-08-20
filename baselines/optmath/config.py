"""OptMATH official-evaluation configuration and provenance."""
from __future__ import annotations

from dataclasses import dataclass


OPTMATH_REPOSITORY = "https://github.com/optsuite/OptMATH"
OPTMATH_UPSTREAM_REVISION = "f15bbc4477c70db85ad148df8bcc1b780bca0f8c"
OPTMATH_PRIMARY_MODEL = "Aurora-Gem/OptMATH-Qwen2.5-7B"
OPTMATH_OPTIONAL_MODEL = "Aurora-Gem/OptMATH-Qwen2.5-32B-Instruct"
OPTMATH_PROMPT_VERSION = "upstream-eval-evaluator-build-cot-prompt-v1"
OPTMATH_SYSTEM_PROMPT = "You are an expert in operations research and optimization."
OPTMATH_USER_TEMPLATE = """Below is an operations research question. Build a mathematical model and corresponding python code using `gurobipy` that appropriately addresses the question.

# Question:
{question}

# Instructions:
1. Output ONLY the Python code within a ```python code block
2. Start your code with: import gurobipy as gp
3. Name your model variable as `model`
4. Use <= instead of < in Gurobi constraints
5. After solving, print the objective value using: print(model.objVal)

# Response:
```python
import gurobipy as gp
from gurobipy import GRB

# Your code here

model.optimize()
print(model.objVal)
```"""


@dataclass(frozen=True)
class OptmathConfig:
    model_id: str = OPTMATH_PRIMARY_MODEL
    model_revision: str | None = None
    upstream_revision: str = OPTMATH_UPSTREAM_REVISION
    prompt_version: str = OPTMATH_PROMPT_VERSION
    temperature: float = 0.8  # Official eval/evaluator.py default.
    max_new_tokens: int = 8192  # Official --max-tokens default.
    top_p: float | None = None  # Official API path leaves this unspecified.
    do_sample: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"
    seed: int = 0
    solver: str = "gurobipy"
    timeout_seconds: int = 100
    numerical_tolerance: float = 0.05  # Official rounded relative tolerance.
    enable_official_conversion_fallback: bool = False

    def generation_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "seed": self.seed,
        }
