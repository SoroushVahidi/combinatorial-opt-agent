"""Configuration for the single GENERAL_PURPOSE_LLM_BASELINE.

This is deliberately NOT ORLM / OptMATH / DeepOR / OR-R1 / PaMOP: it is one
strong general-purpose LLM, prompted zero-shot to formulate the NLP4LP
problem as `gurobipy` code, evaluated with the same machine-checkable
parse/static-validation path as the other full-formulation baselines.

Model selection (recorded 2026-08-15, before any run): Azure OpenAI
deployment `gpt-5.4` was chosen because (1) it is the strongest identifiable
frontier deployment available on this workstation, (2) it is already
verified usable via the PaMOP gpt-5.4 fidelity diagnostic (6/6 completed),
and (3) Azure echoes the exact served model snapshot, giving a stable,
reproducible identity. See docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md.
"""
from __future__ import annotations

from dataclasses import dataclass

GENERIC_LLM_PROVIDER = "azure_openai"
GENERIC_LLM_DEPLOYMENT = "gpt-5.4"  # Azure deployment name (strongest available).
GENERIC_LLM_DEPLOYMENT_ENV = "AZURE_OPENAI_STRONG_DEPLOYMENT"
GENERIC_LLM_PROMPT_VERSION = "generic-llm-zero-shot-gurobipy-v1"
GENERIC_LLM_SYSTEM_PROMPT = "You are an expert in operations research and optimization."

GENERIC_LLM_USER_TEMPLATE = """Below is an optimization problem. Write a complete `gurobipy` Python program that formulates and solves it.

# Problem:
{question}

# Instructions:
1. Output ONLY the Python code within a single ```python code block
2. Start your code with: import gurobipy as gp
3. Name your model variable as `model`
4. After solving, print the objective value using: print(model.objVal)

# Response:
```python
import gurobipy as gp
from gurobipy import GRB

# Your code here

model.optimize()
print(model.objVal)
```"""


@dataclass(frozen=True)
class GenericLLMConfig:
    provider: str = GENERIC_LLM_PROVIDER
    deployment: str = GENERIC_LLM_DEPLOYMENT
    prompt_version: str = GENERIC_LLM_PROMPT_VERSION
    temperature: float = 0.0
    max_tokens: int = 8192
    top_p: float | None = None
    solver: str = "gurobipy"
    timeout_seconds: int = 120
    numerical_tolerance: float = 0.05

    def generation_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "deployment": self.deployment,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }