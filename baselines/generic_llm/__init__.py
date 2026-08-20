"""GENERAL_PURPOSE_LLM_BASELINE package.

One strong general-purpose LLM (Azure OpenAI `gpt-5.4`), zero-shot prompted
to formulate NLP4LP problems as `gurobipy` code. This is a distinct baseline
-- never relabeled as ORLM / OptMATH / DeepOR / OR-R1 / PaMOP.
"""
from baselines.generic_llm.config import GenericLLMConfig
from baselines.generic_llm.result_schema import GenericLLMResult
from baselines.generic_llm.runner import GenerationResult

__all__ = ["GenericLLMConfig", "GenericLLMResult", "GenerationResult"]