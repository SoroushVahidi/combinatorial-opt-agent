"""Provider-agnostic LLM interface for the PaMOP reproduction.

``generate(prompt, model_config) -> LLMResponse`` is the entire contract
(see ``base.LLMProvider.generate``, ``types.LLMResponse``); each of
``openai_provider``, ``gemini_provider``, ``cohere_provider``,
``fireworks_provider``, ``cloudrift_provider`` implements it for one
provider, and ``registry.get_provider(name)`` looks one up by name.

No module in this package prints, logs, or returns an API key/token value.
"""

from .base import LLMProvider
from .registry import get_provider, list_providers
from .types import LLMResponse, ModelConfig, ProviderAuthError, ProviderCallError

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ModelConfig",
    "ProviderAuthError",
    "ProviderCallError",
    "get_provider",
    "list_providers",
]
