"""Provider name -> adapter class registry.

Each entry is instantiated lazily so importing this module never requires
every provider SDK to be installed -- only the one actually requested.
"""

from __future__ import annotations

from .base import LLMProvider

_PROVIDER_IMPORTERS = {
    "openai": ("baselines.pamop.llm.openai_provider", "OpenAIProvider"),
    "gemini": ("baselines.pamop.llm.gemini_provider", "GeminiProvider"),
    "cohere": ("baselines.pamop.llm.cohere_provider", "CohereProvider"),
    "fireworks": ("baselines.pamop.llm.fireworks_provider", "FireworksProvider"),
    "cloudrift": ("baselines.pamop.llm.cloudrift_provider", "CloudRiftProvider"),
}


class UnknownProviderError(ValueError):
    pass


def list_providers() -> list[str]:
    return sorted(_PROVIDER_IMPORTERS)


def get_provider(name: str, **kwargs) -> LLMProvider:
    if name not in _PROVIDER_IMPORTERS:
        raise UnknownProviderError(
            f"unknown provider {name!r}; registered providers: {list_providers()}"
        )
    module_path, class_name = _PROVIDER_IMPORTERS[name]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
