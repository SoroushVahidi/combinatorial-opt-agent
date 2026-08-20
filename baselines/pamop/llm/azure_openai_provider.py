"""Azure OpenAI provider adapter.

Uses the ``openai`` SDK against an OpenAI-compatible Azure endpoint (the
``.../openai/v1`` path style, verified live during this milestone against
this workstation's configured resource -- distinct from the older
``azure_endpoint=``-style ``AzureOpenAI`` client, which would double up the
path given how this workstation's endpoint is already formatted).

Environment variables checked (first non-empty wins for each), matching
both this repo's/workstation's existing naming and the generic
``AZURE_API_*`` convention some other local tooling uses for the same
resource:

  - endpoint:  ``AZURE_OPENAI_ENDPOINT``, then ``AZURE_API_BASE``
  - api key:   ``AZURE_OPENAI_API_KEY``, then ``AZURE_API_KEY``
  - api version: ``AZURE_OPENAI_API_VERSION``, then ``AZURE_API_VERSION``
    (recorded in ``LLMResponse``'s reproducibility metadata via a
    provider-specific field on the raw call result; not required for the
    ``.../openai/v1`` endpoint style itself)

``ModelConfig.model`` is the **Azure deployment name** (e.g.
``"gpt-4.1-mini"``), not necessarily the underlying model id -- Azure lets
a deployment name differ from the model it points at. The actual served
model snapshot (e.g. ``"gpt-4.1-mini-2025-04-14"``), when the API echoes it
back, is recorded separately (see ``deployment`` vs ``underlying_model`` in
``extraction``/``modeling`` call sites that log this).

Some Azure/OpenAI model families (verified this milestone: this
workstation's non-GPT-4 "strong" deployment) reject the standard
``max_tokens`` parameter and require ``max_completion_tokens`` instead.
This adapter tries ``max_tokens`` first and transparently retries once with
``max_completion_tokens`` on that specific error, rather than requiring
every caller to know which convention a given deployment uses.
"""

from __future__ import annotations

from .base import LLMProvider, get_env_token
from .types import ModelConfig, ProviderAuthError

_ENDPOINT_VARS = ("AZURE_OPENAI_ENDPOINT", "AZURE_API_BASE")
_KEY_VARS = ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY")
_API_VERSION_VARS = ("AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION")


class AzureOpenAIProvider(LLMProvider):
    name = "azure_openai"

    def _endpoint(self) -> str:
        return get_env_token(*_ENDPOINT_VARS)

    def _api_version(self) -> str:
        return get_env_token(*_API_VERSION_VARS)

    def _client(self):
        key = get_env_token(*_KEY_VARS)
        endpoint = self._endpoint()
        if not key or not endpoint:
            missing = []
            if not key:
                missing.append("/".join(_KEY_VARS))
            if not endpoint:
                missing.append("/".join(_ENDPOINT_VARS))
            raise ProviderAuthError(f"Azure OpenAI not configured: missing {', '.join(missing)}.")

        from openai import OpenAI

        return OpenAI(api_key=key, base_url=endpoint)

    def _call(self, prompt: str, config: ModelConfig) -> dict:
        client = self._client()
        kwargs: dict = {
            "model": config.model,  # Azure deployment name
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        }
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p

        try:
            response = self._create(client, kwargs, config.max_tokens, use_completion_tokens_param=False)
        except Exception as exc:  # noqa: BLE001 -- inspect, then either retry or re-raise unchanged
            if config.max_tokens is not None and "max_completion_tokens" in str(exc):
                response = self._create(client, kwargs, config.max_tokens, use_completion_tokens_param=True)
            else:
                raise

        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            # Azure/OpenAI echo the exact served snapshot here, which can
            # differ from the deployment name requested -- e.g. deployment
            # "gpt-4.1-mini" -> underlying_model "gpt-4.1-mini-2025-04-14".
            "underlying_model": getattr(response, "model", None),
        }

    @staticmethod
    def _create(client, kwargs: dict, max_tokens, *, use_completion_tokens_param: bool):
        call_kwargs = dict(kwargs)
        if max_tokens is not None:
            call_kwargs["max_completion_tokens" if use_completion_tokens_param else "max_tokens"] = max_tokens
        return client.chat.completions.create(**call_kwargs)
