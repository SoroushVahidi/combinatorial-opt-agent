"""Lazy OptMATH runner with injectable backend and official metadata."""
from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from baselines.optmath.config import OptmathConfig
from baselines.optmath.prompt import PromptBundle


class GenerationBackend(Protocol):
    def generate(self, prompt: PromptBundle, config: OptmathConfig) -> tuple[str, dict[str, int | None]]: ...


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    status: str
    model_id: str
    model_revision: str | None
    prompt_sha256: str
    runtime_seconds: float
    token_counts: dict[str, int | None] = field(default_factory=dict)
    error_category: str | None = None
    error_message: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class TransformersBackend:
    """ADAPTED_OFFICIAL lazy backend for the released Qwen checkpoints."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self._loaded: tuple[str, str | None] | None = None

    def _load(self, config: OptmathConfig) -> None:
        if self._loaded == (config.model_id, config.model_revision):
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers_backend_unavailable") from exc
        dtype = getattr(torch, config.dtype, None)
        kwargs: dict[str, Any] = {"revision": config.model_revision, "device_map": config.device_map}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
        self._loaded = (config.model_id, config.model_revision)

    def generate(self, prompt: PromptBundle, config: OptmathConfig) -> tuple[str, dict[str, int | None]]:
        self._load(config)
        import torch
        torch.manual_seed(config.seed)
        inputs = self.tokenizer(prompt.user, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.do_sample,
                **({"top_p": config.top_p} if config.top_p is not None else {}),
            )
        prompt_len = inputs["input_ids"].shape[-1]
        completion = output[0][prompt_len:]
        return self.tokenizer.decode(completion, skip_special_tokens=True), {
            "prompt_tokens": int(prompt_len), "completion_tokens": int(len(completion)), "total_tokens": int(output.shape[-1])
        }


@dataclass
class OptmathRunner:
    config: OptmathConfig = field(default_factory=OptmathConfig)
    backend: GenerationBackend | None = None

    def generate(self, prompt: PromptBundle) -> GenerationResult:
        prompt_hash = hashlib.sha256(prompt.user.encode()).hexdigest()
        start = time.perf_counter()
        try:
            raw, usage = (self.backend or TransformersBackend()).generate(prompt, self.config)
            return GenerationResult(raw, "COMPLETED", self.config.model_id, self.config.model_revision, prompt_hash, time.perf_counter() - start, usage, environment={"python": platform.python_version(), "platform": platform.platform()})
        except TimeoutError as exc:
            category, message = "generation_timeout", str(exc)
        except Exception as exc:
            category, message = ("transformers_backend_unavailable" if str(exc) == "transformers_backend_unavailable" else "generation_error"), str(exc)
        return GenerationResult("", "FAILED", self.config.model_id, self.config.model_revision, prompt_hash, time.perf_counter() - start, error_category=category, error_message=message[:500], environment={"python": platform.python_version(), "platform": platform.platform()})

    def generate_batch(self, prompts: list[PromptBundle]) -> list[GenerationResult]:
        return [self.generate(prompt) for prompt in prompts]
