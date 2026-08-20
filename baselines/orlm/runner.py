"""Inference-ready ORLM runner with lazy real backend and injectable mocks."""
from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from baselines.orlm.config import OrlmConfig


class GenerationBackend(Protocol):
    def generate(self, prompt: str, config: OrlmConfig) -> tuple[str, dict[str, Any]]: ...


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
        return {
            "raw_output": self.raw_output,
            "status": self.status,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_sha256": self.prompt_sha256,
            "runtime_seconds": self.runtime_seconds,
            "token_counts": self.token_counts,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "environment": self.environment,
        }


class TransformersBackend:
    """Lazy Transformers backend; importing this class does not load weights."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self._loaded_key: tuple[str, str | None] | None = None

    def _load(self, config: OrlmConfig) -> None:
        if self._loaded_key == (config.model_id, config.model_revision):
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
        self._loaded_key = (config.model_id, config.model_revision)

    def generate(self, prompt: str, config: OrlmConfig) -> tuple[str, dict[str, Any]]:
        self._load(config)
        import torch
        if config.seed is not None:
            torch.manual_seed(config.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.decoding_method != "greedy",
        }
        if config.decoding_method != "greedy":
            kwargs.update(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k)
        if config.stop_tokens and hasattr(self.tokenizer, "eos_token_id"):
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop_token in config.stop_tokens:
            if stop_token in text:
                text = text.split(stop_token, 1)[0]
        return text, {"completion_tokens": int(len(new_tokens)), "prompt_tokens": int(prompt_len), "total_tokens": int(output.shape[-1])}


@dataclass
class OrlmRunner:
    config: OrlmConfig = field(default_factory=OrlmConfig)
    backend: GenerationBackend | None = None
    _backend_instance: GenerationBackend | None = field(default=None, init=False, repr=False)

    def generate(self, prompt: str) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            return GenerationResult("", "INPUT_ERROR", self.config.model_id, self.config.model_revision, "", 0.0, error_category="empty_prompt", error_message="prompt must be non-empty")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self.backend is not None:
            backend = self.backend
        else:
            if self._backend_instance is None:
                self._backend_instance = TransformersBackend()
            backend = self._backend_instance
        start = time.perf_counter()
        error_message = ""
        try:
            raw, usage = backend.generate(prompt, self.config)
            return GenerationResult(raw, "COMPLETED", self.config.model_id, self.config.model_revision, prompt_hash, time.perf_counter() - start, usage, environment={"python": platform.python_version(), "platform": platform.platform()})
        except TimeoutError as exc:
            category = "generation_timeout"
            error_message = str(exc)
        except Exception as exc:  # Normalize backend failures for later batch runs.
            category = str(exc) if str(exc) in {"transformers_backend_unavailable"} else "generation_error"
            error_message = str(exc)
        return GenerationResult("", "FAILED", self.config.model_id, self.config.model_revision, prompt_hash, time.perf_counter() - start, error_category=category, error_message=error_message[:500], environment={"python": platform.python_version(), "platform": platform.platform()})

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Stable batch interface; default implementation avoids hidden concurrency."""
        return [self.generate(prompt) for prompt in prompts]
