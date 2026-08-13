"""Lazy, injectable runner; no checkpoint is bundled or downloaded."""
from __future__ import annotations
import platform, random, time
from dataclasses import dataclass, field
from typing import Any, Protocol
from .config import DeepORConfig
from .prompt import PromptBundle

class Backend(Protocol):
    def generate(self, prompt: PromptBundle, config: DeepORConfig) -> tuple[str, dict[str, int | None]]: ...

@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    status: str
    model_id: str | None
    model_revision: str | None
    prompt_sha256: str
    runtime_seconds: float
    token_counts: dict[str, int | None] = field(default_factory=dict)
    rollout_count: int = 1
    error_category: str | None = None
    error_message: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    def to_dict(self): return self.__dict__.copy()

class TransformersBackend:
    def generate(self, prompt: PromptBundle, config: DeepORConfig):
        if not config.model_id and not config.model_path: raise RuntimeError("checkpoint_unavailable")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc: raise RuntimeError("transformers_backend_unavailable") from exc
        identifier = config.model_path or config.model_id
        tokenizer = AutoTokenizer.from_pretrained(identifier, revision=config.model_revision)
        model = AutoModelForCausalLM.from_pretrained(identifier, revision=config.model_revision, device_map="auto")
        random.seed(config.seed); torch.manual_seed(config.seed)
        inputs = tokenizer(prompt.user, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=config.max_new_tokens,
                temperature=config.temperature, top_p=config.top_p,
                do_sample=config.do_sample, repetition_penalty=config.repetition_penalty)
        n = inputs["input_ids"].shape[-1]
        completion = output[0][n:]
        return tokenizer.decode(completion, skip_special_tokens=True), {"prompt_tokens": int(n), "generated_tokens": int(len(completion)), "total_tokens": int(output.shape[-1])}

@dataclass
class DeepORRunner:
    config: DeepORConfig = field(default_factory=DeepORConfig)
    backend: Backend | None = None
    def generate(self, prompt: PromptBundle) -> GenerationResult:
        start = time.perf_counter()
        try:
            raw, usage = (self.backend or TransformersBackend()).generate(prompt, self.config)
            return GenerationResult(raw, "COMPLETED", self.config.model_id or self.config.model_path, self.config.model_revision, prompt.sha256, time.perf_counter()-start, usage, self.config.rollouts, environment={"python": platform.python_version(), "platform": platform.platform()})
        except Exception as exc:
            category = str(exc) if str(exc) in {"checkpoint_unavailable", "transformers_backend_unavailable"} else "generation_error"
            return GenerationResult("", "FAILED", self.config.model_id or self.config.model_path, self.config.model_revision, prompt.sha256, time.perf_counter()-start, rollout_count=self.config.rollouts, error_category=category, error_message=str(exc)[:500], environment={"python": platform.python_version()})
