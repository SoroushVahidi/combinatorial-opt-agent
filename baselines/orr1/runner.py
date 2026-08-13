"""Inference-ready OR-R1 runner with lazy real backends and injectable mocks.

Upstream `eval/generate.py` uses vLLM (`LLM` + `SamplingParams`, `n=topk`)
with a local model path only (`assert os.path.exists(args.model_name_or_path)`)
and applies `tokenizer.apply_chat_template` on top of the `TEMPLATE_q2mc_en`
prompt. `VLLMBackend` mirrors that path exactly; `TransformersBackend` is a
non-official fallback for environments without vLLM, matching the pattern
already used by the ORLM/DeepOR baselines in this repository. No weights are
downloaded or loaded by importing this module.
"""
from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from baselines.orr1.config import OrR1Config


class GenerationBackend(Protocol):
    def generate(self, prompt: str, config: OrR1Config) -> tuple[list[str], dict[str, Any]]: ...


@dataclass(frozen=True)
class GenerationResult:
    raw_outputs: tuple[str, ...]  # One entry per rollout (`config.rollouts`).
    status: str
    model_id: str | None
    model_revision: str | None
    checkpoint_stage: str
    prompt_sha256: str
    runtime_seconds: float
    token_counts: dict[str, int | None] = field(default_factory=dict)
    error_category: str | None = None
    error_message: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["raw_outputs"] = list(self.raw_outputs)
        return d


class VLLMBackend:
    """Lazy vLLM backend mirroring upstream `eval/generate.py` exactly."""

    def __init__(self) -> None:
        self._llm = None
        self._loaded_key: tuple[str, int] | None = None

    def _load(self, config: OrR1Config) -> None:
        identifier = config.model_path or config.model_id
        key = (identifier, config.tensor_parallel_size)
        if self._loaded_key == key:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vllm_backend_unavailable") from exc
        self._llm = LLM(model=identifier, tensor_parallel_size=config.tensor_parallel_size)
        self._loaded_key = key

    def generate(self, prompt: str, config: OrR1Config) -> tuple[list[str], dict[str, Any]]:
        if not config.model_path and not config.model_id:
            raise RuntimeError("checkpoint_unavailable")
        self._load(config)
        from vllm import SamplingParams
        if config.decoding_method == "greedy":
            params = SamplingParams(n=config.topk, temperature=0, top_p=1, max_tokens=config.max_tokens, stop=list(config.stop_tokens))
        else:
            params = SamplingParams(n=config.topk, temperature=config.temperature, top_p=config.top_p, max_tokens=config.max_tokens, stop=list(config.stop_tokens))
        [generation] = self._llm.generate([prompt], params)
        texts = [o.text for o in generation.outputs]
        return texts, {"rollout_count": len(texts)}


class TransformersBackend:
    """Non-official fallback backend; not used by upstream `eval/generate.py`."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self._loaded_key: tuple[str | None, str | None] | None = None

    def _load(self, config: OrR1Config) -> None:
        identifier = config.model_path or config.model_id
        key = (identifier, config.model_revision)
        if self._loaded_key == key:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers_backend_unavailable") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(identifier, revision=config.model_revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(identifier, revision=config.model_revision, trust_remote_code=True, device_map="auto")
        self._loaded_key = key

    def generate(self, prompt: str, config: OrR1Config) -> tuple[list[str], dict[str, Any]]:
        if not config.model_path and not config.model_id:
            raise RuntimeError("checkpoint_unavailable")
        self._load(config)
        import torch
        torch.manual_seed(config.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        outputs: list[str] = []
        with torch.inference_mode():
            for _ in range(config.rollouts):
                generated = self.model.generate(
                    **inputs, max_new_tokens=config.max_tokens,
                    do_sample=config.decoding_method != "greedy",
                    temperature=config.temperature, top_p=config.top_p,
                )
                outputs.append(self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True))
        return outputs, {"prompt_tokens": int(prompt_len), "rollout_count": len(outputs)}


@dataclass
class OrR1Runner:
    config: OrR1Config = field(default_factory=OrR1Config)
    backend: GenerationBackend | None = None

    def generate(self, prompt: str) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            return GenerationResult((), "INPUT_ERROR", self.config.model_id, self.config.model_revision,
                                     self.config.checkpoint_stage, "", 0.0, error_category="empty_prompt",
                                     error_message="prompt must be non-empty")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        backend = self.backend or VLLMBackend()
        start = time.perf_counter()
        try:
            raw_outputs, usage = backend.generate(prompt, self.config)
            return GenerationResult(
                tuple(raw_outputs), "COMPLETED", self.config.model_id, self.config.model_revision,
                self.config.checkpoint_stage, prompt_hash, time.perf_counter() - start, usage,
                environment={"python": platform.python_version(), "platform": platform.platform()},
            )
        except Exception as exc:
            category = str(exc) if str(exc) in {"checkpoint_unavailable", "vllm_backend_unavailable", "transformers_backend_unavailable"} else "generation_error"
            return GenerationResult(
                (), "FAILED", self.config.model_id, self.config.model_revision, self.config.checkpoint_stage,
                prompt_hash, time.perf_counter() - start, error_category=category, error_message=str(exc)[:500],
                environment={"python": platform.python_version(), "platform": platform.platform()},
            )
