#!/usr/bin/env python3
"""LLM baselines for two-stage NLP4LP task:
1) schema retrieval
2) scalar slot instantiation

This module is intentionally interface-light so it can plug into existing
`run_setting(...)` flow without changing metric definitions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - runtime dependency check
    yaml = None


ROOT = Path(__file__).resolve().parent.parent


DEFAULT_CONFIG_PATH = ROOT / "configs" / "llm_baselines.yaml"
DEFAULT_CACHE_DIR = ROOT / "cache" / "llm_baselines"


def _load_yaml_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"LLM baseline config not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required for configs/llm_baselines.yaml (pip install pyyaml)")
    with open(path, encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid YAML config format in {path}")
    return obj


def _sha(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _coerce_number(v: Any) -> float | int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(v, str):
        s = v.strip().lower().replace(",", "")
        if not s:
            return None
        is_percent = s.endswith("%")
        if is_percent:
            s = s[:-1].strip()
        if s.startswith("$"):
            s = s[1:]
        try:
            x = float(s)
        except Exception:
            return None
        if is_percent:
            return x / 100.0
        if float(int(x)) == x:
            return int(x)
        return x
    return None


def _infer_kind(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, str) and "%" in v:
        return "percent"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and float(int(v)) != v:
            return "float"
        return "int"
    return "unknown"


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, p: int = 0, c: int = 0) -> None:
        self.prompt_tokens += max(0, int(p or 0))
        self.completion_tokens += max(0, int(c or 0))


class LLMTwoStageBaseline:
    """Two-stage LLM baseline with on-disk cache and usage accounting."""

    def __init__(
        self,
        method: str,
        catalog: list[dict],
        expected_type_fn,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        if method not in ("openai", "gemini"):
            raise ValueError(f"Unsupported LLM baseline: {method}")
        self.method = method
        self.catalog = catalog
        self.expected_type_fn = expected_type_fn
        self.cache_dir = Path(cache_dir) / method
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = _load_yaml_config(config_path)
        self.method_cfg = (self.cfg.get(method) or {}) if isinstance(self.cfg, dict) else {}
        self.retry_cfg = self.cfg.get("retry", {}) if isinstance(self.cfg, dict) else {}
        self.price_cfg = self.cfg.get("pricing_usd_per_1k_tokens", {}) if isinstance(self.cfg, dict) else {}
        self.usage = LLMUsage()
        self._schema_text = self._build_schema_list_text()
        self._last_schema_for_query: dict[str, str] = {}

        if self.method == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self.client = OpenAI(api_key=api_key)
        else:
            import google.generativeai as genai

            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            genai.configure(api_key=api_key)
            self.client = genai

    def _build_schema_list_text(self) -> str:
        lines: list[str] = []
        for p in self.catalog:
            pid = p.get("id", "")
            desc = (p.get("description") or "").strip().replace("\n", " ")
            if pid:
                lines.append(f"- {pid}: {desc[:600]}")
        return "\n".join(lines)

    def _cache_path(self, stage: str, payload: dict[str, Any]) -> Path:
        key = _sha({"stage": stage, "method": self.method, "payload": payload})
        return self.cache_dir / f"{stage}_{key}.json"

    async def _retry(self, coro_factory):
        max_retries = int(self.retry_cfg.get("max_retries", 3))
        backoff = float(self.retry_cfg.get("base_backoff_seconds", 1.0))
        for i in range(max_retries + 1):
            try:
                return await coro_factory()
            except Exception:
                if i >= max_retries:
                    raise
                await asyncio.sleep(backoff * (2**i))
        raise RuntimeError("unreachable")

    async def _openai_json(self, prompt: str, schema_hint: str) -> dict[str, Any]:
        model = self.method_cfg.get("model", "gpt-4o-mini")
        temp = float(self.method_cfg.get("temperature", 0.0))
        max_tokens = int(self.method_cfg.get("max_tokens", 1024))

        def _call():
            return self.client.chat.completions.create(
                model=model,
                temperature=temp,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return strict JSON only. No prose."},
                    {"role": "user", "content": f"{schema_hint}\n\n{prompt}"},
                ],
            )

        resp = await asyncio.to_thread(_call)
        self.usage.add(
            getattr(resp.usage, "prompt_tokens", 0),
            getattr(resp.usage, "completion_tokens", 0),
        )
        txt = (resp.choices[0].message.content or "{}").strip()
        try:
            return json.loads(txt)
        except Exception:
            return {}

    async def _gemini_json(self, prompt: str, schema_hint: str) -> dict[str, Any]:
        # Allow override without editing YAML: export GEMINI_MODEL=gemini-2.0-flash
        model_name = (
            (os.environ.get("GEMINI_MODEL") or "").strip()
            or self.method_cfg.get("model", "gemini-2.0-flash")
        )
        temp = float(self.method_cfg.get("temperature", 0.0))
        max_tokens = int(self.method_cfg.get("max_tokens", 1024))

        def _call():
            model = self.client.GenerativeModel(model_name)
            return model.generate_content(
                f"{schema_hint}\n\n{prompt}",
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                },
            )

        resp = await asyncio.to_thread(_call)
        um = getattr(resp, "usage_metadata", None)
        self.usage.add(
            getattr(um, "prompt_token_count", 0) if um else 0,
            getattr(um, "candidates_token_count", 0) if um else 0,
        )
        txt = (getattr(resp, "text", None) or "{}").strip()
        try:
            return json.loads(txt)
        except Exception:
            return {}

    async def _llm_json(self, stage: str, payload: dict[str, Any], prompt: str, schema_hint: str) -> dict[str, Any]:
        cpath = self._cache_path(stage, payload)
        if cpath.exists():
            with open(cpath, encoding="utf-8") as f:
                return json.load(f)

        async def _run():
            if self.method == "openai":
                return await self._openai_json(prompt, schema_hint)
            return await self._gemini_json(prompt, schema_hint)

        out = await self._retry(_run)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    def rank(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        payload = {"query": query, "top_k": top_k, "schema_count": len(self.catalog)}
        prompt = (
            "Task: choose exactly one best schema id for this optimization query.\n"
            "Return JSON with key schema_id only.\n\n"
            f"Query:\n{query}\n\n"
            "Candidate schemas (id: description):\n"
            f"{self._schema_text}"
        )
        schema_hint = '{"schema_id": "nlp4lp_test_..."}'
        out = asyncio.run(self._llm_json("stage1_retrieval", payload, prompt, schema_hint))
        sid = str(out.get("schema_id", "")).strip()
        if not sid:
            return []
        self._last_schema_for_query[_sha(query)] = sid
        return [(sid, 1.0)]

    def instantiate(self, query: str, schema_id: str, slots: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"query": query, "schema_id": schema_id, "slots": slots}
        slot_lines = "\n".join([f"- {s['name']} (expected_type={s['expected_type']})" for s in slots])
        prompt = (
            "Task: fill scalar parameter slots from the query text.\n"
            "Return JSON object with key `slot_values` mapping slot name -> numeric value.\n"
            "If a value is missing, omit the slot key.\n\n"
            f"Query:\n{query}\n\n"
            f"Schema id: {schema_id}\n"
            f"Scalar slots:\n{slot_lines}"
        )
        schema_hint = '{"slot_values": {"SlotName": 123}}'
        out = asyncio.run(self._llm_json("stage2_instantiation", payload, prompt, schema_hint))
        sv = out.get("slot_values", {})
        if not isinstance(sv, dict):
            return {}
        return sv

    def llm_assign(self, query: str, schema_id: str, expected_scalar: list[str]) -> tuple[dict[str, Any], int, int]:
        slots = [{"name": p, "expected_type": self.expected_type_fn(p)} for p in expected_scalar]
        raw = self.instantiate(query, schema_id, slots)
        filled: dict[str, Any] = {}
        type_matches = 0
        for p in expected_scalar:
            if p not in raw:
                continue
            v = _coerce_number(raw.get(p))
            if v is None:
                continue
            filled[p] = v
            kind = _infer_kind(raw.get(p))
            et = self.expected_type_fn(p)
            if et == "float" and kind == "int":
                type_matches += 1
            elif et == "int" and kind in ("int",):
                type_matches += 1
            elif et == "percent" and kind in ("percent", "float", "int"):
                type_matches += 1
            elif et == "currency" and kind in ("int", "float"):
                type_matches += 1
            elif et == kind:
                type_matches += 1
        return filled, len(filled), type_matches

    def cost_estimate_usd(self) -> float:
        in_rate = float((self.price_cfg.get(self.method) or {}).get("input_per_1k", 0.0))
        out_rate = float((self.price_cfg.get(self.method) or {}).get("output_per_1k", 0.0))
        return (self.usage.prompt_tokens / 1000.0) * in_rate + (self.usage.completion_tokens / 1000.0) * out_rate

