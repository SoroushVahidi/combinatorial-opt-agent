#!/usr/bin/env python3
"""LLM baselines for two-stage NLP4LP task:
1) schema retrieval
2) scalar slot instantiation

This module is intentionally interface-light so it can plug into existing
`run_setting(...)` flow without changing metric definitions.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import math
import os
import re
import sys
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
# Written by `pick-model --persist-selected` / `--write-selected` for downstream steps.
DEFAULT_GEMINI_MODEL_ARTIFACT = ROOT / "results" / "llm_baselines" / "gemini_selected_model.json"
# Default when YAML/env omit model — prefer free-tier-friendly id; usability is account-specific (run preflight).
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class GeminiHardQuotaError(RuntimeError):
    """Google AI reports zero quota for this model/path (e.g. `limit: 0` on free tier).

    Retries do not help; fix billing, switch project/account, or pick a model with
    visible nonzero limits in AI Studio. Creating another API key in the same project
    does **not** increase quota.
    """


def is_gemini_hard_zero_quota_error(exc: BaseException) -> bool:
    """True when the error text indicates a hard zero quota (not a transient 429)."""
    if isinstance(exc, GeminiHardQuotaError):
        return True
    msg = str(exc)
    # Google often embeds: "Quota exceeded for metric: ... limit: 0"
    if not re.search(r"limit\s*:\s*0\b", msg, flags=re.I):
        return False
    # Avoid false positives: require quota/free-tier context or API exception type
    try:
        from google.api_core import exceptions as gexc

        if isinstance(exc, gexc.ResourceExhausted):
            return True
    except Exception:
        pass
    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 429:
            return True
    except Exception:
        pass
    low = msg.lower()
    if "quota" in low or "free_tier" in low or "resourceexhausted" in type(exc).__name__.lower():
        return True
    return False


def _normalize_gemini_model_name(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("models/"):
        return n[7:]
    return n


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
            _apply_optional_gemini_thread_env()
            from google import genai as google_genai

            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._gemini_client = google_genai.Client(api_key=api_key)

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

    def _retry_sleep_seconds(self, attempt: int, exc: BaseException) -> float:
        """Compute sleep before retry. Honors Gemini/Vertex 'Please retry in Xs' on 429."""
        backoff = float(self.retry_cfg.get("base_backoff_seconds", 1.0))
        base = backoff * (2**attempt)
        msg = str(exc).lower()
        # Google Generative Language API often embeds: "Please retry in 50.12s."
        m = re.search(r"retry in ([\d.]+)\s*s", msg)
        if m:
            return max(base, float(m.group(1)) + 2.0)
        # Quota / rate limit: wait at least ~1 minute if message suggests retry later
        if "resourceexhausted" in type(exc).__name__.lower() or "429" in msg or "quota" in msg:
            return max(base, 60.0)
        return base

    async def _retry(self, coro_factory):
        max_retries = int(self.retry_cfg.get("max_retries", 3))
        for i in range(max_retries + 1):
            try:
                return await coro_factory()
            except GeminiHardQuotaError:
                raise
            except Exception as e:
                if is_gemini_hard_zero_quota_error(e):
                    raise GeminiHardQuotaError(str(e)) from e
                if i >= max_retries:
                    raise
                delay = self._retry_sleep_seconds(i, e)
                await asyncio.sleep(delay)
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
        # Allow override without editing YAML: export GEMINI_MODEL=...
        model_name = (
            (os.environ.get("GEMINI_MODEL") or "").strip()
            or self.method_cfg.get("model", DEFAULT_GEMINI_MODEL)
        )
        temp = float(self.method_cfg.get("temperature", 0.0))
        max_tokens = int(self.method_cfg.get("max_tokens", 1024))

        def _call():
            from google.genai import types as genai_types

            return self._gemini_client.models.generate_content(
                model=model_name,
                contents=f"{schema_hint}\n\n{prompt}",
                config=genai_types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
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


# ── Gemini preflight / model discovery (batch jobs & CLIs) ───────────────────


def resolve_gemini_model_from_config(config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    cfg = _load_yaml_config(config_path)
    gem = (cfg.get("gemini") or {}) if isinstance(cfg, dict) else {}
    raw = (os.environ.get("GEMINI_MODEL") or "").strip() or (
        (gem.get("model") if isinstance(gem, dict) else None) or DEFAULT_GEMINI_MODEL
    )
    return _normalize_gemini_model_name(str(raw))


def gemini_fallback_candidates(config_path: Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Ordered candidates: GEMINI_MODEL (or yaml model), yaml fallback_models, GEMINI_MODEL_FALLBACKS."""
    cfg = _load_yaml_config(config_path)
    gem = (cfg.get("gemini") or {}) if isinstance(cfg, dict) else {}
    out: list[str] = []
    env_m = (os.environ.get("GEMINI_MODEL") or "").strip()
    yaml_m = (gem.get("model") if isinstance(gem, dict) else None) or ""
    if env_m:
        out.append(_normalize_gemini_model_name(env_m))
        if yaml_m and _normalize_gemini_model_name(str(yaml_m)) != _normalize_gemini_model_name(env_m):
            out.append(_normalize_gemini_model_name(str(yaml_m)))
    elif yaml_m:
        out.append(_normalize_gemini_model_name(str(yaml_m)))
    fb = gem.get("fallback_models") if isinstance(gem, dict) else None
    if isinstance(fb, list):
        for x in fb:
            if isinstance(x, str) and x.strip():
                out.append(_normalize_gemini_model_name(x.strip()))
    env_fb = (os.environ.get("GEMINI_MODEL_FALLBACKS") or "").strip()
    if env_fb:
        for part in env_fb.split(","):
            p = part.strip()
            if p:
                out.append(_normalize_gemini_model_name(p))
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _apply_optional_gemini_thread_env() -> None:
    """Optional HPC-friendly defaults before gRPC / client libraries initialize (e.g. Wulver).

    Set ``GEMINI_LIMIT_RUNTIME_THREADS=1`` (default in ``run_gemini_llm_baselines.sbatch``) to
    cap OpenMP/BLAS/NumExpr threads and reduce ``pthread_create failed`` noise. This does **not**
    change API quota, auth, or retry behavior—only process thread fan-out before the ``google.genai`` client loads.

    Explicitly set ``GEMINI_LIMIT_RUNTIME_THREADS=0`` (or ``false``) in the job environment to skip.
    """
    v = (os.environ.get("GEMINI_LIMIT_RUNTIME_THREADS") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    if v not in ("1", "true", "yes", "on"):
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _gemini_supported_methods(model: Any) -> list[str]:
    """Map API-supported methods (new SDK: supported_actions; legacy attr: supported_generation_methods)."""
    actions = getattr(model, "supported_actions", None)
    if actions:
        return list(actions)
    legacy = getattr(model, "supported_generation_methods", None)
    if legacy:
        return list(legacy)
    return []


def _get_gemini_client() -> Any:
    """Configured ``google.genai.Client`` for ``GEMINI_API_KEY`` (Gemini Developer API)."""
    _apply_optional_gemini_thread_env()
    from google import genai as google_genai

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return google_genai.Client(api_key=api_key)


def list_gemini_models_with_generate_content() -> list[dict[str, Any]]:
    """Models visible to this key that advertise generateContent."""
    client = _get_gemini_client()
    rows: list[dict[str, Any]] = []
    for m in client.models.list():
        methods = _gemini_supported_methods(m)
        if "generateContent" not in methods:
            continue
        name = getattr(m, "name", "") or ""
        rows.append(
            {
                "name": _normalize_gemini_model_name(name),
                "full_name": name,
                "display_name": getattr(m, "display_name", "") or "",
                "supported_generation_methods": methods,
            }
        )
    return rows


def gemini_probe_minimal(model_name: str) -> None:
    """Single minimal generateContent; raises GeminiHardQuotaError on hard zero quota."""
    from google.genai import types as genai_types

    client = _get_gemini_client()
    short = _normalize_gemini_model_name(model_name)
    try:
        client.models.generate_content(
            model=short,
            contents="ok",
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4,
            ),
        )
    except Exception as e:
        if is_gemini_hard_zero_quota_error(e):
            raise GeminiHardQuotaError(str(e)) from e
        raise


def pick_first_gemini_model_with_quota(
    candidates: list[str] | None = None,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str | None, str | None, list[tuple[str, str]]]:
    """Try candidates in order with minimal probes.

    Returns (chosen, last_error, failures) where `failures` lists every failed
    (model_name, error_text) attempt (empty if the first candidate succeeds).
    """
    cands = candidates or gemini_fallback_candidates(config_path)
    failures: list[tuple[str, str]] = []
    last_err: str | None = None
    for name in cands:
        try:
            gemini_probe_minimal(name)
            return name, None, failures
        except GeminiHardQuotaError as e:
            msg = str(e)
            failures.append((name, msg))
            last_err = msg
        except Exception as e:
            msg = str(e)
            failures.append((name, msg))
            last_err = msg
    return None, last_err, failures


def default_gemini_model_artifact_path() -> Path:
    env = (os.environ.get("GEMINI_MODEL_SELECTED_FILE") or "").strip()
    if env:
        return Path(env)
    return DEFAULT_GEMINI_MODEL_ARTIFACT


def write_gemini_selection_artifact(
    path: Path,
    *,
    ok: bool,
    model: str | None,
    candidates_ordered: list[str],
    failures: list[tuple[str, str]],
    source: str = "pick-model",
) -> None:
    """Small JSON record so batch jobs / later steps know which model was selected (or that none worked)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ok": ok,
        "model": model,
        "selected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidates_ordered": list(candidates_ordered),
        "failures": [{"model": n, "error": err} for n, err in failures],
        "source": source,
    }
    if not ok:
        payload["reason"] = "all_candidates_failed"
        payload["summary"] = (
            f"All {len(candidates_ordered)} candidate model(s) failed minimal generateContent probes "
            "(hard zero quota, 404, etc.). See failures[]."
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def discover_gemini_models_with_nonzero_quota_probe(
    *,
    max_probes: int = 30,
) -> list[dict[str, Any]]:
    """Probe up to `max_probes` models from list_models (expensive: one request each)."""
    rows = list_gemini_models_with_generate_content()
    out: list[dict[str, Any]] = []
    for r in rows[: max(0, int(max_probes))]:
        name = r["name"]
        try:
            gemini_probe_minimal(name)
            out.append({**r, "quota_probe": "ok"})
        except GeminiHardQuotaError as e:
            out.append({**r, "quota_probe": "limit_zero", "detail": str(e)[:500]})
        except Exception as e:
            out.append({**r, "quota_probe": "error", "detail": str(e)[:500]})
    return out


def gemini_preflight(
    model_name: str | None = None,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    probe_generate: bool = True,
    require_listed: bool = True,
) -> dict[str, Any]:
    """Verify model is listed (optional) and optionally run a minimal generateContent probe.

    Raises GeminiHardQuotaError when the probe hits a hard zero-quota path.
    """
    resolved = _normalize_gemini_model_name(model_name or resolve_gemini_model_from_config(config_path))
    result: dict[str, Any] = {"model": resolved, "listed": False, "probe_ok": False}
    listed = list_gemini_models_with_generate_content()
    listed_names = {r["name"] for r in listed}
    result["listed"] = resolved in listed_names
    if require_listed and not result["listed"]:
        raise RuntimeError(
            f"Model {resolved!r} not in list_models() with generateContent for this key. "
            "Set GEMINI_MODEL to a listed id, or run: python tools/llm_baselines.py list-models"
        )
    if probe_generate:
        gemini_probe_minimal(resolved)
        result["probe_ok"] = True
    return result


def gemini_baseline_should_run(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[bool, str | None]:
    """Return (True, None) to run the Gemini baseline; (False, reason) to skip (soft).

    Raises GeminiHardQuotaError if quota is hard-zero and skipping is not enabled.
    Other errors (e.g. missing key, not listed) propagate.
    """
    if _env_skip_gemini_preflight():
        return True, None
    try:
        gemini_preflight(config_path=config_path, probe_generate=True, require_listed=True)
        return True, None
    except GeminiHardQuotaError as e:
        if _env_skip_on_zero_quota():
            return False, str(e)
        raise
    except Exception as e:
        if is_gemini_hard_zero_quota_error(e):
            if _env_skip_on_zero_quota():
                return False, str(e)
            raise GeminiHardQuotaError(str(e)) from e
        raise


def _env_skip_on_zero_quota() -> bool:
    v = (os.environ.get("GEMINI_SKIP_ON_ZERO_QUOTA") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_skip_gemini_preflight() -> bool:
    v = (os.environ.get("GEMINI_SKIP_PREFLIGHT") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def main(argv: list[str] | None = None) -> None:
    """CLI: preflight, list-models, pick-model, discover-usable."""
    _apply_optional_gemini_thread_env()
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="NLP4LP LLM baselines — Gemini helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pref = sub.add_parser("preflight", help="Verify Gemini model + optional probe (use before batch jobs)")
    p_pref.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p_pref.add_argument("--model", type=str, default=None, help="Override model (default: GEMINI_MODEL / yaml)")
    p_pref.add_argument("--no-probe", action="store_true", help="Only check list_models (no generateContent)")
    p_pref.add_argument(
        "--allow-unlisted",
        action="store_true",
        help="Allow a model not returned by list_models (still runs probe if --no-probe unset)",
    )
    p_pref.add_argument(
        "--write-selected",
        type=Path,
        default=None,
        metavar="PATH",
        help="On success, write JSON artifact (same schema as pick-model --write-selected, source=preflight)",
    )

    p_lm = sub.add_parser("list-models", help="List models for this key with generateContent")
    p_pick = sub.add_parser(
        "pick-model",
        help="Try GEMINI_MODEL / yaml model / fallback_models / GEMINI_MODEL_FALLBACKS until one probe succeeds",
    )
    p_pick.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p_pick.add_argument("--set-env-hint", action="store_true", help="Print export GEMINI_MODEL=... line to stderr")
    p_pick.add_argument(
        "--write-selected",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write JSON artifact (model, timestamp, per-candidate failures). Exit 3 still writes on total failure.",
    )
    p_pick.add_argument(
        "--persist-selected",
        action="store_true",
        help=f"Write artifact to $GEMINI_MODEL_SELECTED_FILE or {DEFAULT_GEMINI_MODEL_ARTIFACT}",
    )

    p_disc = sub.add_parser(
        "discover-usable",
        help="Probe many list_models entries (expensive); finds models with nonzero quota for this account",
    )
    p_disc.add_argument("--max-probes", type=int, default=30)

    args = p.parse_args(argv)

    if args.cmd == "list-models":
        for r in list_gemini_models_with_generate_content():
            print(f"{r['name']}\t{r.get('display_name', '')}")
        return

    if args.cmd == "pick-model":
        cand_list = gemini_fallback_candidates(args.config)
        artifact_path: Path | None = args.write_selected
        if args.persist_selected and artifact_path is None:
            artifact_path = default_gemini_model_artifact_path()
        chosen, err, failures = pick_first_gemini_model_with_quota(config_path=args.config)
        if chosen:
            print(chosen)
            if artifact_path is not None:
                write_gemini_selection_artifact(
                    artifact_path,
                    ok=True,
                    model=chosen,
                    candidates_ordered=cand_list,
                    failures=failures,
                    source="pick-model",
                )
                print(f"Wrote selection artifact: {artifact_path}", file=sys.stderr)
            if args.set_env_hint:
                print(f'export GEMINI_MODEL="{chosen}"', file=sys.stderr)
            return
        lines = [
            "pick-model: every candidate failed the minimal generateContent probe.",
            f"Candidates tried ({len(cand_list)}): {', '.join(cand_list) if cand_list else '(none — configure gemini.model / fallback_models / GEMINI_MODEL_FALLBACKS)'}",
            "Per-model errors:",
        ]
        for name, emsg in failures:
            short = (emsg or "").replace("\n", " ")
            if len(short) > 400:
                short = short[:400] + "…"
            lines.append(f"  - {name}: {short}")
        if err and (not failures or failures[-1][1] != err):
            lines.append(f"Last error: {err}")
        lines.append(
            "Nothing left to try in-process: enable billing, add models with nonzero free-tier quota, "
            "or set GEMINI_SKIP_ON_ZERO_QUOTA=1 to skip Gemini in pipelines."
        )
        msg = "\n".join(lines)
        print(msg, file=sys.stderr)
        if artifact_path is not None:
            write_gemini_selection_artifact(
                artifact_path,
                ok=False,
                model=None,
                candidates_ordered=cand_list,
                failures=failures,
                source="pick-model",
            )
            print(f"Wrote failure artifact: {artifact_path}", file=sys.stderr)
        raise SystemExit(3)

    if args.cmd == "discover-usable":
        for r in discover_gemini_models_with_nonzero_quota_probe(max_probes=args.max_probes):
            probe = r.get("quota_probe", "")
            print(f"{r['name']}\t{probe}\t{r.get('detail', '')}")
        return

    if args.cmd == "preflight":
        try:
            resolved = _normalize_gemini_model_name(
                args.model or resolve_gemini_model_from_config(args.config)
            )
            gemini_preflight(
                args.model,
                config_path=args.config,
                probe_generate=not args.no_probe,
                require_listed=not args.allow_unlisted,
            )
            print(f"OK: Gemini preflight passed for model {resolved!r}")
            if args.write_selected is not None:
                write_gemini_selection_artifact(
                    args.write_selected,
                    ok=True,
                    model=resolved,
                    candidates_ordered=[resolved],
                    failures=[],
                    source="preflight",
                )
                print(f"Wrote selection artifact: {args.write_selected}", file=sys.stderr)
        except GeminiHardQuotaError as e:
            print(f"HARD_QUOTA: {e}", file=sys.stderr)
            # Exit 2: hard zero quota (batch may map to 0 if GEMINI_SKIP_ON_ZERO_QUOTA=1)
            raise SystemExit(2)
        except Exception as e:
            if is_gemini_hard_zero_quota_error(e):
                print(f"HARD_QUOTA: {e}", file=sys.stderr)
                raise SystemExit(2)
            print(f"ERROR: {e}", file=sys.stderr)
            raise SystemExit(1)
        return

    raise SystemExit(1)


if __name__ == "__main__":
    main()
