# Migration plan: `google.generativeai` → `google.genai`

## Current state (this repo)

- **Package:** `google-generativeai` on PyPI (`requirements.txt`).
- **Status:** Google has announced end of feature updates for `google.generativeai` in favor of the **`google.genai`** SDK (see the deprecation notice printed when importing the old package).
- **Usage in-repo:** `tools/llm_baselines.py` — `import google.generativeai as genai`, `genai.configure(api_key=...)`, `GenerativeModel`, `generate_content`, `list_models`.

## Target state

- **Package:** `google-genai` (module `google.genai`) per [Google’s migration guidance](https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md).
- **Auth:** Continue **API key** / AI Studio–style access for LLM baselines unless we add an optional Vertex path later.

## Planned steps (not yet implemented)

1. **Pin and test** — Add `google-genai` alongside `google-generativeai` in a branch; run `preflight`, `pick-model`, and a short NLP4LP baseline smoke test.
2. **Centralize client** — Replace `_ensure_gemini_configured()` / `LLMTwoStageBaseline` Gemini branch with a thin wrapper that:
   - configures the client from `GEMINI_API_KEY`;
   - exposes `list_models` / `generate_content` (or the new SDK equivalents);
   - preserves **`GeminiHardQuotaError`** and **`is_gemini_hard_zero_quota_error()`** behavior.
3. **CLI parity** — Ensure `llm_baselines.py` subcommands (`preflight`, `pick-model`, `list-models`, `discover-usable`) behave identically (exit codes, artifacts).
4. **Remove** `google-generativeai` from `requirements.txt` once stable.
5. **Docs** — Update `docs/gemini_api_quota.md` and batch comments if CLI or env vars change.

## What stays the same

- Hard zero-quota detection (`limit: 0`), no retry on that path.
- Environment variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, fallbacks, preflight/skip flags.
- Batch script flow (`run_gemini_llm_baselines.sbatch`).

## Import / usage inventory (for migration scoping)

| Location | Symbol / usage | Replacement difficulty |
|----------|----------------|-------------------------|
| `tools/llm_baselines.py` | `import google.generativeai as genai` in `LLMTwoStageBaseline.__init__` (Gemini branch) | **Medium** — swap client init; keep `GEMINI_API_KEY` |
| `tools/llm_baselines.py` | `_ensure_gemini_configured()` → `genai.configure`, `list_models`, `GenerativeModel` | **Medium** |
| `tools/llm_baselines.py` | `GenerativeModel.generate_content` in `_gemini_json`, `gemini_probe_minimal` | **Medium** — map to new SDK request/response shapes |
| `batch/learning/run_gemini_llm_baselines.sbatch` | `python -c "import google.generativeai as g; ..."` smoke import | **Trivial** — change to `google.genai` or drop after migration |
| Tests | None import `google.generativeai` directly | **N/A** |

**No other Python modules** in this repo import `google.generativeai` (verified via repository search). All Gemini traffic goes through `tools/llm_baselines.py`.

## References

- Deprecated SDK README: <https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md>
- Gemini API docs: <https://ai.google.dev/gemini-api/docs>
