# Migration: `google.generativeai` → `google.genai` (completed)

## Current state (this repo)

- **Package:** `google-genai` on PyPI (`requirements.txt`); module **`google.genai`** (`from google import genai`).
- **Implementation:** `tools/llm_baselines.py` — `genai.Client(api_key=...)`, `client.models.list()`, `client.models.generate_content(..., config=types.GenerateContentConfig(...))`.
- **Removed:** `google-generativeai` (deprecated `import google.generativeai`).

## API mapping (behavioral parity)

| Old (`google-generativeai`) | New (`google-genai`) |
|----------------------------|----------------------|
| `genai.configure(api_key=...)` | `client = genai.Client(api_key=...)` |
| `genai.list_models()` | `client.models.list()` (pager iterator) |
| Model: `supported_generation_methods` | Model: **`supported_actions`** (API maps `supportedGenerationMethods` → this field) |
| `GenerativeModel(name).generate_content(..., generation_config={...})` | `client.models.generate_content(model=name, contents=..., config=GenerateContentConfig(...))` |
| Response `.usage_metadata.prompt_token_count` / `candidates_token_count` | Same field names on `GenerateContentResponse.usage_metadata` |
| Response `.text` | Same `GenerateContentResponse.text` helper |

## Compatibility / differences

1. **Exceptions:** Failures may surface as **`google.genai.errors.APIError`** (HTTP-style `code`, e.g. 429) instead of **`google.api_core.exceptions.ResourceExhausted`**. Hard zero-quota detection still uses message text (`limit: 0`) and **`is_gemini_hard_zero_quota_error()`** also treats **`APIError` with code 429** when `limit: 0` is present.
2. **Listing:** `list-models` output still exposes `supported_generation_methods` in the JSON-shaped rows for CLI compatibility (values come from **`supported_actions`**).
3. **No Vertex path** in-repo; only API key (Gemini Developer API) as before.

## References

- `google-genai` (python-genai): <https://github.com/googleapis/python-genai>
- Gemini API docs: <https://ai.google.dev/gemini-api/docs>
