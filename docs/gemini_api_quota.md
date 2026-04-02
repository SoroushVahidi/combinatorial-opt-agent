# Gemini API: quota errors (429) and model names

## What went wrong on the cluster

If logs show **`ResourceExhausted` / HTTP 429** with text like:

- `Quota exceeded for metric: ... generate_content_free_tier_requests, limit: 0`

then this is **not** an invalid API key. The request reached Google, but the **project/key has no remaining quota** for that model on the **free tier** (or free tier is not available for that model in your region/account).

**What to do (account-side):**

1. Open [Google AI Studio](https://aistudio.google.com/) → **API keys** / billing for the Gemini API.
2. Check **usage & limits** and enable **paid** usage if your project has **free tier limit 0** for the model you use.
3. See Google’s docs: [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Model name (404 NOT_FOUND)

If logs show **`NotFound`** for `models/...`:

- Pick a model id that supports **`generateContent`** for your API version. Override without editing YAML:

  ```bash
  export GEMINI_MODEL=gemini-2.0-flash
  ```

  Or set another id from **`list_models()`** / AI Studio.

## Repo settings

- Default model: `configs/llm_baselines.yaml` → `gemini.model`
- Env override: **`GEMINI_MODEL`** (see `tools/llm_baselines.py`)

Retries for transient 429s honor **“Please retry in Xs”** in the error message when present (`tools/llm_baselines.py`).
