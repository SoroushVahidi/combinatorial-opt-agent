# DKE External Resource Provenance Audit (2026-08-15)

This document maps all external inference API providers, funding programs, and their integration status in this repository. It provides empirical evidence of usage to establish whether funding declarations or acknowledgments are required for the Data & Knowledge Engineering (DKE) manuscript.

| Provider | Program/support | Integration exists? | API invoked? | Successful response? | Scientific result artifact? | Used in current DKE paper? | Funding connection established? | Recommended treatment | Evidence path |
|---|---|---|---|---|---|---|---|---|---|
| **Microsoft Azure OpenAI / OpenAI** | Azure for Students (USD 100 credit) | Yes (`tools/llm_baselines.py`) | Yes | Yes | Yes | Yes (Generic LLM baseline in Section 4.5) | Highly likely, requires author confirmation | **AUTHOR_CONFIRMATION_REQUIRED** / **FUNDING** | `results/generic_llm/common18_official/results.jsonl`, `manuscript/dke/main.tex` |
| **Google Cloud / GenAI** | Google Cloud Research Credits Program | Yes (`tools/llm_baselines.py` using `google.genai`) | Yes (preflight only) | Yes (preflight) | No | No (no Gemini results reported in tables/figures) | No | **NO_DISCLOSURE_NEEDED** | `docs/GEMINI_RERUN_REPORT.md` (explicitly states "A successful full Gemini benchmark rerun is not claimed here") |
| **Cohere** | Cohere Labs Catalyst Grant Program (USD 1,000 credit) | No active DKE integration | No | No | No | No | No (used for another selective-deferral project) | **NO_DISCLOSURE_NEEDED** | No Cohere artifacts in `results/`, email context |
| **Fireworks / AMD** | AMD AI Developer Program (USD 50 Fireworks credit) | No active DKE integration | No | No | No | No | No | **NO_DISCLOSURE_NEEDED** | No Fireworks artifacts in `results/` |
| **Mistral** | None (standard developer account) | Yes (`tools/llm_baselines.py`, `scripts/mistral_preflight.py`) | Yes (preflight only) | No (key/quota blocked) | No | No | No | **NO_DISCLOSURE_NEEDED** | `docs/MISTRAL_RERUN_REPORT.md` |
| **CloudRift** | None | No | No | No | No | No | No | **NO_DISCLOSURE_NEEDED** | No CloudRift references in codebase |

## Summary of Recommendations

1. **Microsoft Azure OpenAI**: The Generic LLM baseline reported in Section 4.5 (Table 13) is a retained scientific result in the DKE manuscript. It is highly likely that these calls were charged against the author's **Azure for Students** credits. The Funding section in `main.tex` has been updated with a clear commented TODO for the author to confirm this billing relationship. If confirmed, this should appear under Funding.
2. **Google Cloud**: Google Cloud Research Credits were applied, but no Google Gemini-funded inference contributed to any reported results in this paper (Gemini was only smoke-tested). Thus, Google is **not** acknowledged or declared under Funding to keep disclosures strictly connected to this research.
3. **Cohere**: The Cohere Labs Catalyst Grant was used for a completely separate research project regarding selective deferral / budgeted LLM inference. It did not contribute to this DKE paper, so it is **not** disclosed here.
4. **Fireworks / AMD**: No Fireworks AI credits were used for any reported DKE results. No disclosure is needed.
5. **Mistral**: Mistral API preflights were attempted but failed or did not run due to key/quota limitations. No results were generated or reported. No disclosure is needed.
