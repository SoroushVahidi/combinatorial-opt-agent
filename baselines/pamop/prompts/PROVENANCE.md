# Prompt provenance

No PaMOP prompt text has ever been published or recovered (see
`docs/PAMOP_REPRODUCTION_PLAN.md` §1 and §3, and the re-check logged in
§15.3 of that document for this milestone specifically). **Every prompt
template file in this directory is a `REPRODUCTION CHOICE`** — our own
wording, written to satisfy the paper's stated *requirements* for what the
LLM call must produce, not a reconstruction of the authors' actual prompt
text, which does not exist in any public source.

## Paper-derived requirements vs. reconstructed wording

| File | Paper requirement it must satisfy | What's PAPER-SPECIFIED | What's RECONSTRUCTION CHOICE |
|---|---|---|---|
| `extraction_v1.txt` (`G_extr`) | "we derive a structured representation of the problem, extracting textual descriptions for the objective function `t_o`, constraints `t_c`, and parameters and variables `t_v`... we also generate a concise problem summary `g`... We prompt the LLM to assign a vagueness score to each constraint" (paper §3.2) | The four required output fields (`t_o`, `t_c`, `t_v`, `g`) and the per-constraint vagueness score | All wording; the exact output JSON schema field names; the vagueness-score scale (paper never gives one -- this template asks for `[0, 1]`, documented in `extraction.py`); instructions for handling ambiguous input; any few-shot examples (paper doesn't say whether `G_extr` uses any) |

## Versioning

Each template file is content-hashed via `baselines.pamop.llm.base.prompt_hash`
at load time (`prompts/__init__.py::load_prompt`); the hash is recorded in
every `LLMResponse.prompt_hash` produced from a call using that template, so
any future prompt-wording change is traceable in run outputs. Filenames
carry an explicit version suffix (`_v1`, `_v2`, ...) — never edit a shipped
template's *content* in place once anything has been run against it; add a
new version file instead so historical `prompt_hash` values stay
resolvable.
