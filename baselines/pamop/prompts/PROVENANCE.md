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
| `modeling_leaf_v1.txt` (`G_mod`, eq. 3) | `m_{c,i} = G_mod(g, t_v, {t_{c,j} : j in cons_i})` -- inputs are the global summary, the **full global variable list** (not just node-local variables), and only this node's constraint descriptions; output is AMPL code, not prose ("we directly generate code in the modeling language instead of formulas"); "when modeling nodes containing vague constraints, we can incorporate information from their parent and sibling nodes" (§3.3) | The three input categories, their scope (full `t_v`, node-local `t_c` only), AMPL as the output language, and *that* vague constraints get parent/sibling augmentation | All wording; the exact vagueness threshold that triggers augmentation (paper gives no number -- `config.llm.vague_threshold`, itself a reproduction choice); the exact augmentation content/format (paper says "incorporate information from... parent and sibling nodes" but not what form -- this implementation passes their constraint *descriptions*, not full modeled output, since siblings/parent may not be modeled yet in a bottom-up order) |
| `modeling_root_v1.txt` (`G_mod`, eq. 4) | `M = (m_p, m_v, m_o, m_c) = G_mod(g, t_v, t_o, m_c)` -- inputs are the global summary, full variable list, objective text, and the **already-merged** constraint set from eq. 3; this is the one call that produces the objective and completes the model | The four input categories and that this call (not the per-leaf calls) is where the objective gets modeled | All wording; the four-labeled-section output structure (`### PARAMETERS`/`VARIABLES`/`OBJECTIVE`/`CONSTRAINTS`) -- the paper describes `M` only as an abstract tuple, never as a required output *format*; this structuring choice exists purely to give `modeling.py`'s parser reliable section boundaries and to prepare a clean interface for a future AMPL renderer |

## Versioning

Each template file is content-hashed via `baselines.pamop.llm.base.prompt_hash`
at load time (`prompts/__init__.py::load_prompt`); the hash is recorded in
every `LLMResponse.prompt_hash` produced from a call using that template, so
any future prompt-wording change is traceable in run outputs. Filenames
carry an explicit version suffix (`_v1`, `_v2`, ...) — never edit a shipped
template's *content* in place once anything has been run against it; add a
new version file instead so historical `prompt_hash` values stay
resolvable.
