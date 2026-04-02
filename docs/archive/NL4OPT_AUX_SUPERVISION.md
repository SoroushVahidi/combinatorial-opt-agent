# NL4Opt Auxiliary Supervision Audit

Evidence-based audit of the NL4Opt dataset for use as **auxiliary** supervision for the downstream number-to-slot grounding bottleneck. NLP4LP remains the **target** task; NL4Opt is used only where its annotations directly support entity grounding, bound direction, and number-role classification.

## 1. Raw NL4Opt fields (local files)

**Source:** `data_external/raw/nl4opt_competition/generation_data/{train,dev,test}.jsonl`

Each line is a single JSON object keyed by a **hash string** (e.g. `"-804075997"`). The value is one record with these fields (verified from actual file contents):

| Field | Type | Description |
|-------|------|-------------|
| `document` | string | Full NL problem text. |
| `vars` | list[str] | Canonical variable names (e.g. `["cleaners", "receptionists"]`). |
| `var_mentions` | list[str] | All surface forms of variables in document order. |
| `params` | list[str] | List of parameter/limit values as strings (e.g. `["500", "350", "third"]`). |
| `var_mention_to_first_var` | dict[str, str] | Maps each surface form → canonical var (e.g. `{"Cleaners": "cleaners"}`). |
| `first_var_to_mentions` | dict[str, list[str]] | Maps canonical var → list of surface forms. |
| `obj_declaration` | dict | `type` (e.g. `"objective"`), `direction` (`"minimize"`/`"maximize"`), `name`, `terms`: dict mapping **var_mention** → coefficient value string. |
| `const_declarations` | list[dict] | Each: `type`, `direction`, `limit` (value string), `var` (var_mention, optional), `operator` (e.g. `GREATER_OR_EQUAL`, `LESS_OR_EQUAL`). For `xby`: `x_var`, `param`, `y_var`. For linear/sum: `terms` (var_mention → coeff). |
| `spans` | list[dict] | Each: `text`, `start`, `end`, `token_start`, `token_end`, `type` (`"span"`), `label`: one of `VAR`, `PARAM`, `LIMIT`, `CONST_DIR`, `OBJ_NAME`, `OBJ_DIR`. |
| `tokens` | list[dict] | Token-level `text`, `start`, `end`, `id`. |
| `_input_hash` | int | (optional) Same as key. |
| `order_mapping` | dict | (optional) Var mention → index. |

**Constraint types observed:** `sum`, `lowerbound`, `upperbound`, `linear`, `ratio`, `xby`.

**Direction strings observed:** `"minimum"`, `"at least"`, `"below"`, `"up to"`, `"at most"`, `"more than"`, `"budget"`.

**Operator values:** `GREATER_OR_EQUAL`, `LESS_OR_EQUAL`.

---

## 2. What can supervise which bottleneck

### 2.1 Entity / variable association

**Direct supervision:**

- **VAR** spans: character offsets of variable mentions; no numeric value.
- **PARAM** and **LIMIT** spans: numeric (or ratio) values with offsets.
- **obj_declaration.terms**: maps **var_mention** → value string. So for each PARAM span we can match its `text` (normalized) to a value in `terms` and obtain the **var_mention**; then **var_mention_to_first_var** gives the **canonical variable id**.
- **const_declarations**: each has `limit` (value) and often `var` (var_mention). So for each LIMIT span we can match to a constraint by value and read **var**; then map to canonical var via **var_mention_to_first_var**.

**Training target we derive:** For each numeric mention (PARAM or LIMIT span), the **gold variable id** is the canonical var from the declaration that contains that value. Candidate set: **vars** (canonical list). Labels: **gold_variable_id** (single canonical var) or **gold_variable_ids** when a value appears in multiple contexts (we use the one from the declaration that contains this span).

**Approximate / caveats:** (1) Same numeric value can appear in both objective and constraint (e.g. "500" for wage and budget); we assign the variable from the **declaration that this span is annotated to** — we infer which declaration by matching span (start, end, text) to terms/limit. (2) **xby** constraints have `param` (a PARAM span) and `y_var`/`x_var`; the param is a ratio and may not have a single “variable” in the same sense; we can still assign a canonical var (e.g. from x_var or y_var) or mark as “ratio” role.

### 2.2 Lower vs upper bound classification

**Direct supervision:**

- **const_declarations[].direction**: e.g. "at least", "below", "at most", "minimum", "up to", "more than".
- **const_declarations[].operator**: `GREATER_OR_EQUAL` → lower bound; `LESS_OR_EQUAL` → upper bound.
- **const_declarations[].type**: `lowerbound` → lower; `upperbound` → upper; `sum`/`linear`/`ratio`/`xby` get direction from **direction** and **operator**.

**Training target we derive:** For each **LIMIT** span we associate it with the constraint that contains that limit value; then **gold_bound_label** = `lower` | `upper` | `equality` | `other` from direction + operator. PARAM spans in the objective are **not** bounds → we use label **other** (or skip for bound task). Equality: when operator is equality (if present) or direction suggests equality; otherwise we map GREATER_OR_EQUAL + “at least”/“minimum” → **lower**, LESS_OR_EQUAL + “at most”/“below”/“up to” → **upper**.

**Approximate:** “equality” is rare in the data; we use **other** when unclear.

### 2.3 Number-role classification (objective coefficient / limit / rhs / ratio)

**Direct supervision:**

- **Span label**: PARAM vs LIMIT (from **spans[].label**).
- **obj_declaration.terms**: PARAM spans that appear in `terms` are **objective coefficients**.
- **const_declarations**: LIMIT span → constraint; **type** gives role: `sum`/`linear` limit → **limit** (or **rhs_total**); **lowerbound**/**upperbound** → **limit**; **ratio** → **ratio**; **xby** has **param** → ratio-like, and **limit** in other constraints.

**Training target we derive:** **gold_role_label** in: `objective_coeff` | `limit` | `rhs_total` | `ratio` | `other`.

- PARAM in obj_declaration.terms → **objective_coeff**.
- LIMIT in const with type lowerbound/upperbound → **limit**.
- LIMIT in const with type sum/linear (total budget/capacity) → **rhs_total**.
- LIMIT in ratio constraint or PARAM in xby → **ratio** (or **limit** for the numeric limit part).
- Anything else → **other**.

**Approximate:** Some constraints are ambiguous (e.g. linear with both total and per-var); we use the **type** and **direction** as the single source of truth.

---

## 3. Mappings: direct vs approximate

| Supervision | Source | Direct? | Notes |
|-------------|--------|---------|--------|
| Which variable a PARAM belongs to | obj_declaration.terms (var_mention → value); match span text to value; var_mention_to_first_var → canonical var | **Direct** | Same value in two terms (two vars) → we pick one by position or first match. |
| Which variable a LIMIT belongs to | const_declarations (limit, var); match span to limit; read var | **Direct** | Constraint has at most one limit value; var may be missing in sum constraints → we use **other** or skip. |
| Lower vs upper for LIMIT | const direction + operator | **Direct** | direction/operator are explicit. |
| Role (objective_coeff / limit / rhs_total / ratio) | span label PARAM/LIMIT + obj/const type | **Direct** | Type and declaration structure are explicit. |

---

## 4. Exact training targets we will derive from NL4Opt

1. **Entity association task:** One example per (instance_id, mention_id). Fields: problem_text, mention_id, mention_surface, mention_span (start, end), candidate_variable_ids = vars, candidate_variable_texts = from first_var_to_mentions or vars, **gold_variable_id** = canonical var for that PARAM/LIMIT from declarations. Skipped when we cannot uniquely associate (e.g. sum constraint with no var).

2. **Bound direction task:** One example per LIMIT span (and optionally PARAM with bound-like context; we restrict to LIMIT for clarity). **gold_bound_label** in {`lower`, `upper`, `equality`, `other`} from const_declarations.direction and .operator. PARAM-only (objective) examples get **other** or are skipped for this task.

3. **Number-role task:** One example per PARAM and LIMIT span. **gold_role_label** in {`objective_coeff`, `limit`, `rhs_total`, `ratio`, `other`} from span label + obj_declaration / const_declarations type.

---

## 5. What we are NOT claiming NL4Opt can supervise

- **NLP4LP-style slot filling:** NL4Opt does not have “slot names” that match NLP4LP parameters. We do **not** train a single shared slot-fill model on NL4Opt as if it had NLP4LP slot labels. We use NL4Opt only for **auxiliary** heads (entity, bound, role).
- **Exact mention→slot alignment for NLP4LP:** That alignment is only in NLP4LP (gold parameters). NL4Opt gives variable/entity and role/bound at the **NL4Opt** schema level.
- **Retrieval:** NL4Opt has no query–document retrieval; single-document only.
- **Multi-step numerical reasoning:** No derivation or program steps; single-problem only.
