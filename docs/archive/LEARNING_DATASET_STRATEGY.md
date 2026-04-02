# Learning Dataset Strategy: NLP4LP, NL4Opt, TAT-QA, FinQA

Evidence-based audit and recommendations for using the four available datasets to address the downstream number-to-slot bottleneck (wrong variable association, multiple float-like values, lower vs upper bound confusion).

---

## Part 1: Dataset structure and supervision

### 1.1 Local files and schema (evidence from repo)

| Dataset   | Local path(s) | Main data files | Schema / fields (from actual files) |
|-----------|----------------|------------------|-------------------------------------|
| **NLP4LP** | `data/processed/nlp4lp_eval_*.jsonl`, `data/catalogs/nlp4lp_catalog.jsonl`; gold from HF `udell-lab/NLP4LP` or `NLP4LP_GOLD_CACHE` | Eval: `query_id`, `query`, `relevant_doc_id`. Catalog: `doc_id`, `text`, `meta`. Gold (per doc): `parameters` (dict slot→value), `problem_info` (includes `parameters` key with slot names), `solution`, `optimus_code`. | Query = NL problem description. Schema = parameterized template (slot names like `TotalBudget`, `ProfitPerDollarCondos`). Gold = which numbers fill which slots. |
| **NL4Opt** | `data_external/raw/nl4opt_competition/generation_data/{train,dev,test}.jsonl`; merged in `data/processed/all_problems.json` | Each line: one JSON object keyed by hash. Fields: `document`, `vars`, `var_mentions`, `params`, `var_mention_to_first_var`, `first_var_to_mentions`, `obj_declaration` (type, direction, name, terms {var→coeff}), `const_declarations` (type, direction, limit, var, operator, terms), `spans` (text, start, end, label: VAR, PARAM, LIMIT, CONST_DIR, OBJ_NAME, OBJ_DIR), `tokens`. | Full NL document + **span-level labels**: VAR, PARAM, LIMIT, CONST_DIR, OBJ_DIR, OBJ_NAME; **declarations** linking numbers to variables (terms, limit) and min/max (direction, operator). |
| **TAT-QA** | `data_external/raw/tatqa/dataset_raw/tatqa_dataset_{train,dev,test}.json`, `tatqa_dataset_test_gold.json` | List of items. Each: `table` (uid, 2D table), `paragraphs` (uid, order, text), `questions` (uid, order, question, answer, derivation, answer_type e.g. span/arithmetic/multi-span, answer_from, rel_paragraphs, scale). | Hybrid table + paragraphs; QA with **answer** (span or value) and **derivation** (e.g. `(2.9+2.9)/2`). No explicit “slots”; cell/span grounding and multi-step arithmetic. |
| **FinQA** | `data_external/raw/finqa/dataset/{train,dev,test,private_test}.json` | List of items. Each: `pre_text`, `post_text`, `table` (normalized), `table_ori`, `filename`, `qa`: `question`, `answer`, `explanation`, `ann_table_rows`, `ann_text_rows`, `steps` (op, arg1, arg2, res — program steps). | Table + text context; **question**, **answer**, **reasoning program** (steps). No schema slots; grounding via steps and supporting facts. |

### 1.2 What supervision each dataset contains

- **NLP4LP:** (1) Query–document relevance (same doc_id). (2) Gold **parameter** dict: slot name → scalar value (exact slot filling). (3) Schema text per doc (parameter names can be inferred from catalog + problem_info). No span-level alignment in the eval files; alignment is implied by gold parameters.
- **NL4Opt:** (1) **Span labels**: VAR, PARAM, LIMIT, CONST_DIR, OBJ_DIR, OBJ_NAME with character offsets. (2) **Entity alignment**: `var_mention_to_first_var`, `first_var_to_mentions`. (3) **Number–role**: PARAM/LIMIT tied to constraint/objective via `obj_declaration.terms`, `const_declarations.limit`/`.var`. (4) **Min/max**: `const_declarations.direction` (e.g. "at least", "below"), `operator` (GREATER_OR_EQUAL, LESS_OR_EQUAL). (5) No retrieval (single document per example).
- **TAT-QA:** (1) **Answer** (span or value) and **derivation** (arithmetic expression). (2) **answer_from** (table/text). (3) **rel_paragraphs** for supporting text. (4) **answer_type** (span, multi-span, arithmetic, etc.). No slot names; cell/span-level grounding and numerical reasoning.
- **FinQA:** (1) **Answer** and **reasoning steps** (op, arg1, arg2, res). (2) **ann_table_rows**, **ann_text_rows** (supporting facts). (3) No slot/schema; table-cell and text grounding + program execution.

### 1.3 Suitability matrix (evidence-based)

| Task | NLP4LP | NL4Opt | TAT-QA | FinQA |
|------|--------|--------|--------|-------|
| **Schema/problem retrieval** | ✅ Primary: query→doc_id, catalog text | ❌ Single doc per example | ⚠️ Table+passages as “context”; no schema retrieval | ⚠️ Context retrieval (table+text); no schema |
| **Entity/variable grounding** | ⚠️ Implicit (gold params); no spans | ✅ **Excellent**: VAR spans, var_mention_to_first_var | ⚠️ Entity in table headers/paragraphs; no VAR labels | ⚠️ Rows/cells; no explicit variable spans |
| **Objective/constraint role labeling** | ⚠️ Via slot names in schema | ✅ **Excellent**: OBJ_NAME, OBJ_DIR, const_declarations | ❌ QA only; no objective/constraint roles | ❌ QA only |
| **Number–role disambiguation** | ✅ Gold: number→slot name | ✅ **Excellent**: PARAM/LIMIT spans + terms/limit/var | ✅ Which cells/values → answer; derivation | ✅ Steps tie numbers to ops; supporting facts |
| **Multi-step numerical reasoning** | ❌ Single-step slot fill | ❌ Single doc, no multi-step program | ✅ **Strong**: derivation (arithmetic expr) | ✅ **Strong**: program steps |
| **Final slot filling** | ✅ **Target task**: parameters = slot fills | ✅ **Strong**: terms, limit, var = slot-like | ❌ No slot schema; map to “answer” only | ❌ No slots; map to answer + program |

---

## Part 2: Mapping datasets to our three bottlenecks

### 2.1 Wrong variable/entity association

- **NLP4LP:** Directly addresses it: gold is exactly “which number goes to which slot.” No span-level supervision in repo; learning must use (query, schema, gold_parameters) as target. **Use:** Main evaluation and primary learning target for slot filling.
- **NL4Opt:** Best **auxiliary** for this bottleneck: **VAR** spans and **var_mention_to_first_var** give explicit entity/variable grounding; **PARAM** and **LIMIT** tied to **var** in const_declarations and **terms** in obj_declaration show which number belongs to which variable/role. **Use:** Train entity/variable binding and number–variable linking; transfer to NLP4LP slot names.
- **TAT-QA:** Helps indirectly: questions refer to table entities (e.g. “2019 rate of inflation”); answer and derivation show which cells/values are used. **Use:** Augment “which number/span answers which question” and table-cell grounding; no direct slot names.
- **FinQA:** Similar: **steps** and **ann_*** show which numbers participate in reasoning. **Use:** Numerical grounding and disambiguation when many numbers appear; no schema slots.

### 2.2 Multiple float-like values

- **NLP4LP:** Many queries have several similar numbers (e.g. 500, 350, 100, 20, 30000); gold parameters resolve ambiguity. **Use:** Main benchmark for “many numbers → correct slots.”
- **NL4Opt:** Same phenomenon; **PARAM** vs **LIMIT** and linkage to **var** and **const_declarations** disambiguate. **Use:** Train models to associate each numeric span to a role (param vs limit) and variable.
- **TAT-QA:** Tables have many numeric cells; **derivation** and **answer** show which subset is used. **Use:** Train “select the right subset of numbers” and arithmetic over them.
- **FinQA:** **steps** explicitly list which values (arg1, arg2) are combined. **Use:** Train disambiguation of which numbers to use in multi-step reasoning.

### 2.3 Lower vs upper bound confusion

- **NLP4LP:** Slot names and gold distinguish min vs max (e.g. MinimumInvestmentDetachedHouses vs budget cap). **Use:** Main evaluation for min/max correctness.
- **NL4Opt:** **Strongest auxiliary**: `const_declarations` have **direction** (“at least”, “below”, “at most”) and **operator** (GREATER_OR_EQUAL, LESS_OR_EQUAL); **type** (lowerbound, upperbound). **Use:** Train min/max role classification and number–bound linking.
- **TAT-QA:** Some questions compare values (req_comparison); no explicit min/max schema. **Use:** Limited; comparison cues only.
- **FinQA:** No min/max schema. **Use:** Minimal for this bottleneck.

---

## Part 3: Proposed common intermediate format

Single **JSONL** format for training/ablation, one record per example:

```json
{
  "dataset_name": "nlp4lp",
  "example_id": "nlp4lp_test_0",
  "split": "test",
  "raw_text": "Mrs. Watson wants to invest...",
  "schema_or_problem_label": "nlp4lp_test_0",
  "schema_slot_names": ["TotalBudget", "ProfitPerDollarCondos", ...],
  "numeric_mentions": [
    {"raw": "760000", "value": 760000, "char_start": 72, "char_end": 78, "kind": "currency"}
  ],
  "entity_mentions": [],
  "role_labels": null,
  "gold_alignment_targets": {"TotalBudget": 760000, "ProfitPerDollarCondos": 0.5, ...},
  "metadata": {}
}
```

- **dataset_name:** `nlp4lp` | `nl4opt` | `tatqa` | `finqa`
- **example_id:** Unique id (e.g. query_id, NL4Opt hash, TAT-QA uid, FinQA id).
- **split:** train | dev | test
- **raw_text:** Query (NLP4LP, NL4Opt) or question + context serialization (TAT-QA, FinQA).
- **schema_or_problem_label:** For NLP4LP/NL4Opt: doc_id or problem id; for TAT-QA/FinQA: optional context id or empty.
- **schema_slot_names:** For NLP4LP: list of parameter names from problem_info/parameters. For NL4Opt: derived from obj_declaration.terms + const_declarations (slot = variable + role). For TAT-QA/FinQA: null or synthetic “answer” slot.
- **numeric_mentions:** List of {raw, value, char_start, char_end, kind}. From existing extractors (NLP4LP), NL4Opt spans (PARAM, LIMIT), or TAT-QA/FinQA table/cell and text numbers.
- **entity_mentions:** Optional; NL4Opt: VAR spans; others: empty or from table headers/entities.
- **role_labels:** Optional; NL4Opt: per-span or per-number role (objective_coeff, lower_bound, upper_bound, total, etc.); NLP4LP: derivable from slot names; TAT-QA/FinQA: null or “answer”.
- **gold_alignment_targets:** For NLP4LP: `parameters` dict. For NL4Opt: map each PARAM/LIMIT span to (var, role) or slot key. For TAT-QA/FinQA: single key “answer” → value, or derivation string.
- **metadata:** answer_type (TAT-QA), steps (FinQA), derivation, etc.

**Mapping summary:**

- **NLP4LP:** raw_text = query; schema_or_problem_label = relevant_doc_id; schema_slot_names = keys of gold parameters; numeric_mentions = from _extract_num_tokens or similar; gold_alignment_targets = parameters. role_labels can be inferred from slot names (e.g. min/max).
- **NL4Opt:** raw_text = document; schema_slot_names = from obj_declaration.terms keys + const_declarations (e.g. "cleaners_objective_coeff", "receptionists_lower_bound"); numeric_mentions = spans with label PARAM/LIMIT; entity_mentions = VAR spans; role_labels = from const type + direction; gold_alignment_targets = term/limit value per slot.
- **TAT-QA:** raw_text = question + linearized table/paragraphs; schema_slot_names = null or ["answer"]; numeric_mentions = from table cells + text; gold_alignment_targets = {"answer": answer}; metadata.derivation = derivation.
- **FinQA:** raw_text = question + pre_text + table; schema_slot_names = null or ["answer"]; numeric_mentions = from table and pre_text; gold_alignment_targets = {"answer": answer}; metadata.steps = steps.

---

## Part 4: Recommended training strategy

1. **Main target dataset:** **NLP4LP** — it is the in-domain evaluation for retrieval + slot filling; all metrics (exact20_on_hits, instantiation_ready) are defined on it. Keep it as the **final evaluation set** and primary learning target for slot filling.

2. **Best OR-domain auxiliary:** **NL4Opt** — same “NL optimization problem” domain, with span-level supervision (VAR, PARAM, LIMIT, OBJ_*, CONST_*) and explicit min/max (direction, operator). Use it to **pre-train or jointly train** entity/variable grounding, number–role disambiguation, and lower vs upper bound labeling before or alongside NLP4LP.

3. **Numerical reasoning augmentation:** **TAT-QA** and **FinQA** — use for **multi-step reasoning** and **disambiguation of many numbers** (which numbers to use, how to combine them). They do not provide schema/slots; use as **auxiliary tasks** (e.g. answer prediction, derivation/program prediction) or **curriculum**: train on TAT-QA/FinQA for numerical grounding and reasoning, then fine-tune on NL4Opt + NLP4LP for slot filling.

4. **Staged plan:**
   - **Stage 1 (optional):** Train on **TAT-QA** and/or **FinQA** for numerical QA and reasoning (answer + derivation/steps). Output: model that grounds numbers and performs arithmetic. No slot filling yet.
   - **Stage 2:** Train on **NL4Opt** for entity binding (VAR), number–role (PARAM/LIMIT), and min/max (const_declarations). Map to intermediate format with schema_slot_names and gold_alignment_targets derived from terms/limit/var. Optionally **multitask** with Stage 1.
   - **Stage 3:** Fine-tune on **NLP4LP** for end-to-end slot filling (query + schema → parameters). Keep NLP4LP test set **unused** until final evaluation.
   - **Curriculum option:** TAT-QA/FinQA → NL4Opt → NLP4LP (easier numerical reasoning first, then OR-specific roles, then in-domain slots). **Multitask option:** Single model with dataset_name as input; losses per dataset (e.g. alignment loss for NLP4LP/NL4Opt, QA loss for TAT-QA/FinQA).

5. **What remains the final in-domain evaluation set:** **NLP4LP** (e.g. `data/processed/nlp4lp_eval_orig.jsonl` + gold), with metrics as in the current pipeline (schema hit, exact20_on_hits, instantiation_ready).

---

## Part 5: Comparison table and next step

### Dataset comparison

| Dataset | Good for | Not good for |
|---------|----------|--------------|
| **NLP4LP** | Schema retrieval; slot filling (gold parameters); in-domain eval; min/max via slot names | Span-level alignment (no spans in eval); multi-step reasoning; large train set (331 test) |
| **NL4Opt** | Entity/variable grounding (VAR); number–role (PARAM/LIMIT); min/max (direction, operator); slot-like terms/limit/var | Retrieval (single doc); multi-step program; exact match to NLP4LP slot names (need mapping) |
| **TAT-QA** | Multi-step derivation; table+text grounding; many numbers; arithmetic | Schema/slots; objective/constraint roles; OR domain |
| **FinQA** | Reasoning steps (program); number grounding; financial numerical QA | Schema/slots; min/max; OR domain |

### Recommended order to use them

1. **First (auxiliary):** TAT-QA and/or FinQA — numerical reasoning and “which numbers matter.”
2. **Second (auxiliary):** NL4Opt — entity binding, number–role, min/max in OR domain.
3. **Third (main):** NLP4LP — slot filling and final evaluation.

### Proposed common format

As in Part 3: single JSONL with `dataset_name`, `example_id`, `split`, `raw_text`, `schema_or_problem_label`, `schema_slot_names`, `numeric_mentions`, `entity_mentions`, `role_labels`, `gold_alignment_targets`, `metadata`. All four datasets map into it; NLP4LP and NL4Opt populate slot-level alignment; TAT-QA/FinQA populate answer/derivation/steps.

### Recommended next implementation step

**Implement a single pipeline that writes the common intermediate JSONL from all four sources:**

1. Add a script (e.g. `tools/build_learning_corpus.py` or under `training/`) that:
   - Reads NLP4LP (eval + catalog + gold), NL4Opt (generation_data JSONL), TAT-QA (dataset_raw JSON), FinQA (dataset JSON).
   - For each dataset, maps one record per example into the common schema above.
   - Writes one or more JSONL files (e.g. `data/processed/learning_corpus_nlp4lp.jsonl`, `learning_corpus_nl4opt.jsonl`, etc.) or one combined file with `dataset_name` and `split` for filtering.
2. Use existing NLP4LP number extraction and gold parameters; implement NL4Opt span→numeric_mentions and const/obj→schema_slot_names + gold_alignment_targets; implement TAT-QA/FinQA linearization and answer/derivation/steps in metadata.
3. Do **not** start training yet; validate the JSONL (spot-check, counts per dataset/split) and document the mapping in this doc or a short `docs/LEARNING_CORPUS_FORMAT.md`.

---

## Part 6: Direct answers

1. **Main target dataset:** **NLP4LP** — it defines the task (retrieval + slot filling) and the evaluation metrics; all learning should aim to improve on it.
2. **Best OR-domain auxiliary dataset:** **NL4Opt** — same domain and explicit supervision for entity grounding, number–role, and lower vs upper bound; use for pre-training or joint training before NLP4LP.
3. **Used mainly for numerical reasoning augmentation:** **TAT-QA** and **FinQA** — use for multi-step numerical reasoning and disambiguation of many float-like values; not for schema or slot names.
4. **Very next implementation step:** **Build the common intermediate JSONL** from all four datasets (single script that reads NLP4LP, NL4Opt, TAT-QA, FinQA and emits the proposed schema); validate and document; then proceed to model/training design.
