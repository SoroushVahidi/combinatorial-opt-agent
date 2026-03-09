# Common Learning Corpus Format

Single canonical JSONL format for mention-to-slot grounding and related learning. One JSON object per line.

## Required fields (every record)

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | `nlp4lp` \| `nl4opt` \| `tatqa` \| `finqa` |
| `split` | string | `train` \| `dev` \| `test` |
| `instance_id` | string | Unique id within dataset (e.g. query_id, doc hash, item uid). |
| `source_path` | string | Path or identifier of source file/record (for debugging). |
| `problem_text` | string | Full text: query (NLP4LP/NL4Opt), or question + linearized context (TAT-QA/FinQA). |

## Schema / slots

| Field | Type | Description |
|-------|------|-------------|
| `schema_name` | string \| null | Problem/schema identifier (e.g. doc_id for NLP4LP). Null for QA-only datasets. |
| `schema_description` | string \| null | Human-readable schema text if available. |
| `slots` | list of slot objects | See below. Empty list for QA-only when no slot schema. |

**Slot object:**

| Field | Type | Description |
|-------|------|-------------|
| `slot_id` | string | Unique within instance (e.g. parameter name, or `var_role`). |
| `slot_name` | string | Display name. |
| `slot_text` | string \| null | Optional longer description. |
| `slot_role` | string \| null | e.g. `objective_coeff`, `lower_bound`, `upper_bound`, `total`, `ratio`. |
| `expected_type` | string \| null | `int` \| `float` \| `currency` \| `percent` \| `unknown`. |
| `variable_entity` | string \| null | Variable/entity name this slot is associated with (NL4Opt). |

## Numeric mentions

| Field | Type | Description |
|-------|------|-------------|
| `numeric_mentions` | list of mention objects | See below. |

**Mention object:**

| Field | Type | Description |
|-------|------|-------------|
| `mention_id` | string | Unique within instance (e.g. `m0`, `m1`). |
| `surface` | string | Raw span (e.g. `"760000"`, `"$0.50"`). |
| `normalized_value` | number \| null | Numeric value. |
| `type_bucket` | string | `int` \| `float` \| `currency` \| `percent` \| `unknown`. |
| `sentence_id` | int \| null | Sentence index if applicable. |
| `char_start` | int \| null | Start offset in `problem_text`. |
| `char_end` | int \| null | End offset in `problem_text`. |
| `local_context` | string \| null | Surrounding text or token window. |
| `unit` | string \| null | e.g. `dollar`, `percent`. |
| `operator_cues` | list of strings | e.g. `min`, `max`, `at_least`, `at_most`. |

## Supervision

| Field | Type | Description |
|-------|------|-------------|
| `gold_slot_assignments` | object \| null | Map `slot_id` -> `mention_id` (or null if unfilled). For QA-only: slot_id may be `"answer"` -> mention_id or value. |
| `role_labels` | object \| null | Per-mention or per-slot role (e.g. coefficient, bound). |
| `entity_labels` | object \| null | Per-mention variable/entity id where available. |
| `bound_labels` | object \| null | Per-slot or per-mention: `lower` \| `upper` \| `equality` \| null. |

## Metadata

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | object | Dataset-specific: derivation, steps, answer_type, ambiguous flags, etc. |

## Dataset-specific mapping notes

- **NLP4LP:** `problem_text` = query; `schema_name` = relevant_doc_id; slots from gold parameters; gold_slot_assignments = parameter name -> mention_id that fills it (by matching normalized value); no entity_labels in eval data; bound_labels inferred from slot names (min/max).
- **NL4Opt:** `problem_text` = document; slots from obj_declaration.terms + const_declarations; spans give numeric_mentions with char_start/char_end; entity_labels from var_mention_to_first_var; bound_labels from const_declarations.direction/operator.
- **TAT-QA:** `problem_text` = question + linearized table/paragraphs; slots = [] or synthetic `answer`; gold_slot_assignments = `{"answer": value}`; metadata.derivation, metadata.answer_type.
- **FinQA:** `problem_text` = question + pre_text + table; slots = [] or synthetic `answer`; metadata.steps, metadata.answer.

## File layout

- `artifacts/learning_corpus/nlp4lp_{split}.jsonl`
- `artifacts/learning_corpus/nl4opt_{split}.jsonl`
- `artifacts/learning_corpus/tatqa_{split}.jsonl`
- `artifacts/learning_corpus/finqa_{split}.jsonl`
- Or combined: `artifacts/learning_corpus/all_{split}.jsonl` (when `--dataset all`).
