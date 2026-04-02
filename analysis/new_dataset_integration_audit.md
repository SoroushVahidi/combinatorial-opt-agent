# New Dataset Integration Audit

**Generated:** 2026-04-02  
**Purpose:** Pre-integration audit of four new HuggingFace optimization datasets.

---

## 1. Starting Repo State

### Existing dataset adapters (src/data_adapters/registry.py)

| Key | Adapter class | Data root |
|-----|---------------|-----------|
| nlp4lp | NLP4LPAdapter | data/processed/ |
| nl4opt | NL4OptAdapter | data/external/nl4opt/ |
| text2zinc | Text2ZincAdapter | data/external/text2zinc/ |
| optmath | OptMATHAdapter | data/external/optmath/ |
| complexor | ComplexORAdapter | data/external/complexor/ |
| optimus | OptiMUSAdapter | data/external/optimus/ |
| gurobi_modeling_examples | GurobiModelingExamplesAdapter | data/sources/ |
| gurobi_optimods | GurobiOptimodsAdapter | data/sources/ |
| gams_models | GAMSModelsAdapter | data/sources/ |
| miplib | MIPLIBAdapter | data/sources/ |
| or_library | ORLibraryAdapter | data/sources/ |
| pyomo_examples | PyomoExamplesAdapter | data/sources/ |

### Existing download scripts (scripts/)

- `scripts/get_nl4opt.py` – downloads from GitHub raw URLs
- `scripts/get_text2zinc.py` – downloads from HuggingFace datasets API
- `scripts/get_optmath.py` – downloads from HuggingFace datasets API
- `scripts/get_complexor.py` – derives from text2zinc local data

### .gitignore coverage for external data

The following paths are already gitignored:
```
data/external/nl4opt/
data/external/text2zinc/
data/external/optmath/
data/external/complexor/
data/external/optimus/
data/external/*/downloads/
```

The following paths are **NOT** yet gitignored (need adding):
```
data/external/mamo/
data/external/structuredor/
data/external/cardinal_nl4opt/
data/external/industryor/
```

---

## 2. Target Datasets – Pre-Integration Status

### 2.1 MAMO (CardinalOperations/MAMO)

- **Status before this PR:** Not present. No adapter, no script, no data.
- **HuggingFace URL:** https://huggingface.co/datasets/CardinalOperations/MAMO
- **License:** CC-BY-NC-4.0 (non-commercial use only)
- **Splits discovered:** `easy_lp` (file: MAMO_EasyLP.json, 652 rows), `complex_lp` (file: MAMO_ComplexLP.json, 211 rows)
- **Schema:** JSONL, fields: `en_question` (NL problem text), `en_answer` (solution text/answer)
- **No row-level `id` field** – must assign synthetic IDs
- **Scalar gold params:** `en_answer` contains the optimal objective value as a string

### 2.2 StructuredOR (LLM4OR/StructuredOR)

- **Status before this PR:** Not present. No adapter, no script, no data.
- **HuggingFace URL:** https://huggingface.co/datasets/LLM4OR/StructuredOR
- **License:** Unknown (no license in dataset card)
- **Splits discovered:** `train` (86 individual JSON files), `test` (38 individual JSON files)
- **Schema:** One JSON object per file, fields: `question` (NL text), `label` (structured dict with sets/parameters/constraints/variables/objectives), `objective_value` (float)
- **Retrieval method:** Must enumerate file list from HuggingFace siblings API and download individually
- **Note:** The `label` field contains rich structured model representation – useful for formulation mapping

### 2.3 CardinalOperations/NL4OPT (CardinalOperations/NL4OPT)

- **Status before this PR:** Not present as a separate adapter. The existing `nl4opt` adapter uses data from a different source (nl4opt-competition GitHub).
- **HuggingFace URL:** https://huggingface.co/datasets/CardinalOperations/NL4OPT
- **License:** CC-BY-NC-4.0
- **Splits discovered:** `test` (file: NL4OPT_with_optimal_solution.json, 245 rows)
- **Schema:** JSONL, fields: `en_question` (NL text), `en_answer` (optimal value string)
- **No row-level `id` field** – must assign synthetic IDs
- **Note:** This is a different curation than the existing nl4opt adapter. Named `cardinal_nl4opt` to avoid registry key collision.

### 2.4 IndustryOR (CardinalOperations/IndustryOR)

- **Status before this PR:** Not present. No adapter, no script, no data.
- **HuggingFace URL:** https://huggingface.co/datasets/CardinalOperations/IndustryOR
- **License:** CC-BY-NC-4.0
- **Splits discovered:** `test` (file: IndustryOR.json, 100 rows)
- **Schema:** JSONL, fields: `en_question`, `en_answer`, `difficulty`, `id`
- **Has row-level `id` field**

---

## 3. Compatibility Assessment

All four datasets contain natural language optimization problem descriptions and numeric optimal values. Mapping to InternalExample:

| Field | MAMO | StructuredOR | CardinalOperations/NL4OPT | IndustryOR |
|-------|------|-------------|--------------------------|------------|
| `id` | synthetic (row index) | filename stem | synthetic (row index) | provided `id` field |
| `nl_query` | `en_question` | `question` | `en_question` | `en_question` |
| `schema_id` | None | None | None | None |
| `scalar_gold_params` | `{"objective_value": en_answer}` | `{"objective_value": objective_value}` | `{"objective_value": en_answer}` | `{"objective_value": en_answer}` |
| `formulation_text` | None | serialized `label` dict | None | None |
| `structured_gold_params` | None | `label` dict | None | None |
| `metadata.difficulty` | None | None | None | `difficulty` |

**Capability flags for all four:**
- `supports_schema_retrieval`: False (no schema IDs in data)
- `supports_scalar_instantiation`: True (optimal values available)
- `supports_solver_eval`: False (no solver artifacts)
- `supports_full_formulation`: False for MAMO/NL4OPT/IndustryOR; True for StructuredOR (has structured model)

---

## 4. Planned Artifacts

| Artifact | Path | Notes |
|----------|------|-------|
| Fetch script | scripts/get_mamo.py | Downloads from HF raw resolve |
| Fetch script | scripts/get_structuredor.py | Enumerates siblings, downloads per-file |
| Fetch script | scripts/get_cardinal_nl4opt.py | Downloads from HF raw resolve |
| Fetch script | scripts/get_industryor.py | Downloads from HF raw resolve |
| Adapter | src/data_adapters/mamo.py | Maps en_question/en_answer |
| Adapter | src/data_adapters/structuredor.py | Maps question/label/objective_value |
| Adapter | src/data_adapters/cardinal_nl4opt.py | Maps en_question/en_answer |
| Adapter | src/data_adapters/industryor.py | Maps en_question/en_answer/difficulty/id |
| Registry update | src/data_adapters/registry.py | Adds 4 new keys |
| Export script | scripts/export_grounding_training_pairs.py | Generates training pairs |
| Test fixtures | tests/fixtures/datasets/{mamo,structuredor,cardinal_nl4opt,industryor}/test.jsonl | Minimal fixtures |
| Tests | tests/test_dataset_adapters.py | Smoke tests for new adapters |
| .gitignore update | .gitignore | Adds 4 new external data dirs |
| Integration report | docs/NEW_DATASET_INTEGRATION_REPORT.md | Final status report |

---

## 5. Known Risks and Blockers

1. **StructuredOR license is unknown** – dataset card is empty. Proceed with clear documentation noting this.
2. **CC-BY-NC-4.0 on MAMO/NL4OPT/IndustryOR** – non-commercial restriction applies. Data is not vendored into git; users must run download scripts.
3. **No `datasets` or `huggingface_hub` library** in this environment – download scripts use `urllib` directly.
4. **StructuredOR is split across individual JSON files** – the download script must enumerate ~124 files via the API.
5. **MAMO/NL4OPT lack row IDs** – synthetic IDs are generated by position.
