# External Dataset Validation Protocol

**Date:** 2026-08-15
**Goal:** External validation of the retrieval-assisted schema-conditioned scalar-grounding framework on an independent optimization dataset, addressing generalization beyond the NLP4LP benchmark.

## 1. Candidate Dataset Evaluation & Rejection

To ensure a scientifically defensible external validation, we evaluated several candidate optimization-modeling datasets:

- **NL4Opt**: Rejected. The NLP4LP benchmark used in the paper is an expansion of the original NL4Opt dataset. Validating on NL4Opt violates independence because it is a subset of the paper's primary benchmark.
- **Text2Zinc**: Rejected. The dataset is gated on HuggingFace and could not be downloaded.
- **CP-Bench (DCP-Bench-Open)**: Rejected. The dataset provides Python (CPMpy) code solutions but lacks the corresponding natural-language problem descriptions needed to evaluate retrieval from text.
- **MAMO**: Rejected. The raw data URLs yield HTTP 404s. The internal dataset snapshots (`mamo_easy_lp.jsonl`) provide only a question and a single numerical answer string, lacking intermediate structural schemas or scalar parameters required for schema retrieval or grounding.
- **IndustryOR**: Rejected. Similar to MAMO, it provides `en_question` and `en_answer` but lacks gold structured schemas or formulations.
- **OptMATH (`shushulei/OptMATH-Train`)**: Rejected for full pipeline evaluation. OptMATH provides natural language inputs and LLM-generated Python formulations. However, it does not use a fixed catalog of templates, nor does it provide explicit schema IDs or structured scalar parameter mappings. Evaluating "schema retrieval" on OptMATH would require clustering or fabricating arbitrary schema labels, producing a misleading experiment.

**Conclusion:** No publicly accessible, independent dataset supports the *full* fixed-catalog schema retrieval and grounding task without introducing fabricated schema mappings.

## 2. Scientifically Valid External Subtask: Numeric Extraction Recall

While the full pipeline cannot be evaluated without arbitrary schema labels, a critical foundation of the proposed deterministic grounding approach is its **numeric extraction phase**. The framework assumes it can reliably extract digits, spelled-out numbers, percentages, and multiplicative ratio expressions (e.g., "twice", "half") from diverse text without relying on an LLM. 

We define a **Numeric Extraction Recall** subtask on the OptMATH dataset to evaluate whether the deterministic extraction rules used in the paper overfit to NLP4LP's lexical style, or if they generalize to an independent, structurally diverse dataset.

### Protocol Details

- **External Dataset:** `shushulei/OptMATH-Train` (HuggingFace).
- **Sample Size:** 1,000 randomly selected rows (using seed 0).
- **Gold Labels Formulation:** The gold Python code block is extracted from the `output` field. The code is parsed into an Abstract Syntax Tree (AST), and all literal numeric constants (integers and floats) in the code are extracted to form the set of "gold parameters".
- **Inference Task:** Our frozen, deterministic `_extract_num_mentions` function (from `tools/nlp4lp_downstream_utility.py`) is run on the natural language `input` text.
- **Metrics:** 
  - **Extraction Recall:** Fraction of gold unique numeric values (from the AST) that are successfully matched by the extracted mentions (converted to float values).
- **Classification:** `EXTERNAL_ONLY` (This is a component-level diagnostic, not directly comparable to the end-to-end `InstantiationReady` metric on NLP4LP).
- **Limitations:** AST parsing of LLM-generated code may include irrelevant structural numbers (e.g., loop indices like `range(25)`), which artificially lowers recall because these numbers are not explicitly stated in the text. This metric is a strict lower bound on extraction capability.
