# External Validation Claims Audit

Based on the adversarial scientific verification of the OptMATH component-level results, here is the disposition of claims.

## SAFE_TO_CLAIM

* **Claim A:** The deterministic numeric extractor retains substantial recall on an independent optimization-modeling dataset. (RawASTRecall ~0.75, VerifiedTextSupportedRecall ~0.76).
* **Robustness:** The core extraction rules generalize gracefully without major parse failure or catastrophic performance drop when tested on OptMATH natural language inputs.

## MUST_QUALIFY

* **Claim D (Generalization):** The experiment addresses the previous benchmark-generalization criticism *minimally and partially*. It demonstrates that the numeric parameter extraction (a single sub-component) works on external text. It does *not* prove that the full framework (retrieval + grounding) generalizes.

## DO_NOT_CLAIM

* **Claim B:** The deterministic grounding module generalizes beyond NLP4LP. (This experiment only evaluates numeric extraction, not schema-conditioned number-to-slot grounding).
* **Claim C:** The result demonstrates that the framework generalizes beyond NLP4LP. (Extraction != framework).
* **External schema retrieval generalization.**
* **External slot grounding generalization.**
* **External InstantiationReady generalization.**
* **Solver-readiness.**
* Any phrasing suggesting the extraction rules are "**NLP4LP-trained**" or "**learned from NLP4LP**".

## Recommended Manuscript Sentence (Supplementary/Appendix)

"To verify that our component-level numeric extraction heuristics are not overfitted to NLP4LP, we conducted an external audit on a 1,000-instance sample of the OptMATH dataset; the deterministic rules preserved a verified text-supported recall of 80.7% on external code-aligned numeric parameters, confirming basic component generalization without claiming full end-to-end framework applicability."
