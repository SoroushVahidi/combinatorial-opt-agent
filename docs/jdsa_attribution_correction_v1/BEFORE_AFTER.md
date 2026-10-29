# Exact before/after source changes

Baseline: `00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc`. Sequential literal replacements; no numerical observations changed.

## 1. manuscript/jdsa-stage3/additional_evidence.tex

Repair absent register path with immutable provenance links; distinguish frozen wording.

Before:
```latex
The accompanying evidence register, \path{docs/jdsa_revision_phase1_20261029/EXPERIMENT_EVIDENCE_REGISTER.md}, assigns 47 investigation groups to explicit publication destinations.
```

After:
```latex
The \href{https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650/docs/jdsa_revision_phase1_20261029/EXPERIMENT_EVIDENCE_REGISTER.md}{immutable Phase 1 evidence register} assigns 47 investigation groups to publication destinations. The \href{https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json}{released claims manifest} preserves its evidence identities; the \href{https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc/docs/jdsa_manuscript_v1_20261029/CLAIM_LOCATION_CROSSWALK_V1.md}{v1.0.0 location crosswalk} records their placement in that manuscript version. Those frozen records retain the historical oracle wording clarified below.
```

## 2. manuscript/jdsa-stage3/additional_evidence.tex

Define historical metric without changing stored fields or observations.

Before:
```latex
The values below therefore retain aggregate-supported status and are not independent replications of retrieval.
```

After:
```latex
The values below therefore retain aggregate-supported status and are not independent replications of retrieval. Hit@5 averages indicators that the target occurs among the first five results; the historical implementation stores this quantity under the field name \texttt{P@5}. It is not precision among the five retrieved documents.
```

## 3. manuscript/jdsa-stage3/additional_evidence.tex

Correct display heading only.

Before:
```latex
Variant & Method & P@5 & MRR@10
```

After:
```latex
Variant & Method & Hit@5 & MRR@10
```

## 4. manuscript/jdsa-stage3/additional_evidence.tex

Distinguish saved oracle A/C selection rules and the two-case readiness difference; retain B/heuristic results.

Before:
```latex
A gold-schema top-ten oracle yields 270 ordinary-ready queries, while a readiness-maximizing oracle yields 286, including 32 wrong-schema-ready cases. Among tested heuristic rules, the largest observed readiness count is 282, with 20 schema regressions.
```

After:
```latex
Oracle A selects the gold schema whenever it occurs in the top ten, otherwise retaining the baseline: it yields 268/331 ordinary-ready queries and 325/331 correct schemas. Oracle C selects the gold schema only when its grounding is ready, otherwise retaining the baseline: it yields 270/331 ordinary-ready queries and 322/331 correct schemas. Its ready set includes two wrong-schema cases, \path{nlp4lp_test_286} and \path{nlp4lp_test_299}. Their gold candidates, at ranks four and six, have coverage $2/3$ and $1/2$ and fail readiness; oracle A switches to them, whereas C retains the wrong but ready baseline choices. Thus both oracles have 268 schema-correct ready queries. Oracle B maximizes readiness within the top ten without requiring the correct schema and yields 286 ready queries, including 32 wrong-schema-ready cases. Among tested heuristics, the top-five readiness-first rule yields the largest observed readiness count, 282, with 20 schema regressions.
```

## 5. manuscript/jdsa-stage3/main.tex

Scope contribution to the evaluated assisted task.

Before:
```latex
\item An explicit diagnostic protocol linking retrieval, coverage/type readiness and complete stored-reference scalar agreement to their input assistance and denominators.
```

After:
```latex
\item A diagnostic protocol for reference-assisted supplied-slot grounding, linking retrieval, coverage/type readiness and complete stored-reference scalar agreement to their input assistance and denominators.
```

## 6. manuscript/jdsa-stage3/main.tex

Avoid priority claim for component-level evaluation.

Before:
```latex
The contribution is an evaluation methodology and its empirical application, not a new retrieval or matching algorithm.
```

After:
```latex
The contribution is the operationalization and empirical application of these diagnostics to supplied-slot grounding. It claims neither priority for component-level evaluation nor a new retrieval or matching algorithm.
```

## 7. manuscript/jdsa-stage3/main.tex

Position against verified primary-source descriptions without inventing baseline results.

Before:
```latex
Compatibility and validity checks also differ from semantic validation.
```

After:
```latex
Refai and Ahmed evaluate generated formulations at the component level, including decision-variable and constraint precision/recall and constraint/objective root mean squared error \cite{refai2026component}. OPT-Engine varies optimization-problem complexity to diagnose formulation and solving limitations, including constraint formulation \cite{chen2026optengine}. These works share our diagnostic motivation. Here, complete stored-reference agreement is conditioned on readiness in a reference-assisted scalar inventory; ordered gate accounting and fixed-score decoder comparisons delimit the failures and attributions supported by that setting.

OptMATH synthesizes modeling data through back-translation and forward checking \cite{lu2025optmath}; our external extraction check does not evaluate that synthesis framework or full-pipeline transfer. Autoformulation combines LLM-guided Monte Carlo tree search with symbolic pruning over formulations \cite{astorga2025autoformulation}. Its search space includes modeling decisions absent from our supplied-slot assignment controls. These task distinctions do not constitute empirical comparisons with either system.

Compatibility and validity checks also differ from semantic validation.
```

## 8. manuscript/jdsa-stage3/main.tex

State the specific operational distinction in methodological position.

Before:
```latex
We contribute joint endpoint reporting, explicit conditional populations and controlled attribution within saved records.
```

After:
```latex
For reference-assisted supplied slots, we combine complete-reference-agreement reporting, ordered gate accounting, explicit conditional populations and controlled attribution within saved records.
```

## 9. manuscript/jdsa-stage3/main.tex

Connect discussion to prior diagnostic frameworks and delimit novelty.

Before:
```latex
This complements Text2Opt-Bench's modeling--binding analysis \cite{gao2026models}; it adds neither a new matching algorithm nor evidence that numerical binding failures were first identified here.
```

After:
```latex
This complements Text2Opt-Bench's modeling--binding analysis \cite{gao2026models}, Refai and Ahmed's component-level formulation evaluation \cite{refai2026component}, and OPT-Engine's complexity-based diagnosis \cite{chen2026optengine}. Our narrower contribution concerns readiness-conditioned complete reference agreement and fixed-score attribution in supplied-slot grounding; it adds neither a new matching algorithm nor priority for identifying numerical binding failures.
```

## 10. manuscript/jdsa-stage3/main.tex

Link corrected companion PDF rather than the immutable older release.

Before:
```latex
\externaldocument[S-]{supplement}[https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/download/jdsa-manuscript-v1.0.0/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_Supplement.pdf]
```

After:
```latex
\externaldocument[S-]{supplement}[ESM_1.pdf]
```

## 11. manuscript/jdsa-stage3/supplement.tex

Link corrected companion PDF rather than the immutable older release.

Before:
```latex
\externaldocument[M-]{main}[https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/download/jdsa-manuscript-v1.0.0/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf]
```

After:
```latex
\externaldocument[M-]{main}[manuscript.pdf]
```

## 12. manuscript/jdsa-stage3/references.bib

Add verified primary publication metadata. Existing bibliography entries unchanged.

Before:
```latex
(new bibliography entries)
```

After:
```latex

@article{refai2026component,
  author    = {Refai, Dania and Ahmed, Moataz},
  title     = {A component-level evaluation framework for diagnosing {LLM} errors in optimization modelling},
  journal   = {Neurocomputing},
  volume    = {695},
  pages     = {133981},
  year      = {2026},
  doi       = {10.1016/j.neucom.2026.133981}
}

@inproceedings{chen2026optengine,
  author    = {Chen, Yitian and Cheng, Cheng and Sun, Yinan and Ling, Zi and Ge, Dongdong},
  title     = {{OPT}-Engine: Benchmarking the Limits of {LLM}s in Optimization Modeling via Complexity Scaling},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning ({ICML} 2026)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  pages     = {13807--13834},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/chen26o.html}
}

```

## 13. manuscript/jdsa-stage3/main.tex

Remove online-URL destination override for local PDF companion links; avoid double-encoded named destinations.

Before:
```latex
% Use the standard PDF named-destination URL parameter for online companion links.
\makeatletter
\def\XR@@dURL#1#2#3#4#5\\{%
  \unexpanded{{#1}{#2}{#3}{nameddest=#4}}{\XR@URL}}
\makeatother

```

After:
```latex
% Use standard PDF destinations in the corrected local companion document.

```

## 14. manuscript/jdsa-stage3/supplement.tex

Remove online-URL destination override for local PDF companion links; avoid double-encoded named destinations.

Before:
```latex
% Use the standard PDF named-destination URL parameter for online companion links.
\makeatletter
\def\XR@@dURL#1#2#3#4#5\\{%
  \unexpanded{{#1}{#2}{#3}{nameddest=#4}}{\XR@URL}}
\makeatother

```

After:
```latex
% Use standard PDF destinations in the corrected local companion document.

```

## 15. manuscript/jdsa-stage3/additional_evidence.tex

Keep oracle explanation in a separate paragraph and its identifier together at a page break.

Before:
```latex
Thus 325 targets occur in the top ten. Oracle A selects
```

After:
```latex
Thus 325 targets occur in the top ten.

Oracle~A selects
```

## 16. manuscript/jdsa-stage3/additional_evidence.tex

Keep oracle identifiers together in typesetting.

Before:
```latex
Oracle C selects
```

After:
```latex
Oracle~C selects
```

## 17. manuscript/jdsa-stage3/additional_evidence.tex

Keep oracle identifiers together in typesetting.

Before:
```latex
Oracle B maximizes
```

After:
```latex
Oracle~B maximizes
```

## 18. manuscript/jdsa-stage3/main.tex

Distinguish unreleased author-review correction from the preserved manuscript release.

Before:
```latex
Editable manuscript sources, the article and Online Resource 1 are archived in
```

After:
```latex
The baseline manuscript sources, article and Online Resource 1 are archived in
```

## 19. manuscript/jdsa-stage3/main.tex

Distinguish unreleased author-review correction from the preserved manuscript release.

Before:
```latex
The numerical-summary derivation is recorded separately from its authenticated inputs.
```

After:
```latex
The numerical-summary derivation is recorded separately from its authenticated inputs. This attribution correction remains an unreleased author-review revision.
```

## 20. manuscript/jdsa-stage3/reproduction_map.tex

Distinguish unreleased author-review correction from the preserved manuscript release.

Before:
```latex
provides revised sources, a separate claim-location crosswalk and audit reports.
```

After:
```latex
provides the baseline manuscript sources, a separate claim-location crosswalk and audit reports. This attribution correction remains an unreleased author-review revision.
```
