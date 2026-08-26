# SN Computer Science — Stage 5: Reviewer Hardening, Literature Modernization, Venue Positioning

**Date:** 2026-08-27  
**Builds on:** Stages 1–4  
**Starting HEAD:** `ab15970`  
**Authoritative manuscript:** `manuscript/sncs/main.tex`

---

## 1. Executive Verdict

Stage 5 closed the Azure funding TODO (Case A: standard Azure for Students educational credit), updated the software AI disclosure to include Claude, modernized Related Work with three independently verified 2025–2026 systems (OptimAI, AlphaOPT, ORThought), added a task-level comparison-landscape table (not a leaderboard), sharpened novelty and practical-workflow positioning without overclaiming, clarified Exact20’s method-dependent denominator, and synchronized submission metadata/checklist. No new numerical experiments were fabricated. A fair strong new InstantiationReady baseline run was **not** feasible under metric/task incompatibility. Clean-room build succeeds at **40 pages**, 0 undefined citations, 0 unresolved references. Remaining scientific risk is moderate and primarily about task-scope comparison expectations, not provenance or template compliance.

---

## 2. Preflight

| Field | Value |
|---|---|
| HOSTNAME | al-khwarizmi |
| REPO_PATH | /home/soroush/combinatorial-opt-agent |
| BRANCH | main |
| STARTING_HEAD | ab15970 (= origin/main) |
| AHEAD_BEHIND | 0/0 at start |
| WORKTREE | clean except unrelated EAAI figure PDFs |
| GIT_LOCKS | none |
| WORKTREES | single canonical |

EAAI PDFs were not touched by Stage 5 manuscript work (local mtime noise may still appear in `git status`).

---

## 3. Azure Closure

**Classification:** CASE A / `STANDARD_STUDENT_CREDIT`

Author confirmed: Azure for Students subscription with ordinary USD 100 educational credit; not a research-specific Microsoft grant; no award/contract number.

**Manuscript Funding wording (both SNCS and DKE):**
> This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The external large-language-model baseline experiments used Microsoft Azure OpenAI services; associated usage was supported by the standard Azure for Students educational credit available to the author.

`TODO(AUTHOR_CONFIRMATION_REQUIRED)` removed from manuscript sources and submission metadata/checklist.

Placement: Funding declaration (Springer declarations guidance includes research support disclosures in Funding/Declarations). Not labeled as a research grant.

---

## 4. AI Disclosure Update

Verified against Springer Nature AI guidance (authors remain accountable; AI tools are not authors; disclose AI assistance used in content generation / development).

- Writing: ChatGPT, Gemini (unchanged structure)
- Software: **Claude, Cursor, and GitHub Copilot** (Claude added)
- Explicit author review/verification/testing/responsibility retained
- No AI authorship implication

---

## 5. Current SNCS Policy Verification

Re-checked official SN Computer Science aims/scope and Springer Nature journal/AI guidance (2026-08-27):

| Item | Finding |
|---|---|
| Aims/scope | Broad CS including AI, IR, NLP-adjacent, algorithms, mathematical programming — fit remains strong |
| Article type | Original Research |
| Template | `sn-jnl` + `sn-basic` numbered — compliant |
| Abstract | Structured Purpose/Methods/Results/Conclusion; ~249 words after Stage-5 trim; no hard SNCS-specific ≤250 found; Springer general abstract ceilings cited historically up to ~350 |
| Keywords | 6 — within 4–6 convention |
| Page limit | **No SNCS-specific hard page limit located** |
| AI disclosure | Present; updated |
| Funding | Present; Case A resolved |
| Cover letter | Not mandated for standard research articles |

If an SNCS-specific hard abstract/page limit exists only behind Editorial Manager login, it was **not** independently verifiable from public pages in this stage.

---

## 6. Accepted-SNCS Convention Study

Studied recent SN Computer Science / closely related Springer CS articles (approx. 10), including:

1. RAG evaluation framework (SN Computer Science, 2026)
2. Imbalanced vulnerability detection (SN Computer Science, 2026)
3. IDS + SMOTE-IPF optimization (SN Computer Science, 2025)
4. Student outcome ML (SN Computer Science, 2025)
5. Federated-learning anomaly injection (SN Computer Science, 2026)
6–10. Additional Springer AI/IR/optimization-adjacent 2025–2026 articles used for tone/structure cues (Discover Computing / JIIS / related Springer CS venues where SNCS-specific peers were thin)

| FEATURE | COMMON_SNCS_PATTERN | CURRENT_MANUSCRIPT | ACTION_NEEDED |
|---|---|---|---|
| Section names | Intro / Related Work / Method / Experiments / Conclusion (Related Work early or late) | Related Work early; Method; Experiments; Conclusions+Limitations | Keep (already conventional) |
| Related Work | Analytical families, not dump | Families present; strengthened with 2025–26 systems + landscape table | Done |
| Experiments | Setup then results/ablations | Already structured | Keep |
| Limitations | Explicit subsection | Present under Conclusions | Keep |
| Tables | Compact, defined abbreviations | Landscape table added; Exact20 footnote strengthened | Done |
| Tone | Journal article, not rebuttal | Defensive “avoid misunderstanding” novelty sentence rewritten | Done |

---

## 7. Historical Rejection-Risk Audit (pre-edit)

| PREVIOUS_REJECTION_REASON | CURRENT_SNCS_EVIDENCE | RESOLVED | REMAINING_RISK | ACTION |
|---|---|---|---|---|
| KAIS: insufficient comparative studies / outdated baselines | Strong RW + common-18 contextual eval, but missing newest agent/library systems | Partial | Medium | Add landscape table + OptimAI/AlphaOPT/ORThought |
| EAAI: engineering application required | SNCS is not EAAI; no fabricated case study | N/A (non-applicable) | Low | Clarify practical workflow narrative |
| DKE: scope/expertise mismatch | SNCS broad CS; keywords already IR/NLP/KR/opt | Mostly | Low–Medium | Keep CS positioning explicit for reviewer discovery |
| Digital Engineering: reviewer shortage | Process risk, not scientific | N/A | Process | Improve discoverability via title/abstract/keywords/RW |

---

## 8. Pre-edit Reviewer Rubric (0–10)

| Category | Score | Strengths | Weaknesses | Fixable now | Exact action |
|---|---:|---|---|---|---|
| Novelty | 7 | Honest non-claim on TF-IDF/BM25/etc.; diagnostic framing | Novelty still read as defensive | Yes | Rewrite contribution paragraph positively |
| Baselines/comparison | 6 | Lexical+dense+oracle; common-18 context | Newest systems absent; no InstantiationReady peer | Partial | Landscape table + feasibility analysis |
| Related work | 7 | Good taxonomy | Missing 2025–26 agent/library papers | Yes | Add verified refs + analytical contrast |
| Datasets | 8 | NLP4LP scoped; OptMATH component check | Single-benchmark limitation | Yes | Call curated/gated benchmark explicitly |
| Experimental rigor | 8 | Bootstrap/McNemar; residual decomposition | No new comparable baseline | No (infeasible fairly) | Document incompatibility |
| Reproducibility | 9 | Scripts + manifest | — | Docs sync | Sync metadata |
| Technical clarity | 8 | Algorithm present | Exact20 denominator easy to miss | Yes | Numerator/denominator wording |
| Writing | 7 | Precise | Audit/rebuttal tone remnants | Yes | Tone pass on novelty/workflow |
| Formatting | 8 | Stage 4 fixed Table 12 | Landscape table risk | Yes | Fit-to-width table design |
| Limitations | 8 | Explicit | Repetition risk | Light | Prefer one full explanation |
| Practical impact | 6 | Restricted solver evidence | Workflow understated | Yes | Add concrete workflow sentence |
| Missing refs | 6 | Core 2024–26 present | OptimAI/AlphaOPT/ORThought absent | Yes | Verify+cite |

**PRE_EDIT_SCORE_0_TO_100:** 78

---

## 9. Novelty Analysis

Novelty is positioned as:

- retrieval-conditioned schema instantiation with schema-dependent slot inventory;
- deterministic scalar grounding as transparent intermediate task;
- controlled retrieval-vs-grounding separation (lexical/dense/oracle);
- StrictInstantiationReady + exact residual-error decomposition;
- empirical finding that fine-grained value-to-slot accuracy dominates residual failure.

Explicitly **not** claimed: inventing TF-IDF/BM25/LSA/BGE-M3/rule extraction; being first to notice binding is hard (Gao et al. already state this in open-ended settings).

---

## 10–13. Literature Search, References Added/Rejected, Citation Audit

### Search methodology
arXiv HTML/ABS, Crossref, SpringerLink, GitHub release pages, publisher DOI pages; independent verification (not citation-by-suggestion alone). Prefer published version when available.

### Verified candidates

| Title | Authors | Year | Venue/Status | DOI / link | Directly comparable? | Already cited? | Decision |
|---|---|---|---|---|---|---|---|
| OptimAI | Thind et al. | 2025 | arXiv:2504.16918 (preprint) | https://arxiv.org/abs/2504.16918 | NO (end-to-end NLP4LP accuracy) | NO | **ADD** (contextual) |
| AlphaOPT | Kong et al. | 2025 | arXiv:2510.18428 (preprint); code MIT | https://arxiv.org/abs/2510.18428 | NO (experience-library end-to-end) | NO | **ADD** (contextual) |
| ORThought | Yang et al. | 2026 | *Artificial Intelligence for Transportation* 6:100059 | 10.1016/j.ait.2026.100059 | NO (logistics CoT; different primary bench) | NO | **ADD** (contextual) |
| NEMO / related agent systems | Song et al. / others | 2026 | arXiv preprint | — | NO | NO | Mentioned in search; not all added (diminishing returns) |

### Citation correctness
Concrete RW claims for OptiMUS/OptLLM/ORLM/OptMATH/SIRL/OR-R1/DeepOR/Gao binding were spot-checked against prior Stage 1–4 verification trail plus new sources for the three added papers. No unsupported numerical leaderboard claims were attached to the new citations. Citation clusters >4: **0**.

**NEW_VERIFIED_REFERENCES_ADDED:** 3

---

## 14–15. Related Work + Landscape Table

Related Work now analytically contrasts end-to-end generation vs fixed-catalog scalar instantiation for OptimAI/AlphaOPT/ORThought.

Table `tab:comparison_landscape` compares task/output/retrieval/solver-feedback/training/open-artifact/direct-comparability. **No incomparable accuracy numbers** are placed in one column.

---

## 16–17. Baseline Feasibility / New Baseline Run

| Candidate | Public code? | Runnable here fairly? | NLP4LP? | Metric compatible with InstantiationReady? | Decision |
|---|---|---|---|---|---|
| OptimAI | Not a drop-in InstantiationReady evaluator; multi-agent LLM | No (API cost + different task) | Yes (end-to-end) | **No** | Do not run as InstantiationReady baseline |
| AlphaOPT | Yes (MIT) | Requires LLM API + library learning campaign | Uses NLP4LP among others | **No** (end-to-end Acc) | Do not run as InstantiationReady baseline |
| ORThought | Yes | Logistics/LogiOR focus; LLM pipeline | Not InstantiationReady | **No** | Cite only |
| OptiMUS / ORLM / OptMATH | Already contextualized on common-18 | Already reported | Partial | **No** for InstantiationReady cells | Keep existing contextual assessment |

**STRONG_NEW_BASELINE_RUN:** NO  
**Reason:** No verified public system exposes a fair InstantiationReady / Exact20 (on hits) evaluation on the same fixed-catalog scalar-instantiation task without conflating end-to-end solver accuracy with schema-conditioned grounding.

---

## 18–26. Dataset, Practical Impact, Figures/Tables, Pseudocode, Math, Exact20, Rigor, Reproducibility

- Dataset wording: curated/gated public benchmark (not casually “real-world production”).
- Practical workflow sentence added in Introduction.
- Existing pipeline figure retained (authoritative schematic).
- Tables: landscape added; Exact20 denominator made explicit (291/291/303/320); minor overfull (~4.6pt) remains on one significance table — not a serious visual defect.
- Pseudocode: Algorithm already present and aligned with method description — kept.
- Formal equations remain numbered; nonstandard terms defined.
- Experimental claims unchanged numerically; no fabricated stats.
- Reproducibility docs/metadata synchronized for funding/AI; detailed commands remain in `docs/SNCS_REPRODUCIBILITY.md`.

---

## 27–31. Writing, Abstract, Organization, Length, Discoverability

- Novelty paragraph rewritten in journal tone (removed “To avoid any misunderstanding”).
- Abstract self-contained (~249 words), structured, no citations.
- Organization unchanged at high level (already SNCS-conventional).
- Page count: 39 → **40** (landscape table + literature), still no hard limit.
- Reviewer discoverability: title/abstract/keywords/RW now clearly signal NLP/IE, IR, AI-for-optimization, optimization modeling, structured grounding.

---

## 32. Post-edit Previous-Rejection-Risk Matrix

| PREVIOUS_REJECTION_REASON | BEFORE | CHANGE | AFTER | EVIDENCE |
|---|---|---|---|---|
| KAIS comparison | Medium–High | Landscape table + 3 modern systems + feasibility note | Medium (residual: no InstantiationReady peer run) | Table + RW |
| EAAI application | N/A | Practical workflow clarified | Low / addressed without fake case study | Intro workflow |
| DKE scope for SNCS | Low–Medium | Explicit CS multi-area positioning retained/strengthened | Low | Keywords + RW + abstract |
| Reviewer discoverability | Medium | Clearer communities in front matter/RW | Low–Medium | Title/abstract/keywords/RW |

**KAIS_COMPARISON_RISK_RESOLVED:** NO (reduced, not eliminated)  
**EAAI_PRACTICALITY_CONCERN_ADDRESSED:** YES  
**DKE_SCOPE_RISK_FOR_SNCS_LOW:** YES  
**REVIEWER_DISCOVERABILITY_IMPROVED:** YES

---

## 33. Post-edit Reviewer Rubric

| Category | Score |
|---|---:|
| Novelty and contribution | 8.5 |
| Baselines and comparison | 7.5 |
| Related work | 8.5 |
| Datasets/workloads | 8.5 |
| Experimental rigor | 8.0 |
| Reproducibility | 9.0 |
| Technical clarity/correctness | 8.5 |
| Writing/presentation | 8.5 |
| Formatting/journal style | 8.5 |
| Limitations/future work | 8.0 |
| Applications/practical impact | 8.0 |

**POST_EDIT_SCORE_0_TO_100:** 86

Justification: literature/positioning/Exact20/funding/AI/workflow substantially harden rejectability for “outdated/missing comparison” and disclosure issues, but a true InstantiationReady peer baseline remains unavailable, so score stays in 85–89 (moderate residual comparison-scope risk), not ≥90.

---

## 34. Build / Visual Audit

| Check | Result |
|---|---|
| BUILD_SUCCESS | YES |
| FINAL_PAGE_COUNT | 40 |
| UNDEFINED_CITATIONS | 0 |
| UNRESOLVED_REFERENCES | 0 |
| SERIOUS_VISUAL_PROBLEMS | 0 |
| CLEAN_ROOM_BUILD_SUCCESS | YES |
| Overfull | one minor ~4.6pt table warning (not serious) |

---

## 35–36. Repository / Branch / Commits

Wulver: `HUMAN_INTERACTIVE_LOGIN_REQUIRED` (quick check only).  
EAAI PDFs: intentionally untouched as scientific content.  
Commits: Stage-5 logical commits pushed to `origin/main` (see git log).

---

## 37. Remaining Risks

1. Reviewers may still demand head-to-head InstantiationReady numbers against OptimAI/AlphaOPT despite metric incompatibility.
2. Single-benchmark (NLP4LP) dependence remains.
3. Dense-retriever embeddings are not rule-interpretable (already disclosed).
4. Common-18 external context remains small and non-leaderboard by design.

**READY_FOR_STAGE6_FINAL_SUBMISSION_FREEZE:** YES (scientifically), pending only author personal EM entry and optional cover letter.
