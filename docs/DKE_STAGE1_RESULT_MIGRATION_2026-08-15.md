# DKE STAGE 1 RESULT MIGRATION RECORD (2026-08-15)

This record documents how every number-bearing artifact was classified and
migrated into the Data & Knowledge Engineering (DKE) manuscript version under
`manuscript/dke/main.tex`. Classification follows the frozen-result mandate
(CURRENT / STALE / PARTLY STALE / HISTORICAL / UNSUPPORTED).

## 1. Frozen source artifacts (authoritative)

- `results/final_resubmission_method/metrics.json` (rows 1-2)
- `results/final_resubmission_method/strict_metrics.json`
- `results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv` (patched; `per_query.csv` in the same directory is byte-identical to it)
- `results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv` (pre-patch per-query artifact, matches `metrics.json` row 1)
- `results/final_resubmission_method/config.json` (frozen config, git_sha 72f7e29)
- `results/final_resubmission_method/transitions.json` (prepatch/patched transitions)
- `results/oracle_recomputation_2026-08-15/oracle_frozen_summary.csv` (deterministic recomputation, two identical runs)
- `tools/recompute_frozen_oracle.py` (reproduction script, env var NLP4LP_GOLD_CACHE)

### Authoritative frozen numbers used in DKE main.tex

| Method | Coverage | TypeMatch | Exact20 (on hits) | InstReady | Strict |
|--------|----------|-----------|--------------------|-----------|--------|
| TFIDF-TG (pre-patch) | 0.8794 | 0.8515 | 0.2449 | 0.7764 | 0.7462 |
| TFIDF-TG (frozen) | 0.8886 | 0.8665 | 0.2614 | 0.8006 | 0.7704 |
| Oracle-TG | 0.9416 | 0.9230 | 0.2505 | 0.8489 | 0.8489 |

- Schema R@1 (TF--IDF, orig): 0.9094 = 301/331 (CURRENT; retrieval unaffected by grounding patch).
- Prepatch vs patched strict gain: 8/331 queries, 0 losses; McNemar p = 0.0078.
- Frozen bootstrap (B=10000, seed 42): TFIDF vs Oracle InstReady diff -0.0483 CI [-0.0755,-0.0242] p<0.001; Strict diff -0.0785 CI [-0.1088,-0.0514] p<0.001; strict prepatch-vs-patched +0.0242 CI [0.0091,0.0423] p=0.0006.

## 2. Table-by-table migration (base -> DKE)

| Base table | Classification | DKE disposition |
|---|---|---|
| tab:retrieval_backbones | descriptive | KEPT unchanged (tab:retrieval_backbones) |
| tab:exp_blocks | descriptive | KEPT, ablation row relabeled to numeric-extraction ablation |
| tab:nlp4lp-retrieval-main | CURRENT | KEPT unchanged (R@1 0.9094/0.8822/0.8459 ... matches frozen) |
| tab:nlp4lp-downstream-main | PARTLY STALE | REBUILT from frozen (3 rows: prepatch, patched, oracle) + Strict column; non-frozen method rows (BM25-TG, LSA-TG, CON, SIR, ORR, AR, HAR) dropped (HISTORICAL, deferred to Stage 2) |
| tab:nlp4lp-downstream-variants | STALE | DROPPED from DKE body (no frozen noisy/short downstream artifacts) |
| tab:nlp4lp-error-taxonomy | PARTLY STALE | KEPT as approximate diagnostic; wrong-schema count set to ~30 (matches frozen 331-301=30) |
| tab:nlp4lp-prefix-postfix | STALE | REPLACED with frozen numeric-extraction ablation (prepatch vs patched) |
| tab:nlp4lp-newfamily-errorcheck | STALE | DROPPED (GCG/RAL/AAG have no frozen artifacts; deferred to Stage 2) |
| tab:nlp4lp-significance | STALE | REBUILT from frozen bootstrap (3 rows) |
| tab:nlp4lp-overlap | CURRENT | KEPT unchanged (retrieval-only, unaffected by grounding patch) |
| tab:strict-instready | STALE | REBUILT from frozen (3 rows: prepatch, patched, oracle; n_differ 10/10/0) |
| tab:eng-structural / eng-executable / eng-solver | CURRENT | KEPT unchanged (committed camera-ready tables) |

## 3. Stale numbers purged

All occurrences of these stale values were removed from `manuscript/dke/main.tex`
(verified via `pdftotext` grep on the compiled PDF; zero matches):

- 0.5287 (old InstReady), 0.5680 (old oracle), 0.0393 (old oracle gap)
- 0.5196 / 0.5076 (old BM25/LSA InstReady)
- 0.7453 / 0.8609 / 0.1834 (old TypeMatch / Coverage / Exact20)
- 0.3239 / 0.7529 / 0.4985 / 0.4320 / 0.4230 / 0.0272 (non-frozen families)

## 4. Other DKE-version edits

- Document class migrated: sn-jnl -> elsarticle `[3p,times]` + elsarticle-num.bst.
- Front matter migrated: \author/\affil -> elsarticle author/affiliation with
  corresponding-author + ORCID footnote (elsarticle v3.4c has no \credit /
  \orcid / \vitae / statement / dataavailability commands; CRediT written as
  plain text in Declarations).
- Abstract rewritten: 227 words (<=250), no references, frozen numbers.
- Keywords: 6 (allowed 1-7).
- Added highlights.txt (5 bullets, each <=85 chars, verified programmatically).
- References: added paanop2025 (IJCAI-25), huang2025orlm (OR 73(6)), lu2025optmath (ICML PMLR 267), xiao2026deepor (AAAI-26 40(40)); all cited in related work.
- Fixed "no LLM comparison" limitation: now states preliminary common-18
  external-baseline assessment (PaMOP 18/18, OptMATH 15-18/18, generic LLM
  16/18, ORLM blocked on coptpy, DeepOR/OR-R1 no released checkpoints).
- All displayed equations numbered (the top-1 argmax now labeled eq:top1_schema).
- Added AUTHOR_INPUT_REQUIRED.md for vitae/photo, CRediT, data-deposit Option C, funding.

## 5. Compilation

- Toolchain available on this machine: tectonic (no pdflatex). Compiles clean
  with 0 errors via `tectonic main.tex`; 23 pages.
- Fixed during compile: `\botrule` (Springer macro) -> `\bottomrule` (booktabs),
  and added `\usepackage{hyperref}` so the bbl's `\path{...}` fallback does not
  break on DOI underscores.

## 6. Deferred to Stage 2 (full rewrite)

- Non-frozen deterministic variants (BM25-TG, LSA-TG, CON, SIR, ORR, AR, HAR)
  and families (GCG/RAL/AAG): numbers not reported; reconciliation pending.
- Cross-variant (noisy/short) downstream numbers: not reported in DKE body.
- Exact error-taxonomy recount from frozen per-query CSVs.
- Full narrative rewrite beyond number replacement.