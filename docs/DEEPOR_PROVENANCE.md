# DeepOR provenance

## Verification log

- 2026-08-12: initial artifact search (paper, PDF, appendix, code, checkpoint,
  dataset, project page) per the table below.
- 2026-08-15 (recheck for the external-baseline campaign): AAAI proceedings
  page, dblp, and GitHub repository search re-run; no attributable code,
  checkpoint, dataset, or project-page release was found. Classification
  unchanged: `DEEPOR_PAPER_RECONSTRUCTION_READY`, zero empirical rows possible.

## Paper

Xiao et al., “DeepOR: A Deep Reasoning Foundation Model for Optimization
Modeling,” *Proceedings of AAAI-26*, 40(40), 34052–34060, published
2026-03-14. DOI: [10.1609/aaai.v40i40.40699](https://doi.org/10.1609/aaai.v40i40.40699).
The authoritative [proceedings page](https://ojs.aaai.org/index.php/AAAI/article/view/40699)
and [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/40699/44660)
were checked on 2026-08-12.

## Official artifacts

| Artifact | Status | Evidence/notes |
|---|---|---|
| Paper and PDF | AVAILABLE | AAAI proceedings, DOI above |
| Supplementary appendix | NOT FOUND | No separate appendix is linked by the proceedings page |
| Official code | NOT FOUND | Fresh exact-title, author, GitHub, Hugging Face, and ModelScope searches found no attributable release |
| DeepOR checkpoint | NOT FOUND | The paper names Qwen3-8B as its base, but does not identify a released fine-tuned checkpoint |
| Dataset release | NOT FOUND | Paper uses OptMATH’s 210k problem-model pairs; no DeepOR data release was located |
| Project page | NOT FOUND | No attributable project URL was exposed by the proceedings record or searches |

This repository therefore implements **PATH_D_PAPER_RECONSTRUCTION**. It
does not fabricate weights or claim empirical DeepOR results.

## Method architecture

DeepOR trains a reasoning model in two stages. Expertise Tuning synthesizes
long chain-of-thought trajectories through a 34-node expert flowchart (21
thought and 13 decision nodes), then performs SFT. Self-Improvement Learning
uses GRPO and a checklist-based reward shaped from feasibility, correctness,
and robustness, with solver feedback and adaptive resampling. The paper uses
Qwen3-8B, samples 10k instances for SFT from a 210k corpus, and trains RL on
the full corpus.

At evaluation, the paper reports greedy decoding with temperature 0, top-p 1,
and repetition penalty 1.0. The output is a reasoning-bearing optimization
model/program; the case study and checklist identify Pyomo compilation and
solver execution as the relevant execution path. The primary metric is
pass@1 based on the final objective value.

## Training versus inference

Training-time flowchart synthesis, SFT, GRPO, checklist judging, and reward
shaping are recorded for provenance only. The local package provides a
mockable inference interface and conservative parser, not a runnable trained
model.

## Fidelity matrix

| Component | Primary evidence | Local implementation | Fidelity | Notes |
|---|---|---|---|---|
| Base model | Paper training setup | Qwen3-8B target recorded, no weights | PAPER_SPECIFIED | Checkpoint unavailable |
| Reasoning output | Paper abstract/Fig. 2 | Separate reasoning and final answer | PAPER_SPECIFIED / PAPER_RECONSTRUCTED | Literal delimiters are not published |
| Prompt | No literal prompt in available artifacts | Versioned staged optimization prompt | PAPER_RECONSTRUCTED | Not an official prompt |
| Flowchart | Paper expert-flowchart section | Not regenerated at inference | PAPER_SPECIFIED | Training-time component |
| Decoding | Paper evaluation setup | temperature 0, top-p 1, greedy, penalty 1 | PAPER_SPECIFIED | Length remains configurable |
| Solver representation | Paper case study/checklist | Pyomo-oriented static checks/harness | PAPER_RECONSTRUCTED | Exact upstream environment unavailable |
| Metric | Paper evaluation setup | Objective-value proxy explicitly labeled | PAPER_SPECIFIED / LOCAL_ENGINEERING | Semantic equivalence is not inferred |
| Runner | No official code | Lazy Transformers backend plus mock backend | LOCAL_ENGINEERING | No exact execution claim |

## Unresolved ambiguities

The proceedings do not publish the final checkpoint identifier, model
revision, literal inference prompt, tokenizer/chat template, exact output
delimiters, generation length, solver version, timeout, or complete checklist
weights/prompts. The paper’s objective-value pass@1 metric is not
automatically transferable to NLP4LP without a matching gold solver protocol.
