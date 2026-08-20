# DeepOR baseline (paper reconstruction)

This package is a lightweight interface for DeepOR’s published
optimization-modeling method. The AAAI proceedings paper is available, but
the 2026-08-12 primary-source search found no attributable official code,
fine-tuned checkpoint, literal prompt, or supplementary appendix.
Consequently this is **DEEPOR_PAPER_RECONSTRUCTION_READY**, not an empirical
DeepOR result.

The paper specifies Qwen3-8B and greedy evaluation (`temperature=0`,
`top_p=1`, repetition penalty `1.0`). `DeepORConfig` records those settings
while leaving `model_id=None`; the runner fails explicitly with
`checkpoint_unavailable` until an official checkpoint is identified.

The package provides deterministic NLP4LP adaptation, a versioned
paper-reconstructed prompt, reasoning/final-answer parsing, conservative
Pyomo static validation, an opt-in isolated subprocess harness, JSON-friendly
results, offline metrics, a fixed cross-baseline manifest, and mock tests.
Generated code is never executed by ordinary tests. Objective agreement is
reported only as a proxy and never as semantic correctness.
