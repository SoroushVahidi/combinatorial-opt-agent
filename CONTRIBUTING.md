# Contributing

This research repository supports fixed-catalog retrieval and assisted scalar
grounding. Start with the [repository overview](README.md) and
[testing guide](docs/TESTING.md). Use an isolated branch/worktree and preserve
uncommitted, untracked and ignored research material.

## Development and review

Install `requirements-dev.txt` for minimal maintenance tests. Broader research
requirements are optional and not historical environment pins. Do not run models,
solvers, data acquisition or provider calls during routine maintenance. Tests needing
these capabilities must use the explicit markers and opt-ins in the testing guide.

Keep implementation changes, documentation, tests and scientific reanalysis clearly
separated in review. New retrieval baselines belong in `retrieval/`; grounding
methods and diagnostics use existing `tools/` conventions. Add synthetic fixture
coverage before changing adapters or evaluation behavior. Do not silently alter
benchmark results, denominators, schemas or scientific claims.

## Preservation rules

- Permanent manuscript/evidence tags, approved PDFs, source packages, manifests and
  checksums are immutable. New releases require separate review and new versions.
- Preserve historical predictions, statistical inputs, source annotations, negative
  results and environment/solver records even when not cited by the current paper.
- Do not publish raw provider conversations, credentials, excluded M5 archives or
  additional gated/licensing-uncertain source material. See [data restrictions](docs/DATA_ACCESS_AND_LICENSING.md).
- Never run legacy recursive cleanup scripts against an active research worktree.
- Branch/worktree deletion requires positive activity and ancestry checks, including
  owner/cluster evidence. Absence of a local process alone is insufficient.

The [manuscript index](manuscript/MANUSCRIPT_README.md) identifies the final public
pre-submission version and retained earlier journal packages. Historical EAAI/SNCS
"source of truth" documents retain their original scope, not current authority.

Use Python 3.10+ and relative project paths; keep secrets in process environment or
ignored local configuration. Run the minimal offline checks before opening a PR.
