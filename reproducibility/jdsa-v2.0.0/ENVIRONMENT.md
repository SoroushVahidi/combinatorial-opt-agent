# Tested reanalysis environment

Python3.12.3, NumPy2.3.5, Linux x86_64. `requirements-validation.txt` pins the sole nonstandard dependency. A fresh virtual environment and a copied package outside all worktrees are used for verification. The receipt records actual interpreter/package versions and all PASS/FAIL checks.

Primary bootstrap: NumPy default_rng(42),10000 multinomial resamples of327 frozen text groups;95% percentile intervals; exact groupwise sign-swap distribution; Holm family16. Factorial bootstrap: original Python random.Random(42),2000 group-index samples; sorted order statistics50 and1950; separate exact240-test Holm family. These are intentionally different preserved historical procedures.

Known historical model/provider settings are in `provenance/providers.json`. This file does not infer undocumented seeds, prompts, dependency versions or hosted defaults. No historical solver/model environment is falsely represented by this small analysis lockfile.
