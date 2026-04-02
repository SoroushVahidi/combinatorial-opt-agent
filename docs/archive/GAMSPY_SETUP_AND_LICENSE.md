# GAMSPy Setup and License

## What was installed

- **GAMSPy** (PyPI): `gamspy` 1.21.0, with dependencies `gamsapi` 53.2.0, `gamspy_base` 53.2.0, and `pandas`/`numpy` compatible with the rest of the environment.
- **Install command used:**  
  `pip install gamspy`  
  (user install: `--user` because system site-packages was not writable.)  
  After that, numpy/pandas were pinned to fix import errors:  
  `pip install "numpy<2" "pandas>=2.2.2,<2.6" --user`
- **Where it is installed:**  
  User site-packages: `~/.local/lib/python3.11/site-packages/`  
  - `gamspy` package: `~/.local/lib/python3.11/site-packages/gamspy/`  
  - `gamspy_base` (engine, demo license): `~/.local/lib/python3.11/site-packages/gamspy_base/`

## How license activation works

1. **Preferred (online):** You provide either:
   - A **36-character access code** from the GAMS academic portal, or  
   - A **path to an ASCII license file** (e.g. a file you downloaded or received).
2. **Command:**  
   `python -m gamspy install license <access_code_or_path>`  
   - If you pass a path to a file, that file is copied into GAMSPy’s license directory.  
   - If you pass a 36-character code, GAMSPy contacts the license server and writes the license file.
3. **Where the license is stored:**  
   - **After you install your license:**  
     `~/.local/share/GAMSPy/gamspy_license.txt`  
   - **Before that (demo):**  
     The demo license is inside the `gamspy_base` package:  
     `.../site-packages/gamspy_base/gamslice.txt`
4. **Network / checkout (optional):**  
   For a time-limited checkout from a network license:  
   `python -m gamspy install license <access_code> --checkout-duration <hours>`
5. **Verify:**  
   `python -m gamspy show license`

## The one manual step you must do

**Activate your GAMSPy academic/network license** (so it replaces the demo license):

1. Get your **36-character access code** from the GAMS academic portal (the same place you obtained the GAMSPy academic/network license).  
   - Or, if the portal gave you a **license file** (e.g. download or email), save it to a path you know (e.g. `~/gams_license.txt`).

2. Run **one** of the following (from any directory, in the same Python environment where GAMSPy is installed):

   **Option A – Access code (machine has internet):**  
   ```bash
   python -m gamspy install license YOUR_36_CHARACTER_ACCESS_CODE
   ```
   Replace `YOUR_36_CHARACTER_ACCESS_CODE` with your actual code. Do not add extra spaces or quotes unless the code itself contains them.

   **Option B – License file:**  
   ```bash
   python -m gamspy install license /path/to/your/license_file.txt
   ```
   Use the real path to the file (e.g. `$HOME/gams_license.txt`).

3. **Check that it worked:**  
   ```bash
   python -m gamspy show license
   ```  
   You should see your license type (e.g. GAMSPy academic) instead of “Demo license”.

**Where the license is stored after activation:**  
`~/.local/share/GAMSPy/gamspy_license.txt`  
You do **not** need to create this file by hand; `gamspy install license` creates/overwrites it. Do **not** put your access code or license contents in any script or doc in the repo; keep them only in the portal or in that file on your machine.

## Commands to run after you have activated the license

- Verify license:  
  `python -m gamspy show license`
- Run a minimal GAMSPy model (from repo root):  
  `python -c "from gamspy import Container, Set, Variable, Equation, Model, Sum, Sense; m=Container(); i=Set(m,'i',records=['a','b']); j=Set(m,'j',records=['x','y']); x=Variable(m,'x',domain=[i,j]); eq=Equation(m,'eq',domain=j); eq[j]=Sum(i,x[i,j])==1; mod=Model(m,'mod',equations=[eq],problem='lp',sense=Sense.MIN,objective=Sum([i,j],x[i,j])); mod.solve(); print('OK')"`
- Run an example from the cloned repo:  
  `python data_private/gams_models/raw/gamspy-examples/models/trnsport/trnsport.py`  
  (or any other model in `data_private/gams_models/raw/gamspy-examples/models/`.)

## Summary

| Item | Value |
|------|--------|
| GAMSPy installed | Yes |
| Location | `~/.local/lib/python3.11/site-packages/gamspy` (and gamspy_base) |
| License activation | Run `python -m gamspy install license <access_code_or_path>` once |
| License file after activation | `~/.local/share/GAMSPy/gamspy_license.txt` |
| Verify | `python -m gamspy show license` |
