# GAMSPy: The One Manual Step

Everything else is already set up. You only need to **activate your GAMSPy academic/network license** once.

## What to do

1. Get your **36-character access code** from the GAMS academic portal (where you got the GAMSPy academic/network license).  
   - If the portal gave you a **license file** instead, save it somewhere (e.g. `~/gams_license.txt`).

2. In a terminal, run **one** of these (use your real code or path):

   **If you have the access code and this machine has internet:**  
   ```bash
   python -m gamspy install license YOUR_36_CHARACTER_ACCESS_CODE
   ```

   **If you have a license file:**  
   ```bash
   python -m gamspy install license /path/to/your/license_file.txt
   ```

3. Verify:  
   ```bash
   python -m gamspy show license
   ```  
   You should see your license type (e.g. GAMSPy academic), not "Demo license".

**Where the license is stored:**  
`~/.local/share/GAMSPy/gamspy_license.txt`  
(Do not paste your access code or license text into the repo or into any script.)

After this, you can run any GAMSPy model (e.g. under `data_private/gams_models/raw/gamspy-examples/models/`) without further license steps.
