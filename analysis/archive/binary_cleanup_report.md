# Binary cleanup report

Request executed: delete all binary files.

## Removed tracked binary paths

- `figures/nlp4lp_instantiation_pipeline_v2.png`
- `results/eswa_revision/12_figures/*.png` (all tracked files in that folder)
- `results/paper/eaai_camera_ready_figures/*.png`
- `results/paper/eaai_camera_ready_figures/*.pdf`
- `static/favicon.ico`
- `static/icons/*.png`

## Verification command and outcome

```bash
python - <<'PY'
import subprocess, pathlib
files=subprocess.check_output(['git','ls-files'],text=True).splitlines()
bs=[]
for f in files:
    p=pathlib.Path(f)
    try: b=p.read_bytes()[:4096]
    except: continue
    if b'\x00' in b:
        bs.append(f); continue
    try: p.read_text(encoding='utf-8')
    except: bs.append(f)
print('binary_count',len(bs))
PY
```

Observed result: `binary_count 0`.
