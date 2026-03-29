"""Launch Gradio with share=True and write the public URL to a file."""
import re
import subprocess
import sys
from pathlib import Path

Path(__file__).resolve().parent
out_file = Path(__file__).resolve().parent / "data" / "processed" / "public_url.txt"

proc = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd=Path(__file__).resolve().parent,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
url = None
try:
    for line in proc.stdout:
        print(line, end="")
        m = re.search(r"https://[a-zA-Z0-9-]+\.gradio\.live", line)
        if m:
            url = m.group(0)
            break
except (KeyboardInterrupt, Exception):
    pass
finally:
    proc.terminate()
    proc.wait(timeout=5)

if url:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(url)
    print("\nURL written to:", out_file)
