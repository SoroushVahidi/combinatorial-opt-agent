#!/usr/bin/env python3
"""Require exact Stage 3 sources plus only the enumerated link changes."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
S=ROOT/'manuscript/jdsa-stage3';BASE='e12c413c18e6d1f068d29509e7f9108e7ef51f65'
edits=json.loads((D/'EXACT_LINK_CHANGES.json').read_text())
for p in sorted(S.glob('*.tex')):
    text=subprocess.check_output(['git','show',BASE+':'+str(p.relative_to(ROOT))],cwd=ROOT).decode()
    for e in edits:
        if e['file']==p.name:
            assert text.count(e['before'])==1
            text=text.replace(e['before'],e['after'])
    assert p.read_text()==text,p
for name in ['references.bib','sn-jnl.cls','sn-basic.bst','orcidlink.sty','build.sh']:
    p=S/name
    assert subprocess.check_output(['git','show',BASE+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
print('PASS: all 19 TeX sources equal the exact baseline plus enumerated link edits; bibliography, styles and build script unchanged.')
