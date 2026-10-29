#!/usr/bin/env python3
"""Read-only narrative word-count proxy; excludes floats, math and declarations."""
from pathlib import Path
import re,json
B=Path(__file__).resolve().parents[2]
N=B/'manuscript/jdsa-stage2'; O=B/'manuscript/jdsa/source'
def expand(root,name,seen=None):
 seen=set() if seen is None else seen
 if name in seen:raise ValueError('cycle '+name)
 seen.add(name);p=root/(name+'.tex');s=p.read_text();s=re.sub(r'\\input\{([^}]+)\}',lambda m:expand(root,m[1],seen.copy()),s);return s
def prose(s):
 # Narrative-only proxy: remove frontmatter/authorship, floats/captions, equations,
 # reference/declaration backmatter, comments and TeX markup; keep prose arguments.
 s=re.sub(r'(?m)(?<!\\)%.*$','',s)
 s=re.sub(r'\\begin\{(table\*?|figure\*?|algorithm\*?|equation\*?|align\*?|tikzpicture|tabularx?)\}[\s\S]*?\\end\{\1\}',' ',s)
 abstract=re.search(r'\\abstract\{([\s\S]*?)\}\n\n\\keywords',s)
 a=abstract[1] if abstract else ''
 start=s.find(r'\section{');s=s[start:] if start>=0 else s
 s=s.split(r'\backmatter')[0].split(r'\bibliography')[0]
 s=a+'\n'+s
 s=re.sub(r'\$[^$]*\$',' ',s)
 s=re.sub(r'\\(?:label|ref|eqref|cite|path|url)\{[^}]*\}',' ',s)
 s=re.sub(r'\\href\{[^}]*\}\{([^}]*)\}',r'\1',s)
 s=re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?',' ',s)
 return re.findall(r"[A-Za-z]+(?:[-’'][A-Za-z]+)*",s)
result={}
for label,root in [('before',O),('after',N)]:
 result[label]={d:len(prose(expand(root,d))) for d in ['main','supplement']};result[label]['total']=sum(result[label].values())
result['reduction_percent']={k:round(100*(1-result['after'][k]/result['before'][k]),2) for k in result['before']}
print(json.dumps(result,indent=2))
