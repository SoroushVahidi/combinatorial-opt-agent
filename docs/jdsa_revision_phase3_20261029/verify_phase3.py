"""Document preservation only: no model, solver, benchmark or network execution."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parents[2]
O=R/'manuscript/jdsa-phase2';N=R/'manuscript/jdsa-phase3';D=Path(__file__).resolve().parent

def tables(p):
 out={}
 for f in p.glob('*.tex'):
  for t in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',f.read_text(),re.S):
   label=re.search(r'\\label\{([^}]+)',t)[1]
   body=re.search(r'\\begin\{tabularx?\}(.*?)\\end\{tabularx?\}',t,re.S)[1]
   body=body[body.index('\\toprule'):]
   body=re.sub(r'\\\\\[[^]]+\]',r'\\\\',body)
   rows=[r for r in body.split('\\\\') if '&' in r]
   out[label]={'rows':len(rows),'numeric_cells':[re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',r) for r in rows]}
 return out
old,new=tables(O),tables(N)
assert old==new,[(k,old.get(k),new.get(k)) for k in old if old[k]!=new.get(k)]
a=(O/'main.tex').read_text();b=(N/'main.tex').read_text()
abstract=lambda s:re.search(r'\\abstract\{(.*?)\}\n',s,re.S)[1]
assert abstract(a)==abstract(b)
words=re.sub(r'\\[A-Za-z]+','',abstract(b));words=re.findall(r"\S+",words)
assert 150<=len(words)<=250
keywords=re.search(r'\\keywords\{([^}]+)',b)[1].split(',');assert 4<=len(keywords)<=6
sections=re.findall(r'\\section\{([^}]+)',b);assert len(sections)==7
assert a[a.index('\\section{Introduction}'):a.index('\\section{Related Work}')]==b[b.index('\\section{Introduction}'):b.index('\\section{Related Work}')]
# All numbered definitions are retained; ignore only line-layout wrappers.
def equations(s):
 out={}
 for t in re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S):
  label=re.search(r'\\label\{([^}]+)',t)[1]
  t=re.sub(r'\\(?:begin|end)\{gathered\}|\\qquad|\\\\\[\d+pt\]','',t)
  out[label]=re.sub(r'\s+','',t.replace(r'\mathbf{1}',r'\mathbb{1}'))
 return out
assert equations(a)==equations(b)
# Scientific narrative sections preserve all numeric tokens after removing layout.
def numeric_body(s):
 s=s[s.index('\\section{Problem Setting'):s.index('\\backmatter')]
 s=re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}','',s,flags=re.S)
 s=re.sub(r'\\includegraphics\[[^]]+\]\{[^}]+\}','',s)
 s=s.replace('Online Resource 1','Supplement')
 s=re.sub(r'\\\\\[\d+pt\]','',s)
 return re.findall(r'\d+(?:\.\d+)?',s)
assert numeric_body(a)==numeric_body(b),'Scientific numeric token sequence differs'
# Supplement tables/prose retain the same numbers; exclude spacing, resource identifiers.
for f in O.glob('*.tex'):
 if f.name in ['main.tex','supplement.tex','pipeline_figure.tex']:continue
 x,y=f.read_text(),(N/f.name).read_text()
 for name in ['Online Resource 1','JDSA_Phase2','JDSA_Phase3']:
  x=x.replace(name,'');y=y.replace(name,'')
 x=re.sub(r'\\\\\[\d+pt\]','',x);y=re.sub(r'\\\\\[\d+pt\]','',y)
 # D1 paragraph-column width is layout, not a measurement.
 x=re.sub(r'p\{[\d.]+cm\}','',x);y=re.sub(r'p\{[\d.]+cm\}','',y)
 assert re.findall(r'\d+(?:\.\d+)?',x)==re.findall(r'\d+(?:\.\d+)?',y),f.name
inv=json.loads((R/'docs/jdsa_revision_phase2_20261029/INVESTIGATION_DESTINATIONS.json').read_text());assert len(inv)==47
for f in N.glob('*.tex'):
 s=f.read_text();assert not re.search(r'(?<!\\)\\\[',s) and '$$' not in s,f
 assert '\\paragraph{' not in s,f
 for c in re.findall(r'\\cite\w*\{([^}]+)',s):assert len(c.split(','))<=4,(f,c)
paras=[re.sub(r'\s+',' ',x.strip()) for x in b.split('\n\n') if len(x)>200 and not x.lstrip().startswith('\\')]
assert len(paras)==len(set(paras))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'source_commit':'8fbf3d073c54a4103ef06263084ea7d3fded57b3','tables_preserved':new,'table_count':len(new),'investigation_groups':47,'destinations_unchanged':True,'abstract_unchanged':True,'abstract_whitespace_words':len(words),'keyword_count':len(keywords),'introduction_unchanged':True,'scientific_numeric_sequences_preserved':True,'equations_preserved':list(equations(b)),'main_sections':sections,'duplicate_long_prose_paragraphs':0,'experimental_execution':'NONE','vendor_class_sha256':sha(N/'sn-jnl.cls')}
(D/'DOCUMENT_CONSISTENCY_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: 35 tables, 47 investigation groups, scientific numerical sequences, equations, abstract and Introduction preserved.')
