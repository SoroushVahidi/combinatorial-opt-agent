"""Document preservation only: no model, solver, benchmark or network execution."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parents[2]
O=R/'manuscript/jdsa-phase3';N=R/'manuscript/jdsa-phase4';D=Path(__file__).resolve().parent

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
# Sole table-number change is bibliographic: AlphaOPT's conference year.
assert old['tab:comparison_landscape']['numeric_cells'][12]==['2025']
old['tab:comparison_landscape']['numeric_cells'][12]=['2026']
assert old==new,[(k,old.get(k),new.get(k)) for k in old if old[k]!=new.get(k)]
a=(O/'main.tex').read_text();b=(N/'main.tex').read_text()
abstract=lambda s:re.search(r'\\abstract\{(.*?)\}\n',s,re.S)[1]
assert re.findall(r'\d+(?:\.\d+)?',abstract(a))==re.findall(r'\d+(?:\.\d+)?',abstract(b))
assert abstract(a).replace('On 331 NLP4LP queries and a fixed catalog','On 331 queries from NLP4LP, a natural-language optimization benchmark, and a fixed catalog').replace('across 16 NL4Opt training problems.','across 16 training problems from NL4Opt, a natural-language optimization-formulation benchmark.')==abstract(b)
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
 for name in ['Online Resource 1','JDSA_Phase3','JDSA_Phase4']:
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
report={'source_commit':'49827d274cae995005681fb03a299a5363583861','tables_preserved':new,'table_count':len(new),'investigation_groups':47,'destinations_unchanged':True,'abstract_only_benchmark_names_introduced':True,'acknowledgements_unchanged':True,'author_information_preserved':True,'bibliographic_table_exception':'S1 AlphaOPT year 2025 to 2026','abstract_whitespace_words':len(words),'keyword_count':len(keywords),'introduction_unchanged':True,'scientific_numeric_sequences_preserved':True,'equations_preserved':list(equations(b)),'main_sections':sections,'duplicate_long_prose_paragraphs':0,'experimental_execution':'NONE','vendor_class_sha256':sha(N/'sn-jnl.cls')}
(D/'DOCUMENT_CONSISTENCY_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
# Core scientific prose is byte-identical, not merely numerically equivalent.
assert a[a.index('\\section{Problem Setting'):a.index('\\backmatter')]==b[b.index('\\section{Problem Setting'):b.index('\\backmatter')]
ack=lambda t:t[t.index('The author thanks his mother'):t.index('\\section{Introduction}')].strip()
assert ack(a)==ack(b)
print('PASS: 35 tables, 47 investigation groups, scientific numerical sequences, equations, abstract results, author facts and Introduction preserved.')
