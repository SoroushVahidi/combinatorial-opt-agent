"""Presentation-only preservation checks; no experiments or evidence computations.
Usage: python3 verify_terminology.py BUILD_DIRECTORY INDEPENDENT_PACKAGE_OUTPUT
"""
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,re,subprocess,sys,zipfile
import pymupdf
R=Path(__file__).resolve().parents[2];S=R/'manuscript/jdsa-stage3';D=Path(__file__).resolve().parent
BASE='f2b8a94ea01922618e36b667c1d9b5e2c08f893c'
MAIN='Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
SUPP='ESM_1_Retrieval_Assisted_Scalar_Grounding_Supplement.pdf'
ZIP='Retrieval_Assisted_Scalar_Grounding_JDSA_Submission_Source.zip'
checks=[]
def ck(n,v):
 checks.append({'check':n,'pass':bool(v)})
 if not v:raise AssertionError(n)
def old(n):return subprocess.check_output(['git','show',BASE+':manuscript/jdsa-stage3/'+n],cwd=R)
def nums(t):return Counter(re.findall(r'\d+(?:\.\d+)?',t))
def spans(t):return Counter(re.findall(r'(?<!\\)\$(.*?)(?<!\\)\$',t))
def main():
 build,independent=map(Path,sys.argv[1:3]);edits=json.loads((D/'CHANGES.json').read_text());changed={e['file'] for e in edits}
 additions={'additional_evidence.tex':{'10':3},'closed_audits.tex':{'3':1},'common18_overview.tex':{'18':1},'main.tex':{'3':4,'1':1,'25':1,'18':1},'reproduction_map.tex':{'18':1,'3':1},'supplement.tex':{'3':1,'1':1,'4':2,'0.8':1,'20':1,'25':1}}
 for n in changed:
  a=old(n).decode();b=(S/n).read_text();expected=a
  for e in [e for e in edits if e['file']==n]:
   ck('unique logged source segment '+n+':'+e['reason'],expected.count(e['before'])==1);expected=expected.replace(e['before'],e['after'])
  ck('all source edits logged '+n,expected==b)
  ck('no existing numerical tokens removed '+n,not(nums(a)-nums(b)))
  ck('only reviewed definition/identifier numeral additions '+n,dict(nums(b)-nums(a))==additions.get(n,{}))
  ck('every existing inline mathematical expression retained '+n,not(spans(a)-spans(b)))
  for env in ['equation','algorithmic']:
   pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
   ck('protected '+env+' exact '+n,re.findall(pat,a,re.S)==re.findall(pat,b,re.S))
  tab=r'\\begin\{(tabularx?)\}(.*?)\\end\{\1\}'
  ta=re.findall(tab,a,re.S);tb=re.findall(tab,b,re.S)
  ck('all table-body numerical sequences exact '+n,[re.findall(r'\d+(?:\.\d+)?',t[1]) for t in ta]==[re.findall(r'\d+(?:\.\d+)?',t[1]) for t in tb])
  for pat,title in [(r'\\(?:cite|ref|eqref|label)\{[^}]+\}','citations/references/labels'),(r'\\(?:url|href|path|nolinkurl)\{[^}]+\}','URLs and pins')]:ck(title+' exact '+n,re.findall(pat,a)==re.findall(pat,b))
 for p in S.iterdir():
  if p.suffix in {'.tex','.bib','.bbl','.cls','.sty','.bst'} and p.name not in changed|{'main_crossrefs.tex','supplement_crossrefs.tex'}:ck('protected source exact '+p.name,p.read_bytes()==old(p.name))
 a=old('main.tex').decode();b=(S/'main.tex').read_text()
 ck('research questions exact',re.findall(r'\\item \\textbf\{RQ.*',a)==re.findall(r'\\item \\textbf\{RQ.*',b))
 ck('declarations/funding/AI/availability exact',a.split(r'\section*{Statements and Declarations}')[1]==b.split(r'\section*{Statements and Declarations}')[1])
 ck('acknowledgment exact',a.split(r'\bmhead{Acknowledgements}')[1].split(r'\section{Introduction}')[0]==b.split(r'\bmhead{Acknowledgements}')[1].split(r'\section{Introduction}')[0])
 abstract=re.search(r'\\abstract\{(.*?)\}\n',b)[1];words=len(abstract.split());keywords=len(re.search(r'\\keywords\{(.*?)\}',b)[1].split(','))
 ck('abstract range',150<=words<=250);ck('six keywords',keywords==6)
 ck('two-column format exact',b.splitlines()[2]==a.splitlines()[2])
 spec=importlib.util.spec_from_file_location('p3',R/'docs/jdsa_presentation_polish_phase3_20261029/verify_phase3.py');p3=importlib.util.module_from_spec(spec);spec.loader.exec_module(p3)
 tables=[];pdfs={n:pymupdf.open(S/n) for n in [MAIN,SUPP]};names={n:d.resolve_names() for n,d in pdfs.items()}
 fa,fb=p3.floats(a),p3.floats(b);aux=(build/'main.aux').read_text()
 for i,label in enumerate([k for k in fb if k.startswith('tab:')],1):
  ra,rb=p3.rows(fa[label]),p3.rows(fb[label]);ck('all main table cells/highlights exact '+str(i),ra==rb)
  page=int(re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{[^}]+\}\{(\d+)\}',aux)[1]);pdfrows=p3.pdf_numeric_rows(pdfs[MAIN][page-1],0,290 if i in {1,3} else 1000)
  for row in rb:
   expected=[(v,bold) for value,bold in zip(row['values'],row['bold']) for v in re.findall(r'\d+(?:\.\d+)?(?:/\d+)?',value)]
   ck('rendered numerical value/font '+str(i)+':'+row['row'],any(v==expected for _,v in pdfrows))
  tables.append({'number':i,'page':page,'rows':rb})
 ck('140 main numerical cells',sum(len(r['values']) for t in tables for r in t['rows'])==140)
 ck('15 bold highlights',sum(sum(r['bold']) for t in tables for r in t['rows'])==15)
 ck('algorithm complete exact',fa['alg:instantiation']==fb['alg:instantiation'])
 ck('figure source and rendering exact',(S/'Fig1.pdf').read_bytes()==old('Fig1.pdf') and (S/'pipeline_figure.tex').read_bytes()==old('pipeline_figure.tex'))
 result={'baseline':BASE,'abstract_words_whitespace':words,'abstract_words_dash_separated':len(re.sub('---',' ',abstract).split()),'keywords':keywords,'tables':tables,'pdfs':{}}
 for n,d in pdfs.items():
  stem='main' if n==MAIN else 'supplement';log=(build/(stem+'.log')).read_text();blg=(build/(stem+'.blg')).read_text()
  ck('clean diagnostics '+stem,not re.search(r'undefined|Overfull|Missing character:|multiply-defined labels|Rerun to get cross-references|between bookmark levels|^!',log,re.M))
  ck('clean bibliography '+stem,'Warning--' not in blg and 'error message' not in blg)
  ck('independent clean package PDF byte identity '+stem,(S/n).read_bytes()==(independent/n).read_bytes())
  aux=(build/(stem+'.aux')).read_text();cited={k for g in re.findall(r'\\citation\{([^}]+)\}',aux) for k in g.split(',')};keys=re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}',(S/(stem+'.bbl')).read_text(),re.S)
  ck('all references cited/linked '+stem,set(keys)==cited and all('cite.'+k in names[n] for k in keys))
  baseline=pymupdf.open(stream=old(n),filetype='pdf');uris=set();counts=Counter();fonts=set();changedpages=[]
  for p in d:
   ck('text within page '+stem+':'+str(p.number+1),all(w[0]>=0 and w[1]>=0 and w[2]<=p.rect.width and w[3]<=p.rect.height for w in p.get_text('words')))
   for l in p.get_links():
    counts[str(l['kind'])]+=1
    if l.get('file'):
     f,_,dest=l['file'].partition('#nameddest=');ck('companion '+stem+':'+dest,f in names and dest in names[f])
    elif l.get('nameddest'):ck('internal '+stem+':'+l['nameddest'],l['nameddest'] in names[n])
    elif l.get('uri'):uris.add(l['uri'])
   fonts.update(f[0] for f in p.get_fonts())
   if p.number>=len(baseline) or p.get_pixmap().samples!=baseline[p.number].get_pixmap().samples:changedpages.append(p.number+1)
  ck('fonts embedded '+stem,all(bool(d.extract_font(x)[3]) for x in fonts))
  ck('external destination sets exact '+stem,uris=={l['uri'] for p in baseline for l in p.get_links() if l.get('uri')})
  ck('bookmark level correction retained '+stem,not any(b[0]>a[0]+1 for a,b in zip(d.get_toc(),d.get_toc()[1:])))
  result['pdfs'][n]={'pages':len(d),'changed_pages':changedpages,'sha256':hashlib.sha256((S/n).read_bytes()).hexdigest(),'link_counts':dict(counts),'references':len(keys),'warnings':re.findall(r'^.*Warning.*$',log,re.M),'underfull_notices':log.count('Underfull')}
 for n in ['Fig1.pdf','main.bbl','supplement.bbl','main_crossrefs.tex','supplement_crossrefs.tex']:ck('package rebuild byte identity '+n,(S/n).read_bytes()==(independent/n).read_bytes())
 with zipfile.ZipFile(S/ZIP) as z:
  ck('ZIP CRC',z.testzip() is None);ck('flat ZIP',all('/' not in n for n in z.namelist()));ck('all ZIP members current',all(z.read(n)==(S/n).read_bytes() for n in z.namelist()))
  ck('no temp/private/data files',not any(Path(n).suffix in {'.aux','.log','.out','.blg','.json','.csv','.pyc'} or n=='main.pdf' for n in z.namelist()));result['zip_members']=z.namelist()
 for line in (S/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1);ck('checksum '+n,hashlib.sha256((S/n).read_bytes()).hexdigest()==h)
 reg=json.loads((D/'TERMINOLOGY_REGISTER.json').read_text());ck('register first uses and occurrence lists',all('occurrences_before' in x and 'occurrences_after' in x and 'final_action' in x for x in reg))
 result.update(status='PASS',checks_passed=len(checks),checks=checks,register_terms=len(reg),local_edits=len(edits))
 (D/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(checks),'checks;',words,'abstract words;',keywords,'keywords')
if __name__=='__main__':main()
