"""Final presentation/package audit. No inference or scientific solver execution.
Usage: python3 verify_phase4.py BUILD_DIR INDEPENDENT_OUTPUT TEMP_AUDIT_DIR
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, importlib.util, json, re, subprocess, sys, zipfile
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SOURCE=ROOT/'manuscript/jdsa-stage3'
BASE='b0eae08b711185eaf189a50b27bef2b73a04815b'
MAIN='Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
SUPP='ESM_1_Retrieval_Assisted_Scalar_Grounding_Supplement.pdf'
ZIP='Retrieval_Assisted_Scalar_Grounding_JDSA_Submission_Source.zip'
checks=[]
def check(n,v):
 checks.append({'check':n,'pass':bool(v)})
 if not v: raise AssertionError(n)
def old(n):return subprocess.check_output(['git','show',BASE+':manuscript/jdsa-stage3/'+n],cwd=ROOT)
def tokens(s):return Counter(re.findall(r'\d+(?:\.\d+)?',s))
def main():
 build, independent, temp=map(Path,sys.argv[1:4])
 spec=importlib.util.spec_from_file_location('phase3',ROOT/'docs/jdsa_presentation_polish_phase3_20261029/verify_phase3.py');p3=importlib.util.module_from_spec(spec);spec.loader.exec_module(p3)
 edits=json.loads((REPORT/'EDITORIAL_CHANGES.json').read_text()); edited={e['file'] for e in edits}
 for n in edited:
  expected=old(n).decode()
  for e in [e for e in edits if e['file']==n]:
   check('exact logged source segment '+n+':'+e['reason'],expected.count(e['before'])==1)
   expected=expected.replace(e['before'],e['after'])
  check('only logged changes '+n,expected==(SOURCE/n).read_text())
 for p in SOURCE.iterdir():
  if p.suffix in {'.tex','.bib','.cls','.sty','.bst'} and p.name not in edited|{'main_crossrefs.tex','supplement_crossrefs.tex'}:
   check('protected source byte identity '+p.name,p.read_bytes()==old(p.name))
 before=old('main.tex').decode();after=(SOURCE/'main.tex').read_text()
 for env in ['table','equation','algorithm']:
  pattern=r'\\begin\{'+env+r'\*?\}.*?\\end\{'+env+r'\*?\}'
  check('byte-identical complete '+env+' environments',re.findall(pattern,before,re.S)==re.findall(pattern,after,re.S))
 for pattern,n in [(r'\\item \\textbf\{RQ.*','research questions'),(r'\\(?:cite|ref|eqref|label)\{[^}]+\}','citations, references and labels'),(r'\\(?:url|href|path|nolinkurl)\{[^}]+\}','public URLs and evidence pins')]:
  check('exact sequence '+n,re.findall(pattern,before)==re.findall(pattern,after))
 check('all main scientific numeral tokens unchanged',tokens(before.split(r'\begin{document}')[1])==tokens(after.split(r'\begin{document}')[1]))
 check('all funding, disclosures, declarations and availability exact',before.split(r'\section*{Statements and Declarations}')[1]==after.split(r'\section*{Statements and Declarations}')[1])
 check('acknowledgment substance exact',before.split(r'\bmhead{Acknowledgements}')[1].split(r'\section{Introduction}')[0]==after.split(r'\bmhead{Acknowledgements}')[1].split(r'\section{Introduction}')[0])
 abstract=len(re.search(r'\\abstract\{(.*?)\}\n',after)[1].split());keywords=len(re.search(r'\\keywords\{(.*?)\}',after)[1].split(','))
 check('abstract 150–250 words',150<=abstract<=250);check('keywords 4–6',4<=keywords<=6)
 check('Springer two-column numbered basic style retained',r'\documentclass[pdflatex,iicol,sn-basic,Numbered]{sn-jnl}' in after)
 check('headings at most three levels',not re.search(r'\\(?:paragraph|subparagraph)\{',after))
 check('figure caption has no terminal punctuation','uses a different inventory}' in after)
 pdfs={n:pymupdf.open(SOURCE/n) for n in [MAIN,SUPP]};names={n:d.resolve_names() for n,d in pdfs.items()}
 table_report=json.loads((ROOT/'docs/jdsa_presentation_polish_phase3_20261029/VALIDATION.json').read_text())['tables'];floats=p3.floats(after)
 for t in table_report:
  rows=p3.rows(floats[t['label']]);check('Phase 3 table values and highlighting '+str(t['number']),rows==t['rows'])
  x0,x1=(300,1000) if t['number']==1 else (0,290) if t['number']==3 else (0,1000)
  rendered=p3.pdf_numeric_rows(pdfs[MAIN][t['page']-1],x0,x1)
  for row in rows:
   expected=[(v,b) for value,b in zip(row['values'],row['bold']) for v in re.findall(r'\d+(?:\.\d+)?(?:/\d+)?',value)]
   check('PDF table values/fonts '+str(t['number'])+':'+row['row'],any(v==expected for y,v in rendered))
 check('140 table cells',sum(len(r['values']) for t in table_report for r in t['rows'])==140)
 check('15 highlighted table cells',sum(sum(r['bold']) for t in table_report for r in t['rows'])==15)
 result={'baseline':BASE,'abstract_words':abstract,'keywords':keywords,'tables':table_report,'pdfs':{}}
 for n,d in pdfs.items():
  stem='main' if n==MAIN else 'supplement';baseline_name=MAIN if n==MAIN else 'ESM_1.pdf';baseline_pdf=pymupdf.open(stream=old(baseline_name),filetype='pdf')
  check('page count '+stem,len(d)==len(baseline_pdf)==(16 if stem=='main' else 38))
  check('independent clean ZIP build byte identity '+stem,(SOURCE/n).read_bytes()==(independent/n).read_bytes())
  check('standalone journal-style build byte identity '+stem,(SOURCE/n).read_bytes()==(temp/('direct-'+stem)/(stem+'.pdf')).read_bytes())
  for logfile in [build/(stem+'.log'),temp/('direct-'+stem)/(stem+'.log')]:
   log=logfile.read_text();check('clean final diagnostics '+str(logfile),not re.search(r'undefined|Overfull|Missing character:|multiply-defined labels|Rerun to get cross-references|between bookmark levels|^!',log,re.M))
  log=(build/(stem+'.log')).read_text();blg=(build/(stem+'.blg')).read_text();check('BibTeX clean '+stem,'Warning--' not in blg and 'error message' not in blg)
  aux=(build/(stem+'.aux')).read_text();cited={k for g in re.findall(r'\\citation\{([^}]+)\}',aux) for k in g.split(',')};keys=re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}',(SOURCE/(stem+'.bbl')).read_text(),re.S)
  check('all references used and linked '+stem,set(keys)==cited and all('cite.'+k in names[n] for k in keys))
  counts=Counter();visible=defaultdict(list)
  for page in d:
   check('page text in bounds '+stem+':'+str(page.number+1),all(w[0]>=0 and w[1]>=0 and w[2]<=page.rect.width and w[3]<=page.rect.height for w in page.get_text('words')))
   for l in page.get_links():
    counts[str(l['kind'])]+=1
    if l.get('file'):
     f,_,dest=l['file'].partition('#nameddest=');check('companion link '+stem+':'+dest,f in names and dest in names[f])
    elif l.get('nameddest'):check('internal link '+stem+':'+l['nameddest'],l['nameddest'] in names[n])
    elif l.get('uri'):visible[l['uri']].append(page.get_textbox(l['from']))
   for f in page.get_fonts():check('embedded font '+stem+':'+str(f[0]),bool(d.extract_font(f[0])[3]))
  check('external URL destination set unchanged '+stem,set(visible)=={l['uri'] for p in baseline_pdf for l in p.get_links() if l.get('uri')})
  if stem=='main':
   for u,parts in visible.items():
    if u.startswith('https://') and 'orcid.org' not in u:
     displayed=re.sub(r'\s+','',''.join(parts)).rstrip('.,;');expected='f94012872211ab6a828147bb138fbe8ab695b982' if '/tree/f940' in u else u
     check('visible URL/hash '+u,displayed==expected)
   check('Supplementary information top-level bookmark',[1,'Supplementary information',13] in d.get_toc())
  else:
   a='\n'.join(p.get_text() for p in baseline_pdf);b='\n'.join(p.get_text() for p in d)
   check('all supplementary PDF numerical tokens unchanged',tokens(a)==tokens(b))
  result['pdfs'][n]={'pages':len(d),'sha256':hashlib.sha256((SOURCE/n).read_bytes()).hexdigest(),'references':len(keys),'link_counts':dict(counts),'warnings':re.findall(r'^.*Warning.*$',log,re.M),'underfull_notices':log.count('Underfull')}
 for n in ['Fig1.pdf','main.bbl','supplement.bbl','main_crossrefs.tex','supplement_crossrefs.tex']:
  check('clean ZIP regeneration byte identity '+n,(SOURCE/n).read_bytes()==(independent/n).read_bytes())
 check('main bibliography byte-identical', (SOURCE/'main.bbl').read_bytes()==old('main.bbl'))
 check('supplement bibliography changes only two verified capitals',(SOURCE/'supplement.bbl').read_text().replace('{Sisyphus}','sisyphus').replace('{Optimal}','optimal')==old('supplement.bbl').decode())
 with zipfile.ZipFile(SOURCE/ZIP) as z:
  check('ZIP CRC',z.testzip() is None);check('flat source ZIP',all('/' not in n for n in z.namelist()))
  check('ZIP matches all packaged files',all(z.read(n)==(SOURCE/n).read_bytes() for n in z.namelist()))
  check('no temporary/private/unrelated files in ZIP',not any(Path(n).suffix in {'.aux','.log','.out','.blg','.pyc','.csv','.json'} or n=='main.pdf' for n in z.namelist()))
  result['zip_members']=z.namelist()
 for line in (SOURCE/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1);check('checksum '+n,hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()==h)
 evidence=json.loads((temp/'evidence-verification.json').read_text());check('authenticated public saved-record reanalysis',evidence['status']=='PASS');result['evidence_checks']=evidence.get('checks_passed',len(evidence.get('checks',[])))
 result.update(status='PASS',checks_passed=len(checks),checks=checks)
 (REPORT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(checks),'checks; abstract',abstract,'words;',keywords,'keywords')
if __name__=='__main__':main()
