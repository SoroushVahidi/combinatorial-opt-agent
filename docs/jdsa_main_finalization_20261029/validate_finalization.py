"""Check integration identity, clean builds, document links and preserved artifacts.
Reads saved scientific sources only; does not run inference, solvers or evaluators.
"""
from pathlib import Path
import hashlib,json,re,subprocess,sys,zipfile
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
S=ROOT/'manuscript/jdsa-stage3'
SOURCE='d3583fa28fcdb87f7a446f764ca3225ab5f855d3'
BASE='d3071ba31eaf7babdf1e7fe234d8527191ed93de'
MAIN='Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
checks=[]
def check(name,ok):
 checks.append({'check':name,'pass':bool(ok)})
 if not ok: raise AssertionError(name)
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
 build=Path(sys.argv[1]);extracted=Path(sys.argv[2]);frozen=Path(sys.argv[3]);out={'baseline_main':BASE,'scientific_source':SOURCE,'checks':checks,'pdfs':{},'source_comparison':{}}
 edits=json.loads((D/'SOURCE_LINK_CHANGES.json').read_text())
 for f in S.iterdir():
  if f.suffix in {'.tex','.bib','.cls','.bst','.sty'}:
   old=git('show',SOURCE+':manuscript/jdsa-stage3/'+f.name)
   expected=old
   if f.name in edits:
    before=edits[f.name]['before'].encode();after=edits[f.name]['after'].encode();check('unique navigation edit '+f.name,old.count(before)==1);expected=old.replace(before,after)
   check('authoritative scientific source identity '+f.name,expected==f.read_bytes())
   out['source_comparison'][f.name]={'sha256':sha(f.read_bytes()),'source_sha256':sha(old),'navigation_edit':f.name in edits}
 check('scientific/data/code paths unchanged',not git('diff',BASE,'--','results','data','reproducibility','tools','scripts','training','src','retrieval','baselines','batch'))
 old_pdfs=json.loads((D/'PDF_INVENTORY_BEFORE.json').read_text())
 for item in old_pdfs:
  p=ROOT/item['path']
  check('PDF disposition '+item['path'],not p.exists() if item['disposition'].startswith('remove') else p.exists() and sha(p.read_bytes())==item['sha256'])
 f=json.loads(frozen.read_text());check('frozen verifier 9737 PASS',f['status']=='PASS' and f['checks_passed']==9737)
 out['frozen_verifier']={'checks_passed':f['checks_passed'],'status':f['status'],'sha256':sha(frozen.read_bytes()),'executed_from_unchanged_correction_worktree':True}
 with zipfile.ZipFile(S/'JDSA_Stage3_Submission_Source.zip') as z:
  check('ZIP CRC',z.testzip() is None);check('flat ZIP',all('/' not in x for x in z.namelist()))
  for n in z.namelist():check('ZIP member equals final source '+n,z.read(n)==(S/n).read_bytes())
  out['archive_members']=z.namelist()
 pdfs={n:pymupdf.open(S/n) for n in [MAIN,'ESM_1.pdf']};names={n:p.resolve_names() for n,p in pdfs.items()}
 for n,pdf in pdfs.items():
  doc='main' if n==MAIN else 'supplement';check('independent clean build byte identity '+n,(S/n).read_bytes()==(extracted/n).read_bytes())
  log=(build/(doc+'.log')).read_text();blg=(build/(doc+'.blg')).read_text();bad=re.findall(r'^.*(?:undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!).*$',log,re.M)
  check('no unresolved build diagnostics '+doc,not bad);check('BibTeX clean '+doc,'Warning--' not in blg and 'error' not in blg.lower());cross=[];bounds=[]
  for i,page in enumerate(pdf):
   words=page.get_text('words');bounds += [(i+1,w[4]) for w in words if w[0]<-.5 or w[1]<-.5 or w[2]>page.rect.width+.5 or w[3]>page.rect.height+.5]
   check('page numbering '+doc+str(i+1),any(w[4]==str(i+1) and w[1]>page.rect.height*.72 and abs((w[0]+w[2])/2-page.rect.width/2)<15 for w in words))
   for link in page.get_links():
    if link.get('file'):
     target,_,fragment=link['file'].partition('#nameddest=');dest=fragment or link.get('nameddest');check('companion named destination '+doc+str(i+1)+str(dest),target in names and dest in names[target]);cross.append({'page':i+1,'target':target,'destination':dest})
    elif 'nameddest' in link:check('internal named destination '+doc+str(link['nameddest']),link['nameddest'] in names[n])
  check('page text within bounds '+doc,not bounds)
  fonts=subprocess.check_output(['pdffonts',str(S/n)],text=True);check('all fonts embedded '+doc,all(re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',l) for l in fonts.splitlines()[2:]))
  old_name='manuscript.pdf' if n==MAIN else n
  old=pymupdf.open(stream=git('show',SOURCE+':manuscript/jdsa-attribution-correction-v1/'+old_name),filetype='pdf')
  check('same page count as corrected PDF '+doc,len(old)==len(pdf));different_text=[];different_render=[]
  for i in range(len(pdf)):
   if old[i].get_text()!=pdf[i].get_text():different_text.append(i+1)
   if old[i].get_pixmap(matrix=pymupdf.Matrix(1,1)).samples!=pdf[i].get_pixmap(matrix=pymupdf.Matrix(1,1)).samples:different_render.append(i+1)
  if doc=='supplement':check('supplement text and rendering preserved',not different_text and not different_render)
  out['pdfs'][n]={'sha256':sha((S/n).read_bytes()),'pages':len(pdf),'cross_document_links':cross,'text_changed_pages_from_corrected':different_text,'render_changed_pages_from_corrected':different_render,'warnings':re.findall(r'^.*Warning.*$',log,re.M),'underfull_notices':log.count('Underfull'),'metadata':pdf.metadata}
 # Check active entry-point links; historical records are explicitly retired/pinned.
 link_sources=['README.md','manuscript/MANUSCRIPT_README.md','manuscript/jdsa/README.md','manuscript/jdsa-stage3/README.md']
 links=[]
 prefix='https://github.com/SoroushVahidi/combinatorial-opt-agent/'
 for path in link_sources:
  p=ROOT/path
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   local=None
   if target.startswith(prefix):
    for middle in ['raw/refs/heads/main/','blob/main/','tree/main/']:
     if target.startswith(prefix+middle):local=ROOT/target[len(prefix+middle):].split('#')[0]
   elif not re.match(r'[a-z]+:',target):local=p.parent/target.split('#')[0]
   if local is not None:check('navigation target '+path+' '+target,local.exists());links.append({'source':path,'target':target})
 out['navigation_links']=links;out['status']='PASS';(D/'VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(checks),'integration checks')
if __name__=='__main__':main()
