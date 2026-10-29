#!/usr/bin/env python3
"""Validate immutable evidence, numerical preservation and revised source coverage."""
import argparse,collections,hashlib,json,re,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;O=B/'manuscript/jdsa/source';N=B/'manuscript/jdsa-stage2';R=B/'reproducibility/jdsa-v2.0.0'
CHECKS=[]
def check(name,condition):
 CHECKS.append({'check':name,'status':'PASS' if condition else 'FAIL'})
 if not condition:raise AssertionError(name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expand(root,name,stack=()):
 check('acyclic input '+name,name not in stack)
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(root,m[1],stack+(name,)),(root/(name+'.tex')).read_text())
def floats(text):
 out={}
 for m in re.finditer(r'\\begin\{(table\*?|figure\*?|algorithm\*?)\}[\s\S]*?\\end\{\1\}',text):
  ids=re.findall(r'\\label\{([^}]+)\}',m[0]);check('one label per float',len(ids)==1);out[ids[0]]=m[0]
 return out
NUM=r'(?<![A-Za-z\d])[-+]?\d+(?:[.,]\d+)*(?![A-Za-z\d])'
def numbody(text):
 m=re.search(r'\\begin\{tabularx?\}([\s\S]*?)\\end\{tabularx?\}',text)
 return re.findall(NUM,m[1] if m else text)
def clean(text):return re.sub(r'\s+',' ',text).strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build-dir',type=Path);args=ap.parse_args()
 report={'status':'FAIL','scope':'Numerical preservation plus authenticated saved-record/derived diagnostic verification; no new inference or semantic ground truth','baseline':'3296f3bd20128077cd94a4eac82f0b5fe73e2946'}
 try:
  c=json.loads((R/'claims.json').read_text());check('all220 original claim objects parsed',len(c['named_objects'])+len(c['numerical_passages'])==220)
  for s in c['source_of_truth']:check('unchanged canonical source '+s['path'],sha(B/s['path'])==s['sha256'])
  tracked=subprocess.check_output(['git','diff','--name-only',report['baseline'],'--','reproducibility/jdsa-v2.0.0','manuscript/jdsa'],cwd=B,text=True);check('immutable sources/evidence unchanged',tracked=='')
  mf=json.loads((R/'release_manifest.json').read_text())
  for x in mf['files']:check('evidence SHA '+x['path'],sha(R/x['path'])==x['sha256'])
  frozen=D/'FROZEN_REANALYSIS_VERIFICATION.json'
  rr=subprocess.run([sys.executable,'-I',str(R/'verify.py'),'--output',str(frozen)],capture_output=True,text=True)
  check('frozen release verifier PASS',rr.returncode==0);rv=json.loads(frozen.read_text());report['frozen_verification']={'status':rv['status'],'checks_passed':rv['checks_passed'],'sha256':sha(frozen)}
  derive=subprocess.run([sys.executable,'-B',str(D/'derive_ready_conditionals.py')],capture_output=True,text=True);check('two independent conditional derivations and edge tests',derive.returncode==0)
  nv=json.loads((D/'DERIVED_DIAGNOSTICS.json').read_text());report['new_diagnostics']=nv['verification']
  originals={};revised={};expanded={}
  for doc in ['main','supplement']:
   originals.update(floats(expand(O,doc)));expanded[doc]=expand(N,doc);revised.update(floats(expanded[doc]))
  check('all37 original numbered objects retained',set(originals)<=set(revised));check('exactly one new table',set(revised)-set(originals)=={'tab:ready-conditioned'})
  checks=[]
  for label,text in originals.items():
   check('preserved numerical cells '+label,numbody(text)==numbody(revised[label]));checks.append({'id':label,'numeric_cell_tokens':len(numbody(text)),'status':'PASS','baseline_float_sha256':hashlib.sha256(text.encode()).hexdigest(),'revised_float_sha256':hashlib.sha256(revised[label].encode()).hexdigest()})
  report['numbered_objects']=checks
  owners={}
  for doc,s in expanded.items():
   labels=re.findall(r'\\label\{([^}]+)\}',s);check('unique '+doc+' labels',len(labels)==len(set(labels)))
   for l in labels:check('globally distinct label '+l,l not in owners);owners[l]=doc
  for doc,s in expanded.items():
   for ref in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s):
    base=re.sub(r'^[MS]-','',ref);check('defined reference '+doc+':'+ref,base in owners)
    prefix=('M-' if owners[base]=='main' else 'S-') if owners[base]!=doc else ''
    check('correct cross-document prefix '+ref,ref==prefix+base)
  abstract=re.search(r'\\abstract\{([\s\S]*?)\}\n\n\\keywords',expanded['main'])[1];plain=re.sub(r'\\[A-Za-z]+|[{}$]',' ',abstract);words=re.findall(r"[A-Za-z0-9]+(?:[-’'][A-Za-z0-9]+)*",plain)
  check('abstract150to250words',150<=len(words)<=250);report['abstract_words']=len(words)
  check('main two-column Springer class',r'\documentclass[pdflatex,iicol,sn-basic,Numbered]' in expanded['main'])
  for rq in ['RQ1','RQ2','RQ3']:check('RQ in introduction/results '+rq,expanded['main'].count(rq)>=4)
  for literal in ['7/265=2.64','258/265=97.36','7/255=2.75']:check('new reported fraction '+literal,literal in expanded['main'])
  snippets=(N/'ready_conditioned_rows.tex').read_text().strip();check('derived table matches source',snippets in expanded['main'])
  for pattern in [r'40/331',r'41',r'291',r'303',r'320',r'285',r'240',r'16']:check('protected count '+pattern,pattern in expanded['main'])
  for text in ['23{,}241/28{,}799=0.8070','0.8135','withhold its aggregate objective-agreement']:check('protected auxiliary interpretation '+text,text in expanded['supplement'])
  # Parse only source passages for a coverage ledger. Numerals include formatting and IDs;
  # loss of a numeral is reviewed, never silently treated as a scientific correction.
  old_all='\n'.join(expand(O,d) for d in ['main','supplement']);new_all='\n'.join(expanded.values())
  report['numeric_literal_presence_delta']={'removed':sorted(set(re.findall(NUM,old_all))-set(re.findall(NUM,new_all))),'added':sorted(set(re.findall(NUM,new_all))-set(re.findall(NUM,old_all))),'interpretation':'Coverage aid, not semantic equivalence test; see the manually reviewed claim crosswalk.'}
  oldcite={k for a in re.findall(r'\\cite\w*\{([^}]+)\}',old_all) for k in a.split(',')};newcite={k for a in re.findall(r'\\cite\w*\{([^}]+)\}',new_all) for k in a.split(',')};report['citation_delta']={'removed':sorted(oldcite-newcite),'added':sorted(newcite-oldcite)}
  if args.build_dir:
   for doc in ['main','supplement']:
    log=(args.build_dir/(doc+'.log')).read_text(errors='replace');bad=re.findall(r'undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!',log,re.M);check('clean final LaTeX log '+doc,not bad)
  report['status']='PASS'
 except Exception as e:report['error_type']=type(e).__name__;report['error']=str(e)
 report['checks']=CHECKS;report['checks_passed']=sum(c['status']=='PASS' for c in CHECKS);(D/'NUMERICAL_CLAIM_CONSISTENCY.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:report[k] for k in report if k not in ['checks','numbered_objects']},indent=2));return report['status']!='PASS'
if __name__=='__main__':raise SystemExit(main())
