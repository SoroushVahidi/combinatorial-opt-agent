"""Read saved evidence and PDFs; write this correction's audit report only.
No scientific evaluator, prediction, historical expectation or source is modified.
"""
from pathlib import Path
import collections,csv,hashlib,importlib.util,io,json,re,subprocess,sys
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
S=ROOT/'manuscript/jdsa-stage3'
P=ROOT/'manuscript/jdsa-attribution-correction-v1'
BASE='00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc'
ORIGINAL='2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd'
checks=[]
def check(name,ok):
 checks.append({'check':name,'pass':bool(ok)})
 if not ok:raise AssertionError(name)
def blob(path,ref=BASE):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def expand(name,original=False):
 text=blob('manuscript/jdsa-stage3/'+name+'.tex').decode() if original else (S/(name+'.tex')).read_text()
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(m[1],original),text)
def tables(text):
 return {re.search(r'\\label\{([^}]+)\}',m[0])[1]: re.search(r'\\begin\{tabularx?\}([\s\S]*?)\\end\{tabularx?\}',m[0])[1] for m in re.finditer(r'\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}',text)}
def main():
 build=Path(sys.argv[1]);out={'baseline':BASE,'original_topk_commit':ORIGINAL,'checks':checks}
 edits=json.loads((D/'EXACT_SOURCE_CHANGES.json').read_text())
 for f in sorted({x['file'] for x in edits}):
  t=blob(f).decode()
  for e in edits:
   if e['file']==f:
    if e['before']=='':t+=e['after']
    else:check('unique approved source replacement '+f,t.count(e['before'])==1);t=t.replace(e['before'],e['after'])
  check('source equals baseline plus approved edits '+f,t==(ROOT/f).read_text())
 check('README preserves original content',(S/'README.md').read_bytes().endswith(blob('manuscript/jdsa-stage3/README.md')))
 protected=['reproducibility','results','data','tools','training','scripts','manuscript/jdsa','manuscript/jdsa-stage3/JDSA_Stage3_Submission_Source.zip','manuscript/jdsa-stage3/manuscript.pdf','manuscript/jdsa-stage3/ESM_1.pdf','manuscript/jdsa-stage3/main.bbl','manuscript/jdsa-stage3/supplement.bbl']
 check('evidence evaluators datasets archives and historical PDFs unchanged',not subprocess.check_output(['git','diff',BASE,'--',*protected],cwd=ROOT))
 for doc in ['main','supplement']:
  old=expand(doc,True);new=expand(doc);a=tables(old);b=tables(new)
  check('same table labels '+doc,a.keys()==b.keys())
  for k in a:check('unchanged table body '+k,a[k].replace(' & P@5 & ',' & Hit@5 & ')==b[k])
  check('same labels '+doc,re.findall(r'\\label\{([^}]+)\}',old)==re.findall(r'\\label\{([^}]+)\}',new))
 def science_numbers(text):
  text=text.split(r'\begin{document}',1)[1]
  return re.findall(r'(?<![\w.])\d+(?:\.\d+)?',text)
 check('all main scientific numerals preserved',science_numbers(blob('manuscript/jdsa-stage3/main.tex').decode())==science_numbers((S/'main.tex').read_text()))
 rr=list(csv.DictReader(io.StringIO(blob('results/topk_schema_rerank_stage_a/candidate_groundings.csv',ORIGINAL).decode())))
 groups=collections.defaultdict(list)
 for row in rr:groups[row['problem_id']].append(row)
 check('original top-ten frame',len(rr)==3310 and len(groups)==331)
 outcomes={};losses=[]
 for name in ['A','C','B']:
  ready=hit=wrong=0
  for q,rows in groups.items():
   rows.sort(key=lambda x:int(x['rank']));base=rows[0];gold=next((x for x in rows if x['schema_hit']=='1'),None)
   if name=='B':chosen=next((x for x in rows if x['ready']=='1'),base)
   else:chosen=gold if gold is not None and (name=='A' or gold['ready']=='1') else base
   ready+=int(chosen['ready']);hit+=int(chosen['schema_hit']);wrong+=chosen['ready']=='1' and chosen['schema_hit']=='0'
   if name=='A' and gold and base['ready']=='1' and gold['ready']=='0':losses.append({'query':q,'gold_rank':int(gold['rank']),'gold_coverage':float(gold['coverage'])})
  outcomes[name]={'ready':ready,'schema_hits':hit,'wrong_schema_ready':wrong}
 check('oracle A identity',(outcomes['A']['ready'],outcomes['A']['schema_hits'],outcomes['A']['wrong_schema_ready'])==(268,325,0))
 check('oracle C identity',(outcomes['C']['ready'],outcomes['C']['schema_hits'],outcomes['C']['wrong_schema_ready'])==(270,322,2))
 check('oracle B identity',(outcomes['B']['ready'],outcomes['B']['wrong_schema_ready'])==(286,32))
 check('two precise losses',{x['query'] for x in losses}=={'nlp4lp_test_286','nlp4lp_test_299'})
 saved_oracles=list(csv.DictReader(io.StringIO(blob('results/topk_schema_rerank_stage_a/oracle_topk.csv',ORIGINAL).decode())))
 for name,label in [('A','oracle_A_gold_schema'),('C','oracle_C_gold_ready'),('B','oracle_B_readiness')]:
  r=next(r for r in saved_oracles if r['oracle']==label and r['k']=='10')
  check('original oracle CSV '+name,int(r['ready_count'])==outcomes[name]['ready'] and round(float(r['schema_R1'])*331)==outcomes[name]['schema_hits'])
 summary=json.loads(blob('results/topk_schema_rerank_stage_a/summary.json',ORIGINAL))
 for name,label in [('A','oracle_A_gold_schema'),('C','oracle_C_gold_ready'),('B','oracle_B_readiness')]:
  r=summary['oracle_summary'][label+'_k10'];check('original summary oracle '+name,r['ready_count']==outcomes[name]['ready'])
 best=summary['best_reranker'];check('original top-five heuristic unchanged',(best['method'],best['ready_count'],best['schema_regressions'])==('all_k5_R3_ready_cov_type_tfidf',282,20))
 source=blob('tools/topk_schema_rerank_stage_a.py',ORIGINAL).decode()
 check('original A implementation','selected_gold[qid] = gold if gold is not None else base' in source)
 check('original C implementation','selected_gold_ready[qid] = gold if (gold is not None and gold.ready) else base' in source)
 metric=blob('training/metrics.py','e3fdaf49bf507e7eed5f36621e7e4615a3342c5d').decode()
 check('historical Hit@5 definition','return 1.0 if expected_name in top_k else 0.0' in metric and 'p5 = sum(precision_at_k(names, exp, 5) for names, exp in results) / n' in metric)
 out['oracles']=outcomes;out['oracle_A_losses']=losses;out['candidate_csv_sha256']=sha(blob('results/topk_schema_rerank_stage_a/candidate_groundings.csv',ORIGINAL))
 # Reuse existing indicator/set definitions without running their file-writing main().
 f=ROOT/'docs/jdsa_manuscript_v1_20261029/derive_ready_conditionals.py';sp=importlib.util.spec_from_file_location('ready',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 expected=json.loads((ROOT/'reproducibility/jdsa-v2.0.0/expected/SAVED_METRIC_CHECKS.json').read_text());arms={}
 for arm,e in expected['core'].items():
  path=next(p for p,h in expected['inputs'].items() if h==e['sha256']);raw=(ROOT/'reproducibility/jdsa-v2.0.0'/path).read_bytes();check('authenticated metric '+arm,sha(raw)==e['sha256']);rows=list(csv.DictReader(io.StringIO(raw.decode())));ids,hits,ready,strict,full=m.independent_sets(rows)
  check('unchanged independent gate counts '+arm,(len(ids),len(ready),len(strict),len(full))==(331,e['coarse'],e['strict'],e['full_all_key']))
  check('existing indicators agree '+arm,{r['query_id'] for r in rows if m.indicators(r)[4]}==full)
  arms[arm]={'ready':len(ready),'strict':len(strict),'full':len(full),'eligible_full':sum(m.indicators(r)[3] for r in rows)}
 out['protected_arms']=arms
 check('constrained 41 versus 40',(arms['constrained']['eligible_full'],arms['constrained']['full'])==(41,40))
 frozen=json.loads(Path('/tmp/jdsa-attribution-correction-v1-frozen.json').read_text());check('unchanged frozen verifier 9737 PASS',frozen['status']=='PASS' and frozen['checks_passed']==9737);out['frozen_verifier']={'status':frozen['status'],'checks_passed':frozen['checks_passed'],'report_sha256':sha(Path('/tmp/jdsa-attribution-correction-v1-frozen.json').read_bytes())}
 docs={f:pymupdf.open(P/f) for f in ['manuscript.pdf','ESM_1.pdf']};names={f:d.resolve_names() for f,d in docs.items()};out['pdfs']={}
 for f,d in docs.items():
  doc='main' if f=='manuscript.pdf' else 'supplement';log=(build/(doc+'.log')).read_text();blg=(build/(doc+'.blg')).read_text();bad=re.findall(r'^.*(?:undefined|Rerun to get cross-references right|multiply-defined labels|Missing character:|Overfull|^!).*$',log,re.M)
  check('clean final compilation '+doc,not bad);check('clean BibTeX '+doc,'Warning--' not in blg and 'error' not in blg.lower());external=[];bounds=[]
  for i,p in enumerate(d):
   words=p.get_text('words');bounds.extend((i+1,w[4]) for w in words if w[0]<-.5 or w[1]<-.5 or w[2]>p.rect.width+.5 or w[3]>p.rect.height+.5)
   check('page number '+doc+str(i+1),any(w[4]==str(i+1) and w[1]>p.rect.height*.72 and abs((w[0]+w[2])/2-p.rect.width/2)<15 for w in words))
   for l in p.get_links():
    if l.get('file'):
     target,_,fragment=l['file'].partition('#nameddest=');dest=fragment or l.get('nameddest');external.append({'page':i+1,'target':target,'destination':dest,'kind':l['kind']});check('companion destination '+str(l),target in names and dest in names[target])
    elif 'nameddest' in l:check('internal destination '+doc+l['nameddest'],l['nameddest'] in names[f])
  check('no clipped page content '+doc,not bounds)
  fonts=subprocess.check_output(['pdffonts',str(P/f)],text=True);unembedded=[l for l in fonts.splitlines()[2:] if not re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',l)];check('embedded fonts '+doc,not unembedded)
  out['pdfs'][f]={'pages':len(d),'sha256':sha((P/f).read_bytes()),'cross_document_links':external,'warnings':re.findall(r'^.*Warning.*$',log,re.M),'underfull_boxes':len(re.findall('Underfull',log)),'latex_diagnostics':bad,'metadata':d.metadata}
 out['status']='PASS';(D/'VALIDATION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(checks),'checks; no scientific evaluator or numerical expectation modified')
if __name__=='__main__':main()
