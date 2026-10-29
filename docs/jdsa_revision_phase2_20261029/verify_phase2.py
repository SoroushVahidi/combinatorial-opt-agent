"""Read-only evidence aggregation and document-preservation checks.
No model, solver, decoder, training, bootstrap or network calls.
Run with --archive-parent pointing to existing sibling research worktrees;
without it, only public-repository evidence is checked (explicitly reported).
"""
import argparse,csv,hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
P=Path(__file__).resolve().parent
R=P.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--archive-parent',type=Path);args=parser.parse_args()
report={'scope':'Saved observations only; no new experimental measurements','sources':{},'checks':{}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):
 try: name=str(p.relative_to(R))
 except ValueError: name='/'.join(p.parts[-(len(p.parts)-3):])
 report['sources'][name]=sha(p)
def csvrows(p):record(p);return list(csv.DictReader(p.open()))
def tables(folder):
 out={}
 for p in folder.glob('*.tex'):
  for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}',p.read_text(),re.S):
   label=re.search(r'\\label\{([^}]+)\}',t)
   assert label,p
   assert label[1] not in out,label[1]
   out[label[1]]=t
 return out
old=R/'manuscript/jdsa-phase1';new=R/'manuscript/jdsa-phase2'
a,b=tables(old),tables(new)
assert len(a)==33,(len(a),len(b))
assert all(b.get(k)==v for k,v in a.items())
report['checks']['tables']={'phase1_tables_preserved_byte_for_byte':len(a),'new_D1_tables':len(b)-len(a)}
a=(old/'main.tex').read_text();b=(new/'main.tex').read_text()
assert re.search(r'\\abstract\{(.*?)\}\n',a,re.S)[1]==re.search(r'\\abstract\{(.*?)\}\n',b,re.S)[1]
assert a[a.index('\\section{Diagnostic Results}'):a.index('\\section{LLM-Based')].replace('Diagnostic Results','Experimental Results')==b[b.index('\\section{Experimental Results}'):b.index('\\subsection{Complementary Evidence')]
ai=a[a.index('\\section{Introduction}'):a.index('The resulting contribution')]
bi=b[b.index('\\section{Introduction}'):b.index('The resulting contribution')];assert ai==bi
sections=re.findall(r'\\section\{([^}]+)\}',b)
assert len(sections)==7
report['checks']['manuscript']={'abstract_unchanged':True,'introduction_changed_only_roadmap':True,'core_results_unchanged_except_section_title':True,'sections':sections}
# Registered evidence must remain exactly the inspected Phase 1 records.
manifest=list(csv.DictReader((R/'docs/jdsa_revision_phase1_20261029/EVIDENCE_SOURCE_MANIFEST.csv').open()))
checked=[];skipped=[]
for row in manifest:
 p=(R/row['path']) if row['locator'].startswith('current:') else ((args.archive_parent/row['worktree']/row['path']) if args.archive_parent else None)
 if p is None:skipped.append(row['locator']);continue
 assert p.exists(),str(p)
 assert sha(p)==row['sha256'],row['locator']
 checked.append(row['locator'])
report['checks']['phase1_manifest']={'hashes_verified':len(checked),'local_sources_not_checked':skipped}
# Existing extraction measurements: no rerun of extraction or resampling.
base=R/'results/external_validation/optmath';p=base/'audited_per_query_metrics.jsonl';record(p)
rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==1000
sums={k:sum(r[k] for r in rows) for k in ['num_gold_ast','num_matched_ast','num_gold_text_supported','num_matched_text_supported']}
assert sums=={'num_gold_ast':31736,'num_matched_ast':23805,'num_gold_text_supported':29322,'num_matched_text_supported':23805},sums
complete=sum(r['is_complete'] for r in rows);assert complete==591
p=base/'final_verification/verified_metrics.json';record(p);v=json.loads(p.read_text())['VerifiedTextSupportedRecall'];assert math.isclose(v['macro'],23241/28799)
rs=csvrows(base/'final_verification/manual_audit_random150.csv');c=Counter(r['manual_category'] for r in rs)
agree=sum(r['agreement']=='True' for r in rs);assert len(rs)==4801 and agree==4707 and len({r['instance_id'] for r in rs})==150
p=base/'contamination_summary.json';record(p);categories=json.loads(p.read_text())['unique_literals_by_category']
contam=sum(categories[k] for k in ['IMPLEMENTATION_ONLY','SOLVER_OR_API_CONSTANT','INDEX_OR_LOOP_BOUND','ARRAY_OR_DIMENSION_CONSTANT']);assert contam==2310 and round(contam/31736*100,2)==7.28
report['checks']['optmath']={'instances':1000,'audited_sums':sums,'complete':complete,'raw_recall':23805/31736,'text_recall':23805/29322,'corrected_pooled_summary':23241/28799,'corrected_macro_summary':v['micro'],'historical_CI_summary_only':v['95CI'],'classifier_literals':4801,'classifier_agreements':agree,'automatic_1000_instance_constant_count':contam,'constant_denominator':31736,'corrected_per_instance_counts':'Unavailable; CI not re-estimated'}
if args.archive_parent:
 parent=args.archive_parent
 p=parent/'combinatorial-opt-agent-manuscript-m3/docs/manuscript_m3_2026-10-28/COMMON18_CORRECTED_CASES.csv';rs=csvrows(p);assert len(rs)==108
 counts={}
 for system in ['pamop','optmath','orlm','generic','deepor','orr1']:
  rr=[r for r in rs if r['system']==system];assert len(rr)==18
  counts[system]={k:sum(r[k]=='True' for r in rr) for k in ['generation_attempted','parsed','process_success','valid_objective_extraction','reference_available','numerical_agreement','precision_unresolved']}
 assert [counts[s]['process_success'] for s in ['pamop','optmath','orlm','generic']]==[13,15,0,16]
 assert counts['optmath']['numerical_agreement']==11 and counts['generic']['numerical_agreement']==12 and counts['pamop']['precision_unresolved']==9
 assert counts['deepor']['generation_attempted']==counts['orr1']['generation_attempted']==0
 for r in rs:
  if r['system'] in ['generic','optmath'] and r['process_success']=='True':
   a,b=float(r['objective_token']),float(r['reference_objective']);ar,br=round(a),round(b)
   agreed=abs(ar-br)<=.05 if br==0 else abs(ar-br)/abs(br)<=.05
   raw=abs(a-b)<=.05 if b==0 else abs(a-b)/abs(b)<=.05
   assert agreed==(r['numerical_agreement']=='True')==raw
 counts['pamop']['numerical_agreement']='WITHHELD: zero authenticated matches is not an accuracy aggregate'
 report['checks']['common18']=counts
 p=parent/'combinatorial-opt-agent-manuscript-m5/docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/llm217/llm217_per_judgment.csv';rs=csvrows(p);assert len(rs)==651
 by=defaultdict(list)
 for r in rs:by[r['item_id']].append(r)
 assert len(by)==217 and all(len(v)==3 for v in by.values())
 stages={s:{p:sum(r[s+'_valid']=='True' for r in rs if r['provider']==p) for p in ['gemini','ionet','cohere']} for s in ['original','offline','final']}
 assert list(stages['original'].values())==[215,180,105]
 assert list(stages['offline'].values())==[217,213,129]
 assert list(stages['final'].values())==[217,217,197]
 kappas={}
 for stage,label in [('original','original_label'),('final','final_label')]:
  rr=[v for v in by.values() if all(r[stage+'_valid']=='True' for r in v)];n=len(rr);marg=Counter(r[label] for v in rr for r in v)
  po=sum(sum(c*(c-1) for c in Counter(r[label] for r in v).values())/6 for v in rr)/n
  pe=sum((c/(3*n))**2 for c in marg.values());k=(po-pe)/(1-pe)
  kappas[stage]={'complete':n,'kappa':k,'observed':po,'chance':pe,'marginals':dict(marg)}
 assert kappas['original']['complete']==89 and kappas['final']['complete']==197
 assert math.isclose(kappas['final']['kappa'],.3708071786079686)
 paircounts={}
 for p,q in [('gemini','ionet'),('gemini','cohere'),('ionet','cohere')]:
  pairs=[[r for r in v if r['provider'] in [p,q]] for v in by.values()]
  pairs=[v for v in pairs if all(r['final_valid']=='True' for r in v)]
  paircounts[p+'/'+q]=[sum(v[0]['final_label']==v[1]['final_label'] for v in pairs),len(pairs)]
 assert list(paircounts.values())==[[213,217],[123,197],[120,197]]
 report['checks']['relationships']={'stages':stages,'kappa':kappas,'pairs':paircounts,'source_problems':len({r['problem_id'] for r in rs})}
 p=parent/'combinatorial-opt-agent-d1-final-pair-audit/docs/d1_final_pair_audit_2026-10-29/D1_FINAL_PAIR_DECISIONS.csv';rr=csvrows(p);assert len(rr)==12
 labels=Counter(label for r in rr for label in r['Final author labels (base/contrast)'].split('/'))
 assert labels=={'CORRECT':12,'INCORRECT':12}
 assert Counter(r['Contrast type'] for r in rr)=={'MEANING_CHANGING':10,'MEANING_PRESERVING':2}
 d1=(new/'d1_contrasts.tex').read_text()
 for row in rr:
  pair=row['Pair ID'].replace('D1P-','')
  latexrow=next(line for line in d1.splitlines() if line.startswith(pair+' &'))
  assert row['Final author labels (base/contrast)'].replace('INCORRECT','I').replace('CORRECT','C') in latexrow
 p=parent/'combinatorial-opt-agent-d1-final-pair-audit/docs/d1_final_pair_audit_2026-10-29/evidence/local_sources/SELECTED_D1_CASES_UNCHANGED.json';record(p);rr=json.loads(p.read_text())['selected_rows'];assert len(rr)==24 and len({r['pair_id'] for r in rr})==12 and len({r['category'] for r in rr})==8
 report['checks']['D1']={'pairs':12,'samples':24,'categories':8,'status':'Construction and author audit; no provider-performance measurements'}
else:report['checks']['local_archives']='NOT CHECKED: supply --archive-parent; local-only evidence is not released by this package'
destinations=json.loads((P/'INVESTIGATION_DESTINATIONS.json').read_text())
assert {r['id'] for r in destinations}=={f'E{i:02}' for i in range(1,48)}
report['checks']['investigation_destinations']=len(destinations)
# Numbering conventions and long duplicate paragraphs within each document.
for doc in ['main','supplement']:
 def expand(path):
  text=path.read_text()
  return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(new/(m[1]+'.tex')),text)
 text=expand(new/(doc+'.tex'))
 assert not re.search(r'\\\[|\\begin\{(?:equation|align)\*\}',text)
 paragraphs=[re.sub(r'\s+',' ',s).strip() for s in text.split('\n\n') if len(s.split())>=45]
 duplicates=[k for k,v in Counter(paragraphs).items() if v>1];assert not duplicates
 report['checks'][doc+'_duplicate_long_paragraphs']=len(duplicates)
(P/'SCIENTIFIC_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('PASS: preservation, manuscript structure, saved extraction records, and available archival ledgers.')
