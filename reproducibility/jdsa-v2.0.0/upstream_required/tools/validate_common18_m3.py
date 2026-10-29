"""Independent raw-record cross-check of M3 CSVs, without importing their parser."""
import argparse,csv,json,re,math,hashlib
from pathlib import Path
from decimal import Decimal
from collections import Counter

def verify(root,out):
 cases=list(csv.DictReader((out/'COMMON18_CORRECTED_CASES.csv').open()));summary=json.loads((out/'evidence/common18_corrected_summary.json').read_text());checks=[]
 def check(name,condition):
  if not condition:raise AssertionError(name)
  checks.append(name)
 rowsby={}
 for system,folder in [('generic','generic_llm/common18_official'),('optmath','optmath/common18_official_checkpoint'),('orlm','orlm/common18_official_checkpoint')]:
  rowsby[system]={str(r['problem_id']):r for r in (json.loads(line) for line in (root/'results'/folder/'results.jsonl').read_text().splitlines())}
 check('six-system frame108 unique records',len(cases)==len({(r['case_id'],r['system']) for r in cases})==108)
 for r in cases:
  system=r['system'];cid=r['case_id'];name=f'{system}:{cid}'
  if system in ('deepor','orr1'):
   check(name+' no run invented',r['generation_attempted']=='False' and r['corrected_classification']=='N/A_NO_ARTIFACT');continue
  if system=='pamop':
   x=json.loads((root/f'results/pamop/fidelity_diagnostic_gpt5/incremental/problem_{cid}.json').read_text());ex=x['trace']['iterations'][-1]['execution'];check(name+' parse',str(ex['parse_success'])==r['parsed']);check(name+' execution',str(ex['success'])==r['process_success'])
   if not ex['success']:check(name+' no objective result',r['corrected_classification']=='PARSE_FAILED');continue
   log=ex['stdout_tail'];model=x.get('generated_artifacts',{}).get('rendered_model',{}).get('model_text','')
   labeled=re.search(r'(?:maximize|minimize)\s+(\w+)\s*:',model)
   label=labeled[1] if labeled else re.findall(r'^\s*(\w+)\s*=\s*[-+\d.eE]+\s*$',log,re.M)[-1]
   # Separately inspect named display and optimal/presolve solver text, not regex
   # used in the production helper. Select the finest retained decimal token.
   tokens=[]
   for line in log.splitlines():
    if line.strip().startswith(label+' ='):tokens.append(line.split('=',1)[1].strip())
    if 'optimal solution; objective ' in line:tokens.append(line.split('optimal solution; objective ',1)[1].strip())
    if line.strip().startswith('objective '+label+' ='):tokens.append(line.split('=',1)[1].strip().rstrip('.'))
   token=min(tokens,key=lambda t:Decimal(t).as_tuple().exponent);v=Decimal(token);check(name+' highest available precision',v==Decimal(r['objective_token']))
   if r['reference_available']=='False':check(name+' no reference',r['corrected_classification']=='REFERENCE_UNAVAILABLE');continue
   g=Decimal(r['reference_objective']);unit=Decimal(1).scaleb(v.as_tuple().exponent);gap=abs(v-g)
   classification='OBJECTIVE_DISAGREEMENT' if gap-unit>Decimal('1e-6') else ('OBJECTIVE_MATCH' if gap+unit<=Decimal('1e-6') else 'UNRESOLVED_PRECISION')
   check(name+' conservative precision classification',r['corrected_classification']==classification)
  else:
   x=rowsby[system][cid];code=x['parsed'].get('generated_code',x['parsed'].get('coptpy_code',''));check(name+' code extraction',str(bool(code))==r['parsed'])
   if system=='orlm':check(name+' N/A',r['corrected_classification']=='N/A_ENVIRONMENT' and r['process_success']=='');continue
   ex=x['execution'];successful=ex['status']=='COMPLETED' and ex['return_code']==0;check(name+' process status',str(successful)==r['process_success'])
   if not successful:check(name+' excluded failed process',r['numerical_agreement']=='' and r['valid_objective_extraction']=='False');continue
   log=ex['stdout'];printed=Decimal(log.strip().splitlines()[-1]);terminal=re.findall(r'(?:Best objective|Optimal objective)\s+([-+\d.eE]+)',log)[-1]
   check(name+' final independent print witness',Decimal(r['objective_token'])==printed)
   check(name+' terminal crosscheck',math.isclose(float(terminal),float(printed),rel_tol=1e-8,abs_tol=1e-6))
   g=x['gold_objective'];pr,gr=round(float(printed)),round(float(g));agree=abs(pr-gr)<=.05 if gr==0 else abs((pr-gr)/gr)<=.05
   direct=abs(printed-Decimal(str(g))) <= (Decimal('.05') if g==0 else Decimal('.05')*abs(Decimal(str(g))))
   check(name+' agreement independently evaluated',r['numerical_agreement']==str(agree))
   check(name+' unrounded sensitivity',r['unrounded_sensitivity_agreement']==str(direct))
 for system in summary:
  subset=[r for r in cases if r['system']==system]
  check(system+' classification aggregation',dict(Counter(r['corrected_classification'] for r in subset))==summary[system]['classification_counts'])
 check('published counts',summary['generic']['confirmed_agreements']==12 and summary['optmath']['confirmed_agreements']==11)
 check('PaMOP aggregate withheld',summary['pamop']['aggregate_objective_published'] is False and summary['pamop']['precision_unresolved']==9)
 check('unavailable execution never zero rate',summary['orlm']['executed_successfully'] is None)
 report={'checks_passed':len(checks),'cases_checked':108,'archived_generated_cases':72,'solver_or_model_calls':0,'checks':checks};(out/'evidence/independent_common18_validation.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--audit-dir',type=Path,required=True);x=a.parse_args();r=verify(Path(__file__).resolve().parents[1],x.audit_dir);print(json.dumps({k:v for k,v in r.items() if k!='checks'}))
