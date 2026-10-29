#!/usr/bin/env python3
"""Run the unchanged Stage 2 checks against Stage 3; preserve all earlier outputs."""
from pathlib import Path
import importlib.util,json,sys,hashlib,re,subprocess
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
old=ROOT/'docs/jdsa_stage2_20261029/validate_stage2.py'
spec=importlib.util.spec_from_file_location('stage2_checks',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.D=OUT;module.N=ROOT/'manuscript/jdsa-stage3'
status=module.main()
if status:raise SystemExit(status)
report=json.loads((OUT/'NUMERICAL_CLAIM_CONSISTENCY.json').read_text())
checks=[]
def check(name,condition):
 checks.append({'check':name,'status':'PASS' if condition else 'FAIL'})
 if not condition:raise AssertionError(name)
base=ROOT/'manuscript/jdsa-stage2';current=ROOT/'manuscript/jdsa-stage3'
oldtext='\n'.join(module.expand(base,d) for d in ['main','supplement']);newtext='\n'.join(module.expand(current,d) for d in ['main','supplement'])
for doc in ['main','supplement']:
 o=module.floats(module.expand(base,doc));n=module.floats(module.expand(current,doc));check('Stage 2 object locations preserved in '+doc,set(o)==set(n))
 for label in o:check('Stage 2 numerical object content '+label,module.numbody(o[label])==module.numbody(n[label]))
for path in ['manuscript/jdsa','manuscript/jdsa-stage2','reproducibility/jdsa-v2.0.0','docs/jdsa_stage2_20261029']:
 changed=subprocess.check_output(['git','diff','--name-only','60785b0d144365413dd3b44b577bff901043ddd7','--',path],cwd=ROOT,text=True);check('preserved '+path,not changed)
# Stronger checks complement token preservation: fixed empirical equations and table
# rows, with the notation-only assignment equation handled separately.
for label in ['eq:top1_schema','eq:coverage','eq:typematch','eq:bounded_error','eq:exact20','eq:full-scalar','eq:conditional-agreement','eq:snack-overlay','eq:common18-objective-proxy','eq:fleiss']:
 def eq(text):
  for m in re.finditer(r'\\begin\{equation\}[\s\S]*?\\end\{equation\}',text):
   if '\\label{'+label+'}' in m[0]:return re.sub(r'\s+','',m[0])
 check('unchanged scientific equation '+label,eq(oldtext)==eq(newtext) and eq(oldtext) is not None)
getkeys=lambda t:{k for v in re.findall(r'\\cite\w*\{([^}]+)\}',t) for k in v.split(',')}
check('all Stage 2 citation keys preserved',getkeys(oldtext)==getkeys(newtext))
check('no editorial placeholders',not re.search(r'\b(?:TODO|TBD|FIXME|INSERT HERE|PLACEHOLDER)\b',newtext))
report['stage3_extension']={'status':'PASS','baseline':'60785b0d144365413dd3b44b577bff901043ddd7','unchanged_stage2_validator_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'stage2_checks_passed':report['checks_passed'],'checks':checks,'checks_passed':len(checks),'no_new_experiments':True,'scope':'Retargeted unchanged Stage 2 validator plus Stage 2-to-3 preservation. Missing historical originals remain unavailable.'}
(OUT/'NUMERICAL_CONSISTENCY.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('STAGE3',len(checks),'PASS; STAGE2',report['checks_passed'],'PASS; FROZEN',report['frozen_verification']['checks_passed'],'PASS')
