#!/usr/bin/env python3
"""Versioned location overlay; never writes immutable release claims.json."""
import argparse,hashlib,json,re,collections
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;O=B/'manuscript/jdsa/source';N=B/'manuscript/jdsa-stage3';R=B/'reproducibility/jdsa-v2.0.0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build-dir',type=Path,required=True);a=ap.parse_args();old=json.loads((R/'claims.json').read_text());locations={};filedoc={};expanded={}
 def visit(name,doc):
  filedoc[name+'.tex']=doc;p=N/(name+'.tex');s=p.read_text()
  for m in re.finditer(r'\\label\{([^}]+)\}',s):locations[m[1]]={'document':doc,'path':str(p.relative_to(B)),'line':s[:m.start()].count('\n')+1,'source_sha256':sha(p)}
  for x in re.findall(r'\\input\{([^}]+)\}',s):visit(x,doc)
 visit('main','main');visit('supplement','supplement')
 for doc in ['main','supplement']:
  aux=(a.build_dir/(doc+'.aux')).read_text()
  for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}',aux):
   if m[1] in locations:locations[m[1]].update(number=m[2],page=m[3])
 def loc(labels):
  out=[]
  for label in labels:
   assert label in locations,label;out.append({'label':label,**locations[label]})
  return out
 objects=[]
 for c in old['named_objects']:objects.append({'original_claim_id':c['id'],'original_display':c['display'],'original_document':c['document'],'classification_unchanged':c['classification'],'revised_locations':loc([c['id']]),'evidence_reference':'Immutable claims.json entry '+c['id'],'numerical_values':'Preserved; see NUMERICAL_CLAIM_CONSISTENCY.json'})
 # Paragraph mappings are explicit section/claim destinations; exact paragraph matches
 # are recorded where wording survives. Rewritten passages are not falsely byte-matched.
 ranges=[(1,42,['subsec:intro_contributions','subsec:exp_retrieval','subsec:exp_downstream','subsec:grounding-followup']),(43,70,['subsec:intro_contributions']),(71,100,['sec:related_work','app:related-context']),(101,121,['subsec:method_problem']),(122,137,['subsec:assistance-setting','subsec:inference-dependencies']),(138,190,['subsec:metrics']),(191,254,['subsec:method_retrieval','subsec:method_instantiation','subsec:method_controls']),(255,267,['subsec:exp_setup','subsec:exp_significance','app:reproduction']),(268,294,['subsec:exp_retrieval']),(295,326,['subsec:exp_downstream']),(327,360,['subsec:exp_grounding_baselines']),(361,390,['subsec:exp_error']),(391,399,['subsec:grounding-followup','app:factorial-details']),(400,447,['subsec:exp_significance','subsec:exp_error','app:core-support']),(448,470,['app:statistical-history','app:core-support']),(471,474,['app:restricted-details']),(475,477,['app:runtime-details']),(478,484,['sec:llm','app:occurrence-boundary']),(485,510,['app:common18','tab:ext-outcomes']),(511,516,['app:extraction-details']),(517,522,['subsec:d1','app:d1-details']),(523,554,['sec:discussion','subsec:limitations','subsec:reanalysis']),(555,563,['sec:conclusion']),(564,610,['app:reproduction','subsec:reanalysis'])]
 family={'reproduction_map.tex':['app:reproduction','app:input-dependencies'],'reference_conventions.tex':['app:reference-conventions'],'grounding_diagnostics.tex':['subsec:grounding-followup','app:factorial-details'],'supervision_diagnostics.tex':['app:supervision-audit','app:nl4opt-audit'],'llm_evidence.tex':['app:common18','eq:common18-objective-proxy'],'closed_audits.tex':['app:g3'],'relationship_evidence.tex':['app:relationship-overview','app:relationship-evidence'],'relationship_supplement.tex':['app:relationship-evidence','app:future-reference'],'preserved_results_details.tex':['app:restricted-details','app:runtime-details','app:occurrence-boundary','app:statistical-history'],'additional_evidence.tex':['app:additional-evidence'],'d1_contrasts.tex':['app:d1-details'],'pipeline_figure.tex':['fig:instantiation-pipeline']}
 passages=[]
 for c in old['numerical_passages']:
  name=Path(c['source_path']).name;labels=[l for l in c['named_objects'] if l in locations]
  if not labels:
   if name=='main.tex':labels=next(ds for lo,hi,ds in ranges if lo<=c['line_start']<=hi)
   elif name=='supplement.tex':labels=['app:extraction-details'] if 208<=c['line_start']<=213 else ['app:d1-details'] if c['line_start']>=214 else ['tab:comparison_landscape','tab:exp_blocks']
   else:labels=family[name]
  block='\n'.join((B/c['source_path']).read_text().splitlines()[c['line_start']-1:c['line_end']]).strip();normalized=re.sub(r'\s+',' ',block);matches=[]
  for fname,doc in filedoc.items():
   s=(N/fname).read_text()
   if normalized and normalized in re.sub(r'\s+',' ',s):matches.append(str((N/fname).relative_to(B)))
  passages.append({'original_claim_id':c['id'],'original_source':{'path':c['source_path'],'sha256':c['source_sha256'],'line_start':c['line_start'],'line_end':c['line_end'],'passage_sha256':c['passage_sha256']},'original_numeric_literals':c['numeric_literals'],'classification_unchanged':c['classification'],'mapping_status':'Exact normalized passage retained' if matches else 'Rewritten, consolidated or relocated; mapped to explicit evidence discussion','exact_passage_files':matches,'revised_locations':loc(labels),'evidence_domains':c['domains'],'original_investigation_ids':c['investigation_ids'],'limitations':c['limitations']})
 investigation_labels={}
 for ids,labels in [(['E01','E02'],['subsec:exp_retrieval','tab:ranked-retrieval']),(['E03','E06'],['subsec:exp_downstream','subsec:exp_error']),(['E04','E13'],['app:core-support','subsec:exp_significance']),(['E05'],['subsec:exp_grounding_baselines']),(['E07','E09','E10'],['subsec:grounding-followup','app:factorial-details']),(['E08','E11'],['tab:cached-ceilings','tab:full-joint-pilot','app:factorial-details']),(['E12'],['app:reference-conventions']),(['E14'],['tab:nlp4lp-overlap','app:statistical-history']),(['E15','E16','E17'],['app:restricted-details']),(['E18'],['app:common18']),(['E19','E21','E43'],['app:relationship-evidence','app:nl4opt-audit','app:future-reference']),(['E20'],['app:d1-details']),(['E22','E23','E24','E25','E26','E45'],['app:supervision-audit','app:additional-evidence']),(['E27'],['app:extraction-details']),(['E28','E29','E30'],['app:g3']),(['E31','E32','E33','E34','E35','E37','E38','E39','E40','E41','E42','E44'],['app:additional-evidence']),(['E36'],['app:statistical-history']),(['E46','E47'],['app:reproduction'])]:
  for i in ids:investigation_labels[i]=labels
 inv=[{'original_claim_id':x['id'],'name':x['name'],'revised_locations':loc(investigation_labels[x['id']]),'historical_status_preserved':x['status']} for x in old['investigations']]
 deriv=json.loads((D/'DERIVED_DIAGNOSTICS.json').read_text());new={'id':'STAGE2-D01','description':'Complete all-key scalar agreement conditional on ordinary/Strict readiness','locations':loc(['eq:full-scalar','eq:conditional-agreement','tab:ready-conditioned']),'authenticated_input_paths':sorted({x['source_path'] for x in deriv['arms'].values()}),'verification':'DERIVED_DIAGNOSTICS.json','derivation_script':'derive_ready_conditionals.py','classification':'DESCRIPTIVE_SAVED_RECORD_REANALYSIS','new_model_or_solver_experiments':False}
 report={'schema_version':'1','crosswalk_version':'jdsa-stage3-v1','immutable_claim_manifest_path':'reproducibility/jdsa-v2.0.0/claims.json','immutable_claim_manifest_sha256':sha(R/'claims.json'),'baseline_commit':'60785b0d144365413dd3b44b577bff901043ddd7','named_objects':objects,'numerical_passages':passages,'investigations':inv,'new_derived_claims':[new],'coverage':{'original_named_objects':len(objects),'original_numeric_passages':len(passages),'original_investigations':len(inv),'missing_original_ids':[]},'rounding_or_format_equivalences':{'0.370807':'Full precision 0.3708071786 retained in relationship_supplement.tex; main rounds 0.371','0.501643':'Full precision 0.5016425265 retained in relationship_supplement.tex','10,000':'Same historical resample count retained as LaTeX 10{,}000'},'availability_change':'Historical M5-local statements updated to distinguish public v2 projections from unreleased complete originals; no evidence/result overwrite.'}
 (D/'CLAIM_LOCATION_CROSSWALK_V1.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 md=['# Stage 3 claim-location crosswalk v1','','Immutable input: `claims.json` at evidence commit `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`. All 37 numbered objects, 183 numerical passages and 47 investigations retain links in the JSON companion. No immutable release files are changed.','','| Original | Claim ID | Revised document / number / PDF page |','|---|---|---|']
 for x in objects:
  l=x['revised_locations'][0];md.append(f"| {x['original_document']} {x['original_display']} | `{x['original_claim_id']}` | {l['document']} {l.get('number')} / p.{l.get('page')} |")
 md+=['','New claim `STAGE2-D01` supplies the explicitly derived conditional agreement table and definitions. It has no substituted historical observation or new inferential family. The main equation numbers and document prefixes come from the final compiled auxiliary files.','', 'The paragraph overlay records exact retained wording when available and explicit section destinations for consolidated text. This is a location/evidence crosswalk, not a claim that rewritten prose is byte-identical or that historical summary-only observations became independently reproducible.']
 (D/'CLAIM_LOCATION_CROSSWALK_V1.md').write_text('\n'.join(md)+'\n');print('CROSSWALK',report['coverage'])
if __name__=='__main__':main()
