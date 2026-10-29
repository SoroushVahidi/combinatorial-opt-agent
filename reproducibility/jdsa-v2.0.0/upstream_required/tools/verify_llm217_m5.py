#!/usr/bin/env python3
"""Independent, standard-library-only reconstruction from frozen HTTP responses.
No imports of historical validators, consensus calculators, model SDKs or solvers.
Writes derived evidence only to --output; never edits historical records.
"""
import argparse,collections,copy,csv,hashlib,itertools,json,re
from pathlib import Path
P=('gemini','ionet','cohere')
LABELS=('CORRECT','INCORRECT','AMBIGUOUS','CANNOT_DETERMINE')
KEYS={'item_id','decision','explanation','evidence','variable_ids','constraint_ids','missing_relations','missing_implicit_objective_terms','shared_mention_ambiguity','uncertainty'}
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(b):return hashlib.sha256(b).hexdigest()
def main(repo,out):
 out.mkdir(parents=True,exist_ok=True); hashes={};checks=[]
 def read(p):
  p=repo/p;b=p.read_bytes();hashes[str(p.relative_to(repo))]=sha(b);return b.decode()
 def js(p):return json.loads(read(p))
 def rows(p):return list(csv.DictReader(read(p).splitlines()))
 def ck(name,ok):
  if not ok:raise AssertionError(name)
  checks.append(name)
 O=Path('docs/nl4opt_three_model_2026-10-27');R=Path('docs/nl4opt_label_recovery_v2_2026-10-27');V=Path('docs/nl4opt_label_validation_2026-10-27')
 config=js(O/'config.json'); rconfig=js(R/'recovery_config.json'); native=js(R/'recovery_config_native1.json')
 packets=rows(V/'HIGH_PRIORITY_ADJUDICATION_PACKET.csv');endpoints=rows(V/'LABEL_REVIEW_PACKET.csv');complete=js(V/'evidence/review_problem_completeness.json');source=js('docs/nl4opt_label_audit_2026-10-27/evidence/review_source_records.json');provenance={r['problem_id']:r for r in rows(V/'DATASET_PROVENANCE_REGISTER.csv')}
 items={};contexts={}
 for p in packets:
  pid=p['problem_id'];ids=json.loads(p['relationship_ids']);text=p['source_problem_text'];cons=json.loads(p['source_constraints']);variables=complete[pid]['variables']
  pointers={'/obj_declaration'}|{f'/const_declarations/{i}' for i in range(len(cons))}|{e['declaration_pointer'] for e in endpoints if e['problem_id']==pid and e['declaration_pointer']}
  contexts[pid]=(text,variables,pointers)
  ck('train provenance '+pid,provenance[pid]['dataset']=='NL4Opt' and provenance[pid]['partition']=='train')
  ck('source hash '+pid,sha(canonical(source[pid]).encode())==provenance[pid]['source_record_sha256'])
  for rid in ids:
   ck('unique item '+rid,rid not in items);items[rid]=pid
 ck('217 relationships in 16 source problems',len(items)==217 and len(contexts)==16)
 # Independently validate source fields, membership and all literal quotations.
 def validate(j,pid,rid):
  errors=[];q=copy.deepcopy(j)
  if not isinstance(q,dict):return {'valid':False,'judgment':None,'errors':['missing_judgment']}
  q.pop('evidence_positions',None); text,vs,ptrs=contexts[pid]
  if set(q)!=KEYS:errors.append('response_schema')
  label=q.get('decision');label=label.strip().upper().replace(' ','_') if isinstance(label,str) else None
  if label not in LABELS:errors.append('decision_domain')
  else:q['decision']=label
  if q.get('item_id')!=rid:errors.append('item_identifier')
  for k in ('explanation','uncertainty'):
   if not isinstance(q.get(k),str) or not q[k].strip():errors.append('missing_'+k)
  for k in ('variable_ids','constraint_ids','missing_relations','missing_implicit_objective_terms'):
   if not isinstance(q.get(k),list) or any(not isinstance(x,str) for x in q[k]):errors.append('type_'+k)
  if isinstance(q.get('variable_ids'),list) and any(x not in vs for x in q['variable_ids']):errors.append('variable_identifier')
  if isinstance(q.get('constraint_ids'),list) and any(x not in ptrs for x in q['constraint_ids']):errors.append('formulation_identifier')
  if type(q.get('shared_mention_ambiguity')) is not bool:errors.append('boolean_flag')
  ev=q.get('evidence');pos=[]
  if not isinstance(ev,list) or not ev:errors.append('missing_evidence')
  else:
   for e in ev:
    if not isinstance(e,dict) or set(e)!={'quote','kind'} or e.get('kind') not in ('supporting','contradicting','context'):errors.append('evidence_schema');continue
    z=e.get('quote')
    if not isinstance(z,str) or not z.strip() or z not in text:errors.append('nonliteral_evidence');continue
    pos.append({'matches':[[m.start(),m.end()] for m in re.finditer(re.escape(z),text)],'quote':z})
  if not errors:q['evidence_positions']=pos
  return {'valid':not errors,'judgment':q,'errors':sorted(set(errors))}
 def text_of(p,x):
  if p=='gemini':return ''.join(t.get('text','') for c in x.get('candidates',[]) for t in c.get('content',{}).get('parts',[]) if not t.get('thought'))
  if p=='ionet':return ((x.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
  message=x.get('message');return ''.join(t.get('text','') for t in message.get('content',[]) if t.get('type')=='text') if isinstance(message,dict) else ''
 def envelope(text,ids,recover=False):
  if not recover:text=re.sub(r'^```(?:json)?\s*','',text.strip());text=re.sub(r'\s*```$','',text)
  def acceptable(v):return isinstance(v,dict) and set(v)=={'judgments'} and isinstance(v['judgments'],list) and len(v['judgments'])==len(ids) and all(isinstance(j,dict) for j in v['judgments']) and len({j.get('item_id') for j in v['judgments']})==len(ids) and {j.get('item_id') for j in v['judgments']}==set(ids)
  try:
   v=json.loads(text)
   if acceptable(v):return v['judgments']
  except ValueError:pass
  if recover:
   candidates=[];decoder=json.JSONDecoder()
   for start,c in enumerate(text):
    if c!='{':continue
    try:v,n=decoder.raw_decode(text[start:])
    except ValueError:continue
    if acceptable(v):candidates.append(v)
   if len(candidates)==1:return candidates[0]['judgments']
  return []
 def repair(j,pid):
  q=copy.deepcopy(j);aliases=source[pid].get('var_mention_to_first_var',{});vs=contexts[pid][1]
  if isinstance(q.get('variable_ids'),list):q['variable_ids']=[aliases[x] if isinstance(x,str) and x not in vs and x in aliases and aliases[x] in vs else x for x in q['variable_ids']]
  for k in ('missing_relations','missing_implicit_objective_terms'):
   if isinstance(q.get(k),list):q[k]=[canonical(x) if isinstance(x,dict) else x for x in q[k]]
  return q
 orig={p:{} for p in P};offline={p:{} for p in P};requests=[];parse_counts=collections.Counter()
 for p in P:
  seen=[]
  for b in config['batches']:
   cp=js(O/'checkpoints'/p/(b['batch_id']+'.json'));raw=js(O/cp['raw_path']);txt=text_of(p,raw['response']);ids=b['item_ids'];pid=b['problem_id'];seen+=ids
   prompt=read(O/'prompts'/(b['batch_id']+'.txt'));ck('original prompt hash '+p+b['batch_id'],sha(prompt.encode())==config['prompt_sha256'][b['batch_id']])
   strict={j['item_id']:j for j in envelope(txt,ids)} if raw['http_status']==200 else {}
   if raw['http_status']==200:
    parse_counts[p+':original_http200_envelopes_parsed']+=bool(strict)
    parse_counts[p+':original_http200_envelope_failures']+=not bool(strict)
    parse_counts[p+':original_unparsed_items']+=len(ids) if not strict else 0
   else:parse_counts[p+':original_service_unavailable_items']+=len(ids)
   loose={j['item_id']:j for j in envelope(txt,ids,True)} if raw['http_status']==200 else {}
   for rid in ids:
    a=validate(strict.get(rid),pid,rid);orig[p][rid]=a
    archived=cp.get('normalized',{}).get(rid,{})
    ck('original raw/checkpoint validity '+p+rid,a['valid']==bool(archived.get('valid')))
    if a['valid']:ck('original normalized object '+p+rid,a['judgment']==archived['judgment'])
    offline[p][rid]=copy.deepcopy(a) if a['valid'] else validate(repair(loose[rid],pid),pid,rid) if rid in loose else copy.deepcopy(a)
   # Request count is independently enumerated below, including unsuccessful attempts.
  ck('no missing or repeated original targets '+p,len(seen)==len(set(seen))==217 and set(seen)==set(items))
 final=copy.deepcopy(offline)
 for job in rconfig['jobs']+native['jobs']:
  p=job['provider'];b=job['batch'];pid=b['problem_id'];cp=js(R/'replacement_checkpoints'/(b['batch_id']+'.json'))
  ck('replacement request hash '+b['batch_id'],sha(canonical(job['request']).encode())==job['request_sha256'])
  ck('replacement prompt hash '+b['batch_id'],sha(read(R/'prompts'/(b['batch_id']+'.txt')).encode())==job['prompt_sha256'])
  raw=js(Path(cp['raw_path']));ck('replacement raw hash '+b['batch_id'],hashes[cp['raw_path']]==cp['raw_sha256']);ck('raw request identity '+b['batch_id'],raw['request_sha256']==job['request_sha256'])
  found={j['item_id']:j for j in envelope(text_of(p,raw['response']),b['item_ids'],True)} if raw['http_status']==200 else {}
  if raw['http_status']==200:
   parse_counts[p+':recovery_http200_envelopes_parsed']+=bool(found)
   parse_counts[p+':recovery_http200_envelope_failures']+=not bool(found)
  for rid in b['item_ids']:
   ck('only invalid originals targeted '+p+rid,not offline[p][rid]['valid'])
   candidate=validate(repair(found[rid],pid),pid,rid) if rid in found else validate(None,pid,rid)
   ck('replacement raw/checkpoint validity '+p+rid,candidate['valid']==bool(cp['normalized'].get(rid,{}).get('valid')))
   if candidate['valid']:ck('replacement normalized object '+p+rid,candidate['judgment']==cp['normalized'][rid]['judgment'])
   final[p][rid]=candidate
 archive=js(R/'consensus.json')['items']; original_archive=js(O/'consensus.json')['items']
 warnings={rid:[] for rid in items};endpoint_map={e['relationship_id']:e for e in endpoints}
 for rid,pid in items.items():
  e=endpoint_map[rid]
  if not e['span_pointer']:warnings[rid].append('No lexical occurrence; NULL is not verified')
  if not e['declaration_pointer']:warnings[rid].append('No numeric declaration endpoint')
  if not json.loads(next(p['source_constraints'] for p in packets if p['problem_id']==pid)):warnings[rid].append('Source constraint formulation absent; completeness uncertain')
  ck('source warnings '+rid,warnings[rid]==archive[rid]['structural_warnings'])
  for p in P:
   ck('final raw/archive validity '+p+rid,final[p][rid]['valid']==archive[rid]['models'][p]['valid'])
   if final[p][rid]['valid']:ck('final raw/archive object '+p+rid,final[p][rid]['judgment']==archive[rid]['models'][p]['judgment'])
   if orig[p][rid]['valid']:ck('original valid object preserved '+p+rid,original_archive[rid]['models'][p]['judgment']==archive[rid]['models'][p]['judgment'])
 def stats(data,subset=None):
  ids=list(items) if subset is None else list(subset);valid={p:sum(data[p][i]['valid'] for i in ids) for p in P};joint=[i for i in ids if all(data[p][i]['valid'] for p in P)];pairs={};marg=collections.Counter();po=0
  for i in joint:
   c=collections.Counter(data[p][i]['judgment']['decision'] for p in P);marg.update(c);po+=sum(n*(n-1) for n in c.values())/6
  po=po/len(joint) if joint else None;pe=sum((v/(3*len(joint)))**2 for v in marg.values()) if joint else None
  for a,b in itertools.combinations(P,2):
   eligible=[i for i in ids if data[a][i]['valid'] and data[b][i]['valid']];n=sum(data[a][i]['judgment']['decision']==data[b][i]['judgment']['decision'] for i in eligible);pairs[a+'__'+b]={'same':n,'n':len(eligible),'fraction':n/len(eligible)}
  counts=collections.Counter();label_unanimity=0
  for i in ids:
   if not all(data[p][i]['valid'] for p in P):c='INVALID_OR_INCOMPLETE'
   else:
    labels=[data[p][i]['judgment']['decision'] for p in P]
    if len(set(labels))>1:c='DISAGREEMENT'
    else:
     label_unanimity+=1;c='UNRESOLVED' if warnings[i] or labels[0] in LABELS[2:] else 'UNANIMOUS'
   counts[c]+=1
   if data is final:ck('classification '+i,c==archive[i]['classification'])
  return {'n_items':len(ids),'valid':valid,'complete':len(joint),'pairwise':pairs,'marginals':dict(marg),'fleiss_observed':po,'fleiss_expected':pe,'fleiss_kappa':(po-pe)/(1-pe),'classifications':dict(counts),'raw_label_unanimity':label_unanimity,'valid_label_distributions':{p:dict(collections.Counter(data[p][i]['judgment']['decision'] for i in ids if data[p][i]['valid'])) for p in P}}
 stages={'original':stats(orig),'offline_only':stats(offline),'recovery_v2_1':stats(final)}
 common=[i for i in items if all(orig[p][i]['valid'] for p in P)];stages['final_on_original_complete']=stats(final,common)
 origins={p:collections.Counter() for p in P};ledger=[];invalid=[]
 for rid,pid in items.items():
  for p in P:
   a,b,c=orig[p][rid],offline[p][rid],final[p][rid];origin='original_valid' if a['valid'] else 'recovered_original' if b['valid'] else 'new_replacement' if c['valid'] else 'still_invalid';origins[p][origin]+=1
   ck('origin '+p+rid,origin==archive[rid]['models'][p]['origin'])
   row={'item_id':rid,'problem_id':pid,'provider':p,'model':config['providers'][p]['model'],'original_valid':a['valid'],'offline_valid':b['valid'],'final_valid':c['valid'],'original_label':(a['judgment'] or {}).get('decision',''),'final_label':(c['judgment'] or {}).get('decision',''),'origin':origin,'final_errors':';'.join(c['errors']),'classification':archive[rid]['classification'],'warnings':';'.join(warnings[rid]),'parameter_role':endpoint_map[rid]['parameter_role']};ledger.append(row)
   if not c['valid']:invalid.append(row)
 ck('no human reference labels',not any(x['human_verified'] for x in archive.values()))
 for stage,sub in [('original',O/'raw'),('recovery',R/'replacement_raw')]:
  for f in sorted((repo/sub).rglob('*.json')):
   x=js(f.relative_to(repo));resp=x['response'];model_version=resp.get('modelVersion') if x['provider']=='gemini' else resp.get('model') if x['provider']=='ionet' else None
   requests.append({'stage':stage,'provider':x['provider'],'http_status':x['http_status'],'model':x.get('model'),'response_model':model_version,'started_at':x['started_at'],'completed_at':x['completed_at'],'path':str(f.relative_to(repo))})
 problem_counts=collections.Counter(items.values());result={'parse_counts':dict(parse_counts),'stages':stages,'origins':origins,'problems':dict(sorted(problem_counts.items())),'request_status_counts':dict(collections.Counter(f"{x['stage']}:{x['provider']}:{x['http_status']}" for x in requests)),'checks_passed':len(checks),'human_reference_labels':0,'invalid_cases':invalid}
 for name,x in [('llm217_validation.json',result),('llm217_input_hashes.json',hashes),('llm217_checks.json',checks)]: (out/name).write_text(json.dumps(x,indent=2)+'\n')
 for name,records in [('llm217_per_judgment.csv',ledger),('llm217_requests.csv',requests),('llm217_missing20.csv',invalid)]:
  with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 print(json.dumps({'checks_passed':len(checks),'stages':stages,'origins':origins,'request_status_counts':result['request_status_counts']},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);a.add_argument('--output',type=Path,required=True);args=a.parse_args();main(args.repo.resolve(),args.output.resolve())
