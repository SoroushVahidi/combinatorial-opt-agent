#!/usr/bin/env python3
"""Reanalyze authenticated v2 records; never execute an experiment or alter inputs."""
import csv,hashlib,json,re
from decimal import Decimal as D
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'reproducibility/jdsa-v2.0.0'
OUT=Path(__file__).resolve().parent

def indicators(r):
    hit=r['schema_hit']=='1'
    ready=float(r['param_coverage'])>=.8 and float(r['type_match'])>=.8
    eligible=hit and float(r['param_coverage'])==1 and float(r['type_match'])==1 and r['exact20']!='' and float(r['exact20'])==1
    full=eligible and float(r['key_overlap'])==1
    return hit,ready,hit and ready,eligible,full

def independent_sets(rows):
    ids={r['query_id'] for r in rows}
    gate=lambda key,threshold:{r['query_id'] for r in rows if r[key]!='' and D(r[key])>=D(threshold)}
    hit={r['query_id'] for r in rows if r['predicted_doc_id']==r['gold_doc_id']}
    ready=gate('param_coverage','.8')&gate('type_match','.8')
    full=hit&gate('param_coverage','1')&gate('type_match','1')&gate('exact20','1')&gate('key_overlap','1')
    return ids,hit,ready,hit&ready,full

def main():
    v=json.loads((RELEASE/'expected/SAVED_METRIC_CHECKS.json').read_text());manifest=json.loads((RELEASE/'release_manifest.json').read_text());hashes={r['path']:r['sha256'] for r in manifest['files']};report={'status':'PASS','evidence_commit':'209fa982c0a0e2087bd021cbc7a9f57d94256c8f','source_manifest_sha256':hashlib.sha256((RELEASE/'release_manifest.json').read_bytes()).hexdigest(),'definition':'F_G / I and F_G / S; exact all-key success as defined by original metric gates; query-level conditional descriptive fractions, not source-semantic accuracy','uncertainty':'No new significance test or confidence interval. Populations are selected by method-specific gates, share331 exposed queries and327 frozen text groups. No independent Bernoulli sampling assumed.','arms':{}}
    for arm,e in v['core'].items():
        path=next(p for p,h in v['inputs'].items() if h==e['sha256']);assert hashes[path]==e['sha256'];f=RELEASE/path;assert hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256']
        rr=list(csv.DictReader(f.open(newline='')));a=[indicators(r) for r in rr];ids,hits,ready,strict,full=independent_sets(rr)
        assert len(rr)==len(ids)==331
        for column,s in [(0,hits),(1,ready),(2,strict),(4,full)]:assert {r['query_id'] for r,x in zip(rr,a) if x[column]}==s
        assert full<=strict<=ready<=ids
        assert (len(ready),len(strict),len(full))==(e['coarse'],e['strict'],e['full_all_key'])
        report['arms'][arm]={'source_path':path,'source_sha256':e['sha256'],'population':331,'schema_hits':len(hits),'ready':len(ready),'strict':len(strict),'full_all_key':len(full),'full_eligible':sum(x[3] for x in a),'ready_and_full':len(ready&full),'strict_and_full':len(strict&full),'ready_not_full':len(ready-full),'strict_not_full':len(strict-full),'full_given_ready_percent':str(D(100)*D(len(full))/D(len(ready))),'full_given_strict_percent':str(D(100)*D(len(full))/D(len(strict))),'full_ids':sorted(full),'ready_ids':sorted(ready),'strict_ids':sorted(strict)}
    c=report['arms']['constrained'];assert c['full_eligible']==41 and c['full_all_key']==40 and 'nlp4lp_test_102' not in c['full_ids']
    x=report['arms']['ratio'];assert (x['ready_and_full'],x['ready_not_full'],x['strict_not_full'])==(7,258,248)
    # Boundary tests: eligible-only success, abstention, blank numerical evidence, and wrong retrieval.
    base={'schema_hit':'1','param_coverage':'1','type_match':'1','exact20':'1','key_overlap':'1'}
    assert indicators(base)==(True,True,True,True,True)
    assert indicators(dict(base,key_overlap='0.42857142857142855'))[-2:]==(True,False)
    assert indicators(dict(base,param_coverage='0.8'))[1:] == (True,True,False,False)
    assert indicators(dict(base,exact20=''))[-1] is False
    assert indicators(dict(base,schema_hit='0'))[1:] == (True,False,False,False)
    report['verification']={'independent_implementations':'float row indicators versus Decimal set intersections with schema-ID equality','matched_arms':10,'population_subset_checks':10,'boundary_tests':5,'new_model_calls':0,'new_solver_calls':0}
    (OUT/'DERIVED_DIAGNOSTICS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    labels=[('baseline','TF--IDF base'),('ratio','TF--IDF ratio'),('bge','BGE-M3'),('oracle','Oracle')]
    tex=[]
    for arm,label in labels:
        r=report['arms'][arm];tex.append(f"{label} & {r['full_all_key']}/{r['ready']} & {D(r['full_given_ready_percent']):.2f} & {r['full_all_key']}/{r['strict']} & {D(r['full_given_strict_percent']):.2f} \\\\")
    (ROOT/'manuscript/jdsa-stage2/ready_conditioned_rows.tex').write_text('\n'.join(tex)+'\n')
    main_tex=ROOT/'manuscript/jdsa-stage2/main.tex'
    text=main_tex.read_text()
    block='% BEGIN DERIVED READY CONDITIONED ROWS\n'+'\n'.join(tex)+'\n% END DERIVED READY CONDITIONED ROWS'
    text=re.sub(r'% BEGIN DERIVED READY CONDITIONED ROWS[\s\S]*?% END DERIVED READY CONDITIONED ROWS',lambda m:block,text)
    main_tex.write_text(text)
    print(json.dumps({k:{p:v[p] for p in ['ready','strict','full_all_key','full_given_ready_percent']} for k,v in report['arms'].items()},indent=2))
if __name__=='__main__':main()
