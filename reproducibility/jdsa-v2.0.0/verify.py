#!/usr/bin/env python3
"""Deterministic frozen-record reanalysis. No network, model or solver execution.

Use python -I verify.py --output /tmp/jdsa-report.json. Outputs never overwrite inputs.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,itertools,json,math,platform,random,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CHECKS=[]
def check(name,condition):
    CHECKS.append({'check':name,'status':'PASS' if condition else 'FAIL'})
    if not condition:raise AssertionError(name)
def js(p):return json.loads((ROOT/p).read_text())
def rows(p):return list(csv.DictReader((ROOT/p).open(newline='')))
def truth(v):return v is True or str(v).lower() in ['true','1']
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
def swap_p(values):
    dist=collections.Counter({0:1});values=[int(v) for v in values if v]
    for v in values:
        nxt=collections.Counter()
        for s,n in dist.items():nxt[s+v]+=n;nxt[s-v]+=n
        dist=nxt
    t=abs(sum(values));return sum(n for s,n in dist.items() if abs(s)>=t)/sum(dist.values())
def holm(ps):
    out=[0.]*len(ps);last=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):
        last=max(last,min(1.,(len(ps)-rank)*ps[i]));out[i]=last
    return out
def integrity():
    m=js('release_manifest.json');expected={r['path'] for r in m['files']}
    present={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    check('complete release file membership',present==expected|{'release_manifest.json','CHECKSUMS.sha256'})
    for r in m['files']:
        b=(ROOT/r['path']).read_bytes();check('hash '+r['path'],len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'])
    for line in (ROOT/'CHECKSUMS.sha256').read_text().splitlines():
        h,p=line.split('  ',1);check('checksum '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h)
def metric_checks(out):
    r=subprocess.run([sys.executable,'-I',str(ROOT/'verify_saved_metrics.py'),'--output',str(out/'metrics.json')],capture_output=True,text=True)
    check('97 frozen input files and main metric arms',r.returncode==0)
    v=json.loads((out/'metrics.json').read_text());expected=js('expected/SAVED_METRIC_CHECKS.json')
    for k in ['core','retrieval','historical_campaigns','optional_llm','inputs','constrained_vs_typed']:check('frozen aggregation '+k,v[k]==expected[k])
    return v
def stats_checks(metrics):
    paths={k:r['sha256'] for k,r in metrics['core'].items()}
    # Locate core arms by their authenticated byte hashes, not filename guessing.
    arms={}
    for k,h in paths.items():
        p=next(p for p,hh in metrics['inputs'].items() if hh==h);arms[k]={r['query_id']:r for r in rows(p)}
    groups=js('expected/PRESERVED_TEXT_GROUPS.json');ids=sorted(arms['ratio']);gs=sorted(set(groups.values()))
    check('aligned 331 / 327 groups',len(ids)==331 and len(gs)==327 and all(set(a)==set(ids) for a in arms.values()))
    aliases={'tfidf_ratio':'ratio','bge_m3':'bge','constrained_assignment':'constrained','max_weight_matching':'maxweight','optimization_role_repair':'role','search_structured_grounding':'structured','semantic_ir_repair':'semantic','baseline':'baseline','oracle':'oracle'}
    st=rows('expected/PRESERVED_PAIRED_STATISTICS.csv');check('16 primary contrasts',len(st)==16)
    rng=np.random.default_rng(42);w=rng.multinomial(len(gs),np.ones(len(gs))/len(gs),size=10000);sizes=np.array([sum(groups[q]==g for q in ids) for g in gs]);den=w@sizes
    def endpoint(r,k):
        hit=r['schema_hit']=='1';ready=float(r['param_coverage'])>=.8 and float(r['type_match'])>=.8
        return int({'retrieval':hit,'ready':ready,'strict':hit and ready}[k])
    recomputed=[]
    for r in st:
        a,b=arms[aliases[r['left']]],arms[aliases[r['right']]];d={q:endpoint(a[q],r['endpoint'])-endpoint(b[q],r['endpoint']) for q in ids}
        v=np.array([sum(d[q] for q in ids if groups[q]==g) for g in gs]);sample=(w@v)/den
        lo,hi=np.quantile(sample,[.025,.975]);p=swap_p(v)
        check('group bootstrap '+r['comparison'],close(lo,r['ci_low']) and close(hi,r['ci_high']))
        check('exact group swap '+r['comparison'],close(p,r['exact_group_swap_p']))
        recomputed.append(p)
    check('Holm primary family',all(close(a,r['holm_p_16']) for a,r in zip(holm(recomputed),st)))
    return {'contrasts':16,'resamples':10000,'seed':42,'numpy':np.__version__,'all_CIs_and_exact_tests_match':True}
def factorial_checks():
    rr=rows('results/grounding_followup_2026-10-27/audit/factorial_per_query.csv');by=collections.defaultdict(dict)
    for r in rr:check('unique factorial '+r['cell']+r['query_id'],r['query_id'] not in by[r['cell']]);by[r['cell']][r['query_id']]=r
    check('24 x 49 factorial population',len(by)==24 and len(rr)==1176 and all(len(v)==49 for v in by.values()))
    primary=set(next(iter(by.values())));check('excluded known overlap', 'nlp4lp_test_291' not in primary)
    check('identical factorial membership',all(set(v)==primary for v in by.values()))
    groups=js('expected/PRESERVED_TEXT_GROUPS.json')
    for cell,values in by.items():check('277 factorial slots '+cell,sum(int(r['n_gold_slots']) for r in values.values())==277)
    frozen=rows('results/grounding_followup_2026-10-27/audit/single_factor_paired_tests.csv');calculated=[]
    for left,right in itertools.combinations(sorted(by),2):
        if sum(a!=b for a,b in zip(left.split('|'),right.split('|')))!=1:continue
        for metric in ['exact_slots_correct','tolerant_slots_correct','full_exact','full_tolerant']:
            dif=collections.Counter()
            for q in sorted(primary):dif[groups[q]]+=int(by[left][q][metric])-int(by[right][q][metric])
            calculated.append({'left':left,'right':right,'metric':metric,'difference_count':sum(dif.values()),'exact_group_swap_p':swap_p(dif.values())})
    check('240 factorial contrasts',len(calculated)==len(frozen)==240)
    for r,h,p in zip(calculated,holm([r['exact_group_swap_p'] for r in calculated]),frozen):
        check('factorial contrast '+r['left']+r['right']+r['metric'],all(str(r[k])==p[k] for k in ['left','right','metric','difference_count']) and close(r['exact_group_swap_p'],p['exact_group_swap_p']) and close(h,p['holm_p_all_single_factor_endpoints']))
    # Original historical algorithm: Python Random(42), 2000 group draws,
    # sorted order statistics 50 and 1950 (not NumPy interpolated quantiles).
    saved=js('summaries/results/wulver_grounding_attribution_2026-10-27/slurm_1462366/summary.json')
    ids=sorted(primary);gkeys=sorted({groups[q] for q in ids});rng=random.Random(42)
    draws=np.array([[rng.randrange(len(gkeys)) for _ in gkeys] for _ in range(2000)])
    fields=['full_tolerant','full_exact','exact_slots_correct','tolerant_slots_correct','n_gold_slots']
    matrix={cell:np.array([[sum(int(v[q][f]) for q in ids if groups[q]==g) for f in fields]+[sum(groups[q]==g for q in ids)] for g in gkeys],dtype=float) for cell,v in by.items()}
    def rates(t):return {'full_tolerant_rate':t[...,0]/t[...,5],'full_exact_rate':t[...,1]/t[...,5],'exact_slot_accuracy':t[...,2]/t[...,4],'tolerant_slot_accuracy':t[...,3]/t[...,4]}
    for name,r in saved['paired_contrasts'].items():
        points=collections.defaultdict(float);dist=collections.defaultdict(lambda:np.zeros(2000))
        for cell,sign in r['terms']:
            for k,v in rates(matrix[cell].sum(axis=0)).items():points[k]+=sign*v
            for k,v in rates(matrix[cell][draws].sum(axis=1)).items():dist[k]+=sign*v
        for k,v in points.items():check('factorial point '+name+k,close(v,r['absolute_difference'][k]))
        for k,v in dist.items():check('factorial bootstrap '+name+k,all(close(a,b) for a,b in zip(np.sort(v)[[50,1950]],r['group_bootstrap95'][k])))
    checkpoints=js('data/assignment_controls.json');check('61 saved factorial checkpoints',len(checkpoints)==61)
    for cp in checkpoints:
        for cell,r in cp['cells'].items():
            pool,scorer,policy,decoder=cell.split('|');t=cp['matrix_traces'][pool];mat=t['scores'][scorer];choices=r['choice_indices'];non=[i for i in choices if i is not None and i>=0]
            check('assignment dimensions '+cp['query_id']+cell,len(choices)==len(mat)==t['n_slots'] and all(0<=i<t['n_candidates'] for i in non))
            if policy=='one_to_one_diagnostic':check('no mention reuse '+cp['query_id']+cell,len(non)==len(set(non)))
            objective=sum(mat[s][i] for s,i in enumerate(choices) if i is not None and i>=0)
            check('saved score '+cp['query_id']+cell,close(objective,r['objective']))
            if cp['primary']:
                saved=by[cell][cp['query_id']];check('checkpoint/ledger '+cp['query_id']+cell,all(int(saved[k])==r['metrics'][k] for k in ['exact_slots_correct','tolerant_slots_correct','full_exact','full_tolerant']))
    return {'queries':49,'slots':277,'cells':24,'contrasts':240,'checkpoint_count':61,'boundary':'Scores/indices/ledger checks only; source-dependent correctness and optimality not rerun'}
def label_stats(data):
    providers=['gemini','ionet','cohere'];ids=sorted(data);joint=[i for i in ids if all(data[i][p]['valid'] for p in providers)];marg=collections.Counter();po=0
    for i in joint:
        c=collections.Counter(data[i][p]['label'] for p in providers);marg.update(c);po+=sum(n*(n-1) for n in c.values())/6
    po/=len(joint);pe=sum((n/(3*len(joint)))**2 for n in marg.values());counts=collections.Counter();pairs={}
    for a,b in itertools.combinations(providers,2):
        eligible=[i for i in ids if data[i][a]['valid'] and data[i][b]['valid']];n=sum(data[i][a]['label']==data[i][b]['label'] for i in eligible);pairs[a+'__'+b]={'same':n,'n':len(eligible),'fraction':n/len(eligible)}
    for i in ids:
        rr=[data[i][p] for p in providers];ls=[r['label'] for r in rr]
        c='INVALID_OR_INCOMPLETE' if not all(r['valid'] for r in rr) else 'DISAGREEMENT' if len(set(ls))>1 else 'UNRESOLVED' if any(r['has_source_warning'] for r in rr) or ls[0] in ['AMBIGUOUS','CANNOT_DETERMINE'] else 'UNANIMOUS';counts[c]+=1
    return {'n_items':len(ids),'complete':len(joint),'valid':{p:sum(data[i][p]['valid'] for i in ids) for p in providers},'pairwise':pairs,'fleiss_kappa':(po-pe)/(1-pe),'classifications':dict(counts),'valid_label_distributions':{p:dict(collections.Counter(data[i][p]['label'] for i in ids if data[i][p]['valid'])) for p in providers}}
def relationship_checks():
    rr=js('data/relationship_stages.json');st=collections.defaultdict(lambda:collections.defaultdict(dict))
    for r in rr:
        check('unique relationship '+r['stage']+r['item_id']+r['provider'],r['provider'] not in st[r['stage']][r['item_id']]);st[r['stage']][r['item_id']][r['provider']]=r
    expected=js('data/llm217_validation.json');result={}
    for name,data in st.items():
        check('217 x 3 '+name,len(data)==217 and all(set(v)=={'gemini','ionet','cohere'} for v in data.values()));actual=label_stats(data);e=expected['stages'][name]
        for k,v in actual.items():check('relationship '+name+k,close(v,e[k]) if isinstance(v,float) else v==e[k])
        result[name]=actual
    for r in rows('data/llm217_per_judgment.csv'):
        f=st['recovery_v2_1'][r['item_id']][r['provider']];check('published judgment ledger '+r['item_id']+r['provider'],truth(r['final_valid'])==f['valid'] and (not f['valid'] or r['final_label']==f['label']))
    requests=rows('data/llm217_requests.csv');counts=dict(collections.Counter(f"{r['stage']}:{r['provider']}:{r['http_status']}" for r in requests));check('HTTP request denominators',counts==expected['request_status_counts'])
    final=st['recovery_v2_1'];clusters=collections.defaultdict(dict)
    for i,vals in final.items():clusters[str(vals['gemini']['problem_id'])][i]=vals
    frozen=js('expected/manuscript_cells.json')['relationship_clusters']
    check('16 relationship source problems',len(clusters)==16)
    for problem,vals in clusters.items():
        c=collections.Counter();complete=0
        for vv in vals.values():
            rr=list(vv.values());labels=[r['label'] for r in rr];valid=all(r['valid'] for r in rr);complete+=valid
            key='INVALID_OR_INCOMPLETE' if not valid else 'DISAGREEMENT' if len(set(labels))>1 else 'UNRESOLVED' if any(r['has_source_warning'] for r in rr) or labels[0] in ['AMBIGUOUS','CANNOT_DETERMINE'] else 'UNANIMOUS';c[key]+=1
        got=[len(vals),complete]+[c.get(k,0) for k in ['UNANIMOUS','DISAGREEMENT','UNRESOLVED','INVALID_OR_INCOMPLETE']]
        check('Supplement Table S18 '+problem,got==frozen[problem])
    result['boundary']='Reanalysis of validity/label ledgers. Original HTTP/schema validation passed locally (5120 checks); raw response/prose not redistributed.'
    return result
def common18_checks():
    rr=rows('data/common18_cases.csv');by=collections.defaultdict(list)
    for r in rr:by[r['system']].append(r)
    result={};expected=js('expected/common18_summary.json')
    for system,vals in by.items():
        check('common18 unique '+system,len(vals)==len({r['case_id'] for r in vals})==18)
        counts=dict(collections.Counter(r['corrected_classification'] for r in vals));check('common18 dispositions '+system,counts==expected[system]['classification_counts'])
        if system in ['optmath','generic']:
            for r in vals:
                if not truth(r['comparison_eligible']):continue
                a,b=round(float(r['objective_token'])),round(float(r['reference_objective']));match=abs(a-b)<=.05 if b==0 else abs(a-b)/abs(b)<=.05
                check('common18 rounded predicate '+system+r['case_id'],match==truth(r['numerical_agreement']))
        result[system]={'population':18,'classifications':counts}
    check('PaMOP precision withheld',expected['pamop']['aggregate_objective_published'] is False)
    return result
def other_checks():
    pilot=rows('results/joint_grounding_2026-10-27/pilot_per_query.csv');by=collections.defaultdict(list)
    for r in pilot:by[r['method']].append(r)
    check('6 x 50 pilot',len(by)==6 and len(pilot)==300)
    pilot_sums={}
    for method,rr in by.items():
        check('pilot population '+method,len({r['query_id'] for r in rr})==50 and sum(int(r['n_gold_slots']) for r in rr)==280)
        pilot_sums[method]={k:sum(int(r[k]) for r in rr) for k in ['exact_slots_correct','tolerant_slots_correct','full_exact','full_tolerant']}
    manuscript_pilot={'independent_type':(52,57,1,1),'independent_text':(128,139,1,2),'joint_text':(119,130,1,2),'independent_text_units':(134,144,2,3),'joint_text_units':(126,137,2,3),'joint_explicit':(126,137,2,3)}
    for method,expected in manuscript_pilot.items():check('Supplement Table S23 '+method,tuple(pilot_sums[method][k] for k in ['exact_slots_correct','tolerant_slots_correct','full_exact','full_tolerant'])==expected)
    distinct=rows('results/grounding_followup_2026-10-27/slurm_1462557/distinct_retention_per_query.csv')
    check('distinct 49 cases',len(distinct)==len({r['query_id'] for r in distinct})==49)
    check('distinct retained 9 / selected 3 / 23 historical solves',sum(truth(r['distinct_nbest_correct']) for r in distinct)==9 and sum(truth(r['selected_correct_unchanged']) for r in distinct)==3 and sum(int(r['new_solver_calls']) for r in distinct)==23)
    check('distinct zero gains/losses',all(r['saved_exact_nbest_correct']==r['distinct_nbest_correct'] for r in distinct))
    stage=rows('results/sncs_corrections_2026-10-27/cached_stage_a_reference_reanalysis.csv');check('Stage A full 331 frame',len(stage)==len({r['query_id'] for r in stage})==331)
    totals={k:sum(int(truth(r[k])) if r[k] in ['True','False'] else int(r[k]) for r in stage) for k in stage[0] if k!='query_id'}
    e=js('summaries/results/sncs_corrections_2026-10-27/cached_stage_a_reference_summary.json')['totals'];check('Stage A reference-only reanalysis',totals==e)
    cases=js('data/d1_cases.json');dec=rows('data/d1_author_decisions.csv');pairs={r['Pair ID']:r for r in dec};check('D1 24 samples / 12 pairs / 8 categories',len(cases)==len({r['case_id'] for r in cases})==24 and len(pairs)==12 and len({r['category'] for r in cases})==8)
    labels=collections.Counter()
    for r in cases:
        d=pairs[r['pair_id']];label=d['Final author labels (base/contrast)'].split('/')[0 if r['variant']=='base' else 1];labels[label]+=1;check('D1 accepted '+r['case_id'],d['Decision']=='ACCEPT')
    check('D1 label balance',dict(labels)=={'CORRECT':12,'INCORRECT':12})
    check('D1 two equivalence controls',sum(r['Contrast type']=='MEANING_PRESERVING' for r in dec)==2)
    g3=js('data/g3_counts.json');ga=js('summaries/g3_inventory.json')['aggregate']
    check('G3 331 members',len(g3)==len({r['query_id'] for r in g3})==331)
    for field,key in [('reference_scalar_count','reference_scalar_slots'),('schema_scalar_count','schema_scalar_slots'),('intersection_count','intersection_slots'),('additional_count','additional_schema_slots'),('missing_gold_count','missing_gold_slots')]:check('G3 '+key,sum(r[field] for r in g3)==ga[key])
    check('G3 301 equal / 30 unequal',sum(r['additional_count']==0 for r in g3)==301)
    ceilings=rows('results/joint_grounding_2026-10-27/candidate_recall_per_query.csv');cb=collections.defaultdict(list)
    for r in ceilings:cb[r['k']].append(r)
    expected_ceilings={'1':(773,823,22,23),'3':(1227,1256,87,95),'5':(1379,1404,146,155),'8':(1456,1464,195,199),'all':(1467,1473,203,206)}
    for k,rr in cb.items():
        check('candidate ceiling population '+k,len(rr)==331 and sum(int(r['n_gold_slots']) for r in rr)==1847)
        got=(sum(int(r['exact_recalled_slots']) for r in rr),sum(int(r['tolerant_recalled_slots']) for r in rr),sum(truth(r['all_exact_candidates']) for r in rr),sum(truth(r['all_tolerant_candidates']) for r in rr))
        check('candidate ceiling '+k,got==expected_ceilings[k])
    return {'pilot':pilot_sums,'distinct_retention':{'queries':49,'retained':9,'selected':3},'stage_a':totals,'d1':{'pairs':12,'samples':24,'categories':8,'labels':dict(labels),'provider_predictions':0}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
    if out.is_relative_to(ROOT):raise SystemExit('Output must be outside the immutable release directory')
    report={'schema_version':'1','release':'jdsa-evidence-v2.0.0','scope':'Frozen-record reanalysis; not full experimental reproduction','python':platform.python_version(),'numpy':np.__version__,'new_model_calls':0,'new_solver_calls':0}
    try:
        integrity()
        with tempfile.TemporaryDirectory(prefix='jdsa-check-') as tmp:
            metrics=metric_checks(Path(tmp));report['metrics']=metrics;report['primary_statistics']=stats_checks(metrics)
        report['factorial']=factorial_checks();report['relationships']=relationship_checks();report['common18']=common18_checks();report['additional']=other_checks();report['status']='PASS'
    except Exception as e:
        report['status']='FAIL';report['error_type']=type(e).__name__;report['error']=str(e)
    report['checks']=CHECKS;report['checks_passed']=sum(c['status']=='PASS' for c in CHECKS);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:report[k] for k in ['status','checks_passed']}));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
