"""Offline aggregation only: reads committed metric CSVs; never runs a model/decoder.

Run from any directory: python docs/jdsa_revision_phase1_20261029/verify_saved_metrics.py
Writes its deterministic report beside this script. No experimental code is imported.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument("--output",type=Path,required=True)
_args=_parser.parse_args()
ROOT=Path(__file__).resolve().parent
HERE=ROOT/"expected"


def aggregate(path):
    rows = list(csv.DictReader((ROOT / path).open()))
    assert len(rows) == 331 and len({r['query_id'] for r in rows}) == 331, path
    def num(r, key):
        return float(r[key]) if r.get(key, '') != '' else None
    hits = [r['schema_hit'] == '1' for r in rows]
    ready = [num(r, 'param_coverage') >= .8 and num(r, 'type_match') >= .8 for r in rows]
    full = [h and num(r, 'param_coverage') == 1 and num(r, 'type_match') == 1
            and num(r, 'exact20') == 1 for h, r in zip(hits, rows)]
    allkey = [f and num(r, 'key_overlap') == 1 for f, r in zip(full, rows)]
    exact = [num(r, 'exact20') for h, r in zip(hits, rows) if h and num(r, 'exact20') is not None]
    partition = [0] * 5
    for h, r in zip(hits, rows):
        idx = (0 if not h else 1 if num(r, 'param_coverage') < 1 else
               2 if num(r, 'type_match') < 1 else 3 if num(r, 'exact20') != 1 else 4)
        partition[idx] += 1
    return dict(n=len(rows), schema_hits=sum(hits), coarse=sum(ready),
                strict=sum(a and b for a, b in zip(hits, ready)),
                full_eligible=sum(full), full_all_key=sum(allkey),
                coverage=sum(num(r, 'param_coverage') for r in rows) / len(rows),
                type_match=sum(num(r, 'type_match') for r in rows) / len(rows),
                exact20_n=len(exact), exact20_macro=sum(exact) / len(exact) if exact else None,
                residual_partition=partition,
                sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest()), rows, allkey


paths = {
    'baseline': 'results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv',
    'ratio': 'results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv',
    'bge': 'results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv',
    'oracle': 'results/oracle_recomputation_2026-08-15/nlp4lp_downstream_per_query_orig_oracle.csv',
}
for name, suffix in [('typed', ''), ('constrained', '_constrained'), ('maxweight', '_max_weight_matching'),
                     ('role', '_optimization_role_repair'), ('structured', '_search_structured_grounding'),
                     ('semantic', '_semantic_ir_repair')]:
    paths[name] = 'results/stage6_matched_grounding_baselines_2026-08-27/nlp4lp_downstream_per_query_orig_tfidf' + suffix + '.csv'

report = {'scope': 'Saved-record aggregation; no prediction, inference, training, solver, or resampling runs.',
          'core': {}, 'retrieval': {}, 'historical_campaigns': {}, 'optional_llm': {}, 'inputs': {}}
details = {}
for name, path in paths.items():
    result, rows, success = aggregate(path)
    report['core'][name] = result
    details[name] = {r['query_id']: (r, s) for r, s in zip(rows, success)}
    report['inputs'][path] = result['sha256']

expected = {'baseline': (301, 257, 247, 6, 291), 'ratio': (301, 265, 255, 7, 291),
            'bge': (313, 273, 265, 3, 303), 'oracle': (331, 281, 281, 8, 320),
            'typed': (301, 265, 255, 7, 291), 'constrained': (301, 252, 242, 40, 291),
            'maxweight': (301, 246, 236, 37, 291), 'role': (301, 244, 234, 35, 291),
            'structured': (301, 233, 223, 46, 291), 'semantic': (301, 237, 230, 9, 285)}
for name, values in expected.items():
    assert tuple(report['core'][name][k] for k in ['schema_hits', 'coarse', 'strict', 'full_all_key', 'exact20_n']) == values, name
assert report['core']['ratio']['residual_partition'] == [30, 31, 49, 214, 7]
assert report['core']['constrained']['full_eligible'] == 41
gains = sum(details['constrained'][q][1] and not details['typed'][q][1] for q in details['typed'])
losses = sum(details['typed'][q][1] and not details['constrained'][q][1] for q in details['typed'])
assert (gains, losses) == (36, 3)
report['constrained_vs_typed'] = {'full_all_key_gains': gains, 'losses': losses}

for method in ['bm25', 'tfidf', 'lsa', 'bge_m3']:
    report['retrieval'][method] = {}
    for variant in ['orig', 'noisy', 'short']:
        folder = 'results/dense_retrieval_bge_m3' if method == 'bge_m3' else 'results/paper'
        path = f'{folder}/nlp4lp_downstream_per_query_{variant}_{method}.csv'
        result, _, _ = aggregate(path)
        report['inputs'][path] = result['sha256']
        report['retrieval'][method][variant] = result['schema_hits']
assert report['retrieval'] == {'bm25': {'orig': 292, 'noisy': 295, 'short': 254},
                             'tfidf': {'orig': 301, 'noisy': 299, 'short': 258},
                             'lsa': {'orig': 280, 'noisy': 294, 'short': 253},
                             'bge_m3': {'orig': 313, 'noisy': 312, 'short': 270}}

for folder in ['results/eswa_revision/02_downstream_postfix', 'results/baseline_staleness_audit_2026-08-12']:
    for p in sorted((ROOT / folder).glob('nlp4lp_downstream_per_query_*.csv')):
        path = str(p.relative_to(ROOT))
        result, _, _ = aggregate(path)
        report['historical_campaigns'][path] = result
        report['inputs'][path] = result['sha256']
for variant in ['orig', 'noisy', 'short']:
    paths['openai_' + variant] = f'results/paper/nlp4lp_downstream_per_query_{variant}_openai.csv'
paths['mistral_orig'] = 'results/rerun/mistral/run_903088/paper/nlp4lp_downstream_per_query_orig_mistral.csv'
for name in ['openai_orig', 'openai_noisy', 'openai_short', 'mistral_orig']:
    result, _, _ = aggregate(paths[name])
    report['optional_llm'][name] = result
    report['inputs'][paths[name]] = result['sha256']

# Check the previously reported exact tests, without new bootstrap/resampling runs.
groups = json.loads((HERE / 'PRESERVED_TEXT_GROUPS.json').read_text())
assert len(groups) == 331 and len(set(groups.values())) == 327
aliases = {'tfidf_ratio': 'ratio', 'bge_m3': 'bge',
           'constrained_assignment': 'constrained', 'max_weight_matching': 'maxweight',
           'optimization_role_repair': 'role', 'search_structured_grounding': 'structured',
           'semantic_ir_repair': 'semantic', 'baseline': 'baseline', 'oracle': 'oracle'}
def endpoint(row, name):
    hit = row['schema_hit'] == '1'
    ready = float(row['param_coverage']) >= .8 and float(row['type_match']) >= .8
    return int({'retrieval': hit, 'ready': ready, 'strict': hit and ready}[name])

stats = list(csv.DictReader((HERE / 'PRESERVED_PAIRED_STATISTICS.csv').open()))
for row in stats:
    left, right = (details[aliases[row[k]]] for k in ['left', 'right'])
    deltas = {q: endpoint(left[q][0], row['endpoint']) - endpoint(right[q][0], row['endpoint']) for q in left}
    gains, losses = sum(v == 1 for v in deltas.values()), sum(v == -1 for v in deltas.values())
    assert (gains, losses) == (int(row['gains']), int(row['losses']))
    assert math.isclose(sum(deltas.values()) / 331, float(row['difference']), abs_tol=1e-14)
    n = gains + losses
    query_p = min(1, 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)
    assert math.isclose(query_p, float(row['exact_query_McNemar_p']), abs_tol=1e-14)
    grouped = Counter()
    for q, delta in deltas.items():
        grouped[groups[q]] += delta
    distribution = Counter({0: 1})
    for delta in (abs(v) for v in grouped.values() if v):
        nxt = Counter()
        for value, count in distribution.items():
            nxt[value + delta] += count
            nxt[value - delta] += count
        distribution = nxt
    threshold = abs(sum(deltas.values()))
    group_p = sum(c for v, c in distribution.items() if abs(v) >= threshold) / sum(distribution.values())
    assert math.isclose(group_p, float(row['exact_group_swap_p']), abs_tol=1e-14)
adjusted = 0
for rank, row in enumerate(sorted(stats, key=lambda r: float(r['exact_group_swap_p']))):
    adjusted = max(adjusted, min(1, (len(stats) - rank) * float(row['exact_group_swap_p'])))
    assert math.isclose(adjusted, float(row['holm_p_16']), abs_tol=1e-14)
report['paired_statistics'] = {'contrasts_verified': len(stats), 'text_groups': 327,
    'checks': 'gains/losses, differences, exact McNemar, exact group swap, Holm over 16',
    'bootstrap_intervals': 'Archived intervals retained; no new resampling performed.'}

_args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(f'PASS: ten core arms; twelve retrieval cells; {len(report["historical_campaigns"])} historical metric files; four optional LLM files.')
