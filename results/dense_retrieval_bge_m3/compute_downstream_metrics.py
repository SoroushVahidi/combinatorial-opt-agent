import csv
import json
import random

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

tfidf = read_csv("results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv")
bgem3 = read_csv("results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv")

def extract_metrics(rows):
    n = len(rows)
    cov_sum = 0
    tm_sum = 0
    exact20_sum = 0
    hits = 0
    ir = 0
    strict_ir = 0
    for r in rows:
        hit = int(float(r.get("schema_hit", 0) or 0))
        cov = float(r.get("param_coverage", 0) or 0)
        tm = float(r.get("type_match", 0) or 0)
        exact20 = float(r.get("exact20", 0) or 0)
        
        cov_sum += cov
        tm_sum += tm
        if hit == 1:
            exact20_sum += exact20
            hits += 1
        
        if cov >= 0.8 and tm >= 0.8:
            ir += 1
            if hit == 1:
                strict_ir += 1
                
    return {
        "Coverage": cov_sum / n,
        "TypeMatch": tm_sum / n,
        "Exact20_on_hits": exact20_sum / hits if hits > 0 else 0,
        "InstantiationReady": ir / n,
        "StrictInstantiationReady": strict_ir / n
    }

print("=== DOWNSTREAM ===")
print("TF-IDF:", json.dumps(extract_metrics(tfidf), indent=2))
print("BGE-M3:", json.dumps(extract_metrics(bgem3), indent=2))

def get_ir(rows):
    return [1.0 if (float(r.get("param_coverage", 0) or 0) >= 0.8 and float(r.get("type_match", 0) or 0) >= 0.8) else 0.0 for r in rows]

def get_strict_ir(rows):
    return [1.0 if (float(r.get("param_coverage", 0) or 0) >= 0.8 and float(r.get("type_match", 0) or 0) >= 0.8 and int(float(r.get("schema_hit", 0) or 0)) == 1) else 0.0 for r in rows]

def paired_bootstrap_test(vals_a, vals_b, B=10000, seed=42):
    rng = random.Random(seed)
    n = min(len(vals_a), len(vals_b))
    a, b = vals_a[:n], vals_b[:n]
    obs_diff = sum(a) / n - sum(b) / n
    diffs = [av - bv for av, bv in zip(a, b)]
    null_diffs = []
    for _ in range(B):
        s = 0.0
        for d in diffs:
            s += d if rng.random() > 0.5 else -d
        null_diffs.append(s / n)
    null_diffs.sort()
    lo = null_diffs[int(B * 0.025)]
    hi = null_diffs[int(B * 0.975)]
    p = sum(1 for nd in null_diffs if abs(nd) >= abs(obs_diff)) / B
    return obs_diff, obs_diff + lo, obs_diff + hi, p

bge_ir = get_ir(bgem3)
tfidf_ir = get_ir(tfidf)
bge_strict_ir = get_strict_ir(bgem3)
tfidf_strict_ir = get_strict_ir(tfidf)

diff, lo, hi, p = paired_bootstrap_test(bge_ir, tfidf_ir, B=10000, seed=42)
print(f"IR Diff (BGE-M3 vs TFIDF): {diff:.4f}, 95% CI: [{lo:.4f}, {hi:.4f}], p={p:.4f}")

diff, lo, hi, p = paired_bootstrap_test(bge_strict_ir, tfidf_strict_ir, B=10000, seed=42)
print(f"StrictIR Diff (BGE-M3 vs TFIDF): {diff:.4f}, 95% CI: [{lo:.4f}, {hi:.4f}], p={p:.4f}")

print("=== RETRIEVAL McNEMAR ===")
bge_hits = [int(float(r.get("schema_hit", 0) or 0)) for r in bgem3]
tfidf_hits = [int(float(r.get("schema_hit", 0) or 0)) for r in tfidf]
both_correct = 0
bge_only = 0
tfidf_only = 0
both_wrong = 0
for b, t in zip(bge_hits, tfidf_hits):
    if b == 1 and t == 1: both_correct += 1
    elif b == 1 and t == 0: bge_only += 1
    elif b == 0 and t == 1: tfidf_only += 1
    else: both_wrong += 1
print(f"Both correct: {both_correct}")
print(f"BGE only: {bge_only}")
print(f"TFIDF only: {tfidf_only}")
print(f"Both wrong: {both_wrong}")
from scipy.stats import binom
b = min(bge_only, tfidf_only)
n = bge_only + tfidf_only
p = 2 * binom.cdf(b, n, 0.5) if n > 0 else 1.0
print(f"McNemar p-value: {p:.4f}")

