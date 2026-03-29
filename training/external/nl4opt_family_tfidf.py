"""
TF-IDF-based NL4Opt family benchmark (torch-free).

Families are approximate clusters of NL4Opt test descriptions connected in a
similarity graph built from TF-IDF cosine similarity. We then evaluate
retrieval over these families.

Steps:
  - Normalize descriptions (lowercase, numbers -> <NUM>, money -> <MONEY>,
    percents -> <PCT>, collapse whitespace).
  - Vectorize with TF-IDF (word 1–2-grams).
  - For a grid of similarity thresholds tau, build a graph with edges where
    cos_sim >= tau; families are connected components.
  - Keep families with size >= MIN_FAMILY_SIZE; choose the highest tau with
    kept_instances >= 200 if possible, else the tau with maximum kept_instances.

Outputs:
  - data/processed/nl4opt_family_eval_test.jsonl
  - data/processed/nl4opt_family_eval_test_masked.jsonl
  - data/processed/nl4opt_family_catalog.json
  - results/nl4opt_family_coverage.json
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
MIN_FAMILY_SIZE = 5
TAU_CANDIDATES = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]


def _load_nl4opt_test() -> List[dict]:
    path = ROOT / "data" / "raw" / "nl4opt" / "test.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[$£€]\\s*\\d[\\d.,]*", "<MONEY>", t)
    t = re.sub(r"\\d+(?:\\.\\d+)?\\s*%", "<PCT>", t)
    t = re.sub(r"\\d+(?:\\.\\d+)?", "<NUM>", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def _mask_description(text: str, ban_words: set[str]) -> str:
    """Harder masked version: remove numbers/currencies/percents and frequent content words."""
    s = text or ""
    s = re.sub(r"[$£€]\\s*\\d[\\d.,]*", " ", s)
    s = re.sub(r"\\d+(?:\\.\\d+)?\\s*%", " ", s)
    s = re.sub(r"\\d+(?:\\.\\d+)?", " ", s)
    # tokenize and drop top frequent content words (case-insensitive)
    tokens = re.findall(r"[A-Za-z]+", s)
    kept = [tok for tok in tokens if tok.lower() not in ban_words]
    s = " ".join(kept)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _build_similarity_graph(vectors, tau: float) -> List[List[int]]:
    """Return adjacency list for graph where edge(i,j) if cos_sim >= tau."""
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(vectors)
    n = sims.shape[0]
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= tau:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp = []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def main() -> None:
    items = _load_nl4opt_test()
    descriptions: List[str] = []
    for obj in items:
        desc = (obj.get("description") or "").strip()
        if not desc:
            continue
        descriptions.append(desc)
    total = len(descriptions)
    if not descriptions:
        print("No NL4Opt test descriptions found.")
        return

    # Build normalized templates
    templates = [_normalize(d) for d in descriptions]

    # Compute top-50 frequent content words (excluding stopwords)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    freq = Counter()
    for d in descriptions:
        toks = re.findall(r"[A-Za-z]+", d.lower())
        for t in toks:
            if t in ENGLISH_STOP_WORDS:
                continue
            freq[t] += 1
    top50 = [w for w, _ in freq.most_common(50)]
    ban_words = set(top50)

    # TF-IDF vectors (1–2-grams)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
    X = vec.fit_transform(templates)

    best_tau = None
    best_kept = 0
    best_comps: List[List[int]] = []

    for tau in TAU_CANDIDATES:
        adj = _build_similarity_graph(X, tau)
        comps = _connected_components(adj)
        kept_families = [c for c in comps if len(c) >= MIN_FAMILY_SIZE]
        kept_instances = sum(len(c) for c in kept_families)
        print(f"tau={tau:.2f}: families>={MIN_FAMILY_SIZE}={len(kept_families)}, kept_instances={kept_instances}")
        if kept_instances >= 200 and best_tau is None:
            best_tau = tau
            best_kept = kept_instances
            best_comps = kept_families
            break
        if kept_instances > best_kept:
            best_tau = tau
            best_kept = kept_instances
            best_comps = kept_families

    if best_tau is None or best_kept == 0:
        print("No families found with current settings; nl4opt_family benchmark will be empty.")
        cov = {
            "total": total,
            "kept_instances": 0,
            "num_families": 0,
            "avg_family_size": 0.0,
            "min_family_size": MIN_FAMILY_SIZE,
            "tau_used": None,
        }
        cov_path = ROOT / "results" / "nl4opt_family_coverage.json"
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cov_path, "w", encoding="utf-8") as f_cov:
            json.dump(cov, f_cov, indent=2)
        return

    print(f"Using tau={best_tau:.2f} with kept_instances={best_kept}")

    # Map index -> family_id
    eval_path = ROOT / "data" / "processed" / "nl4opt_family_eval_test.jsonl"
    masked_path = ROOT / "data" / "processed" / "nl4opt_family_eval_test_masked.jsonl"
    catalog_path = ROOT / "data" / "processed" / "nl4opt_family_catalog.json"
    cov_path = ROOT / "results" / "nl4opt_family_coverage.json"

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    cov_path.parent.mkdir(parents=True, exist_ok=True)

    families: Dict[str, List[int]] = {}
    for comp in best_comps:
        # representative normalized text is template of first index
        rep_template = templates[comp[0]]
        fam_id = hashlib.sha1(rep_template.encode("utf-8")).hexdigest()[:12]
        families[fam_id] = comp

    # Build catalog
    catalog = []
    for fam_id, idxs in families.items():
        reps = [descriptions[i] for i in idxs[:3]]
        rep_text = " ".join(reps)
        catalog.append(
            {
                "id": fam_id,
                "name": f"NL4Opt family {fam_id}",
                "aliases": [],
                "description": rep_text,
            }
        )

    # Write eval JSONL (normal + masked)
    kept_instances = 0
    with open(eval_path, "w", encoding="utf-8") as f_eval, open(
        masked_path, "w", encoding="utf-8"
    ) as f_masked:
        for fam_id, idxs in families.items():
            for i in idxs:
                desc = descriptions[i]
                kept_instances += 1
                f_eval.write(
                    json.dumps({"query": desc, "problem_id": fam_id}, ensure_ascii=False)
                    + "\n"
                )
                m = _mask_description(desc, ban_words)
                f_masked.write(
                    json.dumps({"query": m, "problem_id": fam_id}, ensure_ascii=False)
                    + "\n"
                )

    with open(catalog_path, "w", encoding="utf-8") as f_cat:
        json.dump(catalog, f_cat, indent=2, ensure_ascii=False)

    num_families = len(families)
    avg_size = kept_instances / num_families if num_families else 0.0
    cov = {
        "total": total,
        "kept_instances": kept_instances,
        "num_families": num_families,
        "avg_family_size": avg_size,
        "min_family_size": MIN_FAMILY_SIZE,
        "tau_used": best_tau,
    }
    with open(cov_path, "w", encoding="utf-8") as f_cov:
        json.dump(cov, f_cov, indent=2)

    print(
        f"NL4Opt TF-IDF family eval: total={total}, kept_instances={kept_instances}, "
        f"num_families={num_families}, avg_family_size={avg_size:.2f}, tau_used={best_tau:.2f}"
    )
    print(f"Wrote {eval_path}, {masked_path}, and {catalog_path}")


if __name__ == "__main__":
    main()

