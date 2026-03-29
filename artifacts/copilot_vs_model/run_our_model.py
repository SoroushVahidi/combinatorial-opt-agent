#!/usr/bin/env python3
"""Run our model on all benchmark cases.

Improvements over v1:
  - Step 2: catalog entries for open-domain schemas have type prefixes → no cross-contamination.
  - Step 3: slot_aware_extraction() — proximity-scored, CamelCase-split matching replaces
            global_consistency_grounding for hand-crafted cases with known numeric parameters.
  - Step 5: hybrid_rank() — Reciprocal Rank Fusion of BM25 + TF-IDF for all queries;
            BM25 handles short queries better than TF-IDF alone.

Writes: artifacts/copilot_vs_model/our_model_outputs.jsonl

Usage:
    python artifacts/copilot_vs_model/run_our_model.py

Requirements:
    pip install scikit-learn rank_bm25
No GPU, no internet, no HF auth required.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

BENCH_FILE = ROOT / "artifacts" / "copilot_vs_model" / "benchmark_cases.jsonl"
OUT_FILE   = ROOT / "artifacts" / "copilot_vs_model" / "our_model_outputs.jsonl"
CATALOG    = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"

STOPWORDS = {
    'The','Each','All','Maximize','Minimize','Given','To','No','In','If','It',
    'On','An','Or','By','At','For','Is','Are','Let','We','Determine','Find',
    'Calculate','Compute','Sum','Also','Total','Note','Both','With','When',
    'They','This','That','These','Then','Thus','Any','One','Two',
}

# ── Step 5: Hybrid BM25 + TF-IDF retrieval via Reciprocal Rank Fusion ─────────

class HybridRetriever:
    """Reciprocal Rank Fusion of BM25 and TF-IDF.

    For each query both retrievers are queried over a large candidate pool (top_k * 10
    or 200 docs) and the per-document scores are combined as:
        rrf_score(doc) = 1/(k + rank_bm25) + 1/(k + rank_tfidf)
    where k=60 (standard RRF constant).
    This fusion is especially helpful for short/noisy queries where BM25 outperforms
    TF-IDF by placing the correct document higher in the ranking.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        self._bm25  = None
        self._tfidf = None
        self._rrf_k = rrf_k

    def fit(self, catalog: list[dict]) -> "HybridRetriever":
        from retrieval.baselines import get_baseline
        self._bm25  = get_baseline("bm25").fit(catalog)
        self._tfidf = get_baseline("tfidf").fit(catalog)
        return self

    def rank(self, query: str, top_k: int = 3, pool: int = 200) -> list[tuple[str, float]]:
        pool = max(pool, top_k * 10)
        bm25_res  = self._bm25.rank(query, top_k=pool)
        tfidf_res = self._tfidf.rank(query, top_k=pool)
        bm25_rank  = {pid: r for r, (pid, _) in enumerate(bm25_res,  1)}
        tfidf_rank = {pid: r for r, (pid, _) in enumerate(tfidf_res, 1)}
        k = self._rrf_k
        all_pids = set(bm25_rank) | set(tfidf_rank)
        rrf_scores = {
            pid: 1 / (k + bm25_rank.get(pid, pool + 1))
               + 1 / (k + tfidf_rank.get(pid, pool + 1))
            for pid in all_pids
        }
        # Primary sort: RRF score (descending).
        # Tiebreaker: TF-IDF rank (ascending) so that when two documents have
        # identical RRF scores the one ranked higher by TF-IDF wins.  This
        # avoids non-deterministic results caused by Python set-iteration order.
        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: (-x[1], tfidf_rank.get(x[0], pool + 1)),
        )
        return [(pid, sc) for pid, sc in ranked[:top_k]]


# ── Step 3: Slot-aware value extraction ───────────────────────────────────────

_NUM_RE = re.compile(r"^\$?-?[\d]+(?:\.\d+)?%?$")
_SENTENCE_BREAKS = frozenset({".", "!", "?", ";"})

# English word-numbers that should be treated as numeric tokens during
# slot-to-value matching.  This allows phrases such as "at least a half"
# (ratio = 0.5), "two different milk teas" (count = 2), etc. to be matched
# to appropriate slots without requiring explicit digit tokens in the text.
_WORD_NUMS: dict[str, float] = {
    "half": 0.5,
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
    "ten": 10.0, "eleven": 11.0, "twelve": 12.0, "fifteen": 15.0,
    "twenty": 20.0, "thirty": 30.0, "forty": 40.0, "fifty": 50.0,
}

# Synonyms for common slot-name keyword components.  When scoring the fit
# between a slot and a nearby number, a synonym match earns SYNONYM_WEIGHT
# of the positional weight; a prefix/stem match earns PREFIX_WEIGHT.
_SLOT_SYNONYMS: dict[str, frozenset[str]] = {
    "min":      frozenset({"min", "least", "minimum", "demand", "require", "requires", "need"}),
    "max":      frozenset({"max", "maximum", "exceed", "most"}),
    "total":    frozenset({"total", "most", "all", "available", "capacity"}),
    "cost":     frozenset({"cost", "costs", "price", "charge", "sold", "sell", "selling"}),
    "profit":   frozenset({"profit", "earn", "earns", "revenue", "sold", "gain"}),
    "labor":    frozenset({"labor", "labour", "work"}),
    "return":   frozenset({"return", "yield", "yields"}),
    "fraction": frozenset({"fraction", "percent", "percentage", "ratio", "rate", "portion"}),
    "rate":     frozenset({"rate", "percent", "percentage", "ratio", "annual"}),
    "budget":   frozenset({"budget", "total", "funds", "capital"}),
    "supply":   frozenset({"supply", "supplies", "provides", "can"}),
    "demand":   frozenset({"demand", "demands", "require", "requires", "need", "needs"}),
    "investment": frozenset({"investment", "invest", "invested", "total"}),
    "price":    frozenset({"price", "prices", "cost", "costs", "sold", "sell", "selling", "charge"}),
    "num":      frozenset({"num", "number", "count", "types", "type", "kinds", "different",
                           "two", "three", "four", "five", "six"}),
}


def _split_camel(name: str) -> list[str]:
    """Split a CamelCase slot name into lowercase keyword tokens.

    Handles compound tokens such as W1, C2 (warehouse/customer labels) as well as
    standard CamelCase words.  Drops trivial connective words.

    Examples
    --------
    "ProfitPerChair"  → ["profit", "chair"]
    "TotalLaborHours" → ["total", "labor", "hours"]
    "SupplyW1"        → ["supply", "w1"]
    "CostW1C1"        → ["cost", "w1", "c1"]
    "MinBondFraction" → ["min", "bond", "fraction"]
    """
    noise = {"per", "the", "a", "an", "and", "or", "of", "in", "is", "at", "by"}
    # Note: this noise set is distinct from the module-level STOPWORDS constant.
    # STOPWORDS filters CamelCase tokens extracted from schema text; this noise
    # set filters trivial connective words from slot names during scoring.
    # Capture label tokens like W1, C2 first, then normal CamelCase words
    parts = re.findall(r"[A-Z][0-9]+|[A-Z][a-z]*", name)
    return [p.lower() for p in parts if p.lower() not in noise]


def slot_aware_extraction(query: str, slots: list[str]) -> dict[str, float]:
    """Proximity-scored slot-to-value matching with synonym expansion.

    For every numeric token in the query, each slot is scored by how close (in
    tokens) the slot's CamelCase-split keyword tokens are to that number.
    Sentence-boundary punctuation is preserved so that context from a different
    sentence cannot contaminate a number's score.  A greedy best-first assignment
    (no slot reuse, no text-position reuse) produces the final mapping.

    Value reuse policy: the same numeric *value* may be assigned to multiple
    slots when those values appear at distinct text positions (e.g. "10 mg of
    gold … at least 10 long cables" → GoldPerLong=10 AND MinLongCables=10).
    Each text position is used at most once, so a single token occurrence
    cannot be claimed by two different slots.

    Scoring weights:
      - Exact token match:   positional weight  = (WINDOW − distance) / WINDOW
      - Synonym match:       positional weight  × SYNONYM_WEIGHT (0.70)
      - Prefix/stem match:   positional weight  × PREFIX_WEIGHT  (0.50)

    Word-numbers ("half", "two", "three", …) are included in the candidate
    pool as if they were digit tokens, enabling extraction of counts and
    fractions expressed in prose (e.g. "at least a half" → 0.5).
    """
    WINDOW         = 7    # tokens either side of the number to consider
    SYNONYM_WEIGHT = 0.70
    PREFIX_WEIGHT  = 0.50

    # Keep sentence-ending punctuation as tokens so cross-sentence context
    # is blocked; commas are stripped so "500,000" parses as a single number.
    raw = re.sub(r"\s+", " ", query.lower().replace(",", ""))
    tokens: list[str] = re.findall(r"\$?-?[\d]+(?:\.\d+)?%?|[.!?;]|\w+", raw)

    # Locate numeric tokens and parse their float values.
    # Word-numbers (e.g. "two" → 2.0, "half" → 0.5) are included alongside
    # ordinary digit tokens so that prose counts and fractions are reachable.
    nums: list[tuple[float, int]] = []
    for idx, tok in enumerate(tokens):
        if tok in _SENTENCE_BREAKS:
            continue
        if _NUM_RE.match(tok):
            s = tok.lstrip("$")
            is_pct = s.endswith("%")
            try:
                v = float(s.rstrip("%"))
                if is_pct:
                    v /= 100.0
                nums.append((v, idx))
            except ValueError:
                pass
        elif tok in _WORD_NUMS:
            nums.append((_WORD_NUMS[tok], idx))

    if not nums or not slots:
        return {}

    def _has_boundary(i: int, j: int) -> bool:
        """Return True if a sentence-break token lies strictly between i and j."""
        lo, hi = (i, j) if i < j else (j, i)
        return any(tokens[k] in _SENTENCE_BREAKS for k in range(lo + 1, hi))

    def context_score(num_idx: int, keywords: list[str]) -> float:
        score = 0.0
        for kw in keywords:
            kw_set = _SLOT_SYNONYMS.get(kw, frozenset()) | {kw}
            best = 0.0
            for offset in range(-WINDOW, WINDOW + 1):
                j = num_idx + offset
                if j < 0 or j >= len(tokens) or j == num_idx:
                    continue
                t = tokens[j]
                if t in _SENTENCE_BREAKS or _NUM_RE.match(t):
                    continue
                if _has_boundary(num_idx, j):
                    continue  # skip cross-sentence tokens
                # Positional weight: preceding tokens carry full weight;
                # following tokens carry half weight (they may belong to the
                # next entity in a list, e.g. "W1 supply 80 units AND W2…").
                base_w = max(0.0, (WINDOW - abs(offset)) / WINDOW)
                if j > num_idx:
                    base_w *= 0.5
                if t == kw:
                    best = max(best, base_w)
                elif t in kw_set:
                    best = max(best, base_w * SYNONYM_WEIGHT)
                elif t.startswith(kw) or (len(kw) >= 3 and t.startswith(kw[:len(t)])):
                    # Bidirectional prefix match: t is a prefix of kw, or kw is a
                    # prefix of t.  Handles plurals ('chairs'/'chair'), inflections
                    # ('demands'/'demand'), and truncated tokens ('lab'/'labor').
                    best = max(best, base_w * PREFIX_WEIGHT)
            score += best
        return score

    slot_kws = {s: _split_camel(s) for s in slots}

    # Build (score, slot, value, text_position) tuples for all viable pairings.
    # Tracking the text position (rather than only the float value) allows the
    # same numeric value that appears at multiple points in the text (e.g. two
    # occurrences of "10") to be assigned to two different slots.
    candidates: list[tuple[float, str, float, int]] = []
    for slot in slots:
        kws = slot_kws[slot]
        if not kws:
            continue
        for val, pos in nums:
            sc = context_score(pos, kws)
            if sc > 0:
                candidates.append((sc, slot, val, pos))

    # Greedy assignment: highest-scoring pair first.
    # Constraints: each slot is used at most once; each text position is used
    # at most once.  The same float value CAN appear in two assignments when
    # it occurs at distinct positions in the text.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_slots:     set[str] = set()
    used_positions: set[int] = set()
    result: dict[str, float] = {}
    for sc, slot, val, pos in candidates:
        if slot not in used_slots and pos not in used_positions:
            result[slot] = val
            used_slots.add(slot)
            used_positions.add(pos)

    # Fallback sharing: for any slot that was not assigned above, allow it to
    # share an already-used text position if its score there is strong enough.
    # This handles cases where two related slots map to the same numeric value
    # (e.g. "TotalGold" and "There" both wanting 1000) without allowing
    # low-quality matches to contaminate the assignment.
    SHARING_THRESHOLD = 0.40
    for sc, slot, val, pos in candidates:
        if slot not in used_slots and sc >= SHARING_THRESHOLD:
            result[slot] = val
            used_slots.add(slot)

    return result


# ── Shared helpers ─────────────────────────────────────────────────────────────

def extract_clean_slots(text: str) -> list[str]:
    raw = re.findall(r'\b([A-Z][A-Za-z][A-Za-z0-9]*)\b', text)
    seen: set[str] = set()
    out = []
    for t in raw:
        if t not in STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def infer_objective(schema_text: str, query: str) -> str:
    q_lo = query.lower()
    s_lo = schema_text.lower()
    if "maximize" in s_lo or "maximize" in q_lo:
        return "maximize"
    if "minimize" in s_lo or "minimize" in q_lo:
        return "minimize"
    return "unknown"


# ── Main runner ────────────────────────────────────────────────────────────────

def _augment_catalog(catalog: list[dict]) -> list[dict]:
    """Expand CamelCase identifiers in catalog descriptions to natural-language tokens.

    Many NLP4LP schema descriptions consist of LP formulation expressions with
    CamelCase slot names like ``SeniorWage``, ``NumberOfSeniorCitizens``.  BM25
    and TF-IDF tokenize these as single opaque tokens and therefore fail to match
    natural-language query words such as "senior", "wage", "workers".

    This function appends word-split expansions for every CamelCase identifier
    found in the description so that, e.g., "SeniorWage" contributes the extra
    tokens "senior" and "wage" to the searchable text without changing the
    original description.
    """
    _NOISE = frozenset({"of", "the", "a", "an", "and", "or", "in", "is", "at",
                        "by", "to", "on", "for", "per", "with"})

    def _camel_to_words(name: str) -> list[str]:
        parts = re.findall(r"[A-Z][a-z0-9]*", name)
        return [p.lower() for p in parts if p.lower() not in _NOISE and len(p) > 1]

    result = []
    for entry in catalog:
        desc = entry.get("description", "")
        camel_ids = re.findall(r"\b[A-Z][A-Za-z][A-Za-z0-9]*\b", desc)
        extra: list[str] = []
        seen: set[str] = set()
        for cid in camel_ids:
            for w in _camel_to_words(cid):
                if w not in seen:
                    extra.append(w)
                    seen.add(w)
        if extra:
            augmented = dict(entry)
            augmented["description"] = desc + " " + " ".join(extra)
            result.append(augmented)
        else:
            result.append(entry)
    return result


def run() -> None:
    from tools.nlp4lp_downstream_utility import (
        _load_catalog_as_problems,
        _run_global_consistency_grounding,
    )

    # Load catalog
    catalog, id_to_text = _load_catalog_as_problems(CATALOG)

    # Step 5: use hybrid BM25 + TF-IDF retriever for all cases.
    # The catalog descriptions are augmented with CamelCase-split words so that
    # LP formula identifiers (e.g. "SeniorWage") can be matched by natural-language
    # query terms ("senior", "wage").
    augmented_catalog = _augment_catalog(catalog)
    retriever = HybridRetriever().fit(augmented_catalog)

    # Load benchmark
    cases = [json.loads(l) for l in BENCH_FILE.open()]
    print(f"Running {len(cases)} benchmark cases …")

    results = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        query    = case["input_text"]
        case_id  = case["case_id"]
        gold_sid = case.get("gold_schema_id", "")

        # Stage 1 – schema retrieval (hybrid BM25+TF-IDF)
        ranked      = retriever.rank(query, top_k=3)
        pred_id     = ranked[0][0] if ranked else ""
        pred_scores = [(pid, round(sc, 6)) for pid, sc in ranked[:3]]
        schema_text  = id_to_text.get(pred_id, "")
        schema_correct = int(pred_id == gold_sid)

        # Stage 2 – slot name extraction from predicted schema
        slots = extract_clean_slots(schema_text)

        # Stage 3 – numeric grounding
        # Step 3: for queries with literal numbers, use slot_aware_extraction
        # (proximity-scored CamelCase-split matching).  Fall back to
        # global_consistency_grounding for NLP4LP-style parameterised schemas
        # where the query uses <num> placeholders or has no extractable numbers.
        filled_values: dict = {}
        diagnostics:   dict = {}
        grounding_method = "none"

        if slots:
            # Try slot-aware extraction first.
            # slot_aware_extraction returns {} (falsy) when the query contains no
            # parseable numeric tokens — e.g. NLP4LP queries that use '<num>'
            # placeholders.  In that case, fall back to global_consistency_grounding.
            sae_result = slot_aware_extraction(query, slots)
            if sae_result:
                filled_values = sae_result
                grounding_method = "slot_aware_extraction"
            else:
                # Fall back to global_consistency_grounding
                try:
                    filled_values, _, diagnostics = _run_global_consistency_grounding(
                        query, "orig", slots
                    )
                    grounding_method = "global_consistency_grounding"
                except Exception as exc:
                    diagnostics = {"error": str(exc)}

        # Collect output
        elapsed = time.perf_counter() - t0
        out = {
            "case_id": case_id,
            "input_text": query,
            "gold_schema_id": gold_sid,
            # retrieval
            "predicted_schema_id": pred_id,
            "retrieval_top3": pred_scores,
            "schema_correct": schema_correct,
            # schema
            "predicted_schema_text": schema_text,
            "predicted_objective_direction": infer_objective(schema_text, query),
            # grounding
            "predicted_slots": slots,
            "slot_value_assignments": {k: v for k, v in filled_values.items() if v is not None},
            # meta
            "method": f"hybrid_bm25_tfidf+{grounding_method}",
            "elapsed_sec": round(elapsed, 4),
            "grounding_diagnostics": {
                "method": grounding_method,
                "n_top_assignments": len(diagnostics.get("top_assignments", [])),
                "top_assignment": (diagnostics.get("top_assignments") or [{}])[0],
            },
        }
        results.append(out)
        status = "✓" if schema_correct else "✗"
        n_filled = len(out["slot_value_assignments"])
        print(f"  [{i:02d}/{len(cases)}] {status} {case_id} | pred={pred_id} | "
              f"filled={n_filled}/{len(slots)} slots [{grounding_method[:3]}]")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_correct = sum(r["schema_correct"] for r in results)
    print(f"\nSchema retrieval accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.1%}")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    run()
