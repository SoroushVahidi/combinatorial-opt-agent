"""Downstream utility demo for NLP4LP: retrieval enables parameter instantiation from NL.

Deterministic, CPU-only: no LLMs, no torch, no solver dependency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NUM_TOKEN_RE = re.compile(r"[$]?\d[\d,]*(?:\.\d+)?%?")

MONEY_CONTEXT = {"budget", "cost", "price", "profit", "revenue", "dollar", "dollars", "$", "€", "usd", "eur"}
PERCENT_CONTEXT = {"percent", "percentage", "rate", "fraction"}

# ── Written-word number recognition ──────────────────────────────────────────
_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_LARGE: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
}

_WORD_TO_NUM: dict[str, int] = {**_ONES, **_TENS, **_LARGE}
for tens_word, tens_val in _TENS.items():
    for ones_word, ones_val in _ONES.items():
        if ones_val > 0:
            _WORD_TO_NUM[f"{tens_word}-{ones_word}"] = tens_val + ones_val

# ── Fraction word recognition ─────────────────────────────────────────────────
# Maps English fraction/rate words to their decimal values (in [0, 1]).
# These produce NumTok(kind="percent") because they represent fractional quantities.
# Keep this table conservative: only unambiguous fraction words.
_WORD_FRACTIONS: dict[str, float] = {
    "half":           0.5,
    "halves":         0.5,
    "third":          1.0 / 3.0,
    "thirds":         1.0 / 3.0,
    "quarter":        0.25,
    "quarters":       0.25,
    "one-half":       0.5,
    "one-third":      1.0 / 3.0,
    "one-quarter":    0.25,
    "two-thirds":     2.0 / 3.0,
    "three-quarters": 0.75,
}

# ── Enumeration-derived count extraction ─────────────────────────────────────
# Words that are never valid enumeration items (articles, prepositions, …).
_ENUM_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "is", "are",
    "be", "been", "was", "were", "has", "have", "had", "do", "does", "did",
    "with", "by", "from", "into", "through", "each", "per", "any", "all",
    "both", "either", "it", "its", "this", "that", "their", "which",
    "can", "could", "should", "would", "may", "might", "must", "shall",
    "not", "no", "also", "some", "more", "other", "such", "than", "if", "as",
    "and", "or", "but", "nor", "so", "yet",
    # Modifier adjectives/determiners never used as standalone enumeration nouns
    # in optimization contexts (e.g. "labor hours, and total budget" should not
    # yield a 2-item enumeration with "hours" and "total").
    "total", "overall", "aggregate", "average", "general",
})

# Regex: "NOUN and NOUN" or "NOUN, NOUN, ..., and NOUN" (1-word items only).
# Using single-word items avoids greedy capture of trailing prepositions/articles.
# Requires "and" so there are always ≥ 2 items.
_ENUM_NOUN_LIST_RE = re.compile(
    r"[A-Za-z][a-zA-Z-]+"                        # first item (1 word, ≥ 2 chars)
    r"(?:\s*,\s*[A-Za-z][a-zA-Z-]+)*"             # 0+ comma-separated middle items
    r"\s*,?\s*and\s+[A-Za-z][a-zA-Z-]+",          # "and LAST" (last item, ≥ 2 chars)
    re.IGNORECASE,
)

# Regex: "N NOUN, N NOUN, ..., and N NOUN" — quantified noun lists.
# Requires at least 2 "N NOUN" pairs (the + after the first comma-group).
_ENUM_QUANT_LIST_RE = re.compile(
    r"[$]?\d[\d,]*(?:\.\d+)?%?\s+[A-Za-z][a-zA-Z]+"
    r"(?:\s*,\s*[$]?\d[\d,]*(?:\.\d+)?%?\s+[A-Za-z][a-zA-Z]+)+"
    r"(?:\s*,?\s*and\s+[$]?\d[\d,]*(?:\.\d+)?%?\s+[A-Za-z][a-zA-Z]+)?",
    re.IGNORECASE,
)


def _extract_enum_derived_counts(query: str) -> list[tuple[float, str, list[str]]]:
    """Detect enumeration patterns and return (count, raw_span, context_tokens).

    Two patterns are recognised:

    1. Unquantified noun lists — "phones and laptops" → 2,
       "apples, bananas, and grapes" → 3, "A, B, and C" → 3.
       Items are single words; articles, stop words, and written numbers
       are silently dropped from the count.

    2. Quantified noun lists — "10 apples, 20 bananas, and 80 grapes" → 3
       (counts the number of distinct item types, not the quantities).

    Conservative guardrails:
    - Only returns counts in [2, 5].
    - Items that are stop-words or written numbers are excluded.
    - Duplicate (count, position) pairs are deduplicated.
    """
    results: list[tuple[float, str, list[str]]] = []

    def _head_word(phrase: str) -> str:
        parts = phrase.strip().lower().split()
        return parts[-1].strip(".,;:()[]{}\"'") if parts else ""

    def _is_valid_item(phrase: str) -> bool:
        hw = _head_word(phrase)
        if not hw or len(hw) < 2:
            return False
        if not hw.replace("-", "").isalpha():
            return False
        if hw in _ENUM_STOP_WORDS:
            return False
        if _WORD_TO_NUM.get(hw) is not None:
            return False
        if hw in _WORD_FRACTIONS:
            return False
        return True

    def _ctx_for(span: str, pos: int) -> list[str]:
        start = max(0, pos - 50)
        end = min(len(query), pos + len(span) + 50)
        return [
            t.lower().strip(".,;:()[]{}\"'")
            for t in query[start:end].split()
            if t.strip(".,;:()[]{}\"'")
        ]

    seen_spans: set[tuple[float, int]] = set()

    # Pattern 1: unquantified noun lists
    for m_obj in _ENUM_NOUN_LIST_RE.finditer(query):
        span = m_obj.group(0)
        items = re.split(r"\s*,\s*|\s+and\s+", span, flags=re.IGNORECASE)
        valid = [it for it in items if it.strip() and _is_valid_item(it)]
        count = len(valid)
        if 2 <= count <= 5:
            pos = m_obj.start()
            key: tuple[float, int] = (float(count), pos)
            if key not in seen_spans:
                seen_spans.add(key)
                results.append((float(count), span, _ctx_for(span, pos)))

    # Pattern 2: quantified noun lists ("10 apples, 20 bananas, and 80 grapes")
    for m_obj in _ENUM_QUANT_LIST_RE.finditer(query):
        span = m_obj.group(0)
        nouns = re.findall(
            r"[$]?\d[\d,]*(?:\.\d+)?%?\s+([A-Za-z][a-zA-Z]*)", span, re.IGNORECASE
        )
        valid = [
            n for n in nouns
            if n.lower() not in _ENUM_STOP_WORDS
            and n.lower().replace("-", "").isalpha()
        ]
        count = len(valid)
        if 2 <= count <= 5:
            pos = m_obj.start()
            key = (float(count), pos)
            if key not in seen_spans:
                seen_spans.add(key)
                results.append((float(count), span, _ctx_for(span, pos)))

    return results

_WRITTEN_NUM_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_WORD_TO_NUM, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _word_to_number(word: str) -> float | None:
    """Return the numeric value of a written English number word, or None."""
    val = _WORD_TO_NUM.get(word.lower())
    return float(val) if val is not None else None


def _parse_word_num_span(toks: list[str], start: int) -> tuple[float | None, int]:
    """Parse a multi-token English word-number phrase beginning at *start*.

    Handles:
    - Simple words:           "two", "twenty", "hundred"
    - Hyphenated compounds:   "twenty-five"  (single token in _WORD_TO_NUM)
    - Multi-token phrases:    "one hundred", "two hundred fifty",
                              "one thousand", "three thousand five hundred"

    The grammar followed is::

        number  = [X million] [Y thousand] [Z hundred] base?
        base    = tens_word ones_word? | ones_word | tens_word

    Multiplier words ("hundred", "thousand", "million") act on the
    accumulated *chunk* so far; if no chunk precedes them the multiplier
    itself is treated as 1 × magnitude (e.g. bare "hundred" → 100).

    Stops at any token not in *_WORD_TO_NUM*.

    Returns ``(value, tokens_consumed)`` or ``(None, 0)`` if no match.
    """
    n = len(toks)
    if start >= n:
        return None, 0

    def _clean(t: str) -> str:
        return t.lower().strip(".,;:()[]{}\"'")

    # Quick check: the first token must be a recognized word-number.
    if _WORD_TO_NUM.get(_clean(toks[start])) is None:
        return None, 0

    total = 0
    chunk = 0
    i = start

    while i < n:
        w = _clean(toks[i])
        v = _WORD_TO_NUM.get(w)
        if v is None:
            break
        if w == "hundred":
            chunk = (chunk if chunk > 0 else 1) * 100
        elif w == "thousand":
            total += (chunk if chunk > 0 else 1) * 1_000
            chunk = 0
        elif w == "million":
            total += (chunk if chunk > 0 else 1) * 1_000_000
            chunk = 0
        else:
            # ones, tens, or a hyphenated combo already in the dict
            chunk += v
        i += 1

    total += chunk
    consumed = i - start
    if consumed == 0:
        return None, 0
    return float(total), consumed


def _classify_word_num_tok(
    raw_surface: str,
    wval: float,
    ctx_set: set[str],
    toks: list[str],
    j: int,
) -> NumTok:
    """Build a ``NumTok`` for a word-number span with type detection.

    Parameters
    ----------
    raw_surface : surface form of the span (e.g. ``"one hundred"``).
    wval        : numeric value of the span (e.g. ``100.0``).
    ctx_set     : lowercased context tokens for the span.
    toks        : full token list of the query.
    j           : index of the first token *after* the span, used to detect
                  trailing ``"percent"`` / ``"per cent"``.

    Detection priority: percent > currency (money context) > int/float.
    """
    is_pct = False
    if j < len(toks):
        next_w = toks[j].lower().strip(".,;:()[]{}\"'")
        if next_w == "percent":
            is_pct = True
        elif (
            next_w == "per"
            and j + 1 < len(toks)
            and toks[j + 1].lower().strip(".,;:()[]{}\"'") == "cent"
        ):
            is_pct = True
    if not is_pct and ("percent" in ctx_set or "percentage" in ctx_set) and wval > 1.0:
        is_pct = True
    if is_pct:
        return NumTok(raw=raw_surface, value=wval / 100.0 if wval > 1.0 else wval, kind="percent")
    if ctx_set & MONEY_CONTEXT:
        return NumTok(raw=raw_surface, value=wval, kind="currency")
    kind = "int" if float(int(wval)) == wval else "float"
    return NumTok(raw=raw_surface, value=wval, kind=kind)


# Cue words and simple operator markers used by constrained assignment.
CUE_WORDS = {
    "budget",
    "cost",
    "profit",
    "revenue",
    "demand",
    "capacity",
    "limit",
    "requirement",
    "requirements",
    "minimum",
    "maximum",
    "least",
    "most",
    "fraction",
    "ratio",
    "percentage",
    "percent",
    "rate",
    "share",
    "total",
    "available",
}

OPERATOR_MIN_WORDS = {"minimum", "min", "least", "lower", "atleast"}
OPERATOR_MAX_WORDS = {"maximum", "max", "most", "upper", "atmost"}


# Weights for interpretable mention-slot compatibility scoring in constrained assignment.
ASSIGN_WEIGHTS = {
    "type_match_bonus": 3.0,
    "type_mismatch_penalty": -4.0,
    "lex_context_overlap": 0.7,
    "lex_sentence_overlap": 0.3,
    "cue_overlap": 1.5,
    "operator_min_bonus": 1.0,
    "operator_max_bonus": 1.0,
    "unit_percent_bonus": 2.0,
    "unit_currency_bonus": 2.0,
    "weak_match_penalty": -1.0,
    # Count-like slot priors: favour small cardinalities, penalise large values.
    "count_small_int_prior": 2.0,        # bonus for int in [1, count_plausible_max]
    "count_large_int_penalty": -2.0,     # penalty for int > count_large_penalty_threshold
    "count_plausible_max": 10,           # integers ≤ this get the small-int bonus
    "count_large_penalty_threshold": 50, # integers > this get the large-int penalty
}


def _safe_json_loads(s: str | None) -> Any:
    if not s:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def _load_eval(eval_path: Path) -> list[dict]:
    items = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "query_id": obj.get("query_id", ""),
                    "query": (obj.get("query") or "").strip(),
                    "relevant_doc_id": obj.get("relevant_doc_id", ""),
                }
            )
    return items


def _load_catalog_as_problems(catalog_path: Path) -> tuple[list[dict], dict[str, str]]:
    """Load catalog JSONL and return list[problem] for baselines + id->text for snippets."""
    catalog: list[dict] = []
    id_to_text: dict[str, str] = {}
    with open(catalog_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("id")
            text = (obj.get("text") or obj.get("description") or "").strip()
            if not doc_id:
                continue
            catalog.append({"id": doc_id, "name": doc_id, "description": text, "aliases": []})
            id_to_text[doc_id] = text
    return catalog, id_to_text


def _apply_low_resource_env() -> None:
    """Set environment variables for low-resource/safe execution (thread limits, no tokenizer parallelism)."""
    env_settings = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    for k, v in env_settings.items():
        if k not in os.environ:
            os.environ[k] = v


def _load_hf_gold(split: str = "test", use_cache: bool = True) -> dict[str, dict]:
    """Load NLP4LP HF split and return doc_id -> parsed fields.
    If NLP4LP_GOLD_CACHE env is set and file exists, load from that JSON (avoids HF threads).
    If use_cache and we load from HF, write to NLP4LP_GOLD_CACHE for next time."""
    cache_path = os.environ.get("NLP4LP_GOLD_CACHE")
    if cache_path and use_cache:
        p = Path(cache_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("split") == split:
                out = data.get("gold_by_id")
                if isinstance(out, dict):
                    return out
    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(f"datasets not available: {e}")

    raw = (
        (os.environ.get("HF_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_TOKEN") or "")
    ).strip()
    kwargs: dict[str, Any] = {"token": raw} if raw else {}
    # Thread limits should be set by caller (OMP_NUM_THREADS etc.) for constrained nodes.
    ds = load_dataset("udell-lab/NLP4LP", split=split, **kwargs)

    gold: dict[str, dict] = {}
    for i, ex in enumerate(ds):
        doc_id = f"nlp4lp_{split}_{i}"
        params = _safe_json_loads(ex.get("parameters"))
        pinfo = _safe_json_loads(ex.get("problem_info"))
        gold[doc_id] = {
            "parameters": params if isinstance(params, dict) else {},
            "problem_info": pinfo if isinstance(pinfo, dict) else {},
            "optimus_code": ex.get("optimus_code") or "",
            "solution": _safe_json_loads(ex.get("solution")),
        }
    if cache_path and use_cache:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"split": split, "gold_by_id": gold}, f, indent=0)
        except Exception:
            pass
    return gold


def _tokens_lower(text: str) -> list[str]:
    return re.findall(r"\w+|<num>|[$]?\d[\d,]*(?:\.\d+)?%?", text.lower())


@dataclass(frozen=True)
class NumTok:
    raw: str
    value: float | None
    kind: str  # percent|currency|int|float|unknown


@dataclass(frozen=True)
class MentionRecord:
    """Numeric mention with local/sentence context for constrained assignment."""

    index: int
    tok: NumTok
    context_tokens: list[str]
    sentence_tokens: list[str]
    cue_words: set[str]


@dataclass(frozen=True)
class SlotRecord:
    """Scalar slot representation."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    aliases: list[str]
    alias_tokens: set[str]
    is_count_like: bool = False


def _parse_num_token(tok: str, context_words: set[str]) -> NumTok:
    # Strip whitespace, then strip trailing punctuation characters so that
    # end-of-sentence numbers like "5000." are recognised correctly.
    # We intentionally strip a broad set of punctuation (not just sentence-ending
    # '.!?') because tokens can have any trailing bracket, comma, or colon.
    t = tok.strip().rstrip(".,;:()[]{}")
    if t == "<num>":
        return NumTok(raw=t, value=None, kind="unknown")
    has_dollar = "$" in t
    is_pct = t.endswith("%")
    num_str = t.replace("$", "").replace("%", "").replace(",", "")
    try:
        val = float(num_str)
    except Exception:
        return NumTok(raw=t, value=None, kind="unknown")

    if is_pct:
        return NumTok(raw=t, value=val / 100.0, kind="percent")

    # Percent context without %: treat e.g. "20 percent" as 0.20.
    if ("percent" in context_words or "percentage" in context_words) and val > 1.0:
        return NumTok(raw=t, value=val / 100.0, kind="percent")

    if 0.0 < val <= 1.0 and (context_words & PERCENT_CONTEXT):
        return NumTok(raw=t, value=val, kind="percent")

    if has_dollar or (context_words & MONEY_CONTEXT):
        return NumTok(raw=t, value=val, kind="currency")

    # Integer vs float
    if float(int(val)) == val:
        return NumTok(raw=t, value=float(int(val)), kind="int")
    return NumTok(raw=t, value=val, kind="float")


def _extract_num_tokens(query: str, variant: str) -> list[NumTok]:
    toks = query.split()
    out: list[NumTok] = []
    i = 0
    while i < len(toks):
        w = toks[i]
        if w == "<num>" and variant in ("noisy", "nonum"):
            out.append(NumTok(raw=w, value=None, kind="unknown"))
            i += 1
            continue
        # local context window (centred on span start)
        ctx = set(x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 3) : i + 4])
        # Digit-based token (single token only)
        m = NUM_TOKEN_RE.fullmatch(w.strip().rstrip(",;:()[]{}").rstrip("."))
        if m:
            out.append(_parse_num_token(w, ctx))
            i += 1
            continue
        # Written-word number — may span multiple tokens ("one hundred fifty").
        wval, consumed = _parse_word_num_span(toks, i)
        if wval is not None:
            j = i + consumed  # index of first token after the span
            raw_surface = " ".join(toks[i:j])
            out.append(_classify_word_num_tok(raw_surface, wval, ctx, toks, j))
            i += consumed
            continue
        # Fraction word (half, one-third, quarter, …) → kind="percent"
        _w_clean = w.lower().strip(".,;:()[]{}\"'")
        _frac_val = _WORD_FRACTIONS.get(_w_clean)
        if _frac_val is not None:
            out.append(NumTok(raw=_w_clean, value=_frac_val, kind="percent"))
            i += 1
            continue
        i += 1
    return out


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _split_camel_case(name: str) -> list[str]:
    """Split a CamelCase or underscore-separated identifier into lowercase tokens.

    Also splits at letter–digit and digit–letter boundaries so that numeric
    suffixes like "Product1" or "Type2" produce separate tokens ("product","1"),
    enabling left-entity-anchor overlap to match "Product 1" in query text
    against the "1" component of slot "LaborProduct1".

    Examples::

        TotalLaborHours      -> ['total', 'labor', 'hours']
        LaborHoursPerProduct -> ['labor', 'hours', 'per', 'product']
        ProteinFeedA         -> ['protein', 'feed', 'a']
        FatFeedA             -> ['fat', 'feed', 'a']
        HeatingHours         -> ['heating', 'hours']
        CoolingHours         -> ['cooling', 'hours']
        LaborProduct1        -> ['labor', 'product', '1']
        LaborProduct2        -> ['labor', 'product', '2']
    """
    # Insert a space between a lowercase letter (or digit) followed by an uppercase letter.
    s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", name)
    # Insert a space between a run of uppercase letters and a following CamelCase word.
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Insert a space at letter–digit and digit–letter transitions (Group 3).
    # This splits suffixes like "product1" → "product 1" and "2type" → "2 type".
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
    # Replace underscores and collapse whitespace, then return lowercase tokens.
    return [t.lower() for t in re.split(r"[\s_]+", s) if t]


def _slot_measure_tokens(name: str) -> list[str]:
    """Return a de-duplicated list of lowercase tokens for a slot name.

    Includes both the full lowercased identifier (for backward-compatible exact
    matching) and the individual camel-case-split component tokens (for
    measure/attribute-aware scoring via narrow_measure_overlap).

    Examples::

        'TotalLaborHours'    -> ['totallaborhours', 'total', 'labor', 'hours']
        'ProteinFeedA'       -> ['proteinfeeda', 'protein', 'feed', 'a']
        'HeatingHours'       -> ['heatinghours', 'heating', 'hours']
    """
    full = name.lower()
    parts = _split_camel_case(name)
    seen: set[str] = {full}
    result: list[str] = [full]
    for t in parts:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _slot_aliases(param_name: str) -> list[str]:
    """Rule-based alias expansion for common optimization slot patterns."""
    n = (param_name or "").lower()
    aliases = [param_name]

    def add_many(xs: Iterable[str]) -> None:
        for x in xs:
            if x not in aliases:
                aliases.append(x)

    if "budget" in n:
        add_many(["budget", "total budget", "available amount", "amount available"])
    if "capacity" in n:
        add_many(["capacity", "limit", "maximum", "available", "upper bound"])
    if "cost" in n or "expense" in n or "price" in n:
        add_many(["cost", "expense", "spending", "spend", "price"])
    if "profit" in n or "revenue" in n:
        add_many(["profit", "gain", "revenue", "return"])
    if "demand" in n or "require" in n or "needed" in n:
        add_many(["demand", "required", "needed", "requirement"])
    if any(w in n for w in ("fraction", "ratio", "percent", "percentage", "rate", "share")):
        add_many(["percentage", "fraction", "share", "ratio", "rate", "proportion"])
    if "min" in n or "minimum" in n or "atleast" in n:
        add_many(["minimum", "at least", "lower bound"])
    if "max" in n or "maximum" in n or "atmost" in n:
        add_many(["maximum", "at most", "upper bound"])

    return aliases


def _extract_num_mentions(query: str, variant: str) -> list[MentionRecord]:
    """Extract numeric mentions with richer context for constrained assignment.

    In addition to digit-based tokens (e.g. "100", "$5000", "20%"), this also
    recognises written-word numbers such as "two", "twenty-five", "one hundred".
    Multi-token word-number spans (e.g. "two hundred fifty") are collapsed into
    a single MentionRecord so that compound values are not split across mentions.
    """
    toks = query.split()
    sent_tokens = [t.lower().strip(".,;:()[]{}") for t in toks]
    mentions: list[MentionRecord] = []
    i = 0
    while i < len(toks):
        w = toks[i]
        # slightly wider context window for constrained assignment
        ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 8) : i + 9]
        ]
        ctx_tokens = [c for c in ctx_tokens if c]
        cue_words = set(ctx_tokens) & CUE_WORDS

        if w == "<num>" and variant in ("noisy", "nonum"):
            tok = NumTok(raw=w, value=None, kind="unknown")
            mentions.append(
                MentionRecord(
                    index=i,
                    tok=tok,
                    context_tokens=ctx_tokens,
                    sentence_tokens=sent_tokens,
                    cue_words=cue_words,
                )
            )
            i += 1
            continue

        # Digit-based token (single token only — no multi-token merging needed).
        m = NUM_TOKEN_RE.fullmatch(w.strip().rstrip(",;:()[]{}").rstrip("."))
        if m:
            tok = _parse_num_token(w, set(ctx_tokens))
            mentions.append(
                MentionRecord(
                    index=i,
                    tok=tok,
                    context_tokens=ctx_tokens,
                    sentence_tokens=sent_tokens,
                    cue_words=cue_words,
                )
            )
            i += 1
            continue

        # Written-word number — may span multiple tokens ("one hundred fifty").
        wval, consumed = _parse_word_num_span(toks, i)
        if wval is not None:
            j = i + consumed  # first token after the span
            raw_surface = " ".join(toks[i:j])
            tok = _classify_word_num_tok(raw_surface, wval, set(ctx_tokens), toks, j)
            mentions.append(
                MentionRecord(
                    index=i,
                    tok=tok,
                    context_tokens=ctx_tokens,
                    sentence_tokens=sent_tokens,
                    cue_words=cue_words,
                )
            )
            i += consumed
            continue

        i += 1
    return mentions


def _expected_type(param_name: str) -> str:
    """Infer the expected scalar type of an optimization parameter from its name.

    Type hierarchy (checked in order):
    1. percent  — clearly represents a rate/fraction/percentage
    2. int      — primary discrete patterns (num, count, items, …)
    3. currency — monetary / budget-like quantities
    4. int      — extended discrete patterns (number-of workers, days, shifts, …)
                  checked *after* currency so "TotalBudget" still → currency
    5. float    — catch-all for continuous real-valued parameters

    Note on float/int: many optimization coefficients (e.g. RequiredEggsPerSandwich=2)
    appear as integer text in queries but are conceptually continuous.  The
    _is_type_match() helper is the authoritative arbiter: it counts an integer
    token as a full type-match for a float slot.
    """
    n = (param_name or "").lower()
    if any(s in n for s in ("percent", "percentage", "fraction", "pct", "ratio", "proportion", "share", "utilisation", "utilization")):
        return "percent"
    # "rate" is percent only when it is the final segment of the name (e.g. DiscountRate,
    # TaxRate, InterestRate).  When "rate" appears as a *prefix* followed by an entity
    # identifier (e.g. RateMachine1, Rate1), the slot holds a processing/production rate
    # (units per hour) which is a plain float, not a percentage.
    if "rate" in n and n.endswith("rate"):
        return "percent"
    # Primary integer indicators (checked before currency to preserve existing behaviour)
    if any(s in n for s in ("num", "count", "types", "items", "ingredients", "nodes", "edges")):
        return "int"
    # Currency / monetary quantities (strictly monetary keywords only).
    # Note: "demand", "capacity", "minimum", "maximum", and "limit" were
    # previously in this list but are NOT monetary — they represent quantity
    # constraints and bounds (e.g. MinimumDemand=100, MaxCapacity=500,
    # TimeLimit=10).  Those slots fall through to "float" so that plain
    # integer tokens (the common case in NL) receive a full type_match.
    if any(
        s in n
        for s in (
            "budget",
            "cost",
            "price",
            "revenue",
            "profit",
            "penalty",
            "investment",
        )
    ):
        return "currency"
    # Extended integer indicators — discrete/countable quantities.
    # Placed *after* currency so "TotalBudget" is not accidentally reclassified.
    _EXT_INT_PATTERNS = (
        "number",    # NumberOfShifts, NumberOfWorkers, NumberOfDays, …
        "workers",   # TotalWorkers
        "employee",  # NumberOfEmployees
        "shifts",    # TotalShifts
        "batches",   # NumberOfBatches
        "rounds",    # NumberOfRounds
        "days",      # NumberOfDays, TotalDays
        "weeks",     # NumberOfWeeks
        "months",    # NumberOfMonths
        "trips",     # NumberOfTrips
        "persons",   # NumberOfPersons
        "patients",  # NumberOfPatients
        "tasks",     # NumberOfTasks
        "machines",  # NumberOfMachines (distinct from cost-related names like MachineCost)
        "factories", # NumberOfFactories
        "farms",     # NumberOfFarms
        "vehicles",  # NumberOfVehicles
        "trucks",    # NumberOfTrucks
        "buses",     # NumberOfBuses
    )
    if any(s in n for s in _EXT_INT_PATTERNS):
        return "int"
    return "float"


# Count-like slot name fragments that indicate a cardinality/count parameter.
# These slots hold the NUMBER OF distinct entities (products, resources, types…),
# which is always a non-negative integer, and typically a small one (2–20).
_COUNT_LIKE_PATTERNS = (
    "num",        # NumProducts, NumTypes, NumItems, NumMixes, …
    "count",      # CountProducts, ProductCount, …
    "types",      # NumCandyTypes, JarTypes, …
    "items",      # NumItems, …
    "ingredients",
    "nodes",
    "edges",
    "number",     # NumberOfProducts, NumberOfResources, …
    "workers",
    "employee",
    "shifts",
    "batches",
    "rounds",
    "machines",
    "factories",
    "farms",
    "vehicles",
    "trucks",
    "buses",
    "tasks",
    "resources",  # NumResources, …
    "products",   # NumProducts, …
    "mixes",      # NumMixes, …
    "materials",  # NumMaterials, …
    "components",
    "categories",
)


def _is_count_like_slot(param_name: str) -> bool:
    """Return True if *param_name* represents a count/cardinality slot.

    Count-like slots hold the number of distinct entities (e.g. NumProducts,
    NumResources, NumberOfMachines).  They always take non-negative integer
    values, and in realistic optimization problems are usually small (2–20).

    The detection is purely name-based so it is fast and explainable.
    """
    n = (param_name or "").lower()
    return any(p in n for p in _COUNT_LIKE_PATTERNS)


def _is_type_match(expected: str, kind: str) -> bool:
    """Return True when *kind* is a full type-match for *expected*.

    Key rule: an integer token is a valid assignment for a float slot.
    Optimization model coefficients (e.g. RequiredEggsPerSandwich = 2.0)
    commonly appear as whole numbers in natural-language descriptions.
    Treating int-as-float as a strict mismatch was the root cause of
    TypeMatch ≈ 0.03 for float-typed parameters.

    Rules:
    - same kind always matches
    - int  → float    = full match  (integer IS a real number)
    - int  → int      = full match
    - pct  → percent  = full match
    - ccy  → currency = full match
    - int  → currency = full match  (monetary values commonly appear as plain
                                     integers in NL, e.g. "budget is 5000")
    - float → currency= full match  (decimal monetary values, e.g. "price is 4.99")
    - anything else   = no match (handled by loose-match or incompatibility logic)
    """
    if expected == kind:
        return True
    if expected == "float" and kind == "int":
        return True
    # Monetary slots (budget, cost, price, …) are often described with plain
    # integers or decimals in NL — no explicit "$" prefix — so int/float tokens
    # ARE valid currency assignments.
    if expected == "currency" and kind in {"int", "float"}:
        return True
    return False


def _choose_token(expected: str, candidates: list[NumTok]) -> tuple[int | None, NumTok | None]:
    """Return (index, token) to use, or (None, None) if no candidates."""
    if not candidates:
        return None, None

    def score(tok: NumTok) -> tuple:
        # higher is better; deterministic tie-break via raw
        val = tok.value if tok.value is not None else 0.0
        absval = abs(val)
        has_decimal = (tok.value is not None) and (float(int(val)) != val)
        if expected == "percent":
            pref = 2 if tok.kind == "percent" else (1 if (tok.value is not None and 0.0 < tok.value <= 1.0) else 0)
            return (pref, absval, tok.raw)
        if expected == "int":
            pref = 2 if tok.kind == "int" else (1 if tok.value is not None and float(int(val)) == val else 0)
            return (pref, absval, tok.raw)
        if expected == "currency":
            # All four scoring functions give int/float tokens the same
            # type_exact_bonus as a currency token for currency slots.
            # True currency tokens (with explicit "$" / "dollar" context) also
            # earn a unit_currency_bonus on top, so they are ranked highest.
            # int/float tokens are valid type-matches and must beat unknown tokens.
            if tok.kind == "currency":
                pref = 2
            elif tok.kind in {"int", "float"}:
                pref = 1
            else:
                pref = 0
            return (pref, absval, tok.raw)
        # float: integer-valued and decimal-valued tokens are equally preferred;
        # both are valid real-number assignments for float-typed slots.
        if tok.kind in {"float", "int"}:
            pref = 2
        elif tok.kind == "currency":
            pref = 1
        else:
            pref = 0
        return (pref, absval, tok.raw)

    best_i = 0
    best_s = score(candidates[0])
    for i in range(1, len(candidates)):
        s = score(candidates[i])
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, candidates[best_i]


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_type_incompatible(expected: str, kind: str) -> bool:
    """Hard incompatibilities between expected slot type and mention kind.

    Rules (extended for type-consistent assignment):
    - percent <-> currency:  hard incompatible (already enforced)
    - int/float -> percent:  hard incompatible (a bare integer/decimal is NOT a
      percentage value; percent tokens are produced by context-aware extraction
      so this would only fire when the token truly lacks a percent signal)
    - percent -> int:        hard incompatible (a percentage fraction is NOT a
      discrete integer count)
    - percent -> float:      hard incompatible (a percent mention cannot fill a
      generic float slot; percent slots are always typed "percent" by _expected_type)
    - float → count-like slot: handled separately in _score_mention_slot and
      _gcg_local_score (non-integer decimals cannot be cardinality counts)
    All other cross-type pairs (e.g. currency <-> int/float) are handled via
    soft scoring (loose-match bonus/penalty).
    """
    if expected == "percent" and kind in {"currency", "int", "float"}:
        return True
    if expected == "currency" and kind in {"percent"}:
        return True
    if expected == "int" and kind == "percent":
        return True
    # A percent mention cannot fill a float slot unless the slot is typed percent.
    # Percent slots always have expected == "percent" (via _expected_type), so this
    # rule fires only for genuinely float-typed slots receiving a percent mention.
    if expected == "float" and kind == "percent":
        return True
    return False


def _build_slot_records(expected_scalar: list[str]) -> list[SlotRecord]:
    slots: list[SlotRecord] = []
    for name in expected_scalar:
        et = _expected_type(name)
        aliases = _slot_aliases(name)
        alias_tokens: set[str] = set()
        for a in aliases:
            alias_tokens.update(_normalize_tokens(a))
        norm_tokens = _slot_measure_tokens(name)
        slots.append(
            SlotRecord(
                name=name,
                norm_tokens=norm_tokens,
                expected_type=et,
                aliases=aliases,
                alias_tokens=alias_tokens,
                is_count_like=_is_count_like_slot(name),
            )
        )
    return slots


def _score_mention_slot(m: MentionRecord, s: SlotRecord) -> tuple[float, dict[str, Any]]:
    """Compute interpretable compatibility score and feature breakdown."""
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.tok.kind
    expected = s.expected_type

    # Hard type incompatibility.
    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return -1e9, features

    # Count-like slot: non-integer decimals and percent values are hard incompatible.
    if s.is_count_like and kind == "float":
        features["count_slot_float_incompatible"] = True
        return -1e9, features

    # Type compatibility.
    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (
            expected == "currency" and kind == "currency"
        ):
            score += ASSIGN_WEIGHTS["type_match_bonus"]
            features["type_match"] = True
        elif expected == "float" and kind in {"float", "int"}:
            # Integer-valued tokens are valid for float slots.  Optimization
            # coefficients (e.g. RequiredEggsPerSandwich = 2) appear as whole
            # numbers in text but are continuous parameters in the model.
            score += ASSIGN_WEIGHTS["type_match_bonus"]
            features["type_match"] = True
        elif expected == "int" and kind == "int":
            score += ASSIGN_WEIGHTS["type_match_bonus"]
            features["type_match"] = True
        elif expected == "int" and kind in {"currency", "float"}:
            score += ASSIGN_WEIGHTS["type_match_bonus"] * 0.5
            features["type_loose_match"] = True
        elif expected == "float" and kind == "currency":
            score += ASSIGN_WEIGHTS["type_match_bonus"] * 0.5
            features["type_loose_match"] = True
        elif expected == "currency" and kind in {"float", "int"}:
            # Monetary values (budget, cost, price, …) often appear as plain
            # integers or decimals without an explicit "$" prefix.  An int/float
            # token IS a valid monetary assignment.
            score += ASSIGN_WEIGHTS["type_match_bonus"]
            features["type_match"] = True

    # Count-like slot: small-integer cardinality prior and large-value penalty.
    if s.is_count_like and kind == "int" and m.tok.value is not None:
        val = m.tok.value
        if 1 <= val <= ASSIGN_WEIGHTS["count_plausible_max"]:
            score += ASSIGN_WEIGHTS["count_small_int_prior"]
            features["count_small_int_prior"] = True
        elif val > ASSIGN_WEIGHTS["count_large_penalty_threshold"]:
            score += ASSIGN_WEIGHTS["count_large_int_penalty"]
            features["count_large_int_penalty"] = True

    # Lexical overlap: context vs slot name/aliases.
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    slot_words = set(s.norm_tokens) | s.alias_tokens

    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += ASSIGN_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap

    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += ASSIGN_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    # Cue-word overlap.
    cue_overlap = len(m.cue_words & slot_words)
    if cue_overlap:
        score += ASSIGN_WEIGHTS["cue_overlap"] * float(cue_overlap)
        features["cue_overlap"] = cue_overlap

    # Operator semantics.
    ctx_tokens = set(m.context_tokens)
    if ctx_tokens & OPERATOR_MIN_WORDS and any(
        w in "".join(s.norm_tokens) for w in ("min", "minimum")
    ):
        score += ASSIGN_WEIGHTS["operator_min_bonus"]
        features["operator_min"] = True
    if ctx_tokens & OPERATOR_MAX_WORDS and any(
        w in "".join(s.norm_tokens) for w in ("max", "maximum")
    ):
        score += ASSIGN_WEIGHTS["operator_max_bonus"]
        features["operator_max"] = True

    # Unit markers.
    if kind == "percent":
        score += ASSIGN_WEIGHTS["unit_percent_bonus"]
        features["unit_percent"] = True
    if kind == "currency":
        score += ASSIGN_WEIGHTS["unit_currency_bonus"]
        features["unit_currency"] = True

    # Weak-match penalty if score is still very small.
    if score <= 0.0:
        score += ASSIGN_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


# --- Semantic IR + repair assignment (Stage A–D) ---

# Rule-based semantic role words -> tag names (easy to extend).
SEMANTIC_ROLE_WORDS: dict[str, str] = {}
for tag, words in [
    ("budget", ["budget", "budgets"]),
    ("cost", ["cost", "costs"]),
    ("expense", ["expense", "expenses"]),
    ("spend", ["spend", "spending", "spent"]),
    ("profit", ["profit", "profits"]),
    ("revenue", ["revenue", "revenues"]),
    ("return", ["return", "returns"]),
    ("demand", ["demand", "demands"]),
    ("requirement", ["requirement", "requirements"]),
    ("need", ["need", "needed", "needs"]),
    ("capacity", ["capacity"]),
    ("limit", ["limit", "limits"]),
    ("available", ["available", "availability"]),
    ("supply", ["supply", "supplies"]),
    ("minimum", ["minimum", "min"]),
    ("lower_bound", ["lower", "least"]),
    ("maximum", ["maximum", "max"]),
    ("upper_bound", ["upper", "most"]),
    ("at_least", ["atleast", "at_least"]),
    ("at_most", ["atmost", "at_most"]),
    ("ratio", ["ratio", "ratios"]),
    ("fraction", ["fraction", "fractions"]),
    ("percentage", ["percentage", "percent"]),
    ("rate", ["rate", "rates"]),
    ("share", ["share", "shares"]),
    ("total", ["total", "totals"]),
    ("fixed_cost", ["fixed", "fixed_cost"]),
    ("variable_cost", ["variable", "variable_cost"]),
    ("penalty", ["penalty", "penalties"]),
    ("resource", ["resource", "resources"]),
    ("time", ["time", "minutes", "hours", "weeks"]),
    ("quantity", ["quantity", "quantities", "amount", "amounts"]),
    ("item_count", ["number", "count", "items", "units"]),
]:
    for w in words:
        SEMANTIC_ROLE_WORDS[w.lower()] = tag

# Phrase-level operator patterns for min/max detection.
# Using individual-word sets that include "at", "no", "than", "to" caused both
# min and max tags to fire for almost every mention (e.g. "to" from "wants to
# maximize" would trigger max; "at" from "at least" would trigger both).
# These phrase-level tuples are matched against a narrow context string so that
# "at most 30 vans" does not contaminate the operator tag of "at least 5000".
_OPERATOR_MIN_PATTERNS: tuple[str, ...] = (
    "at least", "atleast", "at_least",
    "no less than", "not less than",
    "no fewer than", "not fewer than",
    "at minimum", "at a minimum",
    "greater than or equal to", "greater than or equal",
    # Additional min patterns for final-pass coverage
    "minimum of", "a minimum of", "the minimum",
    "at the minimum", "minimum required",
    "must be at least", "should be at least",
)
_OPERATOR_MAX_PATTERNS: tuple[str, ...] = (
    "at most", "atmost", "at_most",
    "no more than", "not more than",
    "cannot exceed", "not to exceed",
    "up to", "upto", "limited to",
    "at maximum", "at a maximum",
    "less than or equal to", "less than or equal",
    "no greater than", "not greater than",
    # Additional max patterns for final-pass coverage
    "maximum of", "a maximum of", "the maximum",
    "at the maximum", "maximum allowed",
    "must not exceed", "should not exceed",
    "no higher than", "not higher than",
)
# Exclusive lower/upper-bound phrases (strict inequality: > or <).
# These are checked AFTER the standard patterns with a negation guard
# so that "no more than" (already an upper-bound pattern) is not
# re-tagged as a lower bound via the substring "more than".
_OPERATOR_MIN_EXCL_PATTERNS: tuple[str, ...] = (
    "more than",
    "greater than",
)
_OPERATOR_MAX_EXCL_PATTERNS: tuple[str, ...] = (
    "fewer than",
    "less than",
)
# Toggle to enable the full bound-role layer (bound_role field, wrong-direction
# penalties, range detection, and swap repair).  Set to False to ablate.
_ENABLE_BOUND_ROLE_LAYER: bool = True
# Narrow context window (tokens each side) for operator-phrase detection.
# A window of 3 keeps operator cues tight to the governing number; this is
# enough to capture "at least NUMBER" (distance 2) and "minimum of NUMBER"
# (distance 2–3) while preventing cross-constraint contamination when two
# "at least"/"at most" clauses appear within the same 14-token span.
_OPERATOR_NARROW_WINDOW: int = 3
# Left-side operator window is wider than the right-side window because operator
# phrases always precede the governed number.  A window of 6 is enough to capture
# the longest standard phrase "greater than or equal to" (5 tokens) while still
# preventing cross-contamination from the NEXT constraint's "at most" / "at least"
# phrase when it appears 7+ tokens to the right of the current number.
_OPERATOR_LEFT_WINDOW: int = 6

# Directional narrow windows for is_per_unit / is_total_like detection.
# The wide ±14 token context is too broad: in short optimization queries,
# a global-total cue ("available") and a per-unit cue ("each") can appear
# within the same window, causing every mention to be flagged as both.
# Using LEFT ±4 for verbs/determiners and RIGHT ±7 for trailing modifiers
# keeps each cue tied to the specific mention it governs.
_LOCALITY_LEFT_WINDOW: int = 4
_LOCALITY_RIGHT_WINDOW: int = 7

# Per-unit governing verbs: appear as the verb immediately before a coefficient.
# "requires 2 hours", "uses 3 liters", "costs $12 per unit"
_PER_UNIT_LEFT_VERBS: frozenset[str] = frozenset({
    "requires", "require", "needed", "takes", "take", "uses", "use",
    "needs", "need", "consumes", "consume", "produces", "produce",
    "contains", "contain", "costs", "cost", "earns", "earn",
    "yields", "yield",
    # Additional per-unit governing verbs for final-pass coverage
    "provides", "provide", "generates", "generate", "allocates", "allocate",
    "contributes", "contribute", "demands", "supplies",
    "processes", "process", "outputs", "output",
    # Note: "demand" and "supply" are intentionally excluded — they are most
    # commonly nouns in optimization queries ("total demand is N", "supply is N")
    # and would falsely flag N as a per-unit coefficient.  Only the unambiguous
    # third-person-singular verb forms "demands" and "supplies" are kept.
})
# Per-unit determiners: appear left of the governed noun/number.
# "each X", "per X", "every X", "for each X"
_PER_UNIT_LEFT_DETERMINERS: frozenset[str] = frozenset({
    "each", "per", "every",
})
# Multi-word per-unit phrases checked against the wider left context string.
# These catch "one unit requires", "a single item uses", "for every product" etc.
_PER_UNIT_LEFT_PHRASES: tuple[str, ...] = (
    "per unit", "per item", "for each", "for every",
    "one unit", "each unit", "per product", "each product",
    "per item", "each item", "per piece", "each piece",
    "unit requires", "unit uses", "unit costs", "unit earns",
    "unit needs", "unit takes", "unit produces",
)
# Total/aggregate left cues: "has N", "budget is N", "spend at most N"
_TOTAL_LEFT_CUES: frozenset[str] = frozenset({
    "total", "budget", "has", "have", "spend", "spent",
    # Additional total-left cues for final-pass coverage
    "overall", "aggregate", "sum", "stock", "stockpile",
    "allocated", "allotted",
})
# Total/aggregate right cues: "N available", "N capacity", "N total"
# Note: "budget" is NOT here — it is a left-side cue ("budget is N", "the budget N"),
# not a word that typically follows the global amount.
_TOTAL_RIGHT_CUES: frozenset[str] = frozenset({
    "available", "capacity", "limit", "total", "supply",
    # Additional total-right cues for final-pass coverage
    "overall", "stock", "remaining", "stored", "on-hand",
    "in-stock", "stocked", "allocated", "allotted",
})
# Multi-word total-like phrases checked against the wide context string.
# "in total", "in all", "total of N", "sum of N", "overall N"
_TOTAL_PHRASE_PATTERNS: tuple[str, ...] = (
    "in total", "in all", "total of", "sum of", "overall",
    "in stock", "on hand", "in supply",
)

# Unit/marker detection.
PERCENT_MARKER_TOKENS = {"%", "percent", "percentage", "pct"}
CURRENCY_MARKER_TOKENS = {"$", "€", "dollar", "dollars", "usd", "eur", "cost", "price", "budget"}

# ── Group 1 role-family lexicons ─────────────────────────────────────────────
# Used to compute tight-window role flags on MentionOptIR (is_cost_like, etc.)
# Each lexicon is checked against the ±2 token window immediately surrounding
# the numeric mention.  Using a tight window prevents cross-mention bleeding
# (e.g. "profit" from the previous mention leaking into the current one).

_COST_CONTEXT_WORDS: frozenset[str] = frozenset({
    "cost", "costs", "expense", "expenses", "price", "prices",
    "spend", "spending", "expenditure", "expenditures",
})

_PROFIT_CONTEXT_WORDS: frozenset[str] = frozenset({
    "profit", "profits", "revenue", "revenues", "return", "returns",
    "earns", "earn", "gain", "gains", "yield", "yields", "earning", "earnings",
})

_DEMAND_CONTEXT_WORDS: frozenset[str] = frozenset({
    "demand", "demands", "required", "needed", "need", "needs",
    "requirement", "requirements", "consume", "consumes",
})

_RESOURCE_CONTEXT_WORDS: frozenset[str] = frozenset({
    "labor", "labour", "material", "materials", "resource", "resources",
    "worker", "workers", "machine", "machines", "manpower",
})

_TIME_CONTEXT_WORDS: frozenset[str] = frozenset({
    "hour", "hours", "time", "day", "days", "minute", "minutes",
    "week", "weeks", "shift", "shifts",
})

# Context nouns that signal a count/cardinality interpretation when appearing
# near a small positive integer (e.g. "three types", "two products").
# Used by the quantity-role layer to derive is_count_like on MentionOptIR.
_COUNT_CONTEXT_NOUNS: frozenset[str] = frozenset({
    "type", "types", "kind", "kinds", "product", "products", "item", "items",
    "resource", "resources", "material", "materials", "ingredient", "ingredients",
    "machine", "machines", "worker", "workers", "factory", "factories",
    "category", "categories", "option", "options", "group", "groups",
    "variant", "variants", "component", "components", "mix", "mixes", "blend", "blends",
    "mode", "modes", "route", "routes", "vehicle", "vehicles", "part", "parts",
    "crop", "crops", "food", "foods", "drug", "drugs", "plant", "plants",
    "class", "classes", "good", "goods", "alloy", "alloys", "fuel", "fuels",
    # Additional count context nouns for final-pass coverage
    "variety", "varieties", "service", "services", "technique", "techniques",
    "method", "methods", "model", "models", "flavor", "flavors", "flavour", "flavours",
    "size", "sizes", "shape", "shapes", "color", "colors", "colour", "colours",
    "grade", "grades", "brand", "brands", "supplier", "suppliers", "vendor", "vendors",
    "department", "departments", "facility", "facilities", "location", "locations",
    "source", "sources", "destination", "destinations", "project", "projects",
    "task", "tasks", "job", "jobs", "shift", "shifts", "period", "periods",
    "warehouse", "warehouses", "depot", "depots", "station", "stations",
    "nutrient", "nutrients", "vitamin", "vitamins", "mineral", "minerals",
})

# Weights for semantic IR scoring (interpretable, additive).
SEMANTIC_IR_WEIGHTS = {
    "type_exact_bonus": 4.0,
    "type_loose_bonus": 1.5,
    "type_incompatible_penalty": -1e9,
    "semantic_tag_overlap": 2.5,
    "lex_context_overlap": 0.6,
    "lex_sentence_overlap": 0.25,
    "operator_min_bonus": 1.2,
    "operator_max_bonus": 1.2,
    "unit_percent_bonus": 2.0,
    "unit_currency_bonus": 2.0,
    "entity_target_bonus": 0.8,
    "weak_match_penalty": -1.0,
}
# Repair / validation weights.
REPAIR_WEIGHTS = {
    "validation_bonus": 0.5,
    "inconsistency_penalty": -2.0,
    "coverage_repair_bonus": 1.5,
    "min_semantic_support": 0.5,
}


@dataclass(frozen=True)
class MentionIR:
    """Lightweight semantic intermediate representation for a numeric mention."""

    mention_id: int
    value: float | None
    type_bucket: str
    raw_surface: str
    context_tokens: list[str]
    sentence_tokens: list[str]
    semantic_role_tags: frozenset[str]
    operator_tags: frozenset[str]  # "min", "max", or empty
    unit_tags: frozenset[str]  # "percent_marker", "currency_marker", "integer_like", "decimal_like"
    polarity_or_bound: str  # "min", "max", or ""
    target_entity_tokens: frozenset[str]
    tok: NumTok


@dataclass(frozen=True)
class SlotIR:
    """Slot with semantic target tags for IR-based matching."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    alias_tokens: set[str]
    semantic_target_tags: frozenset[str]
    operator_preference: frozenset[str]  # "min", "max", or empty
    unit_preference: frozenset[str]


def _context_to_semantic_tags(context_tokens: list[str]) -> frozenset[str]:
    """Rule-based semantic tagging from context tokens."""
    tags: set[str] = set()
    ctx_set = {t.lower().strip(".,;:()[]{}") for t in context_tokens if t}
    for w, tag in SEMANTIC_ROLE_WORDS.items():
        if w in ctx_set:
            tags.add(tag)
    return frozenset(tags)


def _detect_operator_tags(
    context_tokens: list[str],
    narrow_context_tokens: list[str] | None = None,
) -> frozenset[str]:
    """Detect min/max operator cues using phrase-level patterns.

    When *narrow_context_tokens* is supplied (recommended), phrase matching is
    performed on that asymmetric window around the governed number:
      - LEFT  : _OPERATOR_LEFT_WINDOW  (=6) tokens, to capture long phrases like
                "greater than or equal to" (5 tokens) that precede the number
      - RIGHT : _OPERATOR_NARROW_WINDOW (=3) tokens, kept tight to prevent
                contamination from the next constraint's operator phrases

    Single unambiguous words ("minimum", "maximum", "least", "most", "min",
    "max", "lower", "upper") are also checked using the same window.

    Exclusive lower-bound phrases ("more than", "greater than") and exclusive
    upper-bound phrases ("fewer than", "less than") are checked after the
    standard patterns and only when not negated (e.g. "no more than" → max,
    NOT additionally min via "more than").
    """
    ctx = narrow_context_tokens if narrow_context_tokens is not None else context_tokens
    ctx_str = " ".join(ctx)
    ctx_set = set(t.lower() for t in ctx if t)
    out: set[str] = set()

    # Standard inclusive lower-bound patterns.
    if (
        any(p in ctx_str for p in _OPERATOR_MIN_PATTERNS)
        or ctx_set & {"least", "min", "minimum", "lower"}
    ):
        out.add("min")

    # Standard inclusive upper-bound patterns.
    if (
        any(p in ctx_str for p in _OPERATOR_MAX_PATTERNS)
        or ctx_set & {"most", "max", "maximum", "upper"}
    ):
        out.add("max")

    if _ENABLE_BOUND_ROLE_LAYER:
        # Exclusive lower-bound phrases: "more than", "greater than".
        # Guard: skip if negated ("no more than" / "not more than" → already max).
        if "min" not in out:
            for p in _OPERATOR_MIN_EXCL_PATTERNS:
                if p in ctx_str:
                    if not any(f"{neg} {p}" in ctx_str for neg in ("no", "not")):
                        out.add("min")
                        break

        # Exclusive upper-bound phrases: "fewer than", "less than".
        # Guard: skip if negated ("no fewer than" / "not less than" → already min).
        if "max" not in out:
            for p in _OPERATOR_MAX_EXCL_PATTERNS:
                if p in ctx_str:
                    if not any(f"{neg} {p}" in ctx_str for neg in ("no", "not")):
                        out.add("max")
                        break

    return frozenset(out)


def _detect_unit_tags(tok: NumTok, context_tokens: list[str]) -> frozenset[str]:
    out: set[str] = set()
    ctx = set(t.lower() for t in context_tokens if t)
    if tok.kind == "percent" or ctx & PERCENT_MARKER_TOKENS:
        out.add("percent_marker")
    if tok.kind == "currency" or "$" in (tok.raw or "") or ctx & CURRENCY_MARKER_TOKENS:
        out.add("currency_marker")
    if tok.kind == "int" or (tok.value is not None and float(int(tok.value)) == tok.value):
        out.add("integer_like")
    if tok.kind == "float" or (tok.value is not None and float(int(tok.value)) != tok.value):
        out.add("decimal_like")
    return frozenset(out)


def _extract_enriched_mentions(query: str, variant: str) -> list[MentionIR]:
    """Stage A: Enriched mention extraction with semantic tagging."""
    toks = query.split()
    sent_tokens = [t.lower().strip(".,;:()[]{}") for t in toks]
    mentions_ir: list[MentionIR] = []
    mention_id = 0
    for i, w in enumerate(toks):
        if w == "<num>" and variant in ("noisy", "nonum"):
            tok = NumTok(raw=w, value=None, kind="unknown")
        else:
            m = NUM_TOKEN_RE.fullmatch(w.strip().rstrip(",;:()[]{}").rstrip("."))
            if not m:
                continue
            ctx_tokens = [
                x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 12) : i + 13]
            ]
            ctx_tokens = [c for c in ctx_tokens if c]
            ctx_set = set(ctx_tokens)
            tok = _parse_num_token(w, ctx_set)

        ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 12) : i + 13]
        ]
        ctx_tokens = [c for c in ctx_tokens if c]
        semantic_role_tags = _context_to_semantic_tags(ctx_tokens)
        narrow_ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - _OPERATOR_LEFT_WINDOW) : i + _OPERATOR_NARROW_WINDOW + 1]
        ]
        narrow_ctx_tokens = [c for c in narrow_ctx_tokens if c]
        operator_tags = _detect_operator_tags(ctx_tokens, narrow_ctx_tokens)
        unit_tags = _detect_unit_tags(tok, ctx_tokens)
        polarity = "min" if "min" in operator_tags else ("max" if "max" in operator_tags else "")
        target_entity_tokens = frozenset(
            t for t in ctx_tokens if t in SEMANTIC_ROLE_WORDS or len(t) > 2
        )  # simple heuristic: longer tokens / role words

        mentions_ir.append(
            MentionIR(
                mention_id=mention_id,
                value=tok.value,
                type_bucket=tok.kind,
                raw_surface=tok.raw,
                context_tokens=ctx_tokens,
                sentence_tokens=sent_tokens,
                semantic_role_tags=semantic_role_tags,
                operator_tags=operator_tags,
                unit_tags=unit_tags,
                polarity_or_bound=polarity,
                target_entity_tokens=target_entity_tokens,
                tok=tok,
            )
        )
        mention_id += 1
    return mentions_ir


def _slot_semantic_expansion(param_name: str) -> frozenset[str]:
    """Rule-based slot semantic target tags (enriched with per-unit / total semantics)."""
    n = (param_name or "").lower()
    tags: set[str] = set()
    if "budget" in n:
        tags.update(["budget", "total", "available", "spending_limit", "resource_limit"])
    if "capacity" in n or "limit" in n:
        tags.update(["capacity", "limit", "maximum", "available", "upper_bound"])
    if "demand" in n or "require" in n or "need" in n:
        tags.update(["demand", "requirement", "need", "required"])
    if "profit" in n or "revenue" in n:
        tags.update(["profit", "revenue", "gain", "return"])
    if "cost" in n or "expense" in n or "price" in n:
        tags.update(["cost", "expense", "spend", "price"])
    if "min" in n or "minimum" in n or "atleast" in n:
        tags.update(["minimum", "at_least", "lower_bound"])
    if "max" in n or "maximum" in n or "atmost" in n:
        tags.update(["maximum", "at_most", "upper_bound"])
    if any(w in n for w in ("percent", "percentage", "rate", "fraction", "ratio", "share", "pct", "proportion")):
        tags.update(["percentage", "ratio", "share", "rate", "proportion"])
    # Per-unit / coefficient semantics — helps score against "per", "each" cues
    if any(w in n for w in ("per", "each", "unit", "perunit")):
        tags.update(["per_unit", "coefficient", "unit_rate"])
    # Total / aggregate semantics
    if "total" in n or "available" in n or "aggregate" in n:
        tags.update(["total", "aggregate", "available"])
    if not tags:
        tags.add("quantity")
    return frozenset(tags)


def _build_slot_irs(expected_scalar: list[str]) -> list[SlotIR]:
    """Stage B: Build slot IRs with semantic target tags."""
    slot_irs: list[SlotIR] = []
    for name in expected_scalar:
        et = _expected_type(name)
        aliases = _slot_aliases(name)
        alias_tokens = set()
        for a in aliases:
            alias_tokens.update(_normalize_tokens(a))
        norm_tokens = _slot_measure_tokens(name)
        semantic_target_tags = _slot_semantic_expansion(name)
        op_pref: set[str] = set()
        if any(x in name.lower() for x in ("min", "minimum", "atleast")):
            op_pref.add("min")
        if any(x in name.lower() for x in ("max", "maximum", "atmost")):
            op_pref.add("max")
        unit_pref: set[str] = set()
        if et == "percent":
            unit_pref.add("percent_marker")
        if et == "currency":
            unit_pref.add("currency_marker")
        if et == "int":
            unit_pref.add("integer_like")
        slot_irs.append(
            SlotIR(
                name=name,
                norm_tokens=norm_tokens,
                expected_type=et,
                alias_tokens=alias_tokens,
                semantic_target_tags=semantic_target_tags,
                operator_preference=frozenset(op_pref),
                unit_preference=frozenset(unit_pref),
            )
        )
    return slot_irs


def _score_mention_slot_ir(m: MentionIR, s: SlotIR) -> tuple[float, dict[str, Any]]:
    """Stage C: Interpretable compatibility score (mention IR vs slot IR)."""
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.type_bucket
    expected = s.expected_type

    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return SEMANTIC_IR_WEIGHTS["type_incompatible_penalty"], features

    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (
            expected == "currency" and kind == "currency"
        ):
            score += SEMANTIC_IR_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected == "float" and kind in {"float", "int"}:
            # float+float = exact; integer-valued tokens are valid for float slots
            score += SEMANTIC_IR_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected == "int" and kind == "int":
            score += SEMANTIC_IR_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected in ("int", "float") and kind in {"currency"}:
            score += SEMANTIC_IR_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True
        elif expected == "int" and kind == "float":
            score += SEMANTIC_IR_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True
        elif expected == "currency" and kind in {"float", "int"}:
            # Monetary slots filled with a plain integer/float token (no "$" sign)
            # are a full exact match — the value IS a valid monetary quantity.
            score += SEMANTIC_IR_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True

    semantic_overlap = len(m.semantic_role_tags & s.semantic_target_tags)
    if semantic_overlap:
        score += SEMANTIC_IR_WEIGHTS["semantic_tag_overlap"] * float(semantic_overlap)
        features["semantic_tag_overlap"] = semantic_overlap

    slot_words = set(s.norm_tokens) | s.alias_tokens
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += SEMANTIC_IR_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap
    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += SEMANTIC_IR_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    if m.operator_tags & s.operator_preference:
        if "min" in (m.operator_tags & s.operator_preference):
            score += SEMANTIC_IR_WEIGHTS["operator_min_bonus"]
        else:
            score += SEMANTIC_IR_WEIGHTS["operator_max_bonus"]
        features["operator_match"] = True

    if m.unit_tags & s.unit_preference:
        if "percent_marker" in (m.unit_tags & s.unit_preference):
            score += SEMANTIC_IR_WEIGHTS["unit_percent_bonus"]
        else:
            score += SEMANTIC_IR_WEIGHTS["unit_currency_bonus"]
        features["unit_match"] = True

    entity_overlap = len(m.target_entity_tokens & slot_words)
    if entity_overlap:
        score += SEMANTIC_IR_WEIGHTS["entity_target_bonus"] * float(entity_overlap)
        features["entity_overlap"] = entity_overlap

    if score <= 0.0:
        score += SEMANTIC_IR_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


def _semantic_ir_global_assignment(
    mentions_ir: list[MentionIR],
    slots_ir: list[SlotIR],
) -> tuple[dict[str, MentionIR], dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Stage C: Maximum-weight bipartite matching (deterministic)."""
    assignments: dict[str, MentionIR] = {}
    scores_out: dict[str, float] = {}
    debug: dict[str, list[dict[str, Any]]] = {}

    if not mentions_ir or not slots_ir:
        return assignments, scores_out, debug

    m, s = len(mentions_ir), len(slots_ir)
    cost = [[0.0 for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions_ir):
        for j, sr in enumerate(slots_ir):
            sc, feats = _score_mention_slot_ir(mr, sr)
            cost[i][j] = -sc if sc > -1e8 else 1e9
            if sr.name not in debug:
                debug[sr.name] = []
            debug[sr.name].append(
                {"mention_id": mr.mention_id, "mention_raw": mr.raw_surface, "score": sc, "features": feats}
            )

    for name in debug:
        debug[name].sort(key=lambda x: x["score"], reverse=True)

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        cost_arr = np.array(cost)
        row_ind, col_ind = linear_sum_assignment(cost_arr)
        for ri, cj in zip(row_ind, col_ind):
            if cost_arr[ri, cj] < 1e8:
                sr = slots_ir[cj]
                mr = mentions_ir[ri]
                sc, _ = _score_mention_slot_ir(mr, sr)
                assignments[sr.name] = mr
                scores_out[sr.name] = sc
    except ImportError:
        from math import inf
        num_states = 1 << s
        dp = [[-inf] * num_states for _ in range(m + 1)]
        parent = [[(-1, None)] * num_states for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(m):
            for mask in range(num_states):
                if dp[i][mask] == -inf:
                    continue
                if dp[i][mask] > dp[i + 1][mask]:
                    dp[i + 1][mask] = dp[i][mask]
                    parent[i + 1][mask] = (mask, None)
                for j in range(s):
                    if (mask >> j) & 1:
                        continue
                    sc = -cost[i][j]
                    if sc <= 0:
                        continue
                    new_mask = mask | (1 << j)
                    val = dp[i][mask] + sc
                    if val > dp[i + 1][new_mask]:
                        dp[i + 1][new_mask] = val
                        parent[i + 1][new_mask] = (mask, j)
        best_mask = max(range(num_states), key=lambda mk: dp[m][mk])
        i, mask = m, best_mask
        used_slot_for_mention = {}
        while i > 0:
            prev_mask, j = parent[i][mask]
            if j is not None:
                used_slot_for_mention[i - 1] = j
            mask = prev_mask
            i -= 1
        for mi, cj in used_slot_for_mention.items():
            sr = slots_ir[cj]
            mr = mentions_ir[mi]
            sc, _ = _score_mention_slot_ir(mr, sr)
            assignments[sr.name] = mr
            scores_out[sr.name] = sc

    return assignments, scores_out, debug


def _validate_slot_assignment(
    slot_name: str,
    m: MentionIR,
    s: SlotIR,
    score: float,
) -> tuple[bool, float]:
    """Stage D: Plausibility check for one assignment. Returns (valid, adjustment)."""
    if _is_type_incompatible(s.expected_type, m.type_bucket):
        return False, REPAIR_WEIGHTS["inconsistency_penalty"]
    adj = 0.0
    if m.semantic_role_tags & s.semantic_target_tags:
        adj += REPAIR_WEIGHTS["validation_bonus"]
    if (m.operator_tags & s.operator_preference) or (m.unit_tags & s.unit_preference):
        adj += REPAIR_WEIGHTS["validation_bonus"]
    if score < 0.5 and not (m.semantic_role_tags & s.semantic_target_tags):
        adj += REPAIR_WEIGHTS["inconsistency_penalty"]
    return adj >= 0 or score > 1.0, adj


def _validation_and_repair(
    mentions_ir: list[MentionIR],
    slots_ir: list[SlotIR],
    initial_assignments: dict[str, MentionIR],
    initial_scores: dict[str, float],
    debug: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, MentionIR], dict[str, str]]:
    """Stage D: Deterministic validation and coverage-preserving repair."""
    filled = dict(initial_assignments)
    filled_in_repair: dict[str, str] = {}  # slot -> "initial" | "repair"

    for slot in filled:
        filled_in_repair[slot] = "initial"

    used_mention_ids = {m.mention_id for m in filled.values()}
    slot_list = list(slots_ir)
    slot_by_name = {s.name: s for s in slot_list}

    for s in slot_list:
        if s.name in filled:
            m = filled[s.name]
            score = initial_scores.get(s.name, 0.0)
            valid, _ = _validate_slot_assignment(s.name, m, s, score)
            if not valid and score < REPAIR_WEIGHTS["min_semantic_support"]:
                del filled[s.name]
                del filled_in_repair[s.name]
                used_mention_ids.discard(m.mention_id)

    unfilled = [s for s in slot_list if s.name not in filled]
    unfilled.sort(key=lambda x: len(x.semantic_target_tags) + len(x.alias_tokens), reverse=True)

    for s in unfilled:
        candidates = debug.get(s.name, [])
        for cand in candidates:
            mid = cand.get("mention_id")
            if mid is None:
                continue
            mr = next((m for m in mentions_ir if m.mention_id == mid), None)
            if not mr or mr.mention_id in used_mention_ids:
                continue
            sc = cand.get("score", 0.0)
            if sc <= 0 or _is_type_incompatible(s.expected_type, mr.type_bucket):
                continue
            valid, adj = _validate_slot_assignment(s.name, mr, s, sc)
            if valid or sc + adj > 0.5:
                filled[s.name] = mr
                filled_in_repair[s.name] = "repair"
                used_mention_ids.add(mr.mention_id)
                break

    return filled, filled_in_repair


def _run_semantic_ir_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[dict[str, Any], dict[str, MentionIR], dict[str, str]]:
    """Run full pipeline A→B→C→D. Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions_ir = _extract_enriched_mentions(query, variant)
    slots_ir = _build_slot_irs(expected_scalar)
    if not mentions_ir or not slots_ir:
        return filled_values, filled_mentions, filled_in_repair

    initial_assignments, initial_scores, debug = _semantic_ir_global_assignment(
        mentions_ir, slots_ir
    )
    filled, filled_in_repair = _validation_and_repair(
        mentions_ir, slots_ir, initial_assignments, initial_scores, debug
    )

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


# --- Optimization-role-aware assignment (optimization_role_repair) ---

# Words/phrases that imply optimization roles (word or phrase -> role tag). Modular, easy to extend.
OPT_ROLE_WORDS: dict[str, str] = {}
for _tag, _words in [
    ("objective_coeff", ["maximize", "minimize", "profit", "revenue", "return", "earns", "yields", "gain"]),
    ("unit_profit", ["profit", "revenue", "return", "per", "each", "unit", "yields", "earns"]),
    ("unit_revenue", ["revenue", "return", "per", "each", "unit"]),
    ("unit_return", ["return", "per", "each", "unit"]),
    ("unit_cost", ["cost", "costs", "expense", "spend", "per", "each", "unit", "price"]),
    ("resource_consumption", ["requires", "uses", "consumes", "per", "each", "unit"]),
    ("capacity_limit", ["capacity", "available", "can supply", "at most", "no more than", "limit", "limits"]),
    ("demand_requirement", ["demand", "required", "requirement", "must meet", "must satisfy", "needed", "at least", "minimum"]),
    ("total_budget", ["budget", "total budget", "available budget", "spending limit", "available money", "total amount"]),
    ("total_available", ["available", "total capacity", "hours available", "supply available", "available amount"]),
    ("lower_bound", ["at least", "minimum", "no less than", "least", "lower", "min"]),
    ("upper_bound", ["at most", "maximum", "no more than", "most", "upper", "max", "up to", "limited to", "cannot exceed"]),
    ("ratio_constraint", ["ratio", "fraction", "proportion", "share"]),
    ("percentage_constraint", ["percent", "percentage", "rate", "%", "pct"]),
    ("share_constraint", ["share", "fraction", "ratio", "proportion"]),
    ("fixed_cost", ["fixed", "fixed cost", "fixed charge", "setup", "setup cost"]),
    ("penalty", ["penalty", "penalties"]),
    ("setup_cost", ["setup", "setup cost", "fixed charge"]),
    ("time_requirement", ["time", "hours", "days", "minutes", "weeks"]),
    ("quantity_limit", ["quantity", "amount", "number", "count", "limit"]),
    ("cardinality_limit", ["number", "count", "items", "units", "at most", "at least"]),
    ("minimum_requirement", ["minimum", "at least", "least", "min", "required"]),
    ("maximum_allowance", ["maximum", "at most", "most", "max", "up to", "limited"]),
]:
    for _w in _words:
        OPT_ROLE_WORDS[_w.lower().replace(" ", "_")] = _tag
        OPT_ROLE_WORDS[_w.lower().replace(" ", "")] = _tag
        if " " in _w:
            for _t in _w.lower().split():
                OPT_ROLE_WORDS[_t] = _tag

# Per-unit patterns (token sequences).
PER_UNIT_PATTERNS = ["each", "for each", "per", "per unit", "each unit", "each item", "each product", "for every"]
OBJECTIVE_PATTERNS = ["maximize", "minimize", "profit", "revenue", "return", "cost"]
BOUND_PATTERNS = ["at least", "at most", "no less than", "no more than", "minimum", "maximum", "up to", "limited to", "cannot exceed"]
TOTAL_RESOURCE_PATTERNS = ["total budget", "available budget", "hours available", "total capacity", "available amount", "supply available"]
RATIO_PATTERNS = ["percent", "percentage", "ratio", "fraction", "share", "proportion"]
REQUIREMENT_PATTERNS = ["demand", "required", "requirement", "needed", "must satisfy", "must meet"]

# Unit markers for optimization context.
OPT_UNIT_PERCENT = {"%", "percent", "percentage", "pct"}
OPT_UNIT_CURRENCY = {"$", "€", "dollar", "dollars", "usd", "eur", "cost", "price", "budget"}
OPT_UNIT_COUNT = {"number", "count", "items", "units", "quantity"}
OPT_UNIT_TIME = {"hour", "hours", "day", "days", "minute", "minutes", "week", "weeks", "time"}
OPT_UNIT_DECIMAL = set()  # decimal form inferred from value

# Weights for optimization-role scoring (role overlap stronger than plain lexical).
OPT_ROLE_WEIGHTS = {
    "type_exact_bonus": 4.0,
    "type_loose_bonus": 1.5,
    "type_incompatible_penalty": -1e9,
    "opt_role_overlap": 3.0,
    "fragment_compat_bonus": 1.5,
    "operator_match_bonus": 1.2,
    "lex_context_overlap": 0.5,
    "lex_sentence_overlap": 0.2,
    "unit_match_bonus": 2.0,
    "entity_resource_overlap": 0.8,
    "coefficient_vs_total_bonus": 1.2,
    "coeff_to_total_local_penalty": -2.0,  # per-unit mention → total-like slot
    "total_to_coeff_local_penalty": -2.0,  # total mention → coefficient-like slot
    "weak_match_penalty": -1.0,
    "schema_prior_bonus": 0.5,
    # Quantity-role layer bonuses / penalties (added by explicit primary_role field).
    "count_mention_count_slot_bonus": 2.5,   # is_count_like mention → count-like slot
    "bound_direction_bonus": 1.5,            # lower-bound mention → min slot; upper → max
    "bound_direction_penalty": -3.0,         # lower-bound mention → max-only slot (or vice versa)
    "count_mention_non_count_penalty": -1.5, # count-context mention → non-count slot
}
OPT_REPAIR_WEIGHTS = {
    "role_plausibility_bonus": 0.6,
    "total_vs_coeff_penalty": -1.5,
    "bound_plausibility_bonus": 0.5,
    "coverage_repair_bonus": 1.2,
    "min_role_support": 0.4,
}

# ── Global Consistency Grounding (global_consistency_grounding) ──────────────
# Configurable weights / thresholds for the new global-assignment method.

# Local scoring weights (reuse OPT_ROLE_WEIGHTS signals but tuneable separately).
GCG_LOCAL_WEIGHTS: dict[str, float] = {
    "type_exact_bonus": 4.0,
    "type_loose_bonus": 1.5,
    "type_incompatible_penalty": -1e9,
    "opt_role_overlap": 3.0,
    "fragment_compat_bonus": 1.5,
    "operator_match_bonus": 1.2,
    "lex_context_overlap": 0.5,
    "lex_sentence_overlap": 0.2,
    "unit_match_bonus": 2.0,
    "entity_resource_overlap": 0.8,
    "coefficient_vs_total_bonus": 1.2,
    "coeff_to_total_local_penalty": -2.0,  # per-unit mention → total-like slot
    "total_to_coeff_local_penalty": -2.0,  # total mention → coefficient-like slot
    "schema_prior_bonus": 0.5,
    "weak_match_penalty": -1.0,
    # Count-like slot priors: favour small cardinalities, penalise large values.
    "count_small_int_prior": 2.0,        # bonus for int in [1, count_plausible_max]
    "count_large_int_penalty": -2.0,     # penalty for int > count_large_penalty_threshold
    "count_plausible_max": 10,           # integers ≤ this get the small-int bonus
    "count_large_penalty_threshold": 50, # integers > this get the large-int penalty
    # Enumeration-derived count candidates must not fill non-count slots.
    "derived_count_non_count_penalty": -1e9,
    # Quantity-role layer bonuses / penalties (added by explicit primary_role field).
    "count_mention_count_slot_bonus": 2.5,   # is_count_like mention → count-like slot
    "bound_direction_bonus": 1.5,            # lower-bound mention → min slot; upper → max
    "bound_direction_penalty": -3.0,         # lower-bound mention → max-only slot (or vice versa)
    "count_mention_non_count_penalty": -1.5, # count-context mention → non-count slot
    # Narrow-left entity anchor: tight 3-4-token left-window overlap with slot name/aliases.
    # This is a stronger signal than lex_context_overlap (0.5) because the narrow window is
    # entity-discriminating — e.g. narrow_left=("product","a","requires") only overlaps with
    # ProductA slots, not ProductB slots, even when the full context contains both entities.
    "narrow_left_overlap": 2.0,
    # Tight-context cost/profit semantic hints.
    # Fires only when the mention is UNambiguously cost-like (not also profit-like) or
    # profit-like (not also cost-like).  The tight ±2-token window is less prone to cross-
    # mention bleeding than wider contexts, so this flag reliably indicates semantic domain.
    # Example: "profit is 8 and cost is 4" → val=4 has is_cost_like=True, is_profit_like=False
    # → bonus for cost-containing slots, penalty for profit-containing slots.
    "tight_cost_match_bonus": 2.0,
    "tight_cost_mismatch_penalty": -2.5,
    "tight_profit_match_bonus": 2.0,
    "tight_profit_mismatch_penalty": -2.5,
}

# Global consistency rewards/penalties applied to a full assignment as a whole.
GCG_GLOBAL_WEIGHTS: dict[str, float] = {
    "coverage_reward_per_slot": 0.8,     # reward per slot that is filled
    "type_consistency_reward": 0.5,      # reward per slot whose mention type matches expected
    "percent_misuse_penalty": -3.0,      # percent mention → non-percent slot (when pct exists)
    "non_percent_to_pct_slot_penalty": -3.0,  # non-percent mention → percent slot
    "total_to_coeff_penalty": -2.0,      # total-looking mention → per-unit (coeff) slot
    "coeff_to_total_penalty": -2.0,      # per-unit-looking mention → total slot
    "bound_flip_penalty": -2.5,          # min mention → max slot or vice versa
    "duplicate_mention_penalty": -5.0,   # same mention used for two slots (one-to-one violation)
    "plausibility_coverage_bonus": 1.0,  # bonus if ≥ 80 % of slots are filled
    # Entity coherence: when a mention has no direct entity letter anchor in its
    # narrow_left, it should be assigned to the same entity as the immediately
    # preceding mention.  This repairs "Feed B 7 protein AND 15 fat" where fat
    # carries no explicit entity marker but must follow the B-anchored protein.
    "entity_coherence_reward": 1.5,      # unanchored mention → same entity as preceding
    "entity_coherence_penalty": -2.0,    # unanchored mention → different entity than preceding
}

# Candidate pruning: pairs with local score below this are removed from consideration.
GCG_PRUNE_THRESHOLD: float = -0.5

# Beam width for beam search over partial assignments.
GCG_BEAM_WIDTH: int = 8


@dataclass(frozen=True)
class MentionOptIR:
    """Optimization-role intermediate representation for a numeric mention."""

    mention_id: int
    value: float | None
    type_bucket: str
    raw_surface: str
    role_tags: frozenset[str]
    operator_tags: frozenset[str]
    unit_tags: frozenset[str]
    fragment_type: str  # "objective" | "constraint" | "resource" | "ratio" | "bound" | ""
    is_per_unit: bool
    is_total_like: bool
    nearby_entity_tokens: frozenset[str]
    nearby_resource_tokens: frozenset[str]
    nearby_product_tokens: frozenset[str]
    context_tokens: list[str]
    sentence_tokens: list[str]
    tok: NumTok
    # ── Quantity-role layer (explicit primary-role signals) ──────────────────
    # These are derived from type_bucket, operator_tags, role_tags, and context
    # and provide clean boolean access for role-aware scoring without extra lookups.
    is_count_like: bool = False          # small int with count-context nouns nearby, or derived_count
    is_lower_bound_like: bool = False    # operator_tags contains "min"
    is_upper_bound_like: bool = False    # operator_tags contains "max"
    is_percent_like: bool = False        # type_bucket == "percent"
    primary_role: str = "generic"        # dominant role: count | total | coefficient | lower_bound | upper_bound | percent | generic
    # Fine-grained bound role (requires _ENABLE_BOUND_ROLE_LAYER=True).
    # Values: lower_inclusive | lower_exclusive | upper_inclusive | upper_exclusive
    #         range_low | range_high | unknown
    bound_role: str = "unknown"
    # ── Narrow local context (Stage 3 measure/attribute-aware linking) ───────
    # Tokens from the directional narrow windows (±left + ±right) already
    # computed for is_per_unit / is_total_like detection.  This is a MUCH
    # tighter window than context_tokens (±14) and stays within clause
    # boundaries, making it suitable for measure-attribute overlap scoring.
    narrow_context_tokens: tuple[str, ...] = ()
    # ── Group 3: directional left-anchor tokens ───────────────────────────────
    # Only the LEFT portion of the narrow window (tokens to the LEFT of this
    # mention, up to _LOCALITY_LEFT_WINDOW=4).  Keeping left and right separate
    # prevents the right context of one mention from leaking entity cues from
    # the following sibling clause into the entity alignment score.
    # Example: "Product B requires 7 … and product A requires 3"
    #   - narrow_left_tokens of 7 = ("product", "b", "requires")
    #   - narrow_left_tokens of 3 = ("and", "product", "a", "requires")
    # This lets left_entity_anchor_overlap discriminate slot B vs slot A.
    narrow_left_tokens: tuple[str, ...] = ()
    # ── Group 1 role-family flags (Step 2 structured mention representation) ──
    # Derived from the ±2 token tight window immediately surrounding the mention.
    # Using a tight window prevents cross-mention bleeding.  These flags feed:
    #   - role_family_mismatch distractor suppression (Step 4)
    #   - transparency / ablation diagnostics
    is_cost_like: bool = False      # cost/expense/price in tight context
    is_profit_like: bool = False    # profit/revenue/yield in tight context
    is_demand_like: bool = False    # demand/required/needed in tight context
    is_resource_like: bool = False  # labor/material/resource in tight context
    is_time_like: bool = False      # hour/hours/time/day in tight context


@dataclass(frozen=True)
class SlotOptIR:
    """Slot with optimization-role priors for matching."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    alias_tokens: set[str]
    slot_role_tags: frozenset[str]
    operator_preference: frozenset[str]
    unit_preference: frozenset[str]
    is_objective_like: bool
    is_bound_like: bool
    is_total_like: bool
    is_coefficient_like: bool
    is_count_like: bool = False


def _context_to_opt_role_tags(context_tokens: list[str]) -> frozenset[str]:
    """Infer optimization-role tags from context (rule-based)."""
    tags: set[str] = set()
    ctx_lower = [t.lower().strip(".,;:()[]{}") for t in context_tokens if t]
    ctx_set = set(ctx_lower)
    ctx_str = " ".join(ctx_lower)
    for w, tag in OPT_ROLE_WORDS.items():
        if w in ctx_set or w.replace("_", " ") in ctx_str or w in ctx_str:
            tags.add(tag)
    return frozenset(tags)


def _classify_fragment_type(context_tokens: list[str]) -> str:
    """Classify mention context as objective-like, constraint-like, resource-like, ratio-like, or bound-like."""
    ctx = set(t.lower() for t in context_tokens if t)
    ctx_str = " ".join(t.lower() for t in context_tokens if t)
    if ctx & {"maximize", "minimize", "profit", "revenue", "return", "earns", "yields"} or any(
        p in ctx_str for p in OBJECTIVE_PATTERNS
    ):
        return "objective"
    if ctx & {"at most", "at least", "cannot exceed", "limited", "must be", "no more", "no less"} or any(
        p in ctx_str for p in BOUND_PATTERNS
    ):
        return "constraint"
    if ctx & {"available", "capacity", "hours available", "budget", "supply"} or any(
        p in ctx_str for p in TOTAL_RESOURCE_PATTERNS
    ):
        return "resource"
    if ctx & {"percent", "percentage", "ratio", "fraction", "share", "proportion", "%"} or any(
        p in ctx_str for p in RATIO_PATTERNS
    ):
        return "ratio"
    if ctx & {"at least", "at most", "minimum", "maximum", "lower", "upper", "min", "max"}:
        return "bound"
    return ""


def _detect_opt_unit_tags(tok: NumTok, context_tokens: list[str]) -> frozenset[str]:
    """Unit markers for optimization: percent_marker, currency_marker, count_marker, time_marker, decimal_marker."""
    out: set[str] = set()
    ctx = set(t.lower() for t in context_tokens if t)
    if tok.kind == "percent" or ctx & OPT_UNIT_PERCENT or (tok.raw or "").endswith("%"):
        out.add("percent_marker")
    if tok.kind == "currency" or "$" in (tok.raw or "") or ctx & OPT_UNIT_CURRENCY:
        out.add("currency_marker")
    if ctx & OPT_UNIT_COUNT or (tok.kind == "int" and tok.value is not None):
        out.add("count_marker")
    if ctx & OPT_UNIT_TIME:
        out.add("time_marker")
    if tok.kind == "float" or (tok.value is not None and float(int(tok.value)) != tok.value):
        out.add("decimal_marker")
    return frozenset(out)


def _compute_is_count_like_mention(
    tok: NumTok,
    role_tags: frozenset[str],
    context_tokens: list[str],
) -> bool:
    """Return True when this mention is likely a count/cardinality value.

    Criteria (any one is sufficient):
    - Marked as a derived-count from enumeration analysis (reliable signal).
    - Small positive integer (value in [1, 20], int-valued) AND at least one
      count-context noun appears in the nearby context tokens (e.g. "three types").

    Importantly, ``cardinality_limit`` in role_tags is NOT sufficient on its own
    because it can fire for "units" in "100 units must be produced" (a production
    target, not a cardinality count).  Only explicit ``derived_count`` tagging or
    the small-int + count-noun rule produces a reliable count signal.
    """
    if "derived_count" in role_tags:
        return True
    val = tok.value
    if val is None:
        return False
    if not (1 <= val <= 20 and float(int(val)) == val):
        return False
    ctx_set = set(t.lower().strip(".,;:()[]{}") for t in context_tokens if t)
    return bool(ctx_set & _COUNT_CONTEXT_NOUNS)


def _compute_primary_role(
    tok: NumTok,
    is_count_like: bool,
    is_lower_bound_like: bool,
    is_upper_bound_like: bool,
    is_percent_like: bool,
    is_total_like: bool,
    is_per_unit: bool,
) -> str:
    """Synthesize a single primary-role string from the binary role signals.

    Priority order (from most to least specific):
      1. count           – explicit cardinality
      2. percent         – fractional/rate value
      3. lower_bound     – min/at-least constraint
      4. upper_bound     – max/at-most constraint
      5. coefficient     – per-unit coefficient
      6. total           – global total/capacity/budget
      7. generic         – fallback
    """
    if is_count_like:
        return "count"
    if is_percent_like:
        return "percent"
    if is_lower_bound_like and not is_upper_bound_like:
        return "lower_bound"
    if is_upper_bound_like and not is_lower_bound_like:
        return "upper_bound"
    if is_per_unit and not is_total_like:
        return "coefficient"
    if is_total_like and not is_per_unit:
        return "total"
    return "generic"


def _find_range_annotations(toks: list[str]) -> dict[int, str]:
    """Detect range expressions and return a map from token index → bound role.

    Handles:
      - "between X and Y"  →  X: range_low,  Y: range_high
      - "from X to Y"      →  X: range_low,  Y: range_high
      - "X to Y" (bare)    →  X: range_low,  Y: range_high  (only when X is a number)

    Only fires when _ENABLE_BOUND_ROLE_LAYER is True.
    """
    if not _ENABLE_BOUND_ROLE_LAYER:
        return {}
    result: dict[int, str] = {}
    clean = [t.lower().strip(".,;:()[]{}") for t in toks]
    n = len(clean)

    def _is_number_token(idx: int) -> bool:
        raw = toks[idx].strip().rstrip(",;:()[]{}").rstrip(".")
        return bool(NUM_TOKEN_RE.fullmatch(raw)) or clean[idx] in _WORD_FRACTIONS

    for i, tok in enumerate(clean):
        if tok in ("between", "from"):
            bridge_words = {"and", "to", "-"} if tok == "between" else {"to"}
            nums: list[int] = []
            j = i + 1
            while j < n and len(nums) < 2:
                if _is_number_token(j):
                    nums.append(j)
                elif clean[j] not in bridge_words:
                    break
                j += 1
            if len(nums) == 2:
                result[nums[0]] = "range_low"
                result[nums[1]] = "range_high"

    # Bare "X to Y" range: only when the first number token is already known
    # (via digit recognition) and is immediately followed by "to" and another number.
    # This avoids false positives on "send to", "up to N" (already handled above).
    for i in range(n - 2):
        if i in result:
            continue  # already annotated
        if not _is_number_token(i):
            continue
        if clean[i + 1] != "to":
            continue
        if i + 2 < n and _is_number_token(i + 2) and (i + 2) not in result:
            # Guard: ensure "to" is not part of a "up to" / "from X to" phrase already handled.
            if i > 0 and clean[i - 1] in ("up", "upto", "from", "between"):
                continue
            result[i] = "range_low"
            result[i + 2] = "range_high"

    return result


def _compute_bound_role(
    operator_tags: frozenset[str],
    ctx_str: str,
    range_annotation: str = "",
) -> str:
    """Compute fine-grained bound role.

    Returns one of:
      lower_inclusive  – at least, no fewer than, ≥
      lower_exclusive  – more than, greater than, >
      upper_inclusive  – at most, no more than, ≤
      upper_exclusive  – fewer than, less than, <
      range_low        – lower end of "between X and Y"
      range_high       – upper end of "between X and Y"
      unknown          – no operator cue detected
    """
    if not _ENABLE_BOUND_ROLE_LAYER:
        return "unknown"
    if range_annotation in ("range_low", "range_high"):
        return range_annotation
    if "min" in operator_tags and "max" not in operator_tags:
        # If an inclusive standard pattern matched (e.g. "greater than or equal to"),
        # do NOT also fire exclusive via the "greater than" sub-phrase.
        _has_inclusive = any(p in ctx_str for p in _OPERATOR_MIN_PATTERNS)
        if not _has_inclusive and any(p in ctx_str for p in _OPERATOR_MIN_EXCL_PATTERNS):
            return "lower_exclusive"
        return "lower_inclusive"
    if "max" in operator_tags and "min" not in operator_tags:
        _has_inclusive = any(p in ctx_str for p in _OPERATOR_MAX_PATTERNS)
        if not _has_inclusive and any(p in ctx_str for p in _OPERATOR_MAX_EXCL_PATTERNS):
            return "upper_exclusive"
        return "upper_inclusive"
    return "unknown"


def _extract_opt_role_mentions(query: str, variant: str) -> list[MentionOptIR]:
    """Stage 1: Optimization-aware mention extraction with role tags.

    Recognises both digit-based tokens (e.g. "100", "$5000", "20%") and
    written-word numbers (e.g. "two", "twenty-five", "one hundred").
    Multi-token word-number spans such as "two hundred fifty" are collapsed
    into a single mention so that compound values are not split.
    """
    toks = query.split()
    sent_tokens = [t.lower().strip(".,;:()[]{}") for t in toks]
    # Pre-compute range annotations ("between X and Y", "from X to Y").
    range_annotations = _find_range_annotations(toks)
    mentions: list[MentionOptIR] = []
    mention_id = 0
    i = 0
    while i < len(toks):
        w = toks[i]
        span_size = 1  # number of tokens consumed by this mention

        if w == "<num>" and variant in ("noisy", "nonum"):
            tok = NumTok(raw=w, value=None, kind="unknown")
        else:
            # ── digit-based token ────────────────────────────────────────────
            m = NUM_TOKEN_RE.fullmatch(w.strip().rstrip(",;:()[]{}").rstrip("."))
            if m:
                ctx_tokens = [
                    x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 14) : i + 15]
                ]
                ctx_tokens = [c for c in ctx_tokens if c]
                tok = _parse_num_token(w, set(ctx_tokens))
            else:
                # ── fraction word (half, one-third, quarter, …) ─────────────
                # Try the multi-token fraction form first (e.g. "one-half"),
                # then fall back to single-token forms (e.g. "half").
                _w_clean = w.lower().strip(".,;:()[]{}\"'")
                _frac_val = _WORD_FRACTIONS.get(_w_clean)
                if _frac_val is not None:
                    tok = NumTok(raw=_w_clean, value=_frac_val, kind="percent")
                    # span_size stays 1 (single-token fraction word)
                else:
                    # ── written-word number (single or multi-token span) ─────────
                    wval, consumed = _parse_word_num_span(toks, i)
                    if wval is None:
                        i += 1
                        continue
                    span_size = consumed
                    j = i + consumed  # first token after the span
                    # Use wider context centred on span start, same window as digit path.
                    ctx_tokens = [
                        x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 14) : i + 15]
                    ]
                    ctx_tokens = [c for c in ctx_tokens if c]
                    raw_surface = " ".join(toks[i:j])
                    tok = _classify_word_num_tok(raw_surface, wval, set(ctx_tokens), toks, j)

        ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 14) : i + 15]
        ]
        ctx_tokens = [c for c in ctx_tokens if c]
        ctx_set = set(ctx_tokens)
        ctx_str = " ".join(ctx_tokens)
        role_tags = _context_to_opt_role_tags(ctx_tokens)
        # Build operator narrow-context tokens with sentence-boundary stop on the
        # LEFT side to prevent cross-sentence contamination.  For example, in
        # "at least 50 units. The profit is 20 per unit", the left window of "20"
        # would otherwise include "at least" from the previous sentence and
        # incorrectly tag "20" as a lower-bound value.
        _op_left_raw = toks[max(0, i - _OPERATOR_LEFT_WINDOW) : i]
        _op_left_cleaned = [x.lower().strip(".,;:()[]{}") for x in _op_left_raw]
        # Find the rightmost sentence boundary (token ending '.'/'!'/'?' followed by
        # an uppercase token) and discard everything before it.  Scan in reverse so
        # we stop at the first (rightmost) boundary we encounter.
        _op_boundary = 0
        for _bk in range(len(_op_left_raw) - 2, -1, -1):
            _braw = _op_left_raw[_bk].rstrip()
            _braw_next = _op_left_raw[_bk + 1]
            if (
                _braw.endswith((".", "!", "?"))
                and _braw_next[:1].isupper()
                and _braw_next.strip()
            ):
                _op_boundary = _bk + 1
                break
        _op_left_cleaned = _op_left_cleaned[_op_boundary:]
        _op_right_cleaned = [
            x.lower().strip(".,;:()[]{}") for x in toks[i : i + _OPERATOR_NARROW_WINDOW + 1]
        ]
        narrow_ctx_tokens = [c for c in _op_left_cleaned + _op_right_cleaned if c]
        operator_tags = _detect_operator_tags(ctx_tokens, narrow_ctx_tokens)
        unit_tags = _detect_opt_unit_tags(tok, ctx_tokens)
        fragment_type = _classify_fragment_type(ctx_tokens)
        # Use directional narrow windows to detect per-unit vs total-like role,
        # preventing cross-contamination when both cue types appear in the same
        # wide context window (e.g. "2000 hours available" and "each requires 2").
        #
        # Left context: collect up to _LOCALITY_LEFT_WINDOW tokens but discard
        # everything before the most recent sentence boundary (a token ending
        # '.'/'!'/'?' followed by an uppercase token).  This mirrors the
        # sentence-boundary trimming already applied to the right window and
        # prevents tokens from the preceding sentence from contaminating the
        # left anchor (e.g. "hours. Product B requires 5" — "hours" should
        # NOT appear in the left context of "5").
        _left_raw_window = toks[max(0, i - _LOCALITY_LEFT_WINDOW) : i]
        # Find the last sentence boundary inside the left window.
        # A boundary is at position k when _left_raw_window[k] ends with
        # '.'/'!'/'?' AND _left_raw_window[k+1] starts with an uppercase letter.
        _left_start = 0
        for _k in range(len(_left_raw_window) - 1):
            _lw_tok = _left_raw_window[_k].rstrip()
            _lw_next = _left_raw_window[_k + 1]
            if (
                _lw_tok.endswith((".", "!", "?"))
                and _lw_next[:1].isupper()
                and bool(_lw_next.strip())
            ):
                # Discard everything up to and including this boundary token;
                # keep only tokens from _k+1 onward.
                _left_start = _k + 1
        _left_narrow = [
            x.lower().strip(".,;:()[]{}") for x in _left_raw_window[_left_start:]
            if x.strip(".,;:()[]{}").strip()
        ]
        # Right context: collect up to _LOCALITY_RIGHT_WINDOW tokens but stop at
        # sentence boundaries.  A sentence boundary is detected when a token ends
        # with '.' / '!' / '?' AND the *next* token starts with an uppercase letter.
        # This prevents cues in a following sentence (e.g. "available" in
        # "2 hours. There are 2000 hours available.") from contaminating the right
        # context of a per-unit coefficient that belongs to the preceding sentence.
        # Also stop at comma + coordinating conjunction (", and" / ", or" / ", but"
        # etc.) which marks a clause boundary within a single sentence.  This
        # prevents "4 labor hours, and total budget is 5000" from making the "4"
        # appear total-like because "total" leaks into its right window.
        _right_narrow: list[str] = []
        _right_raw_window = toks[i + span_size : i + span_size + _LOCALITY_RIGHT_WINDOW]
        _CLAUSE_CONJUNCTIONS: frozenset[str] = frozenset({"and", "or", "but", "nor", "while", "whereas"})
        for _k, _tok_raw in enumerate(_right_raw_window):
            _tok_clean = _tok_raw.lower().strip(".,;:()[]{}").strip()
            if _tok_clean:
                _right_narrow.append(_tok_clean)
            _stripped_raw = _tok_raw.rstrip()
            # Sentence boundary: period / ! / ? at end of token AND next token is
            # capitalised (new sentence starts).  Abbreviations like "sq." or "ft."
            # are typically followed by lowercase, so they do not trigger a break.
            if (
                _stripped_raw.endswith((".", "!", "?"))
                and _k + 1 < len(_right_raw_window)
                and _right_raw_window[_k + 1][:1].isupper()
                # Guard against an empty next token producing no uppercase char
                and bool(_right_raw_window[_k + 1].strip())
            ):
                break
            # Clause boundary: comma at the end of the current token AND the next
            # token is a coordinating conjunction.  Stops "total" in "4 hours, and
            # total budget is 5000" from contaminating the right context of "4".
            if (
                _stripped_raw.endswith(",")
                and _k + 1 < len(_right_raw_window)
                and _right_raw_window[_k + 1].lower().strip() in _CLAUSE_CONJUNCTIONS
            ):
                break
            # Numeric-token boundary: stop when a new numeric literal appears in the
            # right window.  This prevents cross-measure contamination in patterns
            # like "3 heating hours and 5 cooling hours" where the right context of
            # "3" would otherwise include both "heating" and "cooling".  Stopping
            # at the next number keeps each measure cue local to its own number.
            # Only fires after the first token so that units immediately following
            # the current mention (e.g. "3 hours" → right[0]="hours") are included.
            if _k >= 1 and NUM_TOKEN_RE.fullmatch(_tok_clean):
                # Remove the just-appended numeric token; it is not a measure word.
                _right_narrow.pop()
                break
        _left_set = set(_left_narrow)
        _right_set = set(_right_narrow)
        _left_str = " ".join(_left_narrow)
        # Per-unit determiner window is narrower (±2) than the verb window (±4).
        # Governing verbs like "requires" appear immediately before the number (-1 to -2),
        # but determiners like "each" can appear further back (-3 to -4) in "each X requires N".
        # The narrow determiner window prevents cross-sentence contamination where "each"
        # ends one clause and the next clause starts with a global total.
        _left_det_set = set(
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 2) : i]
            if x.strip(".,;:()[]{}").strip()
        )
        is_per_unit = (
            bool(_left_set & _PER_UNIT_LEFT_VERBS)
            or bool(_left_det_set & _PER_UNIT_LEFT_DETERMINERS)
            or any(p in _left_str for p in _PER_UNIT_LEFT_PHRASES)
        )
        # Wide context string for total-phrase patterns (avoids narrow window issues).
        _ctx_str_wide = " ".join(ctx_tokens)
        is_total_like = bool(
            _left_set & _TOTAL_LEFT_CUES
            or _right_set & _TOTAL_RIGHT_CUES
            or any(p in _ctx_str_wide for p in _TOTAL_PHRASE_PATTERNS)
        )
        entity_tokens = frozenset(t for t in ctx_tokens if len(t) > 2 and t in OPT_ROLE_WORDS)
        resource_tokens = frozenset(
            t for t in ctx_tokens if t in {"capacity", "budget", "available", "limit", "resource", "hours", "time"}
        )
        product_tokens = frozenset(
            t for t in ctx_tokens if t in {"item", "product", "unit", "each", "demand", "quantity"}
        )

        # ── Quantity-role layer: derive explicit boolean role fields ──────────
        _is_count_like = _compute_is_count_like_mention(tok, role_tags, ctx_tokens)
        # Range annotations override operator_tags for "between X and Y" / "from X to Y".
        _range_anno = range_annotations.get(i, "")
        if _range_anno == "range_low":
            operator_tags = frozenset(operator_tags | {"min"})
        elif _range_anno == "range_high":
            operator_tags = frozenset(operator_tags | {"max"})
        _is_lower_bound = "min" in operator_tags
        _is_upper_bound = "max" in operator_tags
        _is_pct = tok.kind == "percent"
        _primary_role = _compute_primary_role(
            tok, _is_count_like, _is_lower_bound, _is_upper_bound, _is_pct, is_total_like, is_per_unit
        )
        # Use the narrow (left-biased) context for bound_role to keep disambiguation
        # tight to the governing operator phrase.
        _narrow_ctx_str = " ".join(narrow_ctx_tokens)
        _bound_role = _compute_bound_role(operator_tags, _narrow_ctx_str, _range_anno)

        # ── Group 1: tight ±2-token role-family flags ─────────────────────────
        # Use a ±2 token window (last 2 left tokens + first 2 right tokens) rather
        # than the full narrow context to prevent cross-mention bleeding.  For
        # example, in "yields 12 profit and costs 5", the full narrow context of "5"
        # would include "profit" from "12"'s territory; the tight window only sees
        # "and costs" (left) + "dollars to" (right), correctly marking 5 as cost_like.
        _tight_left = _left_narrow[-2:]
        _tight_right = _right_narrow[:2]
        _tight_ctx: frozenset[str] = frozenset(_tight_left + _tight_right)
        _is_cost_like_flag = bool(_tight_ctx & _COST_CONTEXT_WORDS)
        _is_profit_like_flag = bool(_tight_ctx & _PROFIT_CONTEXT_WORDS)
        _is_demand_like_flag = bool(_tight_ctx & _DEMAND_CONTEXT_WORDS)
        _is_resource_like_flag = bool(_tight_ctx & _RESOURCE_CONTEXT_WORDS)
        _is_time_like_flag = bool(_tight_ctx & _TIME_CONTEXT_WORDS)

        mentions.append(
            MentionOptIR(
                mention_id=mention_id,
                value=tok.value,
                type_bucket=tok.kind,
                raw_surface=tok.raw,
                role_tags=role_tags,
                operator_tags=operator_tags,
                unit_tags=unit_tags,
                fragment_type=fragment_type,
                is_per_unit=is_per_unit,
                is_total_like=is_total_like,
                nearby_entity_tokens=entity_tokens,
                nearby_resource_tokens=resource_tokens,
                nearby_product_tokens=product_tokens,
                context_tokens=ctx_tokens,
                sentence_tokens=sent_tokens,
                tok=tok,
                is_count_like=_is_count_like,
                is_lower_bound_like=_is_lower_bound,
                is_upper_bound_like=_is_upper_bound,
                is_percent_like=_is_pct,
                primary_role=_primary_role,
                bound_role=_bound_role,
                narrow_context_tokens=tuple(_left_narrow + _right_narrow),
                narrow_left_tokens=tuple(_left_narrow),
                is_cost_like=_is_cost_like_flag,
                is_profit_like=_is_profit_like_flag,
                is_demand_like=_is_demand_like_flag,
                is_resource_like=_is_resource_like_flag,
                is_time_like=_is_time_like_flag,
            )
        )
        mention_id += 1
        i += span_size

    # ── Enumeration-derived count candidates ─────────────────────────────────
    # These handle cases where the count is implicit, e.g.:
    #   "phones and laptops"           → NumProducts = 2
    #   "apples, bananas, and grapes"  → NumItems    = 3
    #   "10 apples, 20 bananas, and 80 grapes" → NumItems = 3
    # Each derived-count mention is tagged as a cardinality candidate and
    # receives a hard penalty for any non-count-like slot (see _gcg_local_score).
    _DERIVED_ROLE_TAGS = frozenset({"cardinality_limit", "quantity_limit", "derived_count"})
    for _ec, _esp, _ectx in _extract_enum_derived_counts(query):
        _etok = NumTok(
            raw=f"derived:{int(_ec)} ({_esp})",
            value=_ec,
            kind="int",
        )
        mentions.append(
            MentionOptIR(
                mention_id=mention_id,
                value=_ec,
                type_bucket="int",
                raw_surface=_etok.raw,
                role_tags=_DERIVED_ROLE_TAGS,
                operator_tags=frozenset(),
                unit_tags=frozenset({"count_marker"}),
                fragment_type="",
                is_per_unit=False,
                is_total_like=False,
                nearby_entity_tokens=frozenset(),
                nearby_resource_tokens=frozenset(),
                nearby_product_tokens=frozenset(),
                context_tokens=_ectx,
                sentence_tokens=sent_tokens,
                tok=_etok,
                is_count_like=True,
                is_lower_bound_like=False,
                is_upper_bound_like=False,
                is_percent_like=False,
                primary_role="count",
            )
        )
        mention_id += 1

    return mentions


def _slot_opt_role_expansion(param_name: str) -> frozenset[str]:
    """Slot name -> optimization-role tags (schema-side priors, enriched with per-unit/total)."""
    n = (param_name or "").lower()
    tags: set[str] = set()
    if "budget" in n:
        tags.update(["total_budget", "upper_bound", "capacity_limit"])
    if "capacity" in n or "limit" in n:
        tags.update(["capacity_limit", "upper_bound", "total_available"])
    if "demand" in n or "require" in n or "need" in n:
        tags.update(["demand_requirement", "lower_bound", "minimum_requirement"])
    if "profit" in n or "revenue" in n or "return" in n:
        tags.update(["objective_coeff", "unit_profit", "unit_revenue", "unit_return"])
    if "cost" in n or "expense" in n or "price" in n:
        if "total" in n or "budget" in n:
            tags.update(["total_budget", "unit_cost"])
        else:
            tags.update(["unit_cost", "objective_coeff"])
    if "percent" in n or "ratio" in n or "fraction" in n or "share" in n or "rate" in n or "pct" in n or "proportion" in n:
        tags.update(["ratio_constraint", "percentage_constraint", "share_constraint"])
    if "min" in n or "minimum" in n or "atleast" in n:
        tags.update(["lower_bound", "minimum_requirement", "demand_requirement"])
    if "max" in n or "maximum" in n or "atmost" in n:
        tags.update(["upper_bound", "maximum_allowance", "capacity_limit"])
    if "penalty" in n or "setup" in n or "fixed" in n:
        tags.update(["penalty", "fixed_cost", "setup_cost"])
    if "time" in n or "hour" in n or "day" in n:
        tags.update(["time_requirement", "resource_consumption"])
    if "number" in n or "count" in n or "item" in n or "quantity" in n or n.startswith("num"):
        tags.update(["quantity_limit", "cardinality_limit"])
    # Per-unit / coefficient semantics — improves match with "per", "each" context cues
    if any(w in n for w in ("per", "each", "unit", "perunit")):
        tags.update(["unit_cost", "objective_coeff", "resource_consumption"])
    # Total / aggregate semantics
    if "total" in n or "available" in n or "aggregate" in n:
        tags.update(["total_available", "capacity_limit"])
    if not tags:
        tags.add("quantity_limit")
    return frozenset(tags)


def _build_slot_opt_irs(expected_scalar: list[str]) -> list[SlotOptIR]:
    """Stage 3: Build slot IRs with optimization-role priors."""
    slot_irs: list[SlotOptIR] = []
    for name in expected_scalar:
        et = _expected_type(name)
        aliases = _slot_aliases(name)
        alias_tokens_flat: set[str] = set()
        for a in aliases:
            alias_tokens_flat.update(_normalize_tokens(a))
        norm_tokens = _slot_measure_tokens(name)
        slot_role_tags = _slot_opt_role_expansion(name)
        op_pref: set[str] = set()
        n_lower_sl = name.lower()
        if any(x in n_lower_sl for x in ("min", "minimum", "atleast")):
            op_pref.add("min")
        if any(x in n_lower_sl for x in ("max", "maximum", "atmost")):
            op_pref.add("max")
        # Extended slot-name cues for bound direction (lower/upper/lowerbound/upperbound).
        if _ENABLE_BOUND_ROLE_LAYER:
            if any(x in n_lower_sl for x in ("lower", "lowerbound", "lower_bound")):
                op_pref.add("min")
            if any(x in n_lower_sl for x in ("upper", "upperbound", "upper_bound")):
                op_pref.add("max")
        unit_pref: set[str] = set()
        if et == "percent":
            unit_pref.add("percent_marker")
        if et == "currency":
            unit_pref.add("currency_marker")
        if et == "int":
            unit_pref.add("count_marker")
        is_objective = bool(slot_role_tags & {"objective_coeff", "unit_profit", "unit_revenue", "unit_return", "unit_cost"})
        is_bound = bool(slot_role_tags & {"lower_bound", "upper_bound", "capacity_limit", "demand_requirement", "minimum_requirement", "maximum_allowance"})
        # Widen is_total detection to cover aggregate capacity/availability slots
        # such as MaxWater, WaterAvailability, PowderedPillAvailability that have
        # "capacity_limit" in their role tags or "available"/"capacity" in their name
        # but lack "budget" or "total" keywords.
        n_lower = name.lower()
        is_total = (
            bool(slot_role_tags & {"total_budget", "total_available", "capacity_limit"})
            or "budget" in n_lower
            or "total" in n_lower
            or "available" in n_lower
            or "availability" in n_lower
            or "capacity" in n_lower
        )
        # A slot is coefficient-like when it holds a per-unit value such as cost/profit per item
        # or resource consumption per unit (e.g. HoursPerProduct, CostPerUnit).
        # However, if the slot is already total-like (e.g. LaborHoursAvailable), the
        # resource_consumption tag reflects the domain topic, not the per-unit role.
        # Exclude total-like slots from coefficient classification to prevent a total-capacity
        # slot from attracting per-unit mentions via coefficient_vs_total_bonus.
        is_coeff = (
            bool(slot_role_tags & {"unit_cost", "unit_profit", "unit_revenue", "resource_consumption"})
            and not is_total
        )

        slot_irs.append(
            SlotOptIR(
                name=name,
                norm_tokens=norm_tokens,
                expected_type=et,
                alias_tokens=alias_tokens_flat,
                slot_role_tags=slot_role_tags,
                operator_preference=frozenset(op_pref),
                unit_preference=frozenset(unit_pref),
                is_objective_like=is_objective,
                is_bound_like=is_bound,
                is_total_like=is_total,
                is_coefficient_like=is_coeff,
                is_count_like=_is_count_like_slot(name),
            )
        )
    return slot_irs


def _score_mention_slot_opt(m: MentionOptIR, s: SlotOptIR) -> tuple[float, dict[str, Any]]:
    """Stage 5: Optimization-role compatibility score."""
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.type_bucket
    expected = s.expected_type

    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return OPT_ROLE_WEIGHTS["type_incompatible_penalty"], features

    # Enumeration-derived counts must not fill non-count-like slots.
    if "derived_count" in m.role_tags and not s.is_count_like:
        features["derived_count_non_count"] = True
        return OPT_ROLE_WEIGHTS["type_incompatible_penalty"], features

    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (expected == "currency" and kind == "currency"):
            score += OPT_ROLE_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected == "float" and kind in {"float", "int"}:
            # float+float = exact; integer-valued tokens are valid for float slots
            score += OPT_ROLE_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected == "int" and kind == "int":
            score += OPT_ROLE_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected in ("int", "float") and kind in {"currency"}:
            score += OPT_ROLE_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True
        elif expected == "int" and kind == "float":
            score += OPT_ROLE_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True
        elif expected == "currency" and kind in {"float", "int"}:
            # Monetary slots filled with a plain integer/float token (no "$" sign)
            # are a full exact match — the value IS a valid monetary quantity.
            score += OPT_ROLE_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True

    role_overlap = len(m.role_tags & s.slot_role_tags)
    if role_overlap:
        score += OPT_ROLE_WEIGHTS["opt_role_overlap"] * float(role_overlap)
        features["opt_role_overlap"] = role_overlap

    if m.fragment_type == "objective" and s.is_objective_like:
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_objective"] = True
    if m.fragment_type in ("constraint", "bound") and s.is_bound_like:
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_bound"] = True
    if m.fragment_type == "resource" and s.is_total_like:
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_resource"] = True
    if m.fragment_type == "ratio" and ("ratio_constraint" in s.slot_role_tags or "percentage_constraint" in s.slot_role_tags):
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_ratio"] = True

    if m.operator_tags & s.operator_preference:
        score += OPT_ROLE_WEIGHTS["operator_match_bonus"]
        features["operator_match"] = True

    slot_words = set(s.norm_tokens) | s.alias_tokens
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += OPT_ROLE_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap
    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += OPT_ROLE_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    if m.unit_tags & s.unit_preference:
        score += OPT_ROLE_WEIGHTS["unit_match_bonus"]
        features["unit_match"] = True

    entity_resource = len(
        (m.nearby_entity_tokens | m.nearby_resource_tokens | m.nearby_product_tokens) & slot_words
    )
    if entity_resource:
        score += OPT_ROLE_WEIGHTS["entity_resource_overlap"] * float(entity_resource)
        features["entity_resource_overlap"] = entity_resource

    if s.is_total_like and m.is_total_like:
        score += OPT_ROLE_WEIGHTS["coefficient_vs_total_bonus"]
        features["total_match"] = True
    if s.is_coefficient_like and m.is_per_unit:
        score += OPT_ROLE_WEIGHTS["coefficient_vs_total_bonus"]
        features["coefficient_match"] = True
    # Local mismatch penalties: per-unit mention → total slot, or total mention → coeff slot.
    if s.is_total_like and not s.is_coefficient_like and m.is_per_unit and not m.is_total_like:
        score += OPT_ROLE_WEIGHTS["coeff_to_total_local_penalty"]
        features["coeff_to_total_penalty"] = True
    if s.is_coefficient_like and not s.is_total_like and m.is_total_like and not m.is_per_unit:
        score += OPT_ROLE_WEIGHTS["total_to_coeff_local_penalty"]
        features["total_to_coeff_penalty"] = True

    # ── Quantity-role layer: primary_role bonuses / penalties ─────────────────
    # count-like mention + count-like slot → strong bonus
    if m.is_count_like and s.is_count_like:
        score += OPT_ROLE_WEIGHTS["count_mention_count_slot_bonus"]
        features["count_role_match"] = True
    # count-context mention (not a percent/currency/derived) → non-count slot: soft penalty
    if m.is_count_like and not s.is_count_like and not m.is_percent_like:
        score += OPT_ROLE_WEIGHTS["count_mention_non_count_penalty"]
        features["count_to_non_count_penalty"] = True
    # lower-bound mention → slot that expects a minimum
    if m.is_lower_bound_like and "min" in s.operator_preference:
        score += OPT_ROLE_WEIGHTS["bound_direction_bonus"]
        features["lower_bound_match"] = True
    # upper-bound mention → slot that expects a maximum
    if m.is_upper_bound_like and "max" in s.operator_preference:
        score += OPT_ROLE_WEIGHTS["bound_direction_bonus"]
        features["upper_bound_match"] = True
    # Wrong-direction bound penalty: strong discouragement for cross-direction assignments.
    if _ENABLE_BOUND_ROLE_LAYER:
        if m.is_lower_bound_like and "max" in s.operator_preference and "min" not in s.operator_preference:
            score += OPT_ROLE_WEIGHTS["bound_direction_penalty"]
            features["bound_direction_wrong"] = True
        elif m.is_upper_bound_like and "min" in s.operator_preference and "max" not in s.operator_preference:
            score += OPT_ROLE_WEIGHTS["bound_direction_penalty"]
            features["bound_direction_wrong"] = True

    score += OPT_ROLE_WEIGHTS["schema_prior_bonus"]
    features["schema_prior"] = True

    if score <= 0.0:
        score += OPT_ROLE_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


# Weights for anchor-linking (context-aware number-to-slot grounding).
ANCHOR_WEIGHTS = {
    "operator_compat_bonus": 1.5,
    "entity_alignment_bonus": 1.2,
    "lex_profile_overlap_bonus": 0.8,
    "edge_prune_penalty": -1e8,
}


def _slot_profile_tokens(s: SlotOptIR) -> set[str]:
    """Slot profile for alignment: name, aliases, and role tag tokens."""
    out: set[str] = set(s.norm_tokens) | s.alias_tokens
    for tag in s.slot_role_tags:
        for part in tag.split("_"):
            if len(part) > 1:
                out.add(part.lower())
    return out


def _score_mention_slot_anchor(
    m: MentionOptIR,
    s: SlotOptIR,
    use_entity_alignment: bool = True,
    use_edge_pruning: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Context-aware anchor-linking score: alignment between mention context and slot profile."""
    base_score, base_feats = _score_mention_slot_opt(m, s)
    if base_score <= -1e8:
        return base_score, {**base_feats, "anchor_pruned": True}

    score = base_score
    features: dict[str, Any] = dict(base_feats)

    # Operator compatibility: lower-bound mention <-> lower-bound slot, etc.
    if m.operator_tags & s.operator_preference:
        score += ANCHOR_WEIGHTS["operator_compat_bonus"]
        features["anchor_operator_compat"] = True
    # Total/budget mention <-> total-like slot; per/each <-> coefficient-like
    if s.is_total_like and m.is_total_like:
        score += ANCHOR_WEIGHTS["operator_compat_bonus"] * 0.5
        features["anchor_total_match"] = True
    if s.is_coefficient_like and m.is_per_unit:
        score += ANCHOR_WEIGHTS["operator_compat_bonus"] * 0.5
        features["anchor_per_unit_match"] = True
    # Percent/fraction mention <-> ratio/percent slot
    if m.type_bucket == "percent" and ("ratio_constraint" in s.slot_role_tags or "percentage_constraint" in s.slot_role_tags or "percent" in (s.name or "").lower()):
        score += ANCHOR_WEIGHTS["operator_compat_bonus"] * 0.5
        features["anchor_percent_match"] = True
    # Objective cue (profit/cost/revenue) <-> objective-like slot
    if m.fragment_type == "objective" and s.is_objective_like:
        score += ANCHOR_WEIGHTS["operator_compat_bonus"] * 0.5
        features["anchor_objective_match"] = True

    # Entity alignment: overlap of mention context + nearby entity/resource with slot name/aliases
    if use_entity_alignment:
        slot_words = set(s.norm_tokens) | s.alias_tokens
        mention_ctx = set(m.context_tokens) | m.nearby_entity_tokens | m.nearby_resource_tokens | m.nearby_product_tokens
        overlap = len(mention_ctx & slot_words)
        if overlap:
            score += ANCHOR_WEIGHTS["entity_alignment_bonus"] * min(overlap, 3)
            features["anchor_entity_overlap"] = overlap

    # Lexical/context similarity: mention context vs slot profile
    slot_profile = _slot_profile_tokens(s)
    ctx_set = set(m.context_tokens)
    profile_overlap = len(ctx_set & slot_profile)
    if profile_overlap:
        score += ANCHOR_WEIGHTS["lex_profile_overlap_bonus"] * min(profile_overlap, 4)
        features["anchor_profile_overlap"] = profile_overlap

    # Edge pruning: strongly implausible pairs get large penalty
    if use_edge_pruning:
        # Percent-like mention to clearly non-percent slot
        if m.type_bucket == "percent" and "ratio_constraint" not in s.slot_role_tags and "percentage_constraint" not in s.slot_role_tags and "percent" not in (s.name or "").lower():
            score = ANCHOR_WEIGHTS["edge_prune_penalty"]
            features["anchor_pruned_percent"] = True
        # At-least/min cue to upper-bound-only slot
        elif "min" in m.operator_tags and "min" not in s.operator_preference and "max" in s.operator_preference:
            score = ANCHOR_WEIGHTS["edge_prune_penalty"]
            features["anchor_pruned_min_to_max"] = True
        # Max cue to lower-bound-only slot
        elif "max" in m.operator_tags and "max" not in s.operator_preference and "min" in s.operator_preference:
            score = ANCHOR_WEIGHTS["edge_prune_penalty"]
            features["anchor_pruned_max_to_min"] = True
        # Per-unit mention to obvious total-budget-only slot (no coefficient role)
        elif m.is_per_unit and s.is_total_like and not s.is_coefficient_like and not (m.role_tags & s.slot_role_tags):
            score = ANCHOR_WEIGHTS["edge_prune_penalty"]
            features["anchor_pruned_per_to_total"] = True
        # Total/budget mention to per-unit-only coefficient slot
        elif m.is_total_like and s.is_coefficient_like and not s.is_total_like:
            score = ANCHOR_WEIGHTS["edge_prune_penalty"]
            features["anchor_pruned_total_to_per"] = True

    features["anchor_total_score"] = score
    return score, features


# --- Entity-semantic beam: semantic role families, entity binding, polarity/scope ---

# Fine-grained semantic role families for mention-slot alignment (beyond coarse type).
SEMANTIC_ROLE_FAMILIES = frozenset({
    "objective_coeff", "lower_bound", "upper_bound", "rhs_total_or_budget",
    "per_unit_rate", "resource_usage_coeff", "ratio_or_share", "fixed_requirement",
})

# Map existing role tags / fragment / operator to semantic family.
def _mention_semantic_families(m: MentionOptIR) -> frozenset[str]:
    """Infer semantic role families from mention context (deterministic)."""
    out: set[str] = set()
    if m.fragment_type == "objective" or (m.role_tags & {"objective_coeff", "unit_profit", "unit_revenue", "unit_return", "unit_cost"}):
        out.add("objective_coeff")
    if m.is_per_unit:
        out.add("per_unit_rate")
    if m.fragment_type == "resource" or m.is_total_like:
        out.add("rhs_total_or_budget")
    if m.fragment_type in ("ratio",) or m.type_bucket == "percent":
        out.add("ratio_or_share")
    if "min" in m.operator_tags or "lower_bound" in m.role_tags or "minimum_requirement" in m.role_tags:
        out.add("lower_bound")
    if "max" in m.operator_tags or "upper_bound" in m.role_tags or "maximum_allowance" in m.role_tags:
        out.add("upper_bound")
    if m.role_tags & {"resource_consumption", "time_requirement"}:
        out.add("resource_usage_coeff")
    if m.role_tags & {"fixed_cost", "setup_cost", "penalty"}:
        out.add("fixed_requirement")
    return frozenset(out) if out else frozenset({"objective_coeff"})


def _slot_semantic_families(s: SlotOptIR) -> frozenset[str]:
    """Infer semantic role families from slot name and existing tags."""
    out: set[str] = set()
    n = (s.name or "").lower()
    if s.is_objective_like or s.is_coefficient_like and ("profit" in n or "revenue" in n or "return" in n or "cost" in n):
        out.add("objective_coeff")
    if s.is_coefficient_like:
        out.add("per_unit_rate")
    if s.is_total_like:
        out.add("rhs_total_or_budget")
    if "percent" in n or "ratio" in n or "fraction" in n or "share" in n:
        out.add("ratio_or_share")
    if "min" in n or "minimum" in n or "least" in n:
        out.add("lower_bound")
    if "max" in n or "maximum" in n or "most" in n:
        out.add("upper_bound")
    if s.slot_role_tags & {"resource_consumption", "time_requirement"}:
        out.add("resource_usage_coeff")
    if s.slot_role_tags & {"fixed_cost", "setup_cost", "penalty"}:
        out.add("fixed_requirement")
    return frozenset(out) if out else frozenset({"objective_coeff"})


def _mention_entity_family_tokens(m: MentionOptIR) -> set[str]:
    """Entity-first: tokens that anchor this mention to variable/owner (for slot binding)."""
    out: set[str] = set()
    out |= m.nearby_entity_tokens
    out |= m.nearby_resource_tokens
    out |= m.nearby_product_tokens
    # Add noun-like or role words from context (longer tokens often entity names)
    for t in m.context_tokens:
        if len(t) >= 3 and t.isalpha():
            out.add(t.lower())
    return out


def _mention_polarity(m: MentionOptIR) -> str:
    """Lower / upper / neutral for operator scope."""
    if "min" in m.operator_tags:
        return "lower"
    if "max" in m.operator_tags:
        return "upper"
    return "neutral"


def _mention_style(m: MentionOptIR) -> str:
    """Total / per_unit / percent / scalar."""
    if m.is_total_like and not m.is_per_unit:
        return "total"
    if m.is_per_unit:
        return "per_unit"
    if m.type_bucket == "percent":
        return "percent"
    return "scalar"


def _slot_polarity(s: SlotOptIR) -> str:
    if "min" in s.operator_preference:
        return "lower"
    if "max" in s.operator_preference:
        return "upper"
    return "neutral"


ENTITY_SEMANTIC_WEIGHTS = {
    "entity_binding_bonus": 2.0,
    "semantic_family_bonus": 1.8,
    "polarity_match_bonus": 1.5,
    "style_match_bonus": 1.0,
    "edge_prune_penalty": -1e8,
}


def _score_mention_slot_entity_semantic(m: MentionOptIR, s: SlotOptIR) -> tuple[float, dict[str, Any]]:
    """Entity-first + semantic role-family + polarity/scope; for use in entity_semantic_beam."""
    base_score, base_feats = _score_mention_slot_anchor(m, s, use_entity_alignment=True, use_edge_pruning=True)
    if base_score <= -1e8:
        return base_score, {**base_feats, "entity_semantic_pruned": True}

    score = base_score
    features: dict[str, Any] = dict(base_feats)

    # A. Entity-first binding: mention entity family vs slot name/aliases
    entity_tokens = _mention_entity_family_tokens(m)
    slot_entity_tokens = set(s.norm_tokens) | s.alias_tokens
    entity_overlap = len(entity_tokens & slot_entity_tokens)
    if entity_overlap:
        score += ENTITY_SEMANTIC_WEIGHTS["entity_binding_bonus"] * min(entity_overlap, 4)
        features["entity_binding_overlap"] = entity_overlap

    # B. Semantic role-family match
    m_families = _mention_semantic_families(m)
    s_families = _slot_semantic_families(s)
    family_overlap = len(m_families & s_families)
    if family_overlap:
        score += ENTITY_SEMANTIC_WEIGHTS["semantic_family_bonus"] * min(family_overlap, 3)
        features["semantic_family_overlap"] = family_overlap

    # C. Lower/upper operator scope
    m_pol = _mention_polarity(m)
    s_pol = _slot_polarity(s)
    if m_pol == s_pol and m_pol != "neutral":
        score += ENTITY_SEMANTIC_WEIGHTS["polarity_match_bonus"]
        features["polarity_match"] = True
    if m_pol != "neutral" and s_pol != "neutral" and m_pol != s_pol:
        score = ENTITY_SEMANTIC_WEIGHTS["edge_prune_penalty"]
        features["polarity_mismatch_pruned"] = True
        return score, features

    # D. Style: total / per_unit / percent consistency
    m_style = _mention_style(m)
    if s.is_total_like and m_style == "total":
        score += ENTITY_SEMANTIC_WEIGHTS["style_match_bonus"]
        features["style_total_match"] = True
    if s.is_coefficient_like and m_style == "per_unit":
        score += ENTITY_SEMANTIC_WEIGHTS["style_match_bonus"]
        features["style_per_unit_match"] = True
    if "ratio_or_share" in s_families and m_style == "percent":
        score += ENTITY_SEMANTIC_WEIGHTS["style_match_bonus"] * 0.5
        features["style_percent_match"] = True

    features["entity_semantic_total_score"] = score
    return score, features


def _run_optimization_role_entity_semantic_beam_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
    beam_width: int = 5,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, str]]:
    """Entity-first + semantic role families + beam over partial assignments. Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, filled_in_repair

    m, s = len(mentions), len(slots)

    # Score matrix: entity_semantic scorer
    score_matrix: list[list[float]] = [[0.0 for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, _ = _score_mention_slot_entity_semantic(mr, sr)
            score_matrix[i][j] = sc

    # Reuse bottom-up beam logic (atoms, extend, admissibility, relation bonus)
    atoms: list[tuple[int, int, float]] = []
    for i in range(m):
        for j in range(s):
            if score_matrix[i][j] > 0 and score_matrix[i][j] < 1e7:
                atoms.append((i, j, score_matrix[i][j]))
    atoms.sort(key=lambda x: -x[2])

    def _bundle_to_partial(bundle: frozenset[tuple[int, int]]) -> dict[str, MentionOptIR]:
        return {slots[j].name: mentions[i] for i, j in bundle}

    def _add_relation_bonus(bundle: frozenset[tuple[int, int]], base_score: float) -> float:
        partial = _bundle_to_partial(bundle)
        bonus = 0.0
        for i, j in bundle:
            bonus += _relation_bonus(mentions[i], slots[j], partial, slots, mentions)
        return base_score + bonus

    beam: list[tuple[frozenset[tuple[int, int]], float]] = []
    for i, j, sc in atoms[: beam_width * 2]:
        bundle = frozenset({(i, j)})
        partial = _bundle_to_partial(bundle)
        if not _is_partial_admissible(partial, slots):
            continue
        beam.append((bundle, sc))
    beam.sort(key=lambda x: -(x[1] + _add_relation_bonus(x[0], 0.0)))
    beam = beam[:beam_width]

    for _ in range(max(0, min(m, s) - 1)):
        next_beam: list[tuple[frozenset[tuple[int, int]], float]] = []
        for bundle, sum_scores in beam:
            used_m = {i for i, _ in bundle}
            used_s = {j for _, j in bundle}
            for i, j, sc in atoms:
                if i in used_m or j in used_s:
                    continue
                new_bundle = bundle | frozenset({(i, j)})
                partial = _bundle_to_partial(new_bundle)
                if not _is_partial_admissible(partial, slots):
                    continue
                new_sum = sum_scores + sc
                next_beam.append((new_bundle, new_sum))
        if not next_beam:
            break
        next_beam.sort(key=lambda x: (-len(x[0]), -(x[1] + _add_relation_bonus(x[0], 0.0))))
        beam = next_beam[:beam_width]

    if not beam:
        return filled_values, filled_mentions, filled_in_repair

    def _rank_score(item: tuple[frozenset[tuple[int, int]], float]) -> tuple[int, float]:
        b, ss = item
        return (len(b), ss + _add_relation_bonus(b, 0.0))

    best_bundle, _ = max(beam, key=_rank_score)
    partial = _bundle_to_partial(best_bundle)
    initial_scores = {slots[j].name: score_matrix[i][j] for i, j in best_bundle}
    debug: dict[str, list[dict[str, Any]]] = {}
    for j, sr in enumerate(slots):
        debug[sr.name] = [
            {"mention_id": mentions[i].mention_id, "mention_raw": mentions[i].raw_surface, "score": score_matrix[i][j], "features": {}}
            for i in range(m)
        ]
        debug[sr.name].sort(key=lambda x: x["score"], reverse=True)
    filled, filled_in_repair = _opt_role_validate_and_repair(mentions, slots, partial, initial_scores, debug)

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


def _opt_role_global_assignment(
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
    score_matrix: list[list[float]] | None = None,
) -> tuple[dict[str, MentionOptIR], dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Stage 5: Maximum-weight bipartite matching for optimization-role assignment.
    If score_matrix is provided, use it (score_matrix[i][j]); else compute from _score_mention_slot_opt."""
    assignments: dict[str, MentionOptIR] = {}
    scores_out: dict[str, float] = {}
    debug: dict[str, list[dict[str, Any]]] = {}

    if not mentions or not slots:
        return assignments, scores_out, debug

    m, s = len(mentions), len(slots)
    cost = [[0.0 for _ in range(s)] for _ in range(m)]
    if score_matrix is not None:
        for i in range(m):
            for j in range(s):
                sc = score_matrix[i][j]
                cost[i][j] = -sc if sc > -1e8 else 1e9
        for j, sr in enumerate(slots):
            debug[sr.name] = [
                {"mention_id": mentions[i].mention_id, "mention_raw": mentions[i].raw_surface, "score": score_matrix[i][j], "features": {}}
                for i in range(m)
            ]
            debug[sr.name].sort(key=lambda x: x["score"], reverse=True)
    else:
        for i, mr in enumerate(mentions):
            for j, sr in enumerate(slots):
                sc, feats = _score_mention_slot_opt(mr, sr)
                cost[i][j] = -sc if sc > -1e8 else 1e9
                if sr.name not in debug:
                    debug[sr.name] = []
                debug[sr.name].append(
                    {"mention_id": mr.mention_id, "mention_raw": mr.raw_surface, "score": sc, "features": feats}
                )
        for name in debug:
            debug[name].sort(key=lambda x: x["score"], reverse=True)

    def _get_score(ri: int, cj: int) -> float:
        if score_matrix is not None:
            return score_matrix[ri][cj]
        sc, _ = _score_mention_slot_opt(mentions[ri], slots[cj])
        return sc

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        cost_arr = np.array(cost)
        row_ind, col_ind = linear_sum_assignment(cost_arr)
        for ri, cj in zip(row_ind, col_ind):
            if cost_arr[ri, cj] < 1e8:
                sr = slots[cj]
                mr = mentions[ri]
                sc = _get_score(ri, cj)
                assignments[sr.name] = mr
                scores_out[sr.name] = sc
    except ImportError:
        from math import inf
        num_states = 1 << s
        dp = [[-inf] * num_states for _ in range(m + 1)]
        parent = [[(-1, None)] * num_states for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(m):
            for mask in range(num_states):
                if dp[i][mask] == -inf:
                    continue
                if dp[i][mask] > dp[i + 1][mask]:
                    dp[i + 1][mask] = dp[i][mask]
                    parent[i + 1][mask] = (mask, None)
                for j in range(s):
                    if (mask >> j) & 1:
                        continue
                    sc = -cost[i][j]
                    if sc <= 0:
                        continue
                    new_mask = mask | (1 << j)
                    val = dp[i][mask] + sc
                    if val > dp[i + 1][new_mask]:
                        dp[i + 1][new_mask] = val
                        parent[i + 1][new_mask] = (mask, j)
        best_mask = max(range(num_states), key=lambda mk: dp[m][mk])
        i, mask = m, best_mask
        used_slot_for_mention = {}
        while i > 0:
            prev_mask, j = parent[i][mask]
            if j is not None:
                used_slot_for_mention[i - 1] = j
            mask = prev_mask
            i -= 1
        for mi, cj in used_slot_for_mention.items():
            sr = slots[cj]
            mr = mentions[mi]
            sc = _get_score(mi, cj)
            assignments[sr.name] = mr
            scores_out[sr.name] = sc

    return assignments, scores_out, debug


def _opt_role_validate_one(slot_name: str, m: MentionOptIR, s: SlotOptIR, score: float) -> tuple[bool, float]:
    """Stage 6: Plausibility check using optimization logic."""
    if _is_type_incompatible(s.expected_type, m.type_bucket):
        return False, OPT_REPAIR_WEIGHTS["total_vs_coeff_penalty"]
    adj = 0.0
    if m.role_tags & s.slot_role_tags:
        adj += OPT_REPAIR_WEIGHTS["role_plausibility_bonus"]
    if s.is_bound_like:
        if ("min" in s.operator_preference and "min" in m.operator_tags) or ("max" in s.operator_preference and "max" in m.operator_tags):
            adj += OPT_REPAIR_WEIGHTS["bound_plausibility_bonus"]
    if s.is_total_like and not m.is_total_like and m.is_per_unit:
        adj += OPT_REPAIR_WEIGHTS["total_vs_coeff_penalty"]
    if s.is_coefficient_like and m.is_total_like and not m.is_per_unit:
        adj += OPT_REPAIR_WEIGHTS["total_vs_coeff_penalty"] * 0.5
    return adj >= -0.5 or score > 1.0, adj


def _opt_role_validate_and_repair(
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
    initial_assignments: dict[str, MentionOptIR],
    initial_scores: dict[str, float],
    debug: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, MentionOptIR], dict[str, str]]:
    """Stage 6: Optimization-aware validation and coverage-preserving repair."""
    filled = dict(initial_assignments)
    filled_in_repair: dict[str, str] = {}

    for slot in filled:
        filled_in_repair[slot] = "initial"

    used_mention_ids = {m.mention_id for m in filled.values()}
    slot_list = list(slots)

    for s in slot_list:
        if s.name not in filled:
            continue
        m = filled[s.name]
        score = initial_scores.get(s.name, 0.0)
        valid, _ = _opt_role_validate_one(s.name, m, s, score)
        if not valid and score < OPT_REPAIR_WEIGHTS["min_role_support"]:
            del filled[s.name]
            if s.name in filled_in_repair:
                del filled_in_repair[s.name]
            used_mention_ids.discard(m.mention_id)

    unfilled = [s for s in slot_list if s.name not in filled]
    unfilled.sort(key=lambda x: len(x.slot_role_tags) + len(x.alias_tokens), reverse=True)

    for s in unfilled:
        for cand in debug.get(s.name, []):
            mid = cand.get("mention_id")
            if mid is None:
                continue
            mr = next((x for x in mentions if x.mention_id == mid), None)
            if not mr or mr.mention_id in used_mention_ids:
                continue
            sc = cand.get("score", 0.0)
            if sc <= 0 or _is_type_incompatible(s.expected_type, mr.type_bucket):
                continue
            valid, adj = _opt_role_validate_one(s.name, mr, s, sc)
            if valid or sc + adj > OPT_REPAIR_WEIGHTS["min_role_support"]:
                filled[s.name] = mr
                filled_in_repair[s.name] = "repair"
                used_mention_ids.add(mr.mention_id)
                break

    # ── Bound-flip swap repair ──────────────────────────────────────────────
    if _ENABLE_BOUND_ROLE_LAYER:
        _bound_swap_repair(filled, filled_in_repair, slots)

    # ── Total vs per-unit contradiction repair ──────────────────────────────
    _total_perunit_swap_repair(filled, filled_in_repair, slots, mentions)

    return filled, filled_in_repair


def _bound_swap_repair(
    filled: dict[str, "MentionOptIR"],
    filled_in_repair: dict[str, str],
    slots: list["SlotOptIR"],
) -> None:
    """Post-assignment repair: swap min/max slot values when the assignment is inverted.

    Triggered only when there is explicit operator evidence that the assignment is
    wrong-direction (the mention assigned to the min slot has a "max" operator tag,
    or the mention assigned to the max slot has a "min" operator tag) AND the
    numerical values are inverted (assigned-min-value > assigned-max-value).

    Conservative by design: if there is no operator evidence, no swap is made even
    when the values look inverted, to avoid inadvertently altering legitimate cases
    where min=max or where ordering is domain-specific.
    """
    min_slots = [s for s in slots if "min" in s.operator_preference and s.name in filled]
    max_slots = [s for s in slots if "max" in s.operator_preference and s.name in filled]
    for s_min in min_slots:
        for s_max in max_slots:
            m_min = filled.get(s_min.name)
            m_max = filled.get(s_max.name)
            if m_min is None or m_max is None:
                continue
            if m_min.value is None or m_max.value is None:
                continue
            if m_min.value <= m_max.value:
                continue  # ordering is already correct
            # Values are inverted (min > max).  Only swap when operator evidence confirms it.
            min_m_has_max_cue = "max" in m_min.operator_tags
            max_m_has_min_cue = "min" in m_max.operator_tags
            if min_m_has_max_cue or max_m_has_min_cue:
                filled[s_min.name] = m_max
                filled[s_max.name] = m_min
                filled_in_repair[s_min.name] = "bound_swap_repair"
                filled_in_repair[s_max.name] = "bound_swap_repair"


def _total_perunit_swap_repair(
    filled: dict[str, "MentionOptIR"],
    filled_in_repair: dict[str, str],
    slots: list["SlotOptIR"],
    mentions: list["MentionOptIR"],
) -> None:
    """Post-assignment repair: correct obvious total↔per-unit confusion.

    Triggered only when there is strong evidence that a total slot received a
    per-unit mention (or vice versa) AND there is an unused mention that is a
    better fit.  Conservative by design: only repairs when both evidence
    conditions are met simultaneously:
      1. The assigned mention's role contradicts the slot's is_total_like /
         is_coefficient_like flag (i.e. explicit per-unit mention → total slot,
         or explicit total mention → coeff slot).
      2. An unused mention exists that has the correct role for the slot and
         has a plausible numerical value (magnitude check: total >> per-unit).
    """
    used_ids: set[int] = {m.mention_id for m in filled.values()}
    for s in slots:
        if s.name not in filled:
            continue
        m_curr = filled[s.name]
        # Check for total-slot assigned a per-unit mention.
        if s.is_total_like and not s.is_coefficient_like and m_curr.is_per_unit and not m_curr.is_total_like:
            # Look for an unused total-like mention with a larger value.
            best: "MentionOptIR | None" = None
            for m_cand in mentions:
                if m_cand.mention_id in used_ids:
                    continue
                if m_cand.is_per_unit and not m_cand.is_total_like:
                    continue  # also per-unit, no help
                if m_cand.value is None or m_curr.value is None:
                    continue
                if m_cand.value <= m_curr.value:
                    continue  # no magnitude evidence that this is the total
                if best is None or m_cand.value > best.value:
                    best = m_cand
            if best is not None:
                used_ids.discard(m_curr.mention_id)
                filled[s.name] = best
                filled_in_repair[s.name] = "total_perunit_swap_repair"
                used_ids.add(best.mention_id)
        # Check for coeff-slot assigned a total-like mention.
        elif s.is_coefficient_like and not s.is_total_like and m_curr.is_total_like and not m_curr.is_per_unit:
            # Look for an unused per-unit mention with a smaller value.
            best = None
            for m_cand in mentions:
                if m_cand.mention_id in used_ids:
                    continue
                if m_cand.is_total_like and not m_cand.is_per_unit:
                    continue  # also total-like, no help
                if m_cand.value is None or m_curr.value is None:
                    continue
                if m_cand.value >= m_curr.value:
                    continue  # no magnitude evidence that this is per-unit
                if best is None or m_cand.value > best.value:  # prefer largest small value
                    best = m_cand
            if best is not None:
                used_ids.discard(m_curr.mention_id)
                filled[s.name] = best
                filled_in_repair[s.name] = "total_perunit_swap_repair"
                used_ids.add(best.mention_id)


# --- Relation-aware + incremental admissible (RAT-SQL / PICARD inspired) ---

# Ordered longest-first so that "minimum" is stripped before "min", etc.
_BOUND_AFFIXES: tuple[str, ...] = (
    "minimum", "maximum", "lower", "upper", "min", "max", "lb", "ub",
)


def _slot_stem(name: str) -> str:
    """Return the *quantity stem* of a bound-slot name.

    Strips leading and trailing min/max/lower/upper affixes so that paired
    slots such as ``MinDemand``/``MaxDemand`` or ``LowerBound``/``UpperBound``
    both reduce to the same stem (``"demand"`` and ``"bound"`` respectively).
    Used by ``_is_partial_admissible`` to decide which min/max pairs should
    have their numeric ordering enforced.

    Examples::

        _slot_stem("MinDemand")       == "demand"
        _slot_stem("MaxDemand")       == "demand"
        _slot_stem("LowerBound")      == "bound"
        _slot_stem("UpperBound")      == "bound"
        _slot_stem("MinimumCapacity") == "capacity"
        _slot_stem("MaximumCapacity") == "capacity"
        _slot_stem("DemandMin")       == "demand"
        _slot_stem("DemandMax")       == "demand"
        _slot_stem("MinHours")        == "hours"
        _slot_stem("MaxHours")        == "hours"
    """
    n = name.lower()
    for affix in _BOUND_AFFIXES:
        if n.startswith(affix) and len(n) > len(affix):
            return n[len(affix):].lstrip("_ ")
        if n.endswith(affix) and len(n) > len(affix):
            return n[: -len(affix)].rstrip("_ ")
    return n


def _slot_slot_relation_tags(s1: SlotOptIR, s2: SlotOptIR) -> frozenset[str]:
    """Relation tags between two slots for relation-aware scoring."""
    out: set[str] = set()
    if s1.name == s2.name:
        return frozenset(out)
    if s1.is_coefficient_like and s2.is_coefficient_like:
        out.add("both_coeff")
    if s1.is_bound_like and s2.is_bound_like:
        if "min" in s1.operator_preference and "max" in s2.operator_preference:
            out.add("min_max_pair")
        elif "max" in s1.operator_preference and "min" in s2.operator_preference:
            out.add("min_max_pair")
    if s1.is_total_like and s2.is_coefficient_like:
        out.add("total_and_coeff")
    if s2.is_total_like and s1.is_coefficient_like:
        out.add("total_and_coeff")
    if (s1.slot_role_tags & s2.slot_role_tags) and s1.slot_role_tags and s2.slot_role_tags:
        out.add("same_role_family")
    return frozenset(out)


def _mention_mention_relation_tags(m1: MentionOptIR, m2: MentionOptIR) -> frozenset[str]:
    """Relation tags between two mentions for relation-aware scoring."""
    out: set[str] = set()
    if m1.mention_id == m2.mention_id:
        return frozenset(out)
    sent1 = set(m1.sentence_tokens)
    sent2 = set(m2.sentence_tokens)
    if sent1 and sent2 and len(sent1 & sent2) >= max(1, min(len(sent1), len(sent2)) // 2):
        out.add("same_sentence")
    if m1.fragment_type and m1.fragment_type == m2.fragment_type:
        out.add("same_fragment_type")
    if m1.is_per_unit and m2.is_per_unit:
        out.add("both_per_unit")
    if m1.is_total_like and m2.is_total_like:
        out.add("both_total")
    if m1.role_tags & m2.role_tags:
        out.add("same_role_family")
    return frozenset(out)


def _is_partial_admissible(
    partial: dict[str, MentionOptIR],
    slots: list[SlotOptIR],
) -> bool:
    """PICARD-style: whether current partial assignment is still valid (hard constraints)."""
    slot_by_name = {s.name: s for s in slots}
    # Use mention_id (hashable) instead of MentionOptIR (has list fields, unhashable)
    assigned_mention_ids = {m.mention_id for m in partial.values()}
    if len(assigned_mention_ids) != len(partial):
        return False
    for slot_name, m in partial.items():
        s = slot_by_name.get(slot_name)
        if not s:
            continue
        if _is_type_incompatible(s.expected_type, m.type_bucket):
            return False
        if s.is_total_like and m.is_per_unit and not m.is_total_like:
            return False
        if s.is_coefficient_like and m.is_total_like and not m.is_per_unit:
            return False
        if s.is_bound_like and ("ratio_constraint" in s.slot_role_tags or "percentage_constraint" in s.slot_role_tags):
            if m.fragment_type == "ratio":
                pass
            elif m.type_bucket == "percent":
                pass
        if m.type_bucket == "percent" and not (
            "ratio_constraint" in s.slot_role_tags or "percentage_constraint" in s.slot_role_tags
            or "percent" in (s.name or "").lower()
        ):
            return False
    min_slots = [s for s in slots if s.name in partial and "min" in s.operator_preference]
    max_slots = [s for s in slots if s.name in partial and "max" in s.operator_preference]
    for s in min_slots:
        m = partial.get(s.name)
        if m and "max" in m.operator_tags and "min" not in m.operator_tags:
            return False
    for s in max_slots:
        m = partial.get(s.name)
        if m and "min" in m.operator_tags and "max" not in m.operator_tags:
            return False
    # Enforce numeric ordering for paired bound slots that share the same
    # quantity stem (e.g. MinDemand/MaxDemand, LowerBound/UpperBound).
    # This rejects partial assignments where the value placed in the min slot
    # is strictly greater than the value in the max slot, preventing the
    # lower_vs_upper_bound failure family without touching other logic.
    for s_min in min_slots:
        m_min = partial[s_min.name]
        if m_min.value is None:
            continue
        for s_max in max_slots:
            m_max = partial[s_max.name]
            if m_max.value is None:
                continue
            if _slot_stem(s_min.name) != _slot_stem(s_max.name):
                continue
            if m_min.value > m_max.value:
                return False
    return True


# Weights for relation-aware bonus/penalty (additive to base score).
RELATION_WEIGHTS = {
    "consistent_pair_bonus": 1.2,
    "inconsistent_pair_penalty": -2.0,
}


def _relation_bonus(
    m: MentionOptIR,
    s: SlotOptIR,
    partial: dict[str, MentionOptIR],
    slots: list[SlotOptIR],
    mentions: list[MentionOptIR],
) -> float:
    """RAT-SQL-style: bonus/penalty from relation consistency with current partial assignment."""
    bonus = 0.0
    slot_by_name = {sx.name: sx for sx in slots}
    for slot_name, m_other in partial.items():
        s_other = slot_by_name.get(slot_name)
        if not s_other or s_other.name == s.name:
            continue
        slot_rel = _slot_slot_relation_tags(s, s_other)
        mention_rel = _mention_mention_relation_tags(m, m_other)
        if "both_coeff" in slot_rel and "both_per_unit" in mention_rel:
            if "same_sentence" in mention_rel or "same_fragment_type" in mention_rel:
                bonus += RELATION_WEIGHTS["consistent_pair_bonus"]
        if "total_and_coeff" in slot_rel:
            if s.is_total_like and m.is_total_like and s_other.is_coefficient_like and m_other.is_per_unit:
                bonus += RELATION_WEIGHTS["consistent_pair_bonus"]
            elif s.is_coefficient_like and m.is_per_unit and s_other.is_total_like and m_other.is_total_like:
                bonus += RELATION_WEIGHTS["consistent_pair_bonus"]
            elif (s.is_total_like and m.is_per_unit) or (s.is_coefficient_like and m.is_total_like):
                bonus += RELATION_WEIGHTS["inconsistent_pair_penalty"]
        if "min_max_pair" in slot_rel:
            if ("min" in s.operator_preference and "min" in m.operator_tags and
                "max" in s_other.operator_preference and "max" in m_other.operator_tags):
                bonus += RELATION_WEIGHTS["consistent_pair_bonus"]
            elif ("max" in s.operator_preference and "max" in m.operator_tags and
                  "min" in s_other.operator_preference and "min" in m_other.operator_tags):
                bonus += RELATION_WEIGHTS["consistent_pair_bonus"]
    return bonus


def _opt_role_incremental_admissible_assignment(
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
    base_score_matrix: list[list[float]],
    debug: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, MentionOptIR], dict[str, float], dict[str, list[dict[str, Any]]]]:
    """
    PICARD-style: build assignment incrementally; only extend with admissible partial states.
    RAT-SQL-style: when choosing candidate for next slot, add relation_bonus from current partial.
    """
    assignments: dict[str, MentionOptIR] = {}
    scores_out: dict[str, float] = {}
    m, s = len(mentions), len(slots)
    slot_order = sorted(
        range(s),
        key=lambda j: (len(slots[j].slot_role_tags) + len(slots[j].alias_tokens), slots[j].name),
        reverse=True,
    )
    used_mention_ids: set[int] = set()
    for j in slot_order:
        sr = slots[j]
        best_i: int | None = None
        best_score = float("-inf")
        for i in range(m):
            if mentions[i].mention_id in used_mention_ids:
                continue
            base_sc = base_score_matrix[i][j]
            if base_sc <= -1e8:
                continue
            tentative = dict(assignments)
            tentative[sr.name] = mentions[i]
            if not _is_partial_admissible(tentative, slots):
                continue
            rel_bonus = _relation_bonus(mentions[i], sr, assignments, slots, mentions)
            total = base_sc + rel_bonus
            if total > best_score:
                best_score = total
                best_i = i
        if best_i is not None and best_score > -1e8:
            assignments[sr.name] = mentions[best_i]
            scores_out[sr.name] = best_score
            used_mention_ids.add(mentions[best_i].mention_id)
    return assignments, scores_out, debug


def _run_optimization_role_relation_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, str]]:
    """
    Relation-aware + incremental admissible assignment (RAT-SQL + PICARD inspired).
    Pipeline: extract opt-role mentions -> build slot opt IRs -> base pair scores ->
    incremental admissible assignment with relation bonus -> validate_and_repair unfilled.
    """
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, filled_in_repair

    m, s = len(mentions), len(slots)
    cost = [[0.0 for _ in range(s)] for _ in range(m)]
    debug: dict[str, list[dict[str, Any]]] = {}
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _score_mention_slot_opt(mr, sr)
            cost[i][j] = -sc if sc > -1e8 else 1e9
            if sr.name not in debug:
                debug[sr.name] = []
            debug[sr.name].append(
                {"mention_id": mr.mention_id, "mention_raw": mr.raw_surface, "score": sc, "features": feats}
            )
    for name in debug:
        debug[name].sort(key=lambda x: x["score"], reverse=True)

    base_score_matrix = [[-cost[i][j] if cost[i][j] < 1e8 else -1e9 for j in range(s)] for i in range(m)]
    initial_assignments, initial_scores, _ = _opt_role_incremental_admissible_assignment(
        mentions, slots, base_score_matrix, debug
    )
    filled, filled_in_repair = _opt_role_validate_and_repair(
        mentions, slots, initial_assignments, initial_scores, debug
    )

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


def _run_optimization_role_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, str]]:
    """Run full optimization-role pipeline. Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, filled_in_repair

    initial_assignments, initial_scores, debug = _opt_role_global_assignment(mentions, slots)
    filled, filled_in_repair = _opt_role_validate_and_repair(
        mentions, slots, initial_assignments, initial_scores, debug
    )

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


def _run_optimization_role_anchor_linking(
    query: str,
    variant: str,
    expected_scalar: list[str],
    use_entity_alignment: bool = True,
    use_edge_pruning: bool = True,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, str]]:
    """Anchor-linking: context-aware number-to-slot grounding with alignment features and edge pruning.
    Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, filled_in_repair

    m, s = len(mentions), len(slots)
    score_matrix: list[list[float]] = [[0.0 for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, _ = _score_mention_slot_anchor(mr, sr, use_entity_alignment=use_entity_alignment, use_edge_pruning=use_edge_pruning)
            score_matrix[i][j] = sc

    initial_assignments, initial_scores, debug = _opt_role_global_assignment(mentions, slots, score_matrix=score_matrix)
    filled, filled_in_repair = _opt_role_validate_and_repair(
        mentions, slots, initial_assignments, initial_scores, debug
    )

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


def _run_optimization_role_bottomup_beam_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
    beam_width: int = 5,
    use_anchor_scores: bool = True,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, str]]:
    """Bottom-up beam over partial assignment bundles. Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, filled_in_repair

    m, s = len(mentions), len(slots)

    # Score matrix: (i, j) -> score
    score_matrix: list[list[float]] = [[0.0 for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            if use_anchor_scores:
                sc, _ = _score_mention_slot_anchor(mr, sr, use_entity_alignment=True, use_edge_pruning=True)
            else:
                sc, _ = _score_mention_slot_opt(mr, sr)
            score_matrix[i][j] = sc

    # Atomic candidates: (mention_idx, slot_idx, score); only admissible pairs with positive score
    atoms: list[tuple[int, int, float]] = []
    for i in range(m):
        for j in range(s):
            if score_matrix[i][j] > 0 and score_matrix[i][j] < 1e7:
                atoms.append((i, j, score_matrix[i][j]))
    atoms.sort(key=lambda x: -x[2])

    # Bundle: (frozenset of (mention_idx, slot_idx), total_score)
    def _bundle_to_partial(bundle: frozenset[tuple[int, int]]) -> dict[str, MentionOptIR]:
        return {slots[j].name: mentions[i] for i, j in bundle}

    def _add_relation_bonus(bundle: frozenset[tuple[int, int]], base_score: float) -> float:
        partial = _bundle_to_partial(bundle)
        bonus = 0.0
        for i, j in bundle:
            bonus += _relation_bonus(mentions[i], slots[j], partial, slots, mentions)
        return base_score + bonus

    # Beam: list of (bundle, sum_of_pair_scores); relation bonus added when ranking
    beam: list[tuple[frozenset[tuple[int, int]], float]] = []
    for i, j, sc in atoms[: beam_width * 2]:
        bundle = frozenset({(i, j)})
        partial = _bundle_to_partial(bundle)
        if not _is_partial_admissible(partial, slots):
            continue
        beam.append((bundle, sc))
    beam.sort(key=lambda x: -(x[1] + _add_relation_bonus(x[0], 0.0)))
    beam = beam[:beam_width]

    # Extend beam iteratively
    for _ in range(max(0, min(m, s) - 1)):
        next_beam: list[tuple[frozenset[tuple[int, int]], float]] = []
        for bundle, sum_scores in beam:
            used_m = {i for i, _ in bundle}
            used_s = {j for _, j in bundle}
            for i, j, sc in atoms:
                if i in used_m or j in used_s:
                    continue
                new_bundle = bundle | frozenset({(i, j)})
                partial = _bundle_to_partial(new_bundle)
                if not _is_partial_admissible(partial, slots):
                    continue
                new_sum = sum_scores + sc
                next_beam.append((new_bundle, new_sum))
        if not next_beam:
            break
        next_beam.sort(key=lambda x: (-len(x[0]), -(x[1] + _add_relation_bonus(x[0], 0.0))))
        beam = next_beam[:beam_width]

    if not beam:
        return filled_values, filled_mentions, filled_in_repair

    # Best bundle: prefer more slots filled, then higher (sum_scores + relation_bonus)
    def _rank_score(item: tuple[frozenset[tuple[int, int]], float]) -> tuple[int, float]:
        b, ss = item
        return (len(b), ss + _add_relation_bonus(b, 0.0))

    best_bundle, best_sum = max(beam, key=_rank_score)
    partial = _bundle_to_partial(best_bundle)
    initial_scores = {slots[j].name: score_matrix[i][j] for i, j in best_bundle}
    debug: dict[str, list[dict[str, Any]]] = {}
    for j, sr in enumerate(slots):
        debug[sr.name] = [
            {"mention_id": mentions[i].mention_id, "mention_raw": mentions[i].raw_surface, "score": score_matrix[i][j], "features": {}}
            for i in range(m)
        ]
        debug[sr.name].sort(key=lambda x: x["score"], reverse=True)
    filled, filled_in_repair = _opt_role_validate_and_repair(mentions, slots, partial, initial_scores, debug)

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


def _gcg_local_score(m: "MentionOptIR", s: "SlotOptIR") -> tuple[float, dict[str, Any]]:
    """Local compatibility score between one mention and one slot.

    Mirrors the logic of _score_mention_slot_opt but uses GCG_LOCAL_WEIGHTS so
    that the two scoring functions can be tuned independently.
    """
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.type_bucket
    expected = s.expected_type

    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return GCG_LOCAL_WEIGHTS["type_incompatible_penalty"], features

    # Count-like slot: non-integer decimals are hard incompatible.
    if s.is_count_like and kind == "float":
        features["count_slot_float_incompatible"] = True
        return GCG_LOCAL_WEIGHTS["type_incompatible_penalty"], features

    # Enumeration-derived counts are hard-incompatible with non-count-like slots.
    # They are cardinality candidates only and must not bleed into numeric slots.
    if "derived_count" in m.role_tags and not s.is_count_like:
        features["derived_count_non_count"] = True
        return GCG_LOCAL_WEIGHTS["derived_count_non_count_penalty"], features

    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (expected == "currency" and kind == "currency"):
            score += GCG_LOCAL_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected == "float" and kind in {"float", "int"}:
            # float+float = exact; integer-valued tokens are valid for float slots
            score += GCG_LOCAL_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected == "int" and kind == "int":
            score += GCG_LOCAL_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected in ("int", "float") and kind in {"currency"}:
            score += GCG_LOCAL_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True
        elif expected == "int" and kind == "float":
            score += GCG_LOCAL_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True
        elif expected == "currency" and kind in {"float", "int"}:
            # Monetary slots filled with a plain integer/float token (no "$" sign)
            # are a full exact match — the value IS a valid monetary quantity.
            score += GCG_LOCAL_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True

    # Count-like slot: small-integer cardinality prior and large-value penalty.
    if s.is_count_like and kind == "int" and m.value is not None:
        val = m.value
        if 1 <= val <= GCG_LOCAL_WEIGHTS["count_plausible_max"]:
            score += GCG_LOCAL_WEIGHTS["count_small_int_prior"]
            features["count_small_int_prior"] = True
        elif val > GCG_LOCAL_WEIGHTS["count_large_penalty_threshold"]:
            score += GCG_LOCAL_WEIGHTS["count_large_int_penalty"]
            features["count_large_int_penalty"] = True

    role_overlap = len(m.role_tags & s.slot_role_tags)
    if role_overlap:
        score += GCG_LOCAL_WEIGHTS["opt_role_overlap"] * float(role_overlap)
        features["opt_role_overlap"] = role_overlap

    if m.fragment_type == "objective" and s.is_objective_like:
        score += GCG_LOCAL_WEIGHTS["fragment_compat_bonus"]
        features["fragment_objective"] = True
    if m.fragment_type in ("constraint", "bound") and s.is_bound_like:
        score += GCG_LOCAL_WEIGHTS["fragment_compat_bonus"]
        features["fragment_bound"] = True
    if m.fragment_type == "resource" and s.is_total_like:
        score += GCG_LOCAL_WEIGHTS["fragment_compat_bonus"]
        features["fragment_resource"] = True
    if m.fragment_type == "ratio" and (
        "ratio_constraint" in s.slot_role_tags or "percentage_constraint" in s.slot_role_tags
    ):
        score += GCG_LOCAL_WEIGHTS["fragment_compat_bonus"]
        features["fragment_ratio"] = True

    if m.operator_tags & s.operator_preference:
        score += GCG_LOCAL_WEIGHTS["operator_match_bonus"]
        features["operator_match"] = True

    slot_words = set(s.norm_tokens) | s.alias_tokens
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += GCG_LOCAL_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap
    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += GCG_LOCAL_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    if m.unit_tags & s.unit_preference:
        score += GCG_LOCAL_WEIGHTS["unit_match_bonus"]
        features["unit_match"] = True

    entity_resource = len(
        (m.nearby_entity_tokens | m.nearby_resource_tokens | m.nearby_product_tokens) & slot_words
    )
    if entity_resource:
        score += GCG_LOCAL_WEIGHTS["entity_resource_overlap"] * float(entity_resource)
        features["entity_resource_overlap"] = entity_resource

    # Narrow-left entity anchor: use ONLY single-character tokens from the tight
    # left-context window.  Single letters (e.g. 'b' from "Feed B contains") and
    # single digits (e.g. '1' from "Machine 1 processes") are entity-identifier
    # suffixes that uniquely discriminate FeedB vs FeedA slots or Machine1 vs
    # Machine2 slots.  Longer tokens (semantic-domain words such as 'labor' in
    # "3 labor hours AND 5 board feet") are excluded because they appear in the
    # narrow_left of the *second* list value due to the preceding clause, and
    # would cause false cross-attribute matches within the same entity.
    _nl_id_tokens = {t for t in m.narrow_left_tokens if len(t) == 1}
    narrow_left_overlap = len(_nl_id_tokens & slot_words)
    if narrow_left_overlap:
        score += GCG_LOCAL_WEIGHTS["narrow_left_overlap"] * float(narrow_left_overlap)
        features["narrow_left_overlap"] = narrow_left_overlap

    # Tight-context cost / profit semantic hints.
    # Only fires when the mention is UNambiguously cost-like (not also profit-like)
    # or profit-like (not also cost-like).  When BOTH flags are set the tight window
    # is too contaminated to be a reliable signal (e.g. right-context of "profit is 8"
    # spills "cost" into its tight window), so we skip the rule.
    if m.is_cost_like and not m.is_profit_like:
        if slot_words & _COST_CONTEXT_WORDS:
            score += GCG_LOCAL_WEIGHTS["tight_cost_match_bonus"]
            features["tight_cost_match"] = True
        elif slot_words & _PROFIT_CONTEXT_WORDS:
            score += GCG_LOCAL_WEIGHTS["tight_cost_mismatch_penalty"]
            features["tight_cost_mismatch"] = True
    if m.is_profit_like and not m.is_cost_like:
        if slot_words & _PROFIT_CONTEXT_WORDS:
            score += GCG_LOCAL_WEIGHTS["tight_profit_match_bonus"]
            features["tight_profit_match"] = True
        elif slot_words & _COST_CONTEXT_WORDS:
            score += GCG_LOCAL_WEIGHTS["tight_profit_mismatch_penalty"]
            features["tight_profit_mismatch"] = True

    if s.is_total_like and m.is_total_like:
        score += GCG_LOCAL_WEIGHTS["coefficient_vs_total_bonus"]
        features["total_match"] = True
    if s.is_coefficient_like and m.is_per_unit:
        score += GCG_LOCAL_WEIGHTS["coefficient_vs_total_bonus"]
        features["coefficient_match"] = True
    # Local mismatch penalties: per-unit mention → total slot, or total mention → coeff slot.
    if s.is_total_like and not s.is_coefficient_like and m.is_per_unit and not m.is_total_like:
        score += GCG_LOCAL_WEIGHTS["coeff_to_total_local_penalty"]
        features["coeff_to_total_penalty"] = True
    if s.is_coefficient_like and not s.is_total_like and m.is_total_like and not m.is_per_unit:
        score += GCG_LOCAL_WEIGHTS["total_to_coeff_local_penalty"]
        features["total_to_coeff_penalty"] = True

    # ── Quantity-role layer: primary_role bonuses / penalties ─────────────────
    if m.is_count_like and s.is_count_like:
        score += GCG_LOCAL_WEIGHTS["count_mention_count_slot_bonus"]
        features["count_role_match"] = True
    if m.is_count_like and not s.is_count_like and not m.is_percent_like:
        score += GCG_LOCAL_WEIGHTS["count_mention_non_count_penalty"]
        features["count_to_non_count_penalty"] = True
    if m.is_lower_bound_like and "min" in s.operator_preference:
        score += GCG_LOCAL_WEIGHTS["bound_direction_bonus"]
        features["lower_bound_match"] = True
    if m.is_upper_bound_like and "max" in s.operator_preference:
        score += GCG_LOCAL_WEIGHTS["bound_direction_bonus"]
        features["upper_bound_match"] = True
    # Wrong-direction bound penalty: strong discouragement for cross-direction assignments.
    if _ENABLE_BOUND_ROLE_LAYER:
        if m.is_lower_bound_like and "max" in s.operator_preference and "min" not in s.operator_preference:
            score += GCG_LOCAL_WEIGHTS["bound_direction_penalty"]
            features["bound_direction_wrong"] = True
        elif m.is_upper_bound_like and "min" in s.operator_preference and "max" not in s.operator_preference:
            score += GCG_LOCAL_WEIGHTS["bound_direction_penalty"]
            features["bound_direction_wrong"] = True

    score += GCG_LOCAL_WEIGHTS["schema_prior_bonus"]
    features["schema_prior"] = True

    if score <= 0.0:
        score += GCG_LOCAL_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


def _gcg_global_penalty(
    assignment: dict[str, "MentionOptIR"],
    slots_by_name: dict[str, "SlotOptIR"],
    all_mentions: list["MentionOptIR"],
) -> tuple[float, list[str]]:
    """Compute global consistency delta (rewards + penalties) for a full/partial assignment.

    Returns (total_delta, list_of_active_reasons) for diagnostics.
    """
    delta = 0.0
    reasons: list[str] = []
    w = GCG_GLOBAL_WEIGHTS

    # Detect global evidence: are there percent mentions in the text?
    has_pct_mention = any(m.type_bucket == "percent" for m in all_mentions)

    # Detect per-slot penalties.
    used_mention_ids: list[int] = []
    for slot_name, m in assignment.items():
        s = slots_by_name.get(slot_name)
        if s is None:
            continue

        mid = m.mention_id
        used_mention_ids.append(mid)

        # Percent misuse.
        if has_pct_mention and s.expected_type != "percent" and m.type_bucket == "percent":
            delta += w["percent_misuse_penalty"]
            reasons.append(f"percent_misuse:{slot_name}")

        if s.expected_type == "percent" and m.type_bucket != "percent" and has_pct_mention:
            delta += w["non_percent_to_pct_slot_penalty"]
            reasons.append(f"non_pct_to_pct_slot:{slot_name}")

        # Total vs coefficient mismatch.
        if s.is_total_like and m.is_per_unit and not m.is_total_like:
            delta += w["total_to_coeff_penalty"]
            reasons.append(f"coeff_to_total:{slot_name}")
        if s.is_coefficient_like and m.is_total_like and not m.is_per_unit:
            delta += w["coeff_to_total_penalty"]
            reasons.append(f"total_to_coeff:{slot_name}")

        # Operator/bound flip.
        if s.operator_preference and m.operator_tags:
            if "min" in s.operator_preference and "max" in m.operator_tags and "min" not in m.operator_tags:
                delta += w["bound_flip_penalty"]
                reasons.append(f"bound_flip_min_to_max:{slot_name}")
            if "max" in s.operator_preference and "min" in m.operator_tags and "max" not in m.operator_tags:
                delta += w["bound_flip_penalty"]
                reasons.append(f"bound_flip_max_to_min:{slot_name}")

        # Coverage reward (per filled slot).
        delta += w["coverage_reward_per_slot"]
        reasons.append(f"coverage_reward:{slot_name}")

        # Type consistency reward.
        if m.type_bucket == s.expected_type or (
            s.expected_type in ("int", "float") and m.type_bucket in ("int", "float", "currency")
        ):
            delta += w["type_consistency_reward"]
            reasons.append(f"type_consistent:{slot_name}")

    # Duplicate mention penalty (one-to-one should hold).
    seen: set[int] = set()
    for mid in used_mention_ids:
        if mid in seen:
            delta += w["duplicate_mention_penalty"]
            reasons.append(f"duplicate_mention:{mid}")
        seen.add(mid)

    # Overall coverage bonus.
    if slots_by_name and len(assignment) / max(1, len(slots_by_name)) >= 0.8:
        delta += w["plausibility_coverage_bonus"]
        reasons.append("plausibility_coverage_bonus")

    # ── Entity-coherence among consecutive-mention pairs ─────────────────────
    # When mention M_{i+1} has no entity-letter anchor in its narrow_left, it
    # should be assigned to the same entity as the preceding mention M_i.
    # This repairs cases like "Feed B contains 7 protein AND 15 fat" where the
    # fat value (15) carries no explicit entity marker in its narrow_left but
    # must follow the Feed-B-anchored protein assignment.
    # Only fires when BOTH the preceding and current slot names contain
    # single-char entity identifiers (preventing spurious penalties for
    # attribute-only slot names such as "TotalBudget").
    _mid_to_slot: dict[int, "SlotOptIR"] = {}
    for _sn, _m in assignment.items():
        _s = slots_by_name.get(_sn)
        if _s is not None:
            _mid_to_slot[_m.mention_id] = _s
    for _sn, _m in assignment.items():
        _prev_mid = _m.mention_id - 1
        if _prev_mid not in _mid_to_slot:
            continue
        # Skip if this mention already has a direct entity-letter anchor.
        _cur_entity_letters = {t for t in _m.narrow_left_tokens if len(t) == 1 and t.isalpha()}
        if _cur_entity_letters:
            continue
        _prev_slot = _mid_to_slot[_prev_mid]
        _cur_slot = slots_by_name.get(_sn)
        if _cur_slot is None:
            continue
        # Extract single-char entity identifiers from both slot names.
        _prev_eids = {t for t in _prev_slot.norm_tokens if len(t) == 1}
        _cur_eids = {t for t in _cur_slot.norm_tokens if len(t) == 1}
        if _prev_eids and _cur_eids:
            if _prev_eids & _cur_eids:
                delta += w["entity_coherence_reward"]
                reasons.append(f"entity_coherent:{_sn}")
            else:
                delta += w["entity_coherence_penalty"]
                reasons.append(f"entity_incoherent:{_sn}")

    return delta, reasons


def _gcg_beam_search(
    mentions: list["MentionOptIR"],
    slots: list["SlotOptIR"],
    local_scores: list[list[float]],
    local_features: list[list[dict[str, Any]]],
    beam_width: int = GCG_BEAM_WIDTH,
) -> tuple[
    dict[str, "MentionOptIR"],
    dict[str, float],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Beam search over partial slot assignments scored by local + global consistency.

    State: frozenset of (slot_index, mention_index) pairs already committed.
    Beam entry: (committed_bundle, sum_local_scores).

    At the end, the beam entry with the highest (local_sum + global_delta) wins.

    Returns:
        assignments        : slot_name -> MentionOptIR
        slot_scores        : slot_name -> local score
        debug              : slot_name -> candidate list (for diagnostics)
        top_assignments    : list of top-k assignment dicts with scores/reasons (diagnostics)
    """
    if not mentions or not slots:
        return {}, {}, {}, []

    slots_by_name = {s.name: s for s in slots}

    # Build debug info: per-slot sorted candidate list.
    debug: dict[str, list[dict[str, Any]]] = {}
    for j, sr in enumerate(slots):
        cands = []
        for i, mr in enumerate(mentions):
            cands.append({
                "mention_id": mr.mention_id,
                "mention_raw": mr.raw_surface,
                "score": local_scores[i][j],
                "features": local_features[i][j],
            })
        cands.sort(key=lambda x: x["score"], reverse=True)
        debug[sr.name] = cands

    # Pruning: build per-slot list of admissible (mention_index, local_score) pairs.
    admissible: list[list[tuple[int, float]]] = []
    for j in range(len(slots)):
        opts = [
            (i, local_scores[i][j])
            for i in range(len(mentions))
            if local_scores[i][j] > GCG_PRUNE_THRESHOLD
        ]
        opts.sort(key=lambda x: -x[1])
        admissible.append(opts)

    # Each beam state: (bundle: frozenset[(slot_j, mention_i)], sum_local)
    # Slots are processed in their original order.
    slot_order = list(range(len(slots)))

    # Beam: list of (bundle, sum_local_score)
    BeamState = tuple[frozenset[tuple[int, int]], float]
    beam: list[BeamState] = [(frozenset(), 0.0)]

    for j in slot_order:
        next_beam: list[BeamState] = []
        for bundle, sum_local in beam:
            # Option 1: skip this slot (leave unassigned).
            next_beam.append((bundle, sum_local))
            # Option 2: assign an admissible mention (one not already used).
            used_mentions = {mi for _, mi in bundle}
            for mi, loc_sc in admissible[j]:
                if mi in used_mentions:
                    continue
                new_bundle = bundle | frozenset([(j, mi)])
                next_beam.append((new_bundle, sum_local + loc_sc))

        # Sort by local sum and prune to beam width.
        next_beam.sort(key=lambda x: -x[1])
        beam = next_beam[:beam_width]

    # Score all final beam states with global consistency delta and pick the best.
    scored_final: list[tuple[float, BeamState, float, list[str]]] = []
    for bundle, sum_local in beam:
        assignment_candidate: dict[str, "MentionOptIR"] = {
            slots[j].name: mentions[mi] for j, mi in bundle
        }
        g_delta, reasons = _gcg_global_penalty(assignment_candidate, slots_by_name, mentions)
        total_score = sum_local + g_delta
        scored_final.append((total_score, (bundle, sum_local), g_delta, reasons))

    if not scored_final:
        return {}, {}, debug, []

    scored_final.sort(key=lambda x: -x[0])
    best_total, (best_bundle, _), best_g_delta, best_reasons = scored_final[0]

    # Build output.
    assignments: dict[str, "MentionOptIR"] = {}
    slot_scores: dict[str, float] = {}
    for j, mi in best_bundle:
        slot_name = slots[j].name
        assignments[slot_name] = mentions[mi]
        slot_scores[slot_name] = local_scores[mi][j]

    # Build top-k diagnostics.
    top_assignments: list[dict[str, Any]] = []
    for rank, (total_sc, (bun, loc_sum), g_dl, rsns) in enumerate(scored_final[:beam_width]):
        asgn_repr = {slots[j].name: mentions[mi].raw_surface for j, mi in bun}
        top_assignments.append({
            "rank": rank,
            "total_score": total_sc,
            "local_sum": loc_sum,
            "global_delta": g_dl,
            "active_reasons": rsns,
            "assignment": asgn_repr,
        })

    return assignments, slot_scores, debug, top_assignments


def _run_global_consistency_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    beam_width: int = GCG_BEAM_WIDTH,
) -> tuple[dict[str, Any], dict[str, "MentionOptIR"], dict[str, Any]]:
    """Global Consistency Grounding (GCG): beam-search global assignment.

    Stages:
      1. Extract optimization-role mentions (reuses existing extractor).
      2. Build slot IRs (reuses existing builder).
      3. Compute local scores for all (mention, slot) pairs.
      4. Prune implausible pairs below GCG_PRUNE_THRESHOLD.
      5. Beam search over partial assignments, scoring by local + global consistency.
      6. Return the assignment with the highest combined score.

    Returns:
        filled_values      : slot_name -> numeric value (float or raw string)
        filled_mentions    : slot_name -> MentionOptIR
        diagnostics        : dict with top_assignments + per-slot debug info
    """
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, "MentionOptIR"] = {}
    diagnostics: dict[str, Any] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, diagnostics

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, diagnostics

    m_count, s_count = len(mentions), len(slots)

    # Precompute local scores and features for all (mention, slot) pairs.
    local_scores: list[list[float]] = [[0.0] * s_count for _ in range(m_count)]
    local_features: list[list[dict[str, Any]]] = [[{} for _ in range(s_count)] for _ in range(m_count)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _gcg_local_score(mr, sr)
            local_scores[i][j] = sc
            local_features[i][j] = feats

    assignments, slot_scores, debug, top_assignments = _gcg_beam_search(
        mentions, slots, local_scores, local_features, beam_width=beam_width
    )

    # ── Bound-flip swap repair: correct inverted min/max assignments ──────────
    gcg_filled_in_repair: dict[str, str] = {k: "initial" for k in assignments}
    if _ENABLE_BOUND_ROLE_LAYER:
        _bound_swap_repair(assignments, gcg_filled_in_repair, slots)

    for slot_name, mr in assignments.items():
        filled_values[slot_name] = mr.tok.value if mr.tok.value is not None else mr.tok.raw
        filled_mentions[slot_name] = mr

    diagnostics["top_assignments"] = top_assignments
    diagnostics["per_slot_candidates"] = debug
    return filled_values, filled_mentions, diagnostics


# ── Maximum-Weight Bipartite Matching Grounding ───────────────────────────────
# Uses the exact Hungarian algorithm (scipy.optimize.linear_sum_assignment) to
# find the globally optimal one-to-one mention-slot assignment.  Unlike
# beam-search methods this approach:
#   - Builds the FULL score matrix for all (mention, slot) pairs.
#   - Enforces the one-to-one constraint exactly.
#   - Finds the globally optimal assignment (under the local score objective)
#     in polynomial time — O(min(m,s)^3) via scipy's implementation.
#
# Falls back to a bitmask-DP assignment if scipy is unavailable
# (_opt_role_global_assignment already handles this internally).


def _run_max_weight_matching_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[dict[str, Any], dict[str, "MentionOptIR"], dict[str, Any]]:
    """Maximum-weight bipartite matching grounding (Hungarian algorithm).

    Computes the globally optimal one-to-one mention-to-slot assignment by:
      1. Extracting optimization-role mentions.
      2. Building slot IRs.
      3. Scoring all (mention, slot) pairs with the opt-role scoring function.
      4. Solving the resulting assignment problem exactly using
         ``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm) or a
         bitmask-DP fallback if scipy is not available.

    This guarantees:
      - A **full score matrix** is computed (no greedy shortcut).
      - A **one-to-one assignment constraint** is enforced exactly.
      - The **globally optimal assignment** under the local score objective is
        returned.

    Returns
    -------
    filled_values   : slot_name -> numeric value (float or raw string)
    filled_mentions : slot_name -> MentionOptIR
    diagnostics     : dict with per_slot_candidates and slot_scores
    """
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, "MentionOptIR"] = {}
    diagnostics: dict[str, Any] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, diagnostics

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, diagnostics

    # _opt_role_global_assignment already implements the full score matrix +
    # Hungarian algorithm (scipy linear_sum_assignment) with a DP fallback.
    assignments, scores, debug = _opt_role_global_assignment(mentions, slots)

    for slot_name, mr in assignments.items():
        filled_values[slot_name] = mr.tok.value if mr.tok.value is not None else mr.tok.raw
        filled_mentions[slot_name] = mr

    diagnostics["per_slot_candidates"] = debug
    diagnostics["slot_scores"] = scores
    return filled_values, filled_mentions, diagnostics


# ── Global Compatibility Grounding (global_compatibility_grounding) ───────────
# Extends GCG with explicit pairwise slot-slot compatibility terms that score
# *pairs* of (slot, mention) assignments jointly.  This implements the
# reviewer-requested "structured slot filling" idea: adjacent slots should be
# coherent with each other, not only individually compatible with their mentions.
#
# Ablation modes (controlled by the ablation_mode argument):
#   "local_only"  — local mention-slot scores only (no pairwise, no global)
#   "pairwise"    — local + pairwise slot-slot terms (no global coverage bonuses)
#   "full"        — local + pairwise + global consistency (complete method)
#
# The pairwise score Phi((si,mi),(sj,mj)) is a sum over pairwise feature detectors.
# Each detector fires based on:
#   - slot role semantics (min/max, total/coeff, percent/scalar, objective/bound)
#   - mention types and per-unit / total flags
#   - relative mention values (ordering coherence for min/max)
#   - mention identity (duplicate penalty)

GCGP_PAIRWISE_WEIGHTS: dict[str, float] = {
    # ── Min/max value ordering ────────────────────────────────────────────────
    "minmax_correct_order_bonus": 2.5,  # min_val <= max_val → bonus
    "minmax_inverted_order_penalty": -4.0,  # min_val > max_val → penalty
    # ── Total vs coefficient separation ──────────────────────────────────────
    "total_coeff_type_distinct_bonus": 1.5,  # total slot + coeff slot → different types/sizes → bonus
    "total_coeff_both_same_type_penalty": -2.0,  # both get same mention type → penalty
    # ── Percent exclusivity ───────────────────────────────────────────────────
    "percent_exclusive_bonus": 1.5,  # percent slot ← pct mention, other ← non-pct → bonus
    "percent_collision_penalty": -3.5,  # two percent slots both get non-pct (when pct exists) → penalty
    # ── Duplicate mention penalty ─────────────────────────────────────────────
    "duplicate_mention_pairwise_penalty": -6.0,  # two slots share the same mention_id → hard penalty
    # ── Semantic role compatibility ───────────────────────────────────────────
    "objective_bound_compatible_bonus": 1.0,  # objective slot + bound slot → different mention roles → bonus
    "objective_objective_collision_penalty": -1.0,  # two objective slots with same-role mention → mild penalty
    # ── Value magnitude plausibility ─────────────────────────────────────────
    "magnitude_budget_gt_coeff_bonus": 1.0,  # total slot value > coeff slot value → bonus
    "magnitude_budget_lt_coeff_penalty": -2.0,  # total slot value < coeff slot value → penalty
}

# Beam width for the pairwise-aware beam search.
GCGP_BEAM_WIDTH: int = 12  # slightly wider than GCG because pairwise reranking helps


def _gcgp_pairwise_score(
    si: "SlotOptIR",
    mi: "MentionOptIR",
    sj: "SlotOptIR",
    mj: "MentionOptIR",
) -> tuple[float, list[str]]:
    """Compute a pairwise compatibility score for the joint assignment (si→mi, sj→mj).

    This scores *cross-slot coherence*: do the two assigned mentions make sense
    together given the semantic relationship between si and sj?

    Returns (delta, reasons) where reasons is a diagnostic list of strings.
    """
    delta = 0.0
    reasons: list[str] = []
    w = GCGP_PAIRWISE_WEIGHTS

    # ── Hard duplicate penalty ────────────────────────────────────────────────
    if mi.mention_id == mj.mention_id:
        delta += w["duplicate_mention_pairwise_penalty"]
        reasons.append(f"duplicate_mention:({si.name},{sj.name})")
        return delta, reasons  # no other terms make sense if same mention

    # ── Min/max value ordering ────────────────────────────────────────────────
    si_is_min = bool(si.operator_preference and "min" in si.operator_preference)
    sj_is_max = bool(sj.operator_preference and "max" in sj.operator_preference)
    si_is_max = bool(si.operator_preference and "max" in si.operator_preference)
    sj_is_min = bool(sj.operator_preference and "min" in sj.operator_preference)

    if si_is_min and sj_is_max and mi.value is not None and mj.value is not None:
        if mi.value <= mj.value:
            delta += w["minmax_correct_order_bonus"]
            reasons.append(f"minmax_correct_order:({si.name},{sj.name})")
        else:
            delta += w["minmax_inverted_order_penalty"]
            reasons.append(f"minmax_inverted_order:({si.name},{sj.name})")
    elif si_is_max and sj_is_min and mi.value is not None and mj.value is not None:
        if mj.value <= mi.value:
            delta += w["minmax_correct_order_bonus"]
            reasons.append(f"minmax_correct_order_rev:({si.name},{sj.name})")
        else:
            delta += w["minmax_inverted_order_penalty"]
            reasons.append(f"minmax_inverted_order_rev:({si.name},{sj.name})")

    # ── Total vs coefficient separation ──────────────────────────────────────
    one_total_one_coeff = (si.is_total_like and sj.is_coefficient_like) or (
        si.is_coefficient_like and sj.is_total_like
    )
    if one_total_one_coeff:
        # Good: different mention kinds (or different is_per_unit / is_total_like flags)
        if mi.is_total_like != mj.is_total_like or mi.is_per_unit != mj.is_per_unit:
            delta += w["total_coeff_type_distinct_bonus"]
            reasons.append(f"total_coeff_distinct:({si.name},{sj.name})")
        elif mi.type_bucket == mj.type_bucket:
            delta += w["total_coeff_both_same_type_penalty"]
            reasons.append(f"total_coeff_same_type:({si.name},{sj.name})")
        # Magnitude plausibility: total slot should have larger value than coeff slot.
        total_m = mi if si.is_total_like else mj
        coeff_m = mj if si.is_total_like else mi
        if total_m.value is not None and coeff_m.value is not None:
            if total_m.value > coeff_m.value:
                delta += w["magnitude_budget_gt_coeff_bonus"]
                reasons.append(f"magnitude_total_gt_coeff:({si.name},{sj.name})")
            elif total_m.value < coeff_m.value:
                delta += w["magnitude_budget_lt_coeff_penalty"]
                reasons.append(f"magnitude_total_lt_coeff:({si.name},{sj.name})")

    # ── Percent exclusivity ───────────────────────────────────────────────────
    si_is_pct = si.expected_type == "percent"
    sj_is_pct = sj.expected_type == "percent"
    if si_is_pct != sj_is_pct:
        # One percent slot, one non-percent slot.
        pct_slot_m = mi if si_is_pct else mj
        non_pct_slot_m = mj if si_is_pct else mi
        if pct_slot_m.type_bucket == "percent" and non_pct_slot_m.type_bucket != "percent":
            delta += w["percent_exclusive_bonus"]
            reasons.append(f"percent_exclusive:({si.name},{sj.name})")
    elif si_is_pct and sj_is_pct:
        # Both percent slots — check they don't both get non-percent mentions.
        if mi.type_bucket != "percent" and mj.type_bucket != "percent":
            delta += w["percent_collision_penalty"]
            reasons.append(f"percent_collision:({si.name},{sj.name})")

    # ── Objective vs bound role compatibility ─────────────────────────────────
    si_obj = si.is_objective_like
    sj_bnd = sj.is_bound_like
    si_bnd = si.is_bound_like
    sj_obj = sj.is_objective_like
    if (si_obj and sj_bnd) or (si_bnd and sj_obj):
        # Objective-like slot and bound-like slot: prefer different fragment types.
        mi_obj = mi.fragment_type == "objective"
        mj_bnd = mj.fragment_type in ("constraint", "bound")
        mi_bnd = mi.fragment_type in ("constraint", "bound")
        mj_obj = mj.fragment_type == "objective"
        if (si_obj and mi_obj and sj_bnd and mj_bnd) or (si_bnd and mi_bnd and sj_obj and mj_obj):
            delta += w["objective_bound_compatible_bonus"]
            reasons.append(f"obj_bound_compat:({si.name},{sj.name})")
    elif si_obj and sj_obj:
        # Both objective: mild penalty if they share the same fragment context.
        if mi.fragment_type == mj.fragment_type == "objective":
            delta += w["objective_objective_collision_penalty"]
            reasons.append(f"obj_obj_collision:({si.name},{sj.name})")

    return delta, reasons


def _gcgp_beam_search(
    mentions: list["MentionOptIR"],
    slots: list["SlotOptIR"],
    local_scores: list[list[float]],
    local_features: list[list[dict[str, Any]]],
    ablation_mode: str = "full",
    beam_width: int = GCGP_BEAM_WIDTH,
) -> tuple[
    dict[str, "MentionOptIR"],
    dict[str, float],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Beam search with pairwise slot-slot compatibility terms.

    ablation_mode controls which scoring components are active:
      "local_only"  — only local mention-slot scores (GCG without global/pairwise)
      "pairwise"    — local + pairwise (no GCG global consistency terms)
      "full"        — local + pairwise + GCG global consistency terms

    Returns same structure as _gcg_beam_search for drop-in compatibility.
    """
    if not mentions or not slots:
        return {}, {}, {}, []

    slots_by_name = {s.name: s for s in slots}

    # Build per-slot candidate debug info.
    debug: dict[str, list[dict[str, Any]]] = {}
    for j, sr in enumerate(slots):
        cands = []
        for i, mr in enumerate(mentions):
            cands.append({
                "mention_id": mr.mention_id,
                "mention_raw": mr.raw_surface,
                "score": local_scores[i][j],
                "features": local_features[i][j],
            })
        cands.sort(key=lambda x: x["score"], reverse=True)
        debug[sr.name] = cands

    # Per-slot admissible (mention_index, local_score) lists, pruned.
    admissible: list[list[tuple[int, float]]] = []
    for j in range(len(slots)):
        opts = [
            (i, local_scores[i][j])
            for i in range(len(mentions))
            if local_scores[i][j] > GCG_PRUNE_THRESHOLD
        ]
        opts.sort(key=lambda x: -x[1])
        admissible.append(opts)

    # Beam state: (bundle: frozenset[(slot_j, mention_i)], running_score)
    # running_score accumulates local + pairwise scores during the search.
    BeamState = tuple[frozenset[tuple[int, int]], float]
    beam: list[BeamState] = [(frozenset(), 0.0)]

    for j in range(len(slots)):
        next_beam: list[BeamState] = []
        for bundle, running_score in beam:
            # Option 1: leave slot j unassigned.
            next_beam.append((bundle, running_score))
            # Option 2: assign an admissible, unused mention.
            used_mentions = {mi for _, mi in bundle}
            for mi, loc_sc in admissible[j]:
                if mi in used_mentions:
                    continue
                new_score = running_score + loc_sc
                # Add pairwise terms against all already-committed slots.
                if ablation_mode in ("pairwise", "full"):
                    for prev_j, prev_mi in bundle:
                        pw_delta, _ = _gcgp_pairwise_score(
                            slots[j], mentions[mi],
                            slots[prev_j], mentions[prev_mi],
                        )
                        new_score += pw_delta
                new_bundle = bundle | frozenset([(j, mi)])
                next_beam.append((new_bundle, new_score))

        next_beam.sort(key=lambda x: -x[1])
        beam = next_beam[:beam_width]

    # Re-score final beam states with global consistency terms (full mode only).
    scored_final: list[tuple[float, BeamState, float, list[str]]] = []
    for bundle, running_score in beam:
        assignment_candidate: dict[str, "MentionOptIR"] = {
            slots[j].name: mentions[mi] for j, mi in bundle
        }
        if ablation_mode == "full":
            g_delta, reasons = _gcg_global_penalty(assignment_candidate, slots_by_name, mentions)
        else:
            g_delta, reasons = 0.0, []
        total_score = running_score + g_delta
        scored_final.append((total_score, (bundle, running_score), g_delta, reasons))

    if not scored_final:
        return {}, {}, debug, []

    scored_final.sort(key=lambda x: -x[0])
    best_total, (best_bundle, _), best_g_delta, best_reasons = scored_final[0]

    assignments: dict[str, "MentionOptIR"] = {}
    slot_scores: dict[str, float] = {}
    for j, mi in best_bundle:
        slot_name = slots[j].name
        assignments[slot_name] = mentions[mi]
        slot_scores[slot_name] = local_scores[mi][j]

    # Top-k diagnostics.
    top_assignments: list[dict[str, Any]] = []
    for rank, (total_sc, (bun, loc_sum), g_dl, rsns) in enumerate(scored_final[:beam_width]):
        asgn_repr = {slots[j].name: mentions[mi].raw_surface for j, mi in bun}
        top_assignments.append({
            "rank": rank,
            "total_score": total_sc,
            "local_sum": loc_sum,
            "global_delta": g_dl,
            "active_reasons": rsns,
            "assignment": asgn_repr,
        })

    return assignments, slot_scores, debug, top_assignments


def _run_global_compatibility_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    ablation_mode: str = "full",
    beam_width: int = GCGP_BEAM_WIDTH,
) -> tuple[dict[str, Any], dict[str, "MentionOptIR"], dict[str, Any]]:
    """Global Compatibility Grounding (GCGP): beam search with pairwise slot-slot terms.

    Extension of Global Consistency Grounding that adds explicit pairwise
    compatibility terms between assigned slot-mention pairs.

    ablation_mode:
      "local_only"  — local scores only (same as simplified GCG)
      "pairwise"    — local + pairwise slot-slot terms
      "full"        — local + pairwise + global consistency (complete method)

    Returns:
        filled_values      : slot_name -> numeric value
        filled_mentions    : slot_name -> MentionOptIR
        diagnostics        : dict with top_assignments, per_slot_candidates, ablation_mode
    """
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, "MentionOptIR"] = {}
    diagnostics: dict[str, Any] = {"ablation_mode": ablation_mode}

    if not expected_scalar:
        return filled_values, filled_mentions, diagnostics

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, diagnostics

    m_count, s_count = len(mentions), len(slots)
    local_scores: list[list[float]] = [[0.0] * s_count for _ in range(m_count)]
    local_features: list[list[dict[str, Any]]] = [[{} for _ in range(s_count)] for _ in range(m_count)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _gcg_local_score(mr, sr)
            local_scores[i][j] = sc
            local_features[i][j] = feats

    assignments, slot_scores, debug, top_assignments = _gcgp_beam_search(
        mentions, slots, local_scores, local_features,
        ablation_mode=ablation_mode,
        beam_width=beam_width,
    )

    for slot_name, mr in assignments.items():
        filled_values[slot_name] = mr.tok.value if mr.tok.value is not None else mr.tok.raw
        filled_mentions[slot_name] = mr

    diagnostics["top_assignments"] = top_assignments
    diagnostics["per_slot_candidates"] = debug
    return filled_values, filled_mentions, diagnostics


def _constrained_assignment(
    mentions: list[MentionRecord],
    slots: list[SlotRecord],
) -> tuple[dict[str, MentionRecord], dict[str, list[dict[str, Any]]]]:
    """
    Deterministic global assignment: at most one mention per slot, one slot per mention.

    Uses simple DP over subsets of slots (sufficient for small numbers of scalar params).
    Returns:
      - mapping slot_name -> assigned MentionRecord (if any)
      - debug info: slot_name -> list of candidate dicts with scores/features
    """
    assignments: dict[str, MentionRecord] = {}
    debug: dict[str, list[dict[str, Any]]] = {}

    if not mentions or not slots:
        return assignments, debug

    m = len(mentions)
    s = len(slots)

    # Precompute scores and feature breakdowns.
    scores: list[list[float]] = [[0.0 for _ in range(s)] for _ in range(m)]
    feat_mat: list[list[dict[str, Any]]] = [[{} for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _score_mention_slot(mr, sr)
            scores[i][j] = sc
            feat_mat[i][j] = feats

    # Build debug info: per-slot candidate list sorted by score.
    for j, sr in enumerate(slots):
        cand: list[dict[str, Any]] = []
        for i, mr in enumerate(mentions):
            cand.append(
                {
                    "mention_index": mr.index,
                    "mention_raw": mr.tok.raw,
                    "mention_kind": mr.tok.kind,
                    "score": scores[i][j],
                    "features": feat_mat[i][j],
                }
            )
        cand.sort(key=lambda x: x["score"], reverse=True)
        debug[sr.name] = cand

    # Dynamic programming over subsets of slots.
    # This is exponential in number of slots; fallback to greedy if too many.
    if s > 15:
        used_mentions: set[int] = set()
        for j, sr in enumerate(slots):
            best_i = None
            best_score = float("-inf")
            for i in range(m):
                if i in used_mentions:
                    continue
                sc = scores[i][j]
                if sc > best_score:
                    best_score = sc
                    best_i = i
            if best_i is not None and best_score > 0.0:
                assignments[slots[j].name] = mentions[best_i]
                used_mentions.add(best_i)
        return assignments, debug

    from math import inf

    num_states = 1 << s
    dp = [[-inf for _ in range(num_states)] for _ in range(m + 1)]
    parent: list[list[tuple[int, int | None]]] = [
        [(-1, None) for _ in range(num_states)] for _ in range(m + 1)
    ]
    dp[0][0] = 0.0

    for i in range(m):
        for mask in range(num_states):
            cur = dp[i][mask]
            if cur == -inf:
                continue
            # Option 1: skip mention i.
            if cur > dp[i + 1][mask]:
                dp[i + 1][mask] = cur
                parent[i + 1][mask] = (mask, None)
            # Option 2: assign mention i to an unused slot j.
            for j in range(s):
                if (mask >> j) & 1:
                    continue
                sc = scores[i][j]
                if sc <= 0.0:
                    continue
                new_mask = mask | (1 << j)
                new_val = cur + sc
                if new_val > dp[i + 1][new_mask]:
                    dp[i + 1][new_mask] = new_val
                    parent[i + 1][new_mask] = (mask, j)

    # Choose best final mask.
    best_mask = 0
    best_val = -inf
    for mask in range(num_states):
        if dp[m][mask] > best_val:
            best_val = dp[m][mask]
            best_mask = mask

    # Backtrack.
    i = m
    mask = best_mask
    used_slot_for_mention: dict[int, int] = {}
    while i > 0:
        prev_mask, chosen_j = parent[i][mask]
        if chosen_j is not None:
            used_slot_for_mention[i - 1] = chosen_j
        mask = prev_mask
        i -= 1

    for mi, sj in used_slot_for_mention.items():
        sr = slots[sj]
        assignments[sr.name] = mentions[mi]

    return assignments, debug


def _rel_err(pred: float, gold: float) -> float:
    return abs(pred - gold) / max(1.0, abs(gold))


def _md5_seed(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _upsert_summary_row(summary_path: Path, row: dict) -> None:
    cols = [
        "variant",
        "baseline",
        "schema_R1",
        "param_coverage",
        "type_match",
        "exact5_on_hits",
        "exact20_on_hits",
        "param_coverage_hits",
        "param_coverage_miss",
        "type_match_hits",
        "type_match_miss",
        "key_overlap",
        "key_overlap_hits",
        "key_overlap_miss",
        "instantiation_ready",
        "n",
    ]
    rows: list[dict] = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    d = {(r.get("variant"), r.get("baseline")): r for r in rows}
    d[(row["variant"], row["baseline"])] = {k: row.get(k, "") for k in cols}

    # deterministic order
    ordered = sorted(d.values(), key=lambda x: (x.get("variant", ""), x.get("baseline", "")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ordered)


def _upsert_types_rows(types_path: Path, types_agg: dict[str, dict]) -> None:
    cols = [
        "variant",
        "baseline",
        "param_type",
        "n_expected",
        "n_filled",
        "param_coverage",
        "type_match",
        "exact5_on_hits",
        "exact20_on_hits",
        "n_queries",
    ]
    rows: list[dict] = []
    if types_path.exists():
        with open(types_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    d = {(r.get("variant"), r.get("baseline"), r.get("param_type")): r for r in rows}
    for t, info in types_agg.items():
        key = (info["variant"], info["baseline"], info["param_type"])
        d[key] = {k: info.get(k, "") for k in cols}

    ordered = sorted(d.values(), key=lambda x: (x.get("variant", ""), x.get("baseline", ""), x.get("param_type", "")))
    types_path.parent.mkdir(parents=True, exist_ok=True)
    with open(types_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ordered)


# --- Acceptance reranking: schema acceptance scoring + optional hierarchy ---
# Weights for interpretable acceptance scoring (additive).
ACCEPTANCE_RERANK_WEIGHTS = {
    "type_coverage_bonus": 1.5,
    "type_missing_penalty": -2.0,
    "role_coverage_bonus": 1.2,
    "operator_compat_bonus": 0.8,
    "fillability_bonus": 1.0,
    "family_consistency_bonus": 1.5,
    "family_mismatch_penalty": -1.5,
    "missing_critical_penalty": -2.5,
    "weak_fit_penalty": -0.8,
}

# Coarse optimization families (rule-based from schema name + slot names).
FAMILY_KEYWORDS: dict[str, list[str]] = {
    "allocation": ["budget", "investment", "allocate", "portfolio", "dollar", "condo", "detached"],
    "production": ["produce", "sandwich", "ingredient", "blend", "production", "recipe"],
    "transportation": ["transport", "flow", "route", "ship", "delivery", "cable", "gold"],
    "scheduling": ["schedule", "assignment", "assign", "worker", "wage", "radio", "ad"],
    "packing": ["pack", "knapsack", "selection", "item", "weight", "capacity"],
    "covering": ["cover", "facility", "location", "warehouse", "customer"],
    "network": ["network", "graph", "path", "edge", "node"],
    "inventory": ["inventory", "supply", "capacity", "available", "stock"],
    "ratio": ["ratio", "fraction", "percentage", "percent", "minimum percentage", "ratio"],
    "generic_lp": ["maximize", "minimize", "subject", "constraint", "linear"],
}


def _schema_family(schema_id: str, schema_text: str, slot_names: list[str]) -> str:
    """Assign coarse optimization family from schema id, description, and slot names."""
    combined = " ".join([schema_id, schema_text] + slot_names).lower()
    best = "generic_lp"
    best_count = 0
    for family, keywords in FAMILY_KEYWORDS.items():
        if family == "generic_lp":
            continue
        c = sum(1 for kw in keywords if kw in combined)
        if c > best_count:
            best_count = c
            best = family
    return best


def _schema_subgroup(schema_id: str, slot_names: list[str]) -> str:
    """Optional subgroup from slot name patterns."""
    slot_str = " ".join(slot_names).lower()
    if any(x in slot_str for x in ["totalbudget", "total budget", "profitper", "profit per"]):
        return "total_budget_per_unit_profit"
    if any(x in slot_str for x in ["capacity", "demand", "available", "required"]):
        return "capacity_demand"
    if any(x in slot_str for x in ["minimum", "maximum", "min", "max", "at least", "at most"]):
        return "min_max_bounds"
    if any(x in slot_str for x in ["ratio", "percentage", "fraction", "percent"]):
        return "ratio_fraction"
    if any(x in slot_str for x in ["fixed", "penalty", "cost"]):
        return "fixed_cost_penalty"
    if any(x in slot_str for x in ["time", "hour", "day", "week", "wage"]):
        return "time_resource"
    if any(x in slot_str for x in ["num", "count", "number", "quantity"]):
        return "item_count"
    return ""


def _query_family_hints(query: str) -> set[str]:
    """Infer likely optimization family from query text."""
    q = query.lower()
    hints: set[str] = set()
    for family, keywords in FAMILY_KEYWORDS.items():
        if family == "generic_lp":
            continue
        if any(kw in q for kw in keywords):
            hints.add(family)
    if not hints:
        hints.add("generic_lp")
    return hints


def _query_subgroup_hints(query: str) -> set[str]:
    """Infer subgroup signals from query."""
    q = query.lower()
    hints: set[str] = set()
    if any(w in q for w in ["budget", "total", "investment", "allocate"]):
        hints.add("total_budget_per_unit_profit")
    if any(w in q for w in ["capacity", "demand", "available", "required", "requirement"]):
        hints.add("capacity_demand")
    if any(w in q for w in ["minimum", "maximum", "at least", "at most", "no less", "no more"]):
        hints.add("min_max_bounds")
    if any(w in q for w in ["ratio", "percentage", "fraction", "percent", "share"]):
        hints.add("ratio_fraction")
    if any(w in q for w in ["fixed", "penalty", "cost"]):
        hints.add("fixed_cost_penalty")
    if any(w in q for w in ["hour", "day", "week", "wage", "time"]):
        hints.add("time_resource")
    if any(w in q for w in ["number of", "count", "quantity", "each type"]):
        hints.add("item_count")
    return hints


def _extract_query_acceptance_features(query: str, variant: str) -> dict[str, Any]:
    """Extract optimization-aware evidence from query for acceptance scoring."""
    toks = query.split()
    q_lower = query.lower()
    tokens_lower = [t.lower().strip(".,;:()[]{}") for t in toks]
    token_set = set(tokens_lower)
    num_toks = _extract_num_tokens(query, variant)
    # Type counts
    type_counts = {"percent": 0, "currency": 0, "int": 0, "float": 0, "unknown": 0}
    for nt in num_toks:
        type_counts[nt.kind] = type_counts.get(nt.kind, 0) + 1
    # Role evidence (cue words present)
    role_evidence = {}
    for role, words in [
        ("budget", ["budget", "total budget", "investment", "available amount"]),
        ("cost", ["cost", "expense", "spend", "price"]),
        ("profit", ["profit", "revenue", "return", "gain"]),
        ("demand", ["demand", "requirement", "required", "needed"]),
        ("capacity", ["capacity", "limit", "maximum", "available"]),
        ("min_max", ["minimum", "maximum", "at least", "at most", "lower", "upper"]),
        ("ratio", ["ratio", "percentage", "fraction", "share", "percent"]),
        ("fixed_penalty", ["fixed", "penalty"]),
        ("time", ["time", "hour", "day", "week", "wage"]),
        ("quantity", ["quantity", "count", "number", "amount"]),
    ]:
        role_evidence[role] = 1 if any(w in q_lower for w in words) else 0
    # Operator evidence
    operator_evidence = {
        "min_like": 1 if (token_set & OPERATOR_MIN_WORDS) else 0,
        "max_like": 1 if (token_set & OPERATOR_MAX_WORDS) else 0,
        "per_unit": 1 if any(p in q_lower for p in ["per ", "each ", "per unit", "for each"]) else 0,
        "total_like": 1 if any(w in token_set for w in ["total", "available", "overall"]) else 0,
    }
    # Structural (objective/constraint/resource language)
    objective_words = {"maximize", "minimize", "maximise", "minimise", "objective", "maximize total", "minimize total"}
    constraint_words = {"subject", "constraint", "must not exceed", "at least", "at most", "cannot exceed"}
    resource_words = {"budget", "capacity", "available", "limit", "resource"}
    structural = {
        "objective_like": 1 if any(w in q_lower for w in objective_words) else 0,
        "constraint_like": 1 if any(w in q_lower for w in constraint_words) else 0,
        "resource_like": 1 if any(w in q_lower for w in resource_words) else 0,
    }
    return {
        "type_counts": type_counts,
        "role_evidence": role_evidence,
        "operator_evidence": operator_evidence,
        "structural": structural,
        "n_numeric": len(num_toks),
        "cue_words": token_set & CUE_WORDS,
    }


def _get_expected_scalar_for_schema(schema_id: str, gold_by_id: dict[str, dict]) -> list[str]:
    """Return list of scalar parameter names for a schema (for acceptance profile)."""
    gold = gold_by_id.get(schema_id) or {}
    pred_params = gold.get("parameters") or {}
    pred_info = gold.get("problem_info") or {}
    expected_params: list[str] = []
    if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
        expected_params = list(pred_info["parameters"].keys())
    elif isinstance(pred_params, dict):
        expected_params = list(pred_params.keys())
    gold_params = gold.get("parameters") or {}
    scalar_keys = [p for p in expected_params if _is_scalar(gold_params.get(p))]
    return scalar_keys


def _build_schema_acceptance_profile(
    schema_id: str,
    gold_entry: dict,
    catalog_entry: dict,
    gold_by_id: dict[str, dict],
) -> dict[str, Any]:
    """Build acceptance profile for a schema: what evidence it expects."""
    slot_names = _get_expected_scalar_for_schema(schema_id, gold_by_id)
    schema_text = (catalog_entry.get("description") or catalog_entry.get("name") or "").lower()
    # Slot/type expectations
    type_expectations = {"percent": 0, "currency": 0, "int": 0, "float": 0}
    expects_percent = False
    expects_currency = False
    expects_count = False
    for p in slot_names:
        et = _expected_type(p)
        if et == "percent":
            type_expectations["percent"] += 1
            expects_percent = True
        elif et == "currency":
            type_expectations["currency"] += 1
            expects_currency = True
        elif et == "int":
            type_expectations["int"] += 1
            expects_count = True
        else:
            type_expectations["float"] += 1
    # Role expectations from slot names
    slot_str = " ".join(slot_names).lower()
    role_expectations = {
        "budget": any(x in slot_str for x in ["budget", "total", "investment"]),
        "demand": any(x in slot_str for x in ["demand", "required", "requirement"]),
        "capacity": any(x in slot_str for x in ["capacity", "available", "limit"]),
        "objective_coeff": any(x in slot_str for x in ["profit", "cost", "revenue", "per", "wage"]),
        "ratio": any(x in slot_str for x in ["ratio", "percent", "fraction", "percentage"]),
        "fixed_penalty": any(x in slot_str for x in ["fixed", "penalty"]),
        "time": any(x in slot_str for x in ["time", "wage", "hour", "day"]),
    }
    # Structural
    structural = {
        "objective_like": "maximize" in schema_text or "minimize" in schema_text,
        "bound_heavy": sum(1 for s in slot_names if "min" in s.lower() or "max" in s.lower()) >= 2,
        "total_budget_like": "budget" in schema_text or "total" in schema_text,
    }
    family = _schema_family(schema_id, schema_text, slot_names)
    subgroup = _schema_subgroup(schema_id, slot_names)
    return {
        "schema_id": schema_id,
        "slot_names": slot_names,
        "type_expectations": type_expectations,
        "expects_percent": expects_percent,
        "expects_currency": expects_currency,
        "expects_count": expects_count,
        "role_expectations": role_expectations,
        "structural": structural,
        "family": family,
        "subgroup": subgroup,
    }


def _acceptance_score(
    query_features: dict[str, Any],
    schema_profile: dict[str, Any],
    query_family_hints: set[str],
    query_subgroup_hints: set[str],
) -> tuple[float, dict[str, Any]]:
    """Compute additive acceptance score and debug breakdown."""
    w = ACCEPTANCE_RERANK_WEIGHTS
    score = 0.0
    debug: dict[str, Any] = {}
    q_types = query_features.get("type_counts") or {}
    s_types = schema_profile.get("type_expectations") or {}
    # Type coverage
    for kind in ("percent", "currency", "int", "float"):
        q_has = (q_types.get(kind) or 0) > 0
        s_expects = (s_types.get(kind) or 0) > 0
        if s_expects and q_has:
            score += w["type_coverage_bonus"]
            debug[f"type_{kind}_ok"] = True
        elif s_expects and not q_has:
            score += w["type_missing_penalty"]
            debug[f"type_{kind}_missing"] = True
    if schema_profile.get("expects_percent") and not (q_types.get("percent") or 0):
        score += w["type_missing_penalty"]
    if schema_profile.get("expects_currency") and not (q_types.get("currency") or 0):
        score += w["type_missing_penalty"]
    # Role coverage
    q_roles = query_features.get("role_evidence") or {}
    s_roles = schema_profile.get("role_expectations") or {}
    for role in s_roles:
        if s_roles.get(role) and q_roles.get(role):
            score += w["role_coverage_bonus"]
            debug[f"role_{role}"] = True
    # Operator compatibility
    q_op = query_features.get("operator_evidence") or {}
    if q_op.get("min_like") and ("min" in str(schema_profile.get("slot_names", [])).lower() or "minimum" in str(schema_profile.get("slot_names", [])).lower()):
        score += w["operator_compat_bonus"]
        debug["op_min"] = True
    if q_op.get("max_like") and ("max" in str(schema_profile.get("slot_names", [])).lower() or "maximum" in str(schema_profile.get("slot_names", [])).lower()):
        score += w["operator_compat_bonus"]
        debug["op_max"] = True
    # Fillability (approximate: we have n_numeric and n slots)
    n_slots = len(schema_profile.get("slot_names") or [])
    n_num = query_features.get("n_numeric") or 0
    if n_slots > 0 and n_num >= n_slots * 0.5:
        score += w["fillability_bonus"]
        debug["fillability_ok"] = True
    elif n_slots > 0 and n_num == 0:
        score += w["missing_critical_penalty"]
        debug["fillability_poor"] = True
    # Family/subgroup consistency
    s_family = schema_profile.get("family") or "generic_lp"
    s_subgroup = schema_profile.get("subgroup") or ""
    if s_family in query_family_hints:
        score += w["family_consistency_bonus"]
        debug["family_match"] = True
    elif query_family_hints and "generic_lp" not in query_family_hints and s_family not in query_family_hints:
        score += w["family_mismatch_penalty"]
        debug["family_mismatch"] = True
    if s_subgroup and s_subgroup in query_subgroup_hints:
        score += w["family_consistency_bonus"] * 0.5
        debug["subgroup_match"] = True
    # Weak fit: only lexical support
    if score <= 0 and n_slots > 0:
        score += w["weak_fit_penalty"]
        debug["weak_fit"] = True
    debug["total_acceptance"] = score
    return score, debug


def make_rerank_rank_fn(
    base_rank_fn,
    gold_by_id: dict[str, dict],
    catalog: list[dict],
    k_retrieval: int = 10,
    use_hierarchy: bool = False,
    variant: str = "orig",
) -> Any:
    """Return a rank_fn(query, top_k) that gets top-k from base, reranks by acceptance, returns top_k."""

    id_to_catalog: dict[str, dict] = {p.get("id"): p for p in catalog if p.get("id")}
    profile_cache: dict[str, dict] = {}

    def get_profile(schema_id: str) -> dict[str, Any]:
        if schema_id in profile_cache:
            return profile_cache[schema_id]
        gold_entry = gold_by_id.get(schema_id) or {}
        catalog_entry = id_to_catalog.get(schema_id) or {"id": schema_id, "name": schema_id, "description": ""}
        profile = _build_schema_acceptance_profile(schema_id, gold_entry, catalog_entry, gold_by_id)
        profile_cache[schema_id] = profile
        return profile

    def rank_fn(query: str, top_k: int) -> list[tuple[str, float]]:
        # Stage A: get top-k candidates from base retriever
        candidates = base_rank_fn(query, top_k=k_retrieval)
        if not candidates:
            return []
        query_features = _extract_query_acceptance_features(query, variant)
        q_family = _query_family_hints(query)
        q_subgroup = _query_subgroup_hints(query)
        # Normalize retrieval scores to [0,1] for combination
        raw_scores = [s for _, s in candidates]
        min_s, max_s = min(raw_scores), max(raw_scores)
        norm = (max_s - min_s) or 1.0
        scored: list[tuple[str, float, float, dict]] = []
        for pid, ret_score in candidates:
            profile = get_profile(pid)
            accept_score, debug = _acceptance_score(
                query_features, profile, q_family, q_subgroup
            )
            norm_ret = (ret_score - min_s) / norm if norm else 0.0
            # Combine: retrieval weight 0.5, acceptance weight 0.5, optional family bonus already in accept_score
            final = 0.5 * norm_ret + 0.5 * max(0.0, (accept_score + 5.0) / 10.0)  # shift accept to non-negative scale
            if use_hierarchy and profile.get("family") not in q_family and "generic_lp" not in q_family:
                final -= 0.2  # extra penalty for hierarchy mismatch
            scored.append((pid, final, accept_score, debug))
        scored.sort(key=lambda x: -x[1])
        return [(pid, score) for pid, score, _, _ in scored[:top_k]]

    return rank_fn


def run_setting(
    variant: str,
    baseline_name: str,
    eval_items: list[dict],
    gold_by_id: dict[str, dict],
    rank_fn,
    doc_ids: list[str],
    random_control: bool,
    assignment_mode: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_path = out_dir / f"nlp4lp_downstream_per_query_{variant}_{baseline_name}.csv"
    json_path = out_dir / f"nlp4lp_downstream_{variant}_{baseline_name}.json"
    summary_path = out_dir / "nlp4lp_downstream_summary.csv"

    def run_one(label: str, mode: str) -> tuple[list[dict], dict, dict]:
        rows: list[dict] = []
        hit_flags: list[int] = []
        cov_vals: list[float] = []
        type_vals: list[float] = []
        exact5_vals: list[float] = []
        exact20_vals: list[float] = []
        inst_ready_flags: list[int] = []
        cov_hits: list[float] = []
        cov_miss: list[float] = []
        type_hits: list[float] = []
        type_miss: list[float] = []
        ko_all: list[float] = []
        ko_hits: list[float] = []
        ko_miss: list[float] = []

        type_names = ["percent", "integer", "currency", "float"]
        type_expected_total = {t: 0 for t in type_names}
        type_filled_total = {t: 0 for t in type_names}
        type_correct_total = {t: 0 for t in type_names}
        type_exact5_num = {t: 0 for t in type_names}
        type_exact5_den = {t: 0 for t in type_names}
        type_exact20_num = {t: 0 for t in type_names}
        type_exact20_den = {t: 0 for t in type_names}

        for ex in eval_items:
            qid = ex["query_id"]
            query = ex["query"]
            gold_id = ex["relevant_doc_id"]
            if mode == "oracle":
                pred_id = gold_id
            elif mode == "random":
                rng = random.Random(_md5_seed(qid))
                pred_id = doc_ids[rng.randrange(len(doc_ids))] if doc_ids else ""
            else:  # retrieval
                ranked = rank_fn(query, top_k=1)
                pred_id = ranked[0][0] if ranked else ""
            schema_hit = 1 if pred_id == gold_id else 0

            # Gold problem (for evaluation) and predicted problem (for schema)
            gold = gold_by_id.get(gold_id) or {}
            gold_params = gold.get("parameters") or {}
            pred = gold_by_id.get(pred_id) or {}
            pred_params = pred.get("parameters") or {}
            pred_info = pred.get("problem_info") or {}

            expected_params: list[str] = []
            if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
                expected_params = list(pred_info["parameters"].keys())
            elif isinstance(pred_params, dict):
                expected_params = list(pred_params.keys())

            # scalar keys based on gold values
            gold_scalar_keys = {p for p, v in (gold_params or {}).items() if _is_scalar(v)}

            def _bucket_type(pname: str) -> str:
                et = _expected_type(pname)
                if et == "percent":
                    return "percent"
                if et == "int":
                    return "integer"
                if et == "currency":
                    return "currency"
                return "float"

            # count expected scalar params per type from gold schema
            for p in gold_scalar_keys:
                t = _bucket_type(p)
                type_expected_total[t] += 1

            pred_scalar_keys = {
                p for p in expected_params if _is_scalar(gold_params.get(p))
            } if isinstance(gold_params, dict) else set()
            # expected scalar params (for filling) are intersection of predicted keys with scalar gold
            expected_scalar = list(pred_scalar_keys)
            n_expected_scalar = len(expected_scalar)
            # schema key overlap (relative to gold scalar keys)
            if gold_scalar_keys:
                key_overlap = len(pred_scalar_keys & gold_scalar_keys) / float(len(gold_scalar_keys))
            else:
                key_overlap = 0.0

            num_toks = _extract_num_tokens(query, variant)
            candidates = list(num_toks)

            filled = {}
            type_matches = 0
            n_filled = 0
            comparable_errs = []

            # per-query per-type counts (optional, used for per-query CSV enrichment if desired later)
            type_expected_q = {t: 0 for t in type_names}
            type_filled_q = {t: 0 for t in type_names}
            type_correct_q = {t: 0 for t in type_names}

            for p in gold_scalar_keys:
                t = _bucket_type(p)
                type_expected_q[t] += 1

            if assignment_mode == "optimization_role_relation_repair" and expected_scalar:
                # Relation-aware + incremental admissible (RAT-SQL + PICARD inspired).
                filled_values, filled_mentions, _ = _run_optimization_role_relation_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "optimization_role_anchor_linking" and expected_scalar:
                filled_values, filled_mentions, _ = _run_optimization_role_anchor_linking(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "optimization_role_bottomup_beam_repair" and expected_scalar:
                filled_values, filled_mentions, _ = _run_optimization_role_bottomup_beam_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "optimization_role_entity_semantic_beam_repair" and expected_scalar:
                filled_values, filled_mentions, _ = _run_optimization_role_entity_semantic_beam_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "optimization_role_repair" and expected_scalar:
                # Optimization-role-aware assignment.
                filled_values, filled_mentions, _ = _run_optimization_role_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "semantic_ir_repair" and expected_scalar:
                # Semantic IR + validation-and-repair assignment.
                filled_values, filled_mentions, _ = _run_semantic_ir_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "global_consistency_grounding" and expected_scalar:
                # Global Consistency Grounding: beam-search assignment with global penalties.
                filled_values, filled_mentions, _diag = _run_global_consistency_grounding(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "max_weight_matching" and expected_scalar:
                # Maximum-weight bipartite matching (Hungarian algorithm / DP fallback).
                filled_values, filled_mentions, _diag = _run_max_weight_matching_grounding(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode in (
                "global_compat_local",
                "global_compat_pairwise",
                "global_compat_full",
            ) and expected_scalar:
                # Global Compatibility Grounding (GCGP): beam search with pairwise terms.
                _gcgp_ablation_map = {
                    "global_compat_local": "local",
                    "global_compat_pairwise": "pairwise",
                    "global_compat_full": "full",
                }
                _gcgp_ablation = _gcgp_ablation_map[assignment_mode]
                filled_values, filled_mentions, _diag = _run_global_compatibility_grounding(
                    query, variant, expected_scalar, ablation_mode=_gcgp_ablation
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode in (
                "relation_aware_basic",
                "relation_aware_ops",
                "relation_aware_semantic",
                "relation_aware_full",
            ) and expected_scalar:
                # Relation-aware greedy grounding.
                _ral_ablation_map = {
                    "relation_aware_basic": "basic",
                    "relation_aware_ops": "ops",
                    "relation_aware_semantic": "semantic",
                    "relation_aware_full": "full",
                }
                _ral_ablation = _ral_ablation_map[assignment_mode]
                from tools.relation_aware_linking import run_relation_aware_grounding
                filled_values, filled_mentions, _diag = run_relation_aware_grounding(
                    query, variant, expected_scalar, ablation_mode=_ral_ablation
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode in (
                "ambiguity_candidate_greedy",
                "ambiguity_aware_beam",
                "ambiguity_aware_abstain",
                "ambiguity_aware_full",
            ) and expected_scalar:
                # Ambiguity-aware grounding (candidate-set + competition reasoning).
                _aag_ablation_map = {
                    "ambiguity_candidate_greedy": "candidate_greedy",
                    "ambiguity_aware_beam": "ambiguity_beam",
                    "ambiguity_aware_abstain": "ambiguity_abstain",
                    "ambiguity_aware_full": "ambiguity_full",
                }
                _aag_mode = _aag_ablation_map[assignment_mode]
                from tools.ambiguity_aware_grounding import run_ambiguity_aware_grounding
                filled_values, filled_mentions, _diag = run_ambiguity_aware_grounding(
                    query, variant, expected_scalar, ablation_mode=_aag_mode
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "constrained" and expected_scalar:
                # Global constrained assignment over mention-slot pairs.
                mention_records = _extract_num_mentions(query, variant)
                slot_records = _build_slot_records(expected_scalar)
                constrained_assignments, _debug = _constrained_assignment(
                    mention_records, slot_records
                )
                # Apply assignments.
                for p in expected_scalar:
                    sr_type = _bucket_type(p)
                    mrec = constrained_assignments.get(p)
                    if mrec is None:
                        continue
                    tok = mrec.tok
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = tok.value if tok.value is not None else tok.raw
                    btype = sr_type
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            else:
                # Original greedy typed / untyped assignment.
                for p in expected_scalar:
                    et = _expected_type(p)
                    if assignment_mode == "untyped":
                        idx, tok = (0, candidates[0]) if candidates else (None, None)
                    else:
                        idx, tok = _choose_token(et, candidates)
                    if tok is None:
                        continue
                    # remove used token
                    if idx is not None and 0 <= idx < len(candidates):
                        candidates.pop(idx)
                    n_filled += 1
                    filled[p] = tok.value if tok.value is not None else tok.raw
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    if _is_type_match(et, tok.kind):
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    # error only if numeric and schema hit
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1

            param_coverage = (n_filled / max(1, n_expected_scalar)) if n_expected_scalar else 0.0
            type_match = (type_matches / max(1, n_filled)) if n_filled else 0.0
            ko_all.append(key_overlap)

            exact5 = ""
            exact20 = ""
            if schema_hit:
                if comparable_errs:
                    exact5 = sum(1 for e in comparable_errs if e <= 0.05) / len(comparable_errs)
                    exact20 = sum(1 for e in comparable_errs if e <= 0.20) / len(comparable_errs)
                else:
                    exact5 = ""
                    exact20 = ""

            rows.append(
                {
                    "query_id": qid,
                    "variant": variant,
                    "baseline": label,
                    "predicted_doc_id": pred_id,
                    "gold_doc_id": gold_id,
                    "schema_hit": schema_hit,
                    "n_expected_scalar": n_expected_scalar,
                    "n_filled": n_filled,
                    "param_coverage": param_coverage,
                    "type_match": type_match,
                    "exact5": exact5,
                    "exact20": exact20,
                    "key_overlap": key_overlap,
                }
            )

            hit_flags.append(schema_hit)
            cov_vals.append(param_coverage)
            type_vals.append(type_match)
            if schema_hit:
                cov_hits.append(param_coverage)
                type_hits.append(type_match)
                ko_hits.append(key_overlap)
            else:
                cov_miss.append(param_coverage)
                type_miss.append(type_match)
                ko_miss.append(key_overlap)
            if isinstance(exact5, float):
                exact5_vals.append(exact5)
            if isinstance(exact20, float):
                exact20_vals.append(exact20)
            inst_ready_flags.append(1 if (param_coverage >= 0.8 and type_match >= 0.8) else 0)

        n = len(rows)
        def _mean(xs: list[float]) -> float | str:
            return (sum(xs) / len(xs)) if xs else ""

        agg = {
            "variant": variant,
            "baseline": label,
            "schema_R1": sum(hit_flags) / n if n else 0.0,
            "param_coverage": sum(cov_vals) / n if n else 0.0,
            "type_match": sum(type_vals) / n if n else 0.0,
            "exact5_on_hits": (sum(exact5_vals) / len(exact5_vals)) if exact5_vals else "",
            "exact20_on_hits": (sum(exact20_vals) / len(exact20_vals)) if exact20_vals else "",
            "param_coverage_hits": _mean(cov_hits),
            "param_coverage_miss": _mean(cov_miss),
            "type_match_hits": _mean(type_hits),
            "type_match_miss": _mean(type_miss),
            "key_overlap": _mean(ko_all),
            "key_overlap_hits": _mean(ko_hits),
            "key_overlap_miss": _mean(ko_miss),
            "instantiation_ready": sum(inst_ready_flags) / n if n else 0.0,
            "n": n,
        }
        # per-type aggregate summary for this (variant, baseline)
        types_agg: dict[str, dict] = {}
        n_queries = len(rows)
        for t in type_names:
            n_exp_t = type_expected_total[t]
            n_fill_t = type_filled_total[t]
            cov_t = (n_fill_t / max(1, n_exp_t)) if n_exp_t else 0.0
            tm_t = (
                (type_correct_total[t] / n_fill_t) if n_fill_t else ""
            )
            e5 = (
                (type_exact5_num[t] / type_exact5_den[t]) if type_exact5_den[t] else ""
            )
            e20 = (
                (type_exact20_num[t] / type_exact20_den[t]) if type_exact20_den[t] else ""
            )
            types_agg[t] = {
                "variant": variant,
                "baseline": label,
                "param_type": t,
                "n_expected": n_exp_t,
                "n_filled": n_fill_t,
                "param_coverage": cov_t,
                "type_match": tm_t,
                "exact5_on_hits": e5,
                "exact20_on_hits": e20,
                "n_queries": n_queries,
            }

        return rows, agg, types_agg

    rows_main, agg_main, types_main = run_one(baseline_name, "oracle" if baseline_name.startswith("oracle") else "retrieval")

    cols = [
        "query_id",
        "variant",
        "baseline",
        "predicted_doc_id",
        "gold_doc_id",
        "schema_hit",
        "n_expected_scalar",
        "n_filled",
        "param_coverage",
        "type_match",
        "exact5",
        "exact20",
        "key_overlap",
    ]
    with open(per_query_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_main)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"variant": variant, "baseline": baseline_name, "k": 1, "random_control": random_control},
                "aggregate": agg_main,
            },
            f,
            indent=2,
        )

    _upsert_summary_row(summary_path, agg_main)

    # Update per-type summary
    types_summary_path = out_dir / "nlp4lp_downstream_types_summary.csv"
    _upsert_types_rows(types_summary_path, types_main)

    if random_control:
        rand_label = "random_untyped" if baseline_name.endswith("_untyped") else "random"
        _, agg_rand, types_rand = run_one(rand_label, "random")
        _upsert_summary_row(summary_path, agg_rand)
        _upsert_types_rows(types_summary_path, types_rand)


def run_single_setting(
    variant: str,
    baseline_arg: str,
    assignment_mode: str,
    out_dir: Path,
    eval_path: Path | None = None,
    catalog_path: Path | None = None,
    acceptance_k: int = 10,
    eval_items: list[dict] | None = None,
    gold_by_id: dict[str, dict] | None = None,
    catalog: list[dict] | None = None,
    doc_ids: list[str] | None = None,
    random_control: bool = False,
) -> bool:
    """Run one (variant, baseline, assignment_mode). If eval_items/gold_by_id/catalog/doc_ids
    are provided, use them (avoids reload; safe for low-resource). Returns True on success."""
    eval_path = eval_path or ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
    catalog_path = catalog_path or ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"
    if eval_items is None:
        eval_items = _load_eval(Path(eval_path))
    if not eval_items:
        return False
    if gold_by_id is None:
        gold_by_id = _load_hf_gold(split="test")
    if catalog is None or doc_ids is None:
        catalog, _ = _load_catalog_as_problems(catalog_path)
        doc_ids = [p["id"] for p in catalog if p.get("id")]

    rank_fn = None
    effective_baseline = baseline_arg
    if baseline_arg == "oracle":
        rank_fn = None
    elif baseline_arg in ("tfidf_acceptance_rerank", "tfidf_hierarchical_acceptance_rerank"):
        from retrieval.baselines import get_baseline
        base = get_baseline("tfidf")
        base.fit(catalog)
        rank_fn = make_rerank_rank_fn(
            base.rank,
            gold_by_id,
            catalog,
            k_retrieval=acceptance_k,
            use_hierarchy=(baseline_arg == "tfidf_hierarchical_acceptance_rerank"),
            variant=variant,
        )
    elif baseline_arg in ("bm25_acceptance_rerank", "bm25_hierarchical_acceptance_rerank"):
        from retrieval.baselines import get_baseline
        base = get_baseline("bm25")
        base.fit(catalog)
        rank_fn = make_rerank_rank_fn(
            base.rank,
            gold_by_id,
            catalog,
            k_retrieval=acceptance_k,
            use_hierarchy=(baseline_arg == "bm25_hierarchical_acceptance_rerank"),
            variant=variant,
        )
    else:
        if baseline_arg not in ("bm25", "tfidf", "lsa"):
            return False
        from retrieval.baselines import get_baseline
        baseline = get_baseline(baseline_arg)
        baseline.fit(catalog)
        rank_fn = baseline.rank
    if assignment_mode == "untyped":
        effective_baseline = f"{baseline_arg}_untyped"
    elif assignment_mode == "constrained":
        effective_baseline = f"{baseline_arg}_constrained"
    elif assignment_mode == "semantic_ir_repair":
        effective_baseline = f"{baseline_arg}_semantic_ir_repair"
    elif assignment_mode == "optimization_role_repair":
        effective_baseline = f"{baseline_arg}_optimization_role_repair"
    elif assignment_mode == "optimization_role_relation_repair":
        effective_baseline = f"{baseline_arg}_optimization_role_relation_repair"
    elif assignment_mode == "optimization_role_anchor_linking":
        effective_baseline = f"{baseline_arg}_optimization_role_anchor_linking"
    elif assignment_mode == "optimization_role_bottomup_beam_repair":
        effective_baseline = f"{baseline_arg}_optimization_role_bottomup_beam_repair"
    elif assignment_mode == "optimization_role_entity_semantic_beam_repair":
        effective_baseline = f"{baseline_arg}_optimization_role_entity_semantic_beam_repair"
    elif assignment_mode == "global_consistency_grounding":
        effective_baseline = f"{baseline_arg}_global_consistency_grounding"
    elif assignment_mode == "max_weight_matching":
        effective_baseline = f"{baseline_arg}_max_weight_matching"
    elif assignment_mode in ("global_compat_local", "global_compat_pairwise", "global_compat_full"):
        effective_baseline = f"{baseline_arg}_{assignment_mode}"
    elif assignment_mode in (
        "relation_aware_basic",
        "relation_aware_ops",
        "relation_aware_semantic",
        "relation_aware_full",
    ):
        effective_baseline = f"{baseline_arg}_{assignment_mode}"
    elif assignment_mode in (
        "ambiguity_candidate_greedy",
        "ambiguity_aware_beam",
        "ambiguity_aware_abstain",
        "ambiguity_aware_full",
    ):
        effective_baseline = f"{baseline_arg}_{assignment_mode}"

    run_setting(
        variant=variant,
        baseline_name=effective_baseline,
        eval_items=eval_items,
        gold_by_id=gold_by_id,
        rank_fn=rank_fn,
        doc_ids=doc_ids,
        random_control=random_control,
        assignment_mode=assignment_mode,
        out_dir=out_dir,
    )
    return True


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="NLP4LP downstream utility demo (retrieval -> parameter instantiation)")
    p.add_argument("--variant", type=str, default="orig", choices=("orig", "noisy", "short"))
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    p.add_argument("--eval", type=Path, default=None)
    p.add_argument(
        "--baseline",
        type=str,
        default="tfidf",
        help="One of: bm25, tfidf, lsa, oracle; or tfidf_acceptance_rerank, tfidf_hierarchical_acceptance_rerank, bm25_acceptance_rerank, bm25_hierarchical_acceptance_rerank",
    )
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--acceptance-k", type=int, default=10, help="Top-k retrieval candidates for acceptance reranking (default 10)")
    p.add_argument("--random-control", action="store_true")
    p.add_argument(
        "--assignment-mode",
        type=str,
        default="typed",
        choices=(
            "typed", "untyped", "constrained", "semantic_ir_repair",
            "optimization_role_repair", "optimization_role_relation_repair",
            "global_consistency_grounding",
            "max_weight_matching",
            "global_compat_local", "global_compat_pairwise", "global_compat_full",
            "relation_aware_basic", "relation_aware_ops",
            "relation_aware_semantic", "relation_aware_full",
            "ambiguity_candidate_greedy", "ambiguity_aware_beam",
            "ambiguity_aware_abstain", "ambiguity_aware_full",
            # Experimental/archived (not in default focused eval; use run_nlp4lp_focused_eval.py --experimental):
            "optimization_role_anchor_linking", "optimization_role_bottomup_beam_repair",
            "optimization_role_entity_semantic_beam_repair",
        ),
    )
    args = p.parse_args()

    if args.k != 1:
        raise SystemExit("This demo currently supports --k 1 only (top-1 schema selection).")

    eval_path = args.eval or (ROOT / "data" / "processed" / f"nlp4lp_eval_{args.variant}.jsonl")
    eval_items = _load_eval(Path(eval_path))
    if not eval_items:
        raise SystemExit(f"No eval instances loaded from {eval_path}")

    # Gold schema/parameters from HF
    gold_by_id = _load_hf_gold(split="test")

    catalog, _id_to_text = _load_catalog_as_problems(args.catalog)
    doc_ids = [p["id"] for p in catalog if p.get("id")]

    rank_fn = None
    effective_baseline = args.baseline
    if args.baseline == "oracle":
        rank_fn = None
    elif args.baseline in ("tfidf_acceptance_rerank", "tfidf_hierarchical_acceptance_rerank"):
        from retrieval.baselines import get_baseline
        base = get_baseline("tfidf")
        base.fit(catalog)
        rank_fn = make_rerank_rank_fn(
            base.rank,
            gold_by_id,
            catalog,
            k_retrieval=args.acceptance_k,
            use_hierarchy=(args.baseline == "tfidf_hierarchical_acceptance_rerank"),
            variant=args.variant,
        )
    elif args.baseline in ("bm25_acceptance_rerank", "bm25_hierarchical_acceptance_rerank"):
        from retrieval.baselines import get_baseline
        base = get_baseline("bm25")
        base.fit(catalog)
        rank_fn = make_rerank_rank_fn(
            base.rank,
            gold_by_id,
            catalog,
            k_retrieval=args.acceptance_k,
            use_hierarchy=(args.baseline == "bm25_hierarchical_acceptance_rerank"),
            variant=args.variant,
        )
    else:
        if args.baseline not in ("bm25", "tfidf", "lsa"):
            raise SystemExit(f"Unknown baseline: {args.baseline}. Use bm25, tfidf, lsa, oracle, or *_acceptance_rerank / *_hierarchical_acceptance_rerank.")
        from retrieval.baselines import get_baseline
        baseline = get_baseline(args.baseline)
        baseline.fit(catalog)
        rank_fn = baseline.rank
    if args.assignment_mode == "untyped":
        effective_baseline = f"{args.baseline}_untyped"
    elif args.assignment_mode == "constrained":
        effective_baseline = f"{args.baseline}_constrained"
    elif args.assignment_mode == "semantic_ir_repair":
        effective_baseline = f"{args.baseline}_semantic_ir_repair"
    elif args.assignment_mode == "optimization_role_repair":
        effective_baseline = f"{args.baseline}_optimization_role_repair"
    elif args.assignment_mode == "optimization_role_relation_repair":
        effective_baseline = f"{args.baseline}_optimization_role_relation_repair"
    elif args.assignment_mode == "optimization_role_anchor_linking":
        effective_baseline = f"{args.baseline}_optimization_role_anchor_linking"
    elif args.assignment_mode == "optimization_role_bottomup_beam_repair":
        effective_baseline = f"{args.baseline}_optimization_role_bottomup_beam_repair"
    elif args.assignment_mode == "optimization_role_entity_semantic_beam_repair":
        effective_baseline = f"{args.baseline}_optimization_role_entity_semantic_beam_repair"
    elif args.assignment_mode == "global_consistency_grounding":
        effective_baseline = f"{args.baseline}_global_consistency_grounding"
    elif args.assignment_mode == "max_weight_matching":
        effective_baseline = f"{args.baseline}_max_weight_matching"
    elif args.assignment_mode in ("global_compat_local", "global_compat_pairwise", "global_compat_full"):
        effective_baseline = f"{args.baseline}_{args.assignment_mode}"
    elif args.assignment_mode in (
        "relation_aware_basic",
        "relation_aware_ops",
        "relation_aware_semantic",
        "relation_aware_full",
    ):
        effective_baseline = f"{args.baseline}_{args.assignment_mode}"
    elif args.assignment_mode in (
        "ambiguity_candidate_greedy",
        "ambiguity_aware_beam",
        "ambiguity_aware_abstain",
        "ambiguity_aware_full",
    ):
        effective_baseline = f"{args.baseline}_{args.assignment_mode}"

    out_dir = ROOT / "results" / "paper"
    run_setting(
        variant=args.variant,
        baseline_name=effective_baseline,
        eval_items=eval_items,
        gold_by_id=gold_by_id,
        rank_fn=rank_fn,
        doc_ids=doc_ids,
        random_control=bool(args.random_control),
        assignment_mode=args.assignment_mode,
        out_dir=out_dir,
    )

    print(f"Wrote {out_dir / f'nlp4lp_downstream_{args.variant}_{effective_baseline}.json'}")
    print(f"Wrote {out_dir / f'nlp4lp_downstream_per_query_{args.variant}_{effective_baseline}.csv'}")
    print(f"Updated {out_dir / 'nlp4lp_downstream_summary.csv'}")


if __name__ == "__main__":
    main()

