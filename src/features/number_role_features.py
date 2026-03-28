"""Number-role feature extraction: operator cues, relevance labels, bound/range detection, quantity families."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

# --- Cue word lists ---

ADDITION_CUES = ["gained", "received", "bought", "added", "more", "another", "increase"]
SUBTRACTION_CUES = ["lost", "spent", "gave away", "sold", "used", "ate", "left", "remaining", "after", "decrease", "reduce"]
RATE_CUES = ["each", "every", "per", "for each", "daily", "weekly", "hourly"]
DIVISION_CUES = ["average", "total", "evenly", "divided"]
COMPARISON_CUES = ["more than", "less than", "difference", "fewer than", "greater than"]
CAPACITY_CUES = ["minimum", "at least", "enough", "max", "maximum"]
TARGET_CUES = ["maximize", "minimize", "optimal", "most", "least", "profit", "cost", "revenue"]
CONSTRAINT_CUES = ["must", "require", "cannot", "limit", "exactly", "constraint"]

LOWER_BOUND_PHRASES = [
    "at least", "no fewer than", "not fewer than", "no less than", "not less than",
    "greater than or equal", "minimum of", "minimum number of", "≥"
]
UPPER_BOUND_PHRASES = [
    "at most", "no more than", "not more than", "no greater than",
    "less than or equal", "maximum of", "maximum number of", "≤"
]

_WRITTEN_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "half": 0.5,
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "but", "or", "nor", "so", "yet", "if", "then", "than", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "we",
    "you", "i", "me", "him", "her", "us", "them", "what", "which", "who",
    "how", "when", "where", "why", "not", "no", "very", "just", "also",
}

_NUMBER_TOKEN_RE = re.compile(r'^-?\d+(?:[.,]\d+)?$')
_RANGE_BETWEEN_RE = re.compile(r'between\s+(\d[\d.,]*)\s+and\s+(\d[\d.,]*)', re.IGNORECASE)
_RANGE_FROM_TO_RE = re.compile(r'from\s+(\d[\d.,]*)\s+to\s+(\d[\d.,]*)', re.IGNORECASE)


@dataclass
class NumberMention:
    """A single numeric mention with enriched role cues."""
    surface: str
    value: float
    position: int
    nearby_operator_cues: list[str] = field(default_factory=list)
    nearby_target_cues: list[str] = field(default_factory=list)
    nearby_constraint_cues: list[str] = field(default_factory=list)
    nearby_quantity_words: list[str] = field(default_factory=list)
    local_direction_cue: str = "none"  # "min_like" | "max_like" | "rate_like" | "none"
    relevance_label: str = "role_unknown"
    bound_role: str = "none"  # "lower" | "upper" | "none"
    range_partner_detected: bool = False
    direction_sensitive_constraint: bool = False
    quantity_family_id: Optional[int] = None


def _parse_number_token(tok: str) -> Optional[float]:
    """Parse a token as a number, returning float or None."""
    tok_clean = tok.strip(".,!?;:")
    if _NUMBER_TOKEN_RE.match(tok_clean):
        try:
            return float(tok_clean.replace(",", ""))
        except ValueError:
            return None
    lower = tok_clean.lower()
    if lower in _WRITTEN_NUMBERS:
        return float(_WRITTEN_NUMBERS[lower])
    return None


def _get_window_text(tokens: list[str], pos: int, radius: int = 6) -> str:
    """Get the joined text of tokens in a ±radius window around pos."""
    start = max(0, pos - radius)
    end = min(len(tokens), pos + radius + 1)
    return " ".join(tokens[start:end])


def _extract_cues(window_text: str, cue_list: list[str]) -> list[str]:
    """Return cues from cue_list found in window_text."""
    found = []
    wt_lower = window_text.lower()
    for cue in cue_list:
        if cue.lower() in wt_lower:
            found.append(cue)
    return found


def _extract_quantity_words(tokens: list[str], pos: int, radius: int = 6) -> list[str]:
    """Extract non-stopword, non-number words from the window."""
    start = max(0, pos - radius)
    end = min(len(tokens), pos + radius + 1)
    words = []
    for i in range(start, end):
        if i == pos:
            continue
        tok = tokens[i].lower().strip(".,!?;:")
        if tok in _STOPWORDS:
            continue
        if _NUMBER_TOKEN_RE.match(tok):
            continue
        if len(tok) < 2:
            continue
        words.append(tok)
    return words


def extract_number_mentions(text: str) -> list[NumberMention]:
    """Extract all number mentions from text with cue annotations."""
    tokens = re.findall(r'\S+', text.lower())
    mentions: list[NumberMention] = []

    # Build set of range pairs to mark range_partner_detected
    range_pairs: set[float] = set()
    for match in _RANGE_BETWEEN_RE.finditer(text):
        try:
            v1 = float(match.group(1).replace(",", ""))
            v2 = float(match.group(2).replace(",", ""))
            range_pairs.add(v1)
            range_pairs.add(v2)
        except ValueError:
            pass
    for match in _RANGE_FROM_TO_RE.finditer(text):
        try:
            v1 = float(match.group(1).replace(",", ""))
            v2 = float(match.group(2).replace(",", ""))
            range_pairs.add(v1)
            range_pairs.add(v2)
        except ValueError:
            pass

    for pos, tok in enumerate(tokens):
        value = _parse_number_token(tok)
        if value is None:
            continue

        surface = tok.strip(".,!?;:")
        window_text = _get_window_text(tokens, pos)

        # Collect operator cues (all types)
        op_cues: list[str] = []
        op_cues.extend(_extract_cues(window_text, ADDITION_CUES))
        op_cues.extend(_extract_cues(window_text, SUBTRACTION_CUES))
        op_cues.extend(_extract_cues(window_text, RATE_CUES))
        op_cues.extend(_extract_cues(window_text, DIVISION_CUES))
        op_cues.extend(_extract_cues(window_text, COMPARISON_CUES))
        op_cues.extend(_extract_cues(window_text, CAPACITY_CUES))

        target_cues = _extract_cues(window_text, TARGET_CUES)
        constraint_cues = _extract_cues(window_text, CONSTRAINT_CUES)
        qty_words = _extract_quantity_words(tokens, pos)

        # Bound detection
        bound_role = "none"
        for phrase in LOWER_BOUND_PHRASES:
            if phrase.lower() in window_text.lower():
                bound_role = "lower"
                break
        if bound_role == "none":
            for phrase in UPPER_BOUND_PHRASES:
                if phrase.lower() in window_text.lower():
                    bound_role = "upper"
                    break

        # Direction cue
        if target_cues:
            if any(c in ("maximize", "most", "revenue") for c in target_cues):
                local_direction = "max_like"
            elif any(c in ("minimize", "least", "cost") for c in target_cues):
                local_direction = "min_like"
            else:
                local_direction = "none"
        elif any(c in op_cues for c in RATE_CUES):
            local_direction = "rate_like"
        else:
            local_direction = "none"

        # Range partner
        range_partner = value in range_pairs

        # Direction sensitive constraint
        direction_sensitive = bound_role != "none" and bool(target_cues)

        m = NumberMention(
            surface=surface,
            value=value,
            position=pos,
            nearby_operator_cues=op_cues,
            nearby_target_cues=target_cues,
            nearby_constraint_cues=constraint_cues,
            nearby_quantity_words=qty_words,
            local_direction_cue=local_direction,
            relevance_label="role_unknown",
            bound_role=bound_role,
            range_partner_detected=range_partner,
            direction_sensitive_constraint=direction_sensitive,
        )
        mentions.append(m)

    # Apply initial relevance labels
    for m in mentions:
        _assign_relevance(m)

    return mentions


def _assign_relevance(m: NumberMention) -> None:
    """Assign relevance_label based on cues."""
    has_target = bool(m.nearby_target_cues)
    has_constraint = bool(m.nearby_constraint_cues)
    has_rate = any(c in m.nearby_operator_cues for c in RATE_CUES)

    if has_target or has_constraint or has_rate:
        m.relevance_label = "role_required"
        return

    # Year-like
    if 1900 <= m.value <= 2099 and not has_target and not has_constraint:
        m.relevance_label = "role_irrelevant"
        return

    # Very large number
    if m.value >= 1_000_000_000:
        m.relevance_label = "role_irrelevant"
        return

    has_add_sub = any(
        c in m.nearby_operator_cues
        for c in ADDITION_CUES + SUBTRACTION_CUES
    )
    if has_add_sub:
        m.relevance_label = "role_optional"
        return

    m.relevance_label = "role_unknown"


def annotate_relevance(mentions: list[NumberMention], question_text: str) -> list[NumberMention]:
    """Re-annotate relevance labels on mentions using question context."""
    import copy
    result = [copy.copy(m) for m in mentions]
    tokens = re.findall(r'\S+', question_text.lower())

    for m in result:
        # Re-examine window in question_text (if mention position valid)
        if m.position < len(tokens):
            window_text = _get_window_text(tokens, m.position)
            # Re-check target and constraint cues from question context
            target_cues = _extract_cues(window_text, TARGET_CUES)
            constraint_cues = _extract_cues(window_text, CONSTRAINT_CUES)
            if target_cues and not m.nearby_target_cues:
                m.nearby_target_cues = target_cues
            if constraint_cues and not m.nearby_constraint_cues:
                m.nearby_constraint_cues = constraint_cues

        _assign_relevance(m)
    return result


def detect_quantity_families(mentions: list[NumberMention], text: str) -> list[NumberMention]:
    """Assign quantity_family_id to mentions that form bound pairs or share units."""
    import copy
    result = [copy.copy(m) for m in mentions]
    tokens = re.findall(r'\S+', text.lower())

    next_family_id = 0
    assigned: dict[int, int] = {}  # index → family_id

    # Strategy 1: lower+upper pair sharing a nearby noun within ±4 tokens
    lowers = [(i, m) for i, m in enumerate(result) if m.bound_role == "lower"]
    uppers = [(i, m) for i, m in enumerate(result) if m.bound_role == "upper"]

    def get_nearby_nouns(pos: int, radius: int = 4) -> set[str]:
        start = max(0, pos - radius)
        end = min(len(tokens), pos + radius + 1)
        nouns: set[str] = set()
        for tok in tokens[start:end]:
            t = tok.strip(".,!?;:")
            if t in _STOPWORDS or len(t) < 3:
                continue
            if _NUMBER_TOKEN_RE.match(t):
                continue
            nouns.add(t)
        return nouns

    paired: set[int] = set()
    for li, lm in lowers:
        best_ui = None
        best_shared: set[str] = set()
        for ui, um in uppers:
            if ui in paired:
                continue
            ln = get_nearby_nouns(lm.position)
            un = get_nearby_nouns(um.position)
            shared = ln & un
            if shared and len(shared) > len(best_shared):
                best_shared = shared
                best_ui = ui
        if best_ui is not None:
            fid = next_family_id
            next_family_id += 1
            assigned[li] = fid
            assigned[best_ui] = fid
            paired.add(best_ui)

    # Strategy 2: same unit word, within 10 tokens
    for i, mi in enumerate(result):
        for j, mj in enumerate(result):
            if j <= i:
                continue
            if i in assigned and j in assigned and assigned[i] == assigned[j]:
                continue
            # Check token distance
            if abs(mi.position - mj.position) > 10:
                continue
            # Check shared unit words
            wi = set(mi.nearby_quantity_words)
            wj = set(mj.nearby_quantity_words)
            shared_units = wi & wj
            if shared_units:
                if i in assigned:
                    fid = assigned[i]
                elif j in assigned:
                    fid = assigned[j]
                else:
                    fid = next_family_id
                    next_family_id += 1
                assigned[i] = fid
                assigned[j] = fid

    for idx, fid in assigned.items():
        result[idx].quantity_family_id = fid

    return result
