"""Text normalization and heuristic parsing helpers."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^A-Za-z0-9.+/-]+")


def normalize_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = _SPACE_RE.sub(" ", text.strip())
    return text


# BOL description cleaning -----------------------------------------------------

# Shipping/packaging units only -- these are quantities, never product specs.
# Embedded measurements like "15kg", "200KGS", "1100g", "32L" are NOT in this
# list and survive cleaning untouched.
_BOL_QTY_UNITS = (
    r"EA|PCS?|PIECES?|CTN|BOX(?:ES)?|PALLETS?|PALETS?|PALLET|PKG|PKGS|"
    r"PACKAGES?|UNITS?|BAGS?|BOTTLES?|DRUMS?|ROLLS?|CARTONS?|PACKS?|"
    r"SETS?|SHEETS?"
)
_BOL_QTY_RE = re.compile(
    rf"(?<![A-Za-z0-9])\d+\s*({_BOL_QTY_UNITS})\b",
    flags=re.IGNORECASE,
)
_BOL_PAREN_QTY_RE = re.compile(
    rf"\(\s*\d+\s*({_BOL_QTY_UNITS})\s*\)",
    flags=re.IGNORECASE,
)
_BOL_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")
_BOL_LETTER_CLUSTER_RE = re.compile(r"[A-Za-z]{4,}")


def _is_partno_only_segment(segment: str) -> bool:
    """True if the segment looks like a part number with no real prose."""
    stripped = segment.strip()
    if not stripped:
        return True
    if " " in stripped:
        return False
    if len(stripped) > 16:
        return False
    return _BOL_LETTER_CLUSTER_RE.search(stripped) is None


def _pick_best_segment(segments: List[str]) -> str:
    """Pick the most description-like segment from a `#&`-split BOL string."""
    cleaned = [seg.strip() for seg in segments if seg and seg.strip()]
    if not cleaned:
        return ""
    candidates: List[str] = []
    for seg in cleaned:
        if seg == ".":
            continue
        if len(seg) <= 3 and seg.isalpha():  # country/region code (VN, TH, JP)
            continue
        if _is_partno_only_segment(seg):
            continue
        candidates.append(seg)
    if candidates:
        return max(candidates, key=len)
    return max(cleaned, key=len)


def _strip_repeated_prefix(text: str, min_repeat: int = 24) -> str:
    """Drop a duplicated description: if the opening `min_repeat` chars
    of `text` reappear later, truncate at the start of the second copy.

    Robust to truncation in the second copy (BOL exports often pack two
    differently-truncated copies of the same description into one cell).
    """
    n = len(text)
    if n < 2 * min_repeat:
        return text
    head = text[:min_repeat]
    idx = text.find(head, min_repeat)
    if idx == -1:
        return text
    return text[:idx].rstrip(" -,;:")


def clean_bol_description(text: str) -> str:
    """Clean a BOL-sourced product description.

    Removes the BOL export `#&` separator, picks one product name when
    multiple are concatenated, strips shipping-quantity tokens, and
    collapses whitespace. Does NOT strip embedded measurements
    ("15kg", "99.999%", "32L").
    """
    if not text:
        return ""
    base = normalize_text(text)
    # Step 1: split on `#&` first, then any remaining `#`, and pick best.
    if "#&" in base:
        base = _pick_best_segment(base.split("#&"))
    if "#" in base:
        base = _pick_best_segment(base.split("#"))
    base = base.replace("#", " ")
    # Step 2: remove duplicated prefix (concatenated product names).
    base = _strip_repeated_prefix(normalize_text(base))
    # Step 3: strip shipping-quantity tokens, including parenthetical wrappers.
    base = _BOL_PAREN_QTY_RE.sub(" ", base)
    base = _BOL_QTY_RE.sub(" ", base)
    base = _BOL_EMPTY_PAREN_RE.sub(" ", base)
    return normalize_text(base)


def truncate_text(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip(" ,;:-")
    return clipped


def truncate_text_word_safe(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip(" ,;:-")
    if " " not in clipped:
        return clipped
    safe = clipped.rsplit(" ", 1)[0].rstrip(" ,;:-")
    return safe or clipped


def substantive_word_count(text: str) -> int:
    return len([token for token in tokenize(text) if len(token) > 2])


def tokenize(text: str) -> List[str]:
    cleaned = _NON_WORD_RE.sub(" ", normalize_text(text)).strip().lower()
    if not cleaned:
        return []
    return [token for token in cleaned.split(" ") if token]


def remove_phrases(text: str, phrases: Iterable[str]) -> str:
    result = normalize_text(text)
    for phrase in phrases:
        result = re.sub(re.escape(phrase), " ", result, flags=re.IGNORECASE)
    return normalize_text(result)


def extract_pattern_matches(text: str, patterns: Dict[str, str]) -> List[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
    for key, pattern in patterns.items():
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append({key: found.group(0).strip()})
    return matches


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = normalize_text(text).lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
