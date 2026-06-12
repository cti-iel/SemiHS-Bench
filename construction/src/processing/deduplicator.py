"""Record deduplication and quality filtering."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


AUTHORITY_RANK = {"CROSS": 3, "EBTI": 2, "catalog": 1}


# Masks tokens that look like part / model / SKU identifiers: alphanumeric
# runs of length >= 4 that contain at least one digit and at least one letter
# (or that are pure-digit runs >= 5 chars). Punctuation inside the token (-, /,
# .) is allowed. This isolates the prose from the SKU so two distinct
# products with the same boilerplate description are not collapsed.
_PARTNUM_RE = re.compile(
    r"\b(?=[A-Za-z0-9./-]*\d)(?=[A-Za-z0-9./-]*[A-Za-z])[A-Za-z0-9./-]{4,}\b"
    r"|\b\d{5,}\b"
)


def mask_part_numbers(text: str) -> str:
    """Replace part/model number tokens with a placeholder for dedup similarity.

    Used to compare descriptions ignoring SKU differences. ``MODEL: ABC123``
    and ``MODEL: XYZ456`` collapse to ``MODEL: <PN> <PN>``.
    """
    return _PARTNUM_RE.sub("<PN>", text)


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left.lower(), b=right.lower()).ratio()


def masked_similarity(left: str, right: str) -> float:
    """Similarity computed after masking part-number tokens in both strings."""
    return similarity(mask_part_numbers(left), mask_part_numbers(right))


_WORD_RE = re.compile(r"[A-Za-z0-9./'\-]+")


def _word_set(text: str) -> set:
    return {tok.lower() for tok in _WORD_RE.findall(text)}


def _diff_is_prose_only(left: str, right: str) -> bool:
    """True if every token that differs between the two strings is letter-only.

    A "letter-only" diff means the strings differ only in prose words like
    paraphrases ("supplied AS A cartridge" vs "supplied IN cartridge form").
    Any digit-bearing diff (model numbers, sizes, dimensions) means the
    products are functionally distinct.
    """
    diff = _word_set(left).symmetric_difference(_word_set(right))
    if not diff:
        return True
    return all(any(c.isalpha() for c in tok) and not any(c.isdigit() for c in tok)
               for tok in diff)


def is_near_duplicate(
    left: str,
    right: str,
    *,
    high_threshold: float = 0.95,
    low_threshold: float = 0.92,
) -> bool:
    """Decide whether two descriptions are near-duplicates.

    Two descriptions are duplicates if either:

    * raw similarity >= ``high_threshold`` (paraphrases with identical SKUs), or
    * raw similarity in [``low_threshold``, ``high_threshold``) AND every
      token that differs between them is letter-only (no digits — so the
      diff is pure prose, not a model number or dimension).
    """
    raw = similarity(left, right)
    if raw >= high_threshold:
        return True
    if raw >= low_threshold and _diff_is_prose_only(left, right):
        return True
    return False


def deduplicate_records(records: Iterable[Dict[str, object]], threshold: float = 0.95) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for record in records:
        duplicate_index = None
        for index, existing in enumerate(kept):
            if record["hs6_label"] != existing["hs6_label"]:
                continue
            if similarity(str(record["tier1_description"]), str(existing["tier1_description"])) >= threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(record)
            continue
        current_rank = AUTHORITY_RANK.get(str(record.get("tier1_source", "")), 0)
        existing_rank = AUTHORITY_RANK.get(str(kept[duplicate_index].get("tier1_source", "")), 0)
        if current_rank > existing_rank:
            kept[duplicate_index] = record
        elif current_rank == existing_rank and len(str(record["tier1_description"])) > len(str(kept[duplicate_index]["tier1_description"])):
            kept[duplicate_index] = record
    return kept


def find_near_duplicate_groups(
    records: Sequence[Mapping[str, object]],
    *,
    high_threshold: float = 0.95,
    low_threshold: float = 0.92,
    description_field: str = "tier1_description",
    hs_field: str = "hs6_label",
) -> List[List[int]]:
    """Group record indices into near-duplicate clusters.

    Two records belong to the same group when they share the same
    ``hs_field`` value AND :func:`is_near_duplicate` returns True for
    their ``description_field``. Returns a list of index-lists;
    singletons are omitted.
    """
    by_hs: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        code = str(record.get(hs_field) or "")
        if not code:
            continue
        by_hs.setdefault(code, []).append(idx)

    groups: List[List[int]] = []
    for indices in by_hs.values():
        parent = list(range(len(indices)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(indices)):
            desc_i = str(records[indices[i]].get(description_field) or "")
            if not desc_i:
                continue
            for j in range(i + 1, len(indices)):
                desc_j = str(records[indices[j]].get(description_field) or "")
                if not desc_j:
                    continue
                if is_near_duplicate(
                    desc_i,
                    desc_j,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                ):
                    union(i, j)
        clusters: Dict[int, List[int]] = {}
        for k in range(len(indices)):
            clusters.setdefault(find(k), []).append(indices[k])
        for members in clusters.values():
            if len(members) >= 2:
                groups.append(sorted(members))
    return groups


def pick_keeper(
    members: Sequence[Mapping[str, object]],
    *,
    description_field: str = "tier1_description",
) -> int:
    """Index of the record to keep within a duplicate cluster.

    Tie-breakers: longest description first; fall back to lowest index
    for stability.
    """
    best_idx = 0
    best_len = -1
    for i, record in enumerate(members):
        desc_len = len(str(record.get(description_field) or ""))
        if desc_len > best_len:
            best_idx = i
            best_len = desc_len
    return best_idx


def quality_filter(records: Iterable[Dict[str, object]], max_tier2_length: int = 80, max_tier2_words: int = 5) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for record in records:
        tier2 = record.get("tier2_minimal") or {}
        minimal_name = str(tier2.get("part_name", ""))
        if len(minimal_name) > max_tier2_length:
            continue
        if minimal_name == str(record.get("tier1_description", "")):
            continue
        if len(minimal_name.split()) > max_tier2_words:
            continue
        filtered.append(record)
    return filtered

