"""Construct per-record HS6 candidate sets for constrained top-k eval.

Each test record gets a 4-element `candidate_set`: the gold HS6 code plus
three confusable distractors. Distractor selection prefers, in order:

1. **Boundary expansion** — HS6s drawn from the headings named in the record's
   `difficulty_tags` / `annotation.boundary_tags` (e.g. tag `8541_vs_8542`
   on a record in 8541 contributes 8542 children; tag
   `8542.31_vs_8542.32_vs_8542.33_vs_8542.39` contributes those four HS6s
   directly).
2. **Sibling heading** — other HS6s under the same HS4 as the gold code.
3. **Random chapter** — HS6s under the same HS2 chapter, uniformly sampled.

Selection is deterministic per record (RNG seeded on the record id), so
repeated runs produce identical candidate sets.

The presented order is shuffled (also deterministic) so the gold's index is
not a constant — `gold_rank_in_candidates` records the post-shuffle position
for sanity checks (always non-negative; -1 would indicate the gold leaked
out of the slate, which the builder treats as an error).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


CANDIDATE_SIZE = 4

# Construction labels recorded on the candidate_set. We tag each set with the
# strongest source used (boundary > sibling > chapter). The set is constructed
# in priority order, so this label lines up with the dominant distractor source.
CONSTRUCTION_BOUNDARY = "boundary_expansion"
CONSTRUCTION_SIBLING = "sibling_heading"
CONSTRUCTION_CHAPTER = "random_chapter"


_BOUNDARY_TAG_RE = re.compile(r"_vs_")
_DIGITS_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class CandidateSet:
    """Immutable 4-element HS6 candidate slate."""

    codes: Tuple[str, ...]
    construction: str
    gold_rank_in_candidates: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": len(self.codes),
            "codes": list(self.codes),
            "construction": self.construction,
            "gold_rank_in_candidates": self.gold_rank_in_candidates,
        }


# ----- helpers ---------------------------------------------------------------


def _record_seed(record_id: str) -> int:
    """Stable integer seed from record id. Avoids Python's randomized hash()."""
    h = 0
    for ch in record_id:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _collect_boundary_tags(record: Mapping[str, Any]) -> List[str]:
    tags: List[str] = []
    for tag in record.get("difficulty_tags") or []:
        if isinstance(tag, str) and _BOUNDARY_TAG_RE.search(tag):
            tags.append(tag)
    annotation = (record.get("source_metadata") or {}).get("annotation") or {}
    for tag in annotation.get("boundary_tags") or []:
        if isinstance(tag, str) and _BOUNDARY_TAG_RE.search(tag) and tag not in tags:
            tags.append(tag)
    return tags


def _expand_tag(
    tag: str,
    *,
    gold_hs6: str,
    hs6_pool_by_hs4: Mapping[str, Set[str]],
    in_scope_hs6: Set[str],
) -> List[str]:
    """Translate one boundary tag into a list of candidate HS6 codes,
    interleaving across the tag's parts so that each named class is
    represented before any one class is exhausted.

    A tag part may be either a 4-digit HS4 (expand to all in-scope HS6s under
    it) or a 6-digit HS6 (use directly). Dotted forms like ``8542.31`` are
    accepted.
    """
    parts = [
        "".join(_DIGITS_RE.findall(part))
        for part in tag.split("_vs_")
    ]
    # Build a per-part queue of candidate HS6s.
    queues: List[List[str]] = []
    for part in parts:
        queue: List[str] = []
        if len(part) >= 6:
            code = part[:6]
            if code != gold_hs6 and code in in_scope_hs6:
                queue.append(code)
        elif len(part) == 4:
            queue = sorted(c for c in hs6_pool_by_hs4.get(part, ()) if c != gold_hs6)
        queues.append(queue)

    # Round-robin across queues so 8541 vs 8542 contributes one of each first.
    candidates: List[str] = []
    seen: Set[str] = set()
    while any(queues):
        progressed = False
        for queue in queues:
            if not queue:
                continue
            code = queue.pop(0)
            if code in seen:
                continue
            candidates.append(code)
            seen.add(code)
            progressed = True
        if not progressed:
            break
    return candidates


def _index_taxonomy(in_scope_hs6: Iterable[str]) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    in_scope_set = set(in_scope_hs6)
    by_hs4: Dict[str, Set[str]] = {}
    by_hs2: Dict[str, Set[str]] = {}
    for code in in_scope_set:
        if len(code) != 6:
            continue
        by_hs4.setdefault(code[:4], set()).add(code)
        by_hs2.setdefault(code[:2], set()).add(code)
    return in_scope_set, by_hs4, by_hs2


# ----- public API ------------------------------------------------------------


def build_candidate_set(
    record: Mapping[str, Any],
    *,
    in_scope_hs6: Iterable[str],
    size: int = CANDIDATE_SIZE,
) -> CandidateSet:
    """Build a candidate set for a single record.

    Raises ValueError if there aren't enough in-scope HS6 codes in the record's
    chapter to fill ``size``. (At the benchmark chapter scope this never triggers.)
    """
    if size < 2:
        raise ValueError("candidate set size must be at least 2 (gold + distractor)")

    gold_hs6 = str(record["hs6_label"])
    if len(gold_hs6) != 6:
        raise ValueError(f"record {record.get('id')!r}: hs6_label must be 6 digits")

    in_scope_set, by_hs4, by_hs2 = _index_taxonomy(in_scope_hs6)
    if gold_hs6 not in in_scope_set:
        # Gold is allowed to live outside the in-scope pool (BOL_new_hs6_audit),
        # but the candidate set must still draw distractors from the pool.
        in_scope_set = in_scope_set | {gold_hs6}
        by_hs4.setdefault(gold_hs6[:4], set()).add(gold_hs6)
        by_hs2.setdefault(gold_hs6[:2], set()).add(gold_hs6)

    distractor_target = size - 1
    rng = random.Random(_record_seed(str(record["id"])))

    distractors: List[str] = []
    seen: Set[str] = {gold_hs6}
    used_sources: Set[str] = set()

    # 1. Boundary expansion.
    for tag in _collect_boundary_tags(record):
        for code in _expand_tag(
            tag,
            gold_hs6=gold_hs6,
            hs6_pool_by_hs4=by_hs4,
            in_scope_hs6=in_scope_set,
        ):
            if code in seen:
                continue
            distractors.append(code)
            seen.add(code)
            used_sources.add(CONSTRUCTION_BOUNDARY)
            if len(distractors) >= distractor_target:
                break
        if len(distractors) >= distractor_target:
            break

    # 2. Sibling HS6s under the same HS4 (sorted for determinism, then shuffled).
    if len(distractors) < distractor_target:
        siblings = sorted(by_hs4.get(gold_hs6[:4], set()) - seen)
        rng.shuffle(siblings)
        for code in siblings:
            distractors.append(code)
            seen.add(code)
            used_sources.add(CONSTRUCTION_SIBLING)
            if len(distractors) >= distractor_target:
                break

    # 3. Random in-chapter codes.
    if len(distractors) < distractor_target:
        chapter = sorted(by_hs2.get(gold_hs6[:2], set()) - seen)
        rng.shuffle(chapter)
        for code in chapter:
            distractors.append(code)
            seen.add(code)
            used_sources.add(CONSTRUCTION_CHAPTER)
            if len(distractors) >= distractor_target:
                break

    if len(distractors) < distractor_target:
        raise ValueError(
            f"record {record.get('id')!r}: only {len(distractors)} distractors "
            f"available in chapter {gold_hs6[:2]}; need {distractor_target}"
        )

    if CONSTRUCTION_BOUNDARY in used_sources:
        construction = CONSTRUCTION_BOUNDARY
    elif CONSTRUCTION_SIBLING in used_sources:
        construction = CONSTRUCTION_SIBLING
    else:
        construction = CONSTRUCTION_CHAPTER

    # Shuffle gold + distractors together so gold position is not a constant.
    slate = [gold_hs6] + distractors
    rng.shuffle(slate)
    gold_rank = slate.index(gold_hs6)

    return CandidateSet(
        codes=tuple(slate),
        construction=construction,
        gold_rank_in_candidates=gold_rank,
    )


def attach_candidate_sets(
    records: Sequence[Mapping[str, Any]],
    *,
    in_scope_hs6: Iterable[str],
    size: int = CANDIDATE_SIZE,
) -> List[Dict[str, Any]]:
    """Return a copy of ``records`` with `candidate_set` populated on each."""
    in_scope_list = list(in_scope_hs6)
    out: List[Dict[str, Any]] = []
    for record in records:
        slate = build_candidate_set(record, in_scope_hs6=in_scope_list, size=size)
        new_record = dict(record)
        new_record["candidate_set"] = slate.to_dict()
        out.append(new_record)
    return out
