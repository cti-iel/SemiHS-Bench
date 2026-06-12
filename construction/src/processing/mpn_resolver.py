"""Matching helpers for catalog and BOL variants."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from src.models import RawAuxiliaryRecord
from src.utils.text_utils import tokenize


def _compact(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _hs4(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:4]


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return float(intersection) / float(union)


def _record_search_text(record: Dict[str, object]) -> str:
    tier2 = record.get("tier2_minimal") or {}
    return " ".join(
        [
            str(record.get("tier1_description", "")),
            str(tier2.get("part_name", "")),
            " ".join(str(keyword) for keyword in record.get("keywords", [])),
        ]
    ).lower()


def _manufacturer_match(record_text: str, candidate: RawAuxiliaryRecord) -> bool:
    manufacturer = candidate.manufacturer.lower()
    if not manufacturer:
        return False
    return manufacturer in record_text or _compact(candidate.manufacturer) in _compact(record_text)


def _exact_mpn_match(record_text: str, candidate: RawAuxiliaryRecord) -> bool:
    if not candidate.mpn:
        return False
    return _compact(candidate.mpn) in _compact(record_text)


def _catalog_hs4_gate(record: Dict[str, object], candidate: RawAuxiliaryRecord) -> bool:
    metadata = candidate.metadata or {}
    target_hs4 = _hs4(metadata.get("target_hs_family"))
    record_hs4 = _hs4(record.get("hs4_label"))
    if not target_hs4 or not record_hs4:
        return True
    return target_hs4 == record_hs4


def _catalog_heading_mismatch(record: Dict[str, object], candidate: RawAuxiliaryRecord) -> bool:
    metadata = candidate.metadata or {}
    actual_hs4 = _hs4(metadata.get("hs_heading") or metadata.get("hs_code"))
    record_hs4 = _hs4(record.get("hs4_label"))
    return bool(actual_hs4 and record_hs4 and actual_hs4 != record_hs4)


def _token_overlap_score(record: Dict[str, object], candidate: RawAuxiliaryRecord) -> float:
    tier2 = record.get("tier2_minimal") or {}
    return max(
        _token_overlap(str(record.get("tier1_description", "")), candidate.description),
        _token_overlap(" ".join(str(keyword) for keyword in record.get("keywords", [])), candidate.description),
        _token_overlap(str(tier2.get("part_name", "")), candidate.description),
    )


def _confidence_label(score: float, exact_mpn_match: bool) -> str:
    if exact_mpn_match:
        return "exact_mpn"
    if score >= 0.7:
        return "high"
    if score >= 0.45:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def score_match_details(record: Dict[str, object], candidate: RawAuxiliaryRecord) -> Dict[str, object]:
    record_text = _record_search_text(record)
    token_overlap = _token_overlap_score(record, candidate)
    exact_mpn_match = _exact_mpn_match(record_text, candidate)
    manufacturer_match = _manufacturer_match(record_text, candidate)
    hs4_gate_pass = _catalog_hs4_gate(record, candidate)
    heading_mismatch = _catalog_heading_mismatch(record, candidate)

    score = 0.0
    if hs4_gate_pass:
        if exact_mpn_match:
            score += 0.7
        if manufacturer_match:
            score += 0.2
        score += 0.65 * token_overlap
    if heading_mismatch:
        score = max(0.0, score - 0.1)
    score = min(score, 1.0)

    return {
        "score": round(score, 4),
        "token_overlap": round(token_overlap, 4),
        "exact_mpn_match": exact_mpn_match,
        "manufacturer_match": manufacturer_match,
        "hs4_gate_pass": hs4_gate_pass,
        "heading_mismatch": heading_mismatch,
        "target_hs4": _hs4((candidate.metadata or {}).get("target_hs_family")),
        "actual_hs4": _hs4((candidate.metadata or {}).get("hs_heading") or (candidate.metadata or {}).get("hs_code")),
        "confidence": _confidence_label(score, exact_mpn_match),
    }


def score_auxiliary_match(record: Dict[str, object], candidate: RawAuxiliaryRecord) -> float:
    return float(score_match_details(record, candidate)["score"])


def best_match_details(
    record: Dict[str, object],
    candidates: Iterable[RawAuxiliaryRecord],
) -> Tuple[Optional[RawAuxiliaryRecord], float, Dict[str, object]]:
    best_candidate = None
    best_details: Dict[str, object] = {}
    best_rank = (-1.0, False, False, -1.0, 0)
    for candidate in candidates:
        details = score_match_details(record, candidate)
        rank = (
            float(details["score"]),
            bool(details["exact_mpn_match"]),
            bool(details["manufacturer_match"]),
            float(details["token_overlap"]),
            len(candidate.description),
        )
        if rank > best_rank:
            best_candidate = candidate
            best_rank = rank
            best_details = details
    return best_candidate, float(best_details.get("score", 0.0)), best_details


def best_match(
    record: Dict[str, object],
    candidates: Iterable[RawAuxiliaryRecord],
) -> Tuple[Optional[RawAuxiliaryRecord], float]:
    best_candidate, best_score, _ = best_match_details(record, candidates)
    return best_candidate, best_score
