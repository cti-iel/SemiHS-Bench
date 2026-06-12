"""Generate benchmark-style degraded tiers from canonical products."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.utils.abbreviation_engine import abbreviate_text, load_abbreviations
from src.utils.config_loader import load_mapping
from src.utils.text_utils import (
    clean_bol_description,
    normalize_text,
    remove_phrases,
    tokenize,
    truncate_text_word_safe,
)


LOW_INFORMATION_TOKENS = {
    "a",
    "an",
    "and",
    "apparatus",
    "article",
    "articles",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "including",
    "instrument",
    "instruments",
    "is",
    "item",
    "items",
    "module",
    "modules",
    "of",
    "or",
    "other",
    "part",
    "parts",
    "product",
    "products",
    "system",
    "systems",
    "that",
    "the",
    "this",
    "used",
    "with",
}
AGGRESSIVE_LOW_INFORMATION_TOKENS = LOW_INFORMATION_TOKENS | {
    "absolute",
    "appearance",
    "double",
    "electrical",
    "electric",
    "handle",
    "handheld",
    "liquid",
    "low",
    "physical",
    "position",
    "profile",
    "properties",
    "single",
    "solid",
    "travel",
    "very",
}
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
DATE_RE = re.compile(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$")
MEASUREMENT_RE = re.compile(
    r"^\d+(?:\.\d+)?(?:V|MV|A|MA|W|MW|HZ|KHZ|MHZ|GHZ|KB|MB|GB|KBIT|MBIT|GBIT|MM|CM|IN|MG|G|KG|PSI|PIN|PINS)$",
    flags=re.IGNORECASE,
)
PUNCT_STRIP = " ,;:.-_()/[]{}"
CAS_INLINE_RE = re.compile(r"\bCAS(?:\s+NO\.?:?)?\s*\d{2,7}-\d{2}-\d\b", flags=re.IGNORECASE)
QUALITY_LABEL_RE = re.compile(r"\b(?:APPEARANCE|PHYSICAL PROPERT(?:Y|IES))\b\s*:?", flags=re.IGNORECASE)


def canonical_to_benchmark_record(product: Mapping[str, object]) -> Dict[str, object]:
    canonical_id = str(product.get("canonical_id", "")).strip()
    benchmark_id = canonical_id.replace("CP-", "SH-", 1) if canonical_id.startswith("CP-") else "SH-" + canonical_id
    primary_source = str(product.get("primary_source", ""))
    raw_description = str(product.get("canonical_description", ""))
    if primary_source == "BOL":
        tier1 = clean_bol_description(raw_description)
    else:
        tier1 = normalize_text(raw_description)
    return {
        "id": benchmark_id,
        "canonical_id": canonical_id,
        "keywords": list(product.get("keywords", [])),
        "tier1_description": tier1,
        "tier1_source": primary_source,
        "tier2_minimal": {"part_name": "", "manufacturer": ""},
        "tier2_provenance": "",
        "hs6_label": str(product.get("hs6_label", "")),
        "hs4_label": str(product.get("hs4_label", "")),
        "hs2_label": str(product.get("hs2_label", "")),
        "label_source": str(product.get("label_source", "") or product.get("primary_source", "")),
        "difficulty_tags": [],
        "justification_text": normalize_text(str(product.get("justification_text", ""))),
        "bol_metadata": None,
        "source_reference": str(product.get("primary_reference", "")),
        "source_metadata": {
            "canonical_id": canonical_id,
            "sampling_metadata": copy.deepcopy(product.get("sampling_metadata", {})),
            "source_evidence": copy.deepcopy(product.get("source_evidence", [])),
            "merge_confidence": product.get("merge_confidence"),
            "keywords": list(product.get("keywords", [])),
            "manufacturer_hint": str(product.get("manufacturer_hint", "")),
            "manufacturer_hint_source": str(product.get("manufacturer_hint_source", "")),
        },
    }


def _manufacturer_hint(record: Mapping[str, object]) -> str:
    metadata = record.get("source_metadata") or {}
    if not isinstance(metadata, dict):
        return ""
    explicit_keys = ("manufacturer_hint", "manufacturer", "brand", "producer", "maker", "vendor")
    for key in explicit_keys:
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    for evidence in metadata.get("source_evidence", []):
        if not isinstance(evidence, dict):
            continue
        source_metadata = evidence.get("source_metadata") or {}
        if not isinstance(source_metadata, dict):
            continue
        for key in explicit_keys:
            value = str(source_metadata.get(key, "")).strip()
            if value:
                return value
    return ""


def _remove_phrase_groups(text: str, rules: Mapping[str, object]) -> str:
    cleaned = normalize_text(text)
    for key in ("remove_tariff_phrases", "remove_use_case_phrases", "remove_connective_phrases"):
        cleaned = remove_phrases(cleaned, rules.get(key, []))
    cleaned = CAS_INLINE_RE.sub(" ", cleaned)
    cleaned = QUALITY_LABEL_RE.sub(" ", cleaned)
    return normalize_text(cleaned)


def _extract_specs(text: str, patterns: Mapping[str, str]) -> Tuple[List[Dict[str, str]], str]:
    extra_details: List[Dict[str, str]] = []
    source_text = normalize_text(text)
    cleaned = source_text
    seen = set()
    for spec_name, pattern in patterns.items():
        matches = list(re.finditer(pattern, source_text, flags=re.IGNORECASE))
        for match in matches:
            value = normalize_text(match.group(0)).strip(PUNCT_STRIP)
            if not value:
                continue
            key = (spec_name, value.lower())
            if key in seen:
                continue
            seen.add(key)
            extra_details.append({str(spec_name): value})
        if matches:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
            cleaned = normalize_text(cleaned)
    return extra_details, cleaned


def _clean_part_name(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,;:.])", r"\1", cleaned)
    cleaned = cleaned.strip(PUNCT_STRIP)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return normalize_text(cleaned)


def _compress_informative_prefix(text: str, target_limit: int) -> str:
    candidate = _clean_part_name(text)
    clauses = [normalize_text(part) for part in re.split(r"[.;]\s+|\n+", candidate) if normalize_text(part)]
    if clauses:
        candidate = clauses[0]
        if len(candidate) < 24 and len(clauses) > 1:
            candidate = normalize_text(candidate + " " + clauses[1])
    if len(candidate) > target_limit:
        filtered = [token for token in candidate.split() if token.lower().strip(PUNCT_STRIP) not in LOW_INFORMATION_TOKENS]
        if filtered:
            candidate = " ".join(filtered)
    if len(candidate) > target_limit:
        words = candidate.split()
        shortened: List[str] = []
        for word in words:
            next_text = " ".join(shortened + [word])
            if len(next_text) > target_limit:
                break
            shortened.append(word)
        if shortened:
            candidate = " ".join(shortened)
    return _clean_part_name(candidate)


def _safe_tier2_part_name(
    text: str,
    original: str,
    abbreviations: List[Tuple[str, str]],
    limit: int,
    target_limit: int,
) -> str:
    candidate = _compress_informative_prefix(text, target_limit=target_limit)
    if not candidate:
        candidate = abbreviate_text(original, abbreviations)
        candidate = _compress_informative_prefix(candidate, target_limit=target_limit)
    return truncate_text_word_safe(candidate, limit)


def _aggressive_tier2_part_name(
    text: str,
    abbreviations: List[Tuple[str, str]],
    limit: int,
    target_limit: int,
) -> str:
    cleaned = QUALITY_LABEL_RE.sub(" ", CAS_INLINE_RE.sub(" ", normalize_text(text)))
    abbreviated = abbreviate_text(cleaned, abbreviations)
    tokens = [
        token
        for token in abbreviated.split()
        if token.lower().strip(PUNCT_STRIP) not in AGGRESSIVE_LOW_INFORMATION_TOKENS
    ]
    candidate = _compress_informative_prefix(" ".join(tokens), target_limit=target_limit)
    if not candidate:
        candidate = _compress_informative_prefix(abbreviated, target_limit=target_limit)
    return truncate_text_word_safe(candidate, limit)


def _candidate_texts_for_mpn(record: Mapping[str, object]) -> List[Tuple[int, str]]:
    texts: List[Tuple[int, str]] = []
    texts.append((3, str(record.get("tier1_description", ""))))
    for keyword in record.get("keywords", []):
        texts.append((5, str(keyword)))
    metadata = record.get("source_metadata") or {}
    if isinstance(metadata, dict):
        for keyword in metadata.get("keywords", []):
            texts.append((4, str(keyword)))
        for evidence in metadata.get("source_evidence", []):
            if not isinstance(evidence, dict):
                continue
            texts.append((2, str(evidence.get("description", ""))))
            for keyword in evidence.get("keywords", []):
                texts.append((4, str(keyword)))
    return texts


def _looks_like_real_mpn(candidate: str) -> bool:
    token = candidate.strip(PUNCT_STRIP)
    if len(token) < 4 or len(token) > 24:
        return False
    if CAS_RE.match(token) or DATE_RE.match(token):
        return False
    if token.isdigit():
        return False
    compact = token.replace("-", "").replace("/", "")
    if compact.isdigit():
        return False
    if len(compact) == 6 and compact.isdigit():
        return False
    upper = compact.upper()
    if MEASUREMENT_RE.match(upper):
        return False
    if not any(char.isalpha() for char in token) or not any(char.isdigit() for char in token):
        return False
    # Short tokens must have at least 2 digits to avoid generic 3-char alpha+1-digit noise
    if len(token) <= 5 and sum(1 for c in token if c.isdigit()) < 2:
        return False
    # Reject common generic tokens that match patterns but aren't MPNs
    lower = token.lower()
    if lower in {"grade1", "grade2", "grade3", "class1", "class2", "type1", "type2", "iso9001"}:
        return False
    return True


def _extract_real_mpn(record: Mapping[str, object], patterns: Sequence[str]) -> Optional[str]:
    candidates: List[Tuple[int, str]] = []
    for priority, text in _candidate_texts_for_mpn(record):
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                token = normalize_text(match.group(0)).strip(PUNCT_STRIP)
                if not _looks_like_real_mpn(token):
                    continue
                score = priority
                if "-" in token:
                    score += 1
                if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
                    score += 2
                candidates.append((score, token))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    return candidates[0][1]


def _compress_to_tier3(part_name: str, abbreviations: List[Tuple[str, str]], limit: int) -> str:
    abbreviated = _clean_part_name(abbreviate_text(part_name, abbreviations))
    target_limit = min(limit, 30)
    if len(abbreviated) <= target_limit:
        return abbreviated
    tokens = abbreviated.split()
    important = [token for token in tokens if token.lower() not in LOW_INFORMATION_TOKENS]
    current = important or tokens
    while current and len(" ".join(current)) > target_limit:
        current = current[:-1]
    if current:
        return truncate_text_word_safe(" ".join(current), target_limit)
    return truncate_text_word_safe(abbreviated, target_limit)


def _weak_reasons(
    record: Mapping[str, object],
    tier2: Mapping[str, object],
    tier3: Mapping[str, object],
    rules: Mapping[str, object],
    mpn_extracted: bool,
) -> List[str]:
    reasons: List[str] = []
    tier2_part = str(tier2.get("part_name", ""))
    tier3_part = str(tier3.get("part_name", ""))
    if len(tier2_part) > int(rules.get("tier2_weak_length", 70)):
        reasons.append("tier2_too_long")
    if normalize_text(tier2_part).lower() == normalize_text(str(record.get("tier1_description", ""))).lower():
        reasons.append("tier2_unchanged")
    if len(tier3_part) > int(rules.get("tier3_weak_length", 35)) and not mpn_extracted:
        reasons.append("tier3_long_non_mpn")
    return reasons


def _build_rule_based_tier2(
    record: Mapping[str, object],
    rules: Mapping[str, object],
    abbreviations: List[Tuple[str, str]],
) -> Dict[str, object]:
    description = str(record.get("tier1_description", ""))
    cleaned = _remove_phrase_groups(description, rules)
    abbreviated = abbreviate_text(cleaned, abbreviations)
    extra_details, without_specs = _extract_specs(abbreviated, rules.get("spec_patterns", {}))
    part_name = _safe_tier2_part_name(
        without_specs,
        original=description,
        abbreviations=abbreviations,
        limit=int(rules.get("tier2_max_length", 80)),
        target_limit=int(rules.get("tier2_target_length", 60)),
    )
    if normalize_text(part_name).lower() == normalize_text(description).lower():
        part_name = _aggressive_tier2_part_name(
            description,
            abbreviations=abbreviations,
            limit=int(rules.get("tier2_max_length", 80)),
            target_limit=int(rules.get("tier2_target_length", 60)),
        )
    return {
        "part_name": part_name,
        "manufacturer": _manufacturer_hint(record),
        "extra_details": extra_details,
    }


def _build_rule_based_tier3(
    record: Mapping[str, object],
    tier2: Mapping[str, object],
    rules: Mapping[str, object],
    abbreviations: List[Tuple[str, str]],
) -> Tuple[Dict[str, object], bool, bool]:
    mpn = _extract_real_mpn(record, rules.get("mpn_patterns", []))
    if mpn:
        return (
            {
                "part_name": mpn,
                "manufacturer": str(tier2.get("manufacturer", "")),
            },
            True,
            False,
        )
    compact = _compress_to_tier3(
        str(tier2.get("part_name", "")),
        abbreviations=abbreviations,
        limit=int(rules.get("tier3_max_length", 40)),
    )
    return (
        {
            "part_name": compact,
            "manufacturer": str(tier2.get("manufacturer", "")),
        },
        False,
        True,
    )


def _review_row(record: Mapping[str, object]) -> Dict[str, object]:
    metadata = record.get("degradation_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "id": record.get("id"),
        "canonical_id": metadata.get("canonical_id", record.get("canonical_id")),
        "hs6_label": record.get("hs6_label"),
        "tier1_description": record.get("tier1_description"),
        "tier2_minimal": copy.deepcopy(record.get("tier2_minimal")),
        "mpn_extracted": metadata.get("mpn_extracted", False),
        "tier2_equals_tier1": metadata.get("tier2_equals_tier1", False),
        "tier3_is_truncation": metadata.get("tier3_is_truncation", False),
        "weak_degradation_reasons": list(metadata.get("weak_degradation_reasons", [])),
    }


def _summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    total = len(records)
    tier2_lengths = [len(str((record.get("tier2_minimal") or {}).get("part_name", ""))) for record in records]
    mpn_hits = sum(
        1
        for record in records
        if bool((record.get("degradation_metadata") or {}).get("mpn_extracted", False))
    )
    unchanged = sum(
        1
        for record in records
        if bool((record.get("degradation_metadata") or {}).get("tier2_equals_tier1", False))
    )
    weak_count = sum(
        1
        for record in records
        if bool((record.get("degradation_metadata") or {}).get("weak_degradation_reasons"))
    )
    manufacturer_count = sum(
        1
        for record in records
        if bool(normalize_text(str((record.get("tier2_minimal") or {}).get("manufacturer", ""))))
    )
    return {
        "total_records": total,
        "average_tier2_part_name_length": round(sum(tier2_lengths) / total, 2) if total else 0.0,
        "mpn_extraction_rate": round(mpn_hits / total, 4) if total else 0.0,
        "tier2_unchanged_rate": round(unchanged / total, 4) if total else 0.0,
        "weak_record_count": weak_count,
        "manufacturer_hint_rate": round(manufacturer_count / total, 4) if total else 0.0,
    }


def generate_tiers(
    records: Iterable[Mapping[str, object]],
    abbreviations_path: str,
    rules_path: str,
) -> Dict[str, object]:
    rules = load_mapping(rules_path)
    abbreviations = load_abbreviations(abbreviations_path)

    degraded: List[Dict[str, object]] = []
    review_rows: List[Dict[str, object]] = []
    for product in records:
        benchmark_record = canonical_to_benchmark_record(product)
        tier2 = _build_rule_based_tier2(benchmark_record, rules, abbreviations)
        tier3, mpn_extracted, tier3_is_truncation = _build_rule_based_tier3(
            benchmark_record,
            tier2,
            rules,
            abbreviations,
        )
        weak_reasons = _weak_reasons(benchmark_record, tier2, tier3, rules, mpn_extracted)
        tier2_equals_tier1 = normalize_text(str(tier2.get("part_name", ""))).lower() == normalize_text(
            str(benchmark_record.get("tier1_description", ""))
        ).lower()

        # The released record carries two tiers; ``tier2`` (the structured form)
        # remains an internal build intermediate used to derive the minimal Tier 2.
        tier2_prov = "natural_mpn" if mpn_extracted else "degraded_manual"
        benchmark_record["tier2_minimal"] = tier3
        benchmark_record["tier2_provenance"] = tier2_prov
        benchmark_record["degradation_metadata"] = {
            "canonical_id": benchmark_record["canonical_id"],
            "mpn_extracted": mpn_extracted,
            "tier2_equals_tier1": tier2_equals_tier1,
            "tier3_is_truncation": tier3_is_truncation,
            "weak_degradation_reasons": weak_reasons,
        }
        degraded.append(benchmark_record)
        review_rows.append(_review_row(benchmark_record))

    return {
        "records": degraded,
        "review_rows": review_rows,
        "summary": _summary(degraded),
    }
