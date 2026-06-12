"""Difficulty, ambiguity, and classifiability heuristics."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

from src.annotation.boundary_detector import BOUNDARY_TAGS, detect_boundaries
from src.collectors.hts_taxonomy import TaxonomyEntry, TaxonomyIndex
from src.processing.degrader import _looks_like_real_mpn
from src.utils.text_utils import normalize_text, tokenize


STOPWORDS = {
    "a",
    "an",
    "and",
    "apparatus",
    "article",
    "articles",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "instrument",
    "instruments",
    "is",
    "it",
    "its",
    "machine",
    "machines",
    "of",
    "on",
    "or",
    "other",
    "product",
    "products",
    "system",
    "systems",
    "that",
    "the",
    "this",
    "to",
    "used",
    "whether",
    "with",
    "without",
}
MATERIAL_KEYWORDS = {
    "silicon",
    "ceramic",
    "aluminum",
    "aluminium",
    "copper",
    "gallium",
    "arsenide",
    "phosphide",
    "germanium",
    "metal",
    "photoresist",
    "chemical",
    "wafer",
    "wafers",
}
FUNCTION_KEYWORDS = {
    "amplify",
    "amplifier",
    "controller",
    "controllers",
    "convert",
    "converter",
    "converters",
    "measure",
    "measurement",
    "monitor",
    "process",
    "processor",
    "processors",
    "sensor",
    "store",
    "switch",
    "switching",
}
USE_KEYWORDS = {
    "for vehicles",
    "for telecom",
    "telecommunications",
    "for semiconductor manufacturing",
    "for industrial automation",
    "used in automotive",
    "used for",
    "designed for",
    "application",
    "applications",
    "solely or principally",
}
PRODUCT_TYPE_KEYWORDS = {
    "amplifier",
    "capacitor",
    "controller",
    "diode",
    "igbt",
    "integrated",
    "led",
    "memory",
    "mcu",
    "microcontroller",
    "module",
    "mosfet",
    "processor",
    "prober",
    "resistor",
    "sensor",
    "switch",
    "transistor",
    "wafer",
}
SPEC_HINT_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s?(?:v|mv|a|ma|w|mw|hz|khz|mhz|ghz|kb|mb|gb|mm|cm|g|kg|pin|pins)|"
    r"qfn|bga|lqfp|soic|dip|qfp|sot|tqfp|tsop|plcc|csp|lga|pga)\b",
    flags=re.IGNORECASE,
)
DISTINCTIVE_TOKENS = {"silicon", "memory", "processor", "controllers", "controller", "amplifiers", "amplifier", "wafer", "wafers"}


def _record_text(record: Mapping[str, object]) -> str:
    parts = [str(record.get("tier1_description", ""))]
    parts.extend(str(item) for item in record.get("keywords", []) if str(item).strip())
    return normalize_text(" ".join(parts))


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    }


def _taxonomy_entries_by_hs4(taxonomy: TaxonomyIndex) -> Dict[str, List[TaxonomyEntry]]:
    entries: Dict[str, List[TaxonomyEntry]] = defaultdict(list)
    for entry in taxonomy.all_entries():
        entries[entry.hs4].append(entry)
    return entries


def _plausible_sibling(record_tokens: set[str], entry: TaxonomyEntry, actual_hs6: str) -> bool:
    if entry.hs6 == actual_hs6:
        return True
    entry_tokens = _content_tokens(entry.description)
    overlap = record_tokens & entry_tokens
    if len(overlap) >= 2:
        return True
    if any(token in DISTINCTIVE_TOKENS or len(token) >= 6 for token in overlap):
        return True
    return False


def infer_classification_driver(record: Mapping[str, object]) -> str:
    justification = normalize_text(str(record.get("justification_text", ""))).lower()
    tier1_text = normalize_text(str(record.get("tier1_description", ""))).lower()
    text = justification or tier1_text

    if "gri 3" in text or "gri iii" in text:
        return "combination"

    categories = []
    if any(keyword in text for keyword in MATERIAL_KEYWORDS):
        categories.append("material")
    if any(keyword in text for keyword in FUNCTION_KEYWORDS):
        categories.append("function")
    if any(keyword in text for keyword in USE_KEYWORDS):
        categories.append("use")

    if len(categories) >= 2:
        return "combination"
    if categories:
        return categories[0]
    return "function"


def infer_ambiguity(
    record: Mapping[str, object],
    taxonomy: TaxonomyIndex,
    entries_by_hs4: Mapping[str, Sequence[TaxonomyEntry]] | None = None,
) -> object:
    hs4 = str(record.get("hs4_label", ""))
    hs6 = str(record.get("hs6_label", ""))
    siblings = list((entries_by_hs4 or _taxonomy_entries_by_hs4(taxonomy)).get(hs4, []))
    if not siblings:
        return 1
    record_tokens = _content_tokens(_record_text(record))
    plausible = sum(1 for entry in siblings if _plausible_sibling(record_tokens, entry, hs6))
    plausible = max(1, plausible)
    if plausible >= 3:
        return "3+"
    return plausible


def infer_tier2_classifiable(record: Mapping[str, object]) -> str:
    part_name = normalize_text(str((record.get("tier2_minimal") or {}).get("part_name", "")))
    if not part_name:
        return "no"
    for token in re.split(r"\s+", part_name):
        if _looks_like_real_mpn(token):
            return "no"
    lowered = part_name.lower()
    tokens = set(tokenize(lowered))
    has_product_type = any(keyword in tokens for keyword in PRODUCT_TYPE_KEYWORDS)
    has_specs = bool(SPEC_HINT_RE.search(part_name))
    if has_product_type and has_specs:
        return "yes"
    if has_product_type:
        return "partial"
    return "no"


def build_annotation_report(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    boundary_counts = Counter()
    for record in records:
        for tag in record.get("boundary_tags", []):
            boundary_counts[str(tag)] += 1
    for tag in BOUNDARY_TAGS:
        boundary_counts.setdefault(tag, 0)
    return {
        "total_records": len(records),
        "ambiguity_score_distribution": dict(sorted(Counter(str(record.get("ambiguity_score", "")) for record in records).items())),
        "classification_driver_distribution": dict(
            sorted(Counter(str(record.get("classification_driver", "")) for record in records).items())
        ),
        "boundary_case_counts": dict(sorted(boundary_counts.items())),
        "tier2_classifiable_distribution": dict(
            sorted(Counter(str(record.get("tier2_classifiable", "")) for record in records).items())
        ),
    }


def annotate_records(records: Iterable[Dict[str, object]], taxonomy: TaxonomyIndex) -> List[Dict[str, object]]:
    entries_by_hs4 = _taxonomy_entries_by_hs4(taxonomy)
    annotated: List[Dict[str, object]] = []
    for record in records:
        new_record = copy.deepcopy(record)
        boundary_tags = detect_boundaries(new_record)
        new_record["boundary_tags"] = list(boundary_tags)
        new_record["difficulty_tags"] = list(boundary_tags)
        new_record["classification_driver"] = infer_classification_driver(new_record)
        new_record["ambiguity_score"] = infer_ambiguity(new_record, taxonomy, entries_by_hs4=entries_by_hs4)
        new_record["tier2_classifiable"] = infer_tier2_classifiable(new_record)

        source_metadata = new_record.get("source_metadata")
        if isinstance(source_metadata, dict):
            annotation_metadata = dict(source_metadata.get("annotation", {}))
            annotation_metadata.update(
                {
                    "boundary_tags": list(boundary_tags),
                    "ambiguity_score": new_record["ambiguity_score"],
                    "classification_driver": new_record["classification_driver"],
                    "tier2_classifiable": new_record["tier2_classifiable"],
                }
            )
            source_metadata["annotation"] = annotation_metadata
            new_record["source_metadata"] = source_metadata

        annotated.append(new_record)
    return annotated
