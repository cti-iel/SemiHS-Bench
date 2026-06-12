"""Core data models for SemiHS-Bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


AUTHORITATIVE_SOURCES = {"EBTI", "CROSS", "JP_CUSTOMS"}
TIER2_PROVENANCE = {"natural_mpn", "degraded_manual", "degraded_llm"}
CLASSIFICATION_DRIVERS = {"material", "function", "use", "combination"}
TIER2_CLASSIFIABILITY = {"yes", "no", "partial"}


@dataclass
class RawAuthoritativeRecord:
    source: str
    reference: str
    description: str
    hs_code: str
    justification: str = ""
    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScrapeTask:
    source: str
    query_type: str
    query_value: str
    task_id: str
    pagination_cursor: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResultItem:
    source: str
    reference: str
    detail_url: str
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    source: str
    reference: str
    success: bool
    record: Optional[RawAuthoritativeRecord] = None
    raw_payload: Optional[Dict[str, Any]] = None
    reason: str = ""
    retryable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["record"] = self.record.to_dict() if self.record else None
        return data


@dataclass
class RunReport:
    source: str
    tasks_total: int = 0
    search_pages_fetched: int = 0
    detail_pages_fetched: int = 0
    discovered_items: int = 0
    normalized_records: int = 0
    duplicates_skipped: int = 0
    retries: int = 0
    terminal_failures: int = 0
    blocked_items: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawAuxiliaryRecord:
    source: str
    reference: str
    description: str
    manufacturer: str = ""
    mpn: str = ""
    hs_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2Minimal:
    part_name: str
    manufacturer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewItem:
    queue: str
    reference: str
    reason: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuthoritativeEvidence:
    source: str
    reference: str
    description: str
    hs6_label: str
    language: str
    justification_text: str = ""
    keywords: List[str] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalProduct:
    canonical_id: str
    canonical_description: str
    hs6_label: str
    hs4_label: str
    hs2_label: str
    primary_source: str
    primary_reference: str
    language: str
    justification_text: str
    keywords: List[str] = field(default_factory=list)
    source_evidence: List[AuthoritativeEvidence] = field(default_factory=list)
    manufacturer_hint: str = ""
    manufacturer_hint_source: str = ""
    merge_confidence: float = 1.0
    selection_score: float = 0.0
    label_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_evidence"] = [evidence.to_dict() for evidence in self.source_evidence]
        return data


@dataclass
class BenchmarkRecord:
    id: str
    tier1_description: str
    tier1_source: str
    tier2_minimal: Tier2Minimal
    tier2_provenance: str
    hs6_label: str
    hs4_label: str
    hs2_label: str
    label_source: str
    difficulty_tags: List[str]
    ambiguity_score: Union[int, str]
    classification_driver: str
    tier2_classifiable: str
    justification_text: str
    bol_metadata: Optional[Dict[str, Any]] = None
    source_reference: str = ""
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tier2_minimal"] = self.tier2_minimal.to_dict()
        return data
