"""Overlay natural BOL and catalog variants onto degraded benchmark records."""

from __future__ import annotations

import csv
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.collectors.bol_collector import is_specific_bol_record
from src.models import RawAuxiliaryRecord
from src.processing.mpn_resolver import best_match_details
from src.utils.text_utils import normalize_text


DEFAULT_GENERIC_BOL_TERMS = [
    "electronic parts",
    "electronic component",
    "electronic components",
    "semiconductor devices nos",
    "semiconductor parts",
    "parts",
]
EXACT_MPN_REVIEW_FIELDS = [
    "id",
    "canonical_id",
    "hs4_label",
    "hs6_label",
    "tier1_description",
    "catalog_candidate_reference",
    "catalog_candidate_provider",
    "catalog_candidate_mpn",
    "catalog_candidate_manufacturer",
    "catalog_candidate_description",
    "catalog_candidate_product_url",
    "catalog_score",
    "catalog_confidence",
    "catalog_hs4_gate_pass",
    "catalog_heading_mismatch",
    "catalog_target_hs4",
    "catalog_actual_hs4",
    "catalog_block_reason",
    "catalog_block_reason_detail",
    "reviewer_decision",
    "reviewer_note",
]


def _declared_hs(candidate: RawAuxiliaryRecord) -> str:
    return str((candidate.metadata or {}).get("declared_hs", "")).replace(".", "").strip()


def _bol_hs_verified(record: Mapping[str, object], candidate: RawAuxiliaryRecord) -> bool:
    declared_hs = _declared_hs(candidate)
    hs4_label = str(record.get("hs4_label", "")).strip()
    return bool(declared_hs and hs4_label and declared_hs.startswith(hs4_label))


def _apply_bol_variant(record: Dict[str, object], candidate: RawAuxiliaryRecord, score: float) -> None:
    metadata = candidate.metadata or {}
    declared_hs = _declared_hs(candidate)
    bol_metadata = {
        "shipper": normalize_text(str(metadata.get("shipper", ""))),
        "consignee": normalize_text(str(metadata.get("consignee", ""))),
        "port_origin": normalize_text(str(metadata.get("port_origin", ""))),
        "port_dest": normalize_text(str(metadata.get("port_dest", ""))),
        "arrival_date": str(metadata.get("arrival_date", "")).strip(),
        "bol_description": candidate.description,
        "declared_hs": declared_hs,
        "hs_verified": _bol_hs_verified(record, candidate),
    }
    record["bol_metadata"] = bol_metadata
    record["auxiliary_match_metadata"]["bol"] = {
        "reference": candidate.reference,
        "score": round(score, 4),
        "declared_hs": declared_hs,
        "hs_verified": bol_metadata["hs_verified"],
    }


def _apply_catalog_variant(record: Dict[str, object], candidate: RawAuxiliaryRecord, score: float) -> None:
    if candidate.mpn:
        record["tier2_minimal"] = {
            "part_name": candidate.mpn,
            "manufacturer": candidate.manufacturer or str((record.get("tier2_minimal") or {}).get("manufacturer", "")),
        }
        record["tier2_provenance"] = "natural_mpn"

    record["auxiliary_match_metadata"]["catalog"] = {
        "reference": candidate.reference,
        "score": round(score, 4),
        "provider": str((candidate.metadata or {}).get("provider", "")),
        "mpn": candidate.mpn,
        "confidence": str(((record.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("confidence", "")),
    }


def _catalog_block_reason(
    candidate: RawAuxiliaryRecord | None,
    score: float,
    min_score: float,
    hs4_gate_pass: bool,
    heading_mismatch: bool,
) -> tuple[str, str]:
    if not candidate:
        return "", ""
    reasons: List[str] = []
    details: List[str] = []
    if not hs4_gate_pass:
        reasons.append("target_hs4_mismatch")
        details.append("target_hs4 gate failed")
    if heading_mismatch:
        reasons.append("heading_mismatch")
        details.append("actual heading mismatched record heading")
    if score < min_score:
        reasons.append("below_score")
        details.append("score below threshold")
    if not reasons:
        return "", ""
    return "|".join(reasons), "; ".join(details)


def build_exact_mpn_review_rows(review_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in review_rows:
        if not bool(row.get("catalog_exact_mpn_match")):
            continue
        if bool(row.get("catalog_applied")):
            continue
        rows.append(
            {
                "id": row.get("id"),
                "canonical_id": row.get("canonical_id"),
                "hs4_label": row.get("hs4_label"),
                "hs6_label": row.get("hs6_label"),
                "tier1_description": row.get("tier1_description"),
                "catalog_candidate_reference": row.get("catalog_candidate_reference"),
                "catalog_candidate_provider": row.get("catalog_candidate_provider"),
                "catalog_candidate_mpn": row.get("catalog_candidate_mpn"),
                "catalog_candidate_manufacturer": row.get("catalog_candidate_manufacturer"),
                "catalog_candidate_description": row.get("catalog_candidate_description"),
                "catalog_candidate_product_url": row.get("catalog_candidate_product_url"),
                "catalog_score": row.get("catalog_score"),
                "catalog_confidence": row.get("catalog_confidence"),
                "catalog_hs4_gate_pass": row.get("catalog_hs4_gate_pass"),
                "catalog_heading_mismatch": row.get("catalog_heading_mismatch"),
                "catalog_target_hs4": row.get("catalog_target_hs4"),
                "catalog_actual_hs4": row.get("catalog_actual_hs4"),
                "catalog_block_reason": row.get("catalog_block_reason"),
                "catalog_block_reason_detail": row.get("catalog_block_reason_detail"),
                "reviewer_decision": "",
                "reviewer_note": "",
            }
        )
    return rows


def write_exact_mpn_review_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXACT_MPN_REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EXACT_MPN_REVIEW_FIELDS})


def enrich_records_with_auxiliary(
    records: Iterable[Mapping[str, object]],
    bol_candidates: Sequence[RawAuxiliaryRecord],
    catalog_candidates: Sequence[RawAuxiliaryRecord],
    *,
    bol_generic_terms: Sequence[str] | None = None,
    bol_min_score: float = 0.2,
    catalog_min_score: float = 0.3,
) -> Dict[str, object]:
    generic_terms = [term.lower() for term in (bol_generic_terms or DEFAULT_GENERIC_BOL_TERMS)]
    filtered_bol_candidates = [
        candidate
        for candidate in bol_candidates
        if candidate.description and is_specific_bol_record(candidate, generic_terms)
    ]
    filtered_catalog_candidates = [candidate for candidate in catalog_candidates if candidate.description or candidate.mpn]

    enriched_records: List[Dict[str, object]] = []
    review_rows: List[Dict[str, object]] = []

    for record in records:
        enriched = copy.deepcopy(dict(record))
        enriched["auxiliary_match_metadata"] = {}

        bol_candidate, bol_score, _ = best_match_details(enriched, filtered_bol_candidates)
        bol_applied = bool(
            bol_candidate
            and bol_score >= bol_min_score
            and _bol_hs_verified(enriched, bol_candidate)
        )
        if bol_candidate:
            enriched["auxiliary_match_metadata"]["bol_candidate"] = {
                "reference": bol_candidate.reference,
                "score": round(bol_score, 4),
                "hs_verified": _bol_hs_verified(enriched, bol_candidate),
            }
        if bol_applied:
            _apply_bol_variant(enriched, bol_candidate, bol_score)

        catalog_candidate, catalog_score, catalog_details = best_match_details(enriched, filtered_catalog_candidates)
        catalog_heading_mismatch = bool(catalog_details.get("heading_mismatch", False))
        catalog_hs4_gate_pass = bool(catalog_details.get("hs4_gate_pass", True))
        catalog_applied = bool(
            catalog_candidate
            and catalog_score >= catalog_min_score
            and catalog_hs4_gate_pass
            and not catalog_heading_mismatch
        )
        catalog_block_reason, catalog_block_reason_detail = _catalog_block_reason(
            catalog_candidate,
            catalog_score,
            catalog_min_score,
            catalog_hs4_gate_pass,
            catalog_heading_mismatch,
        )
        if catalog_candidate:
            enriched["auxiliary_match_metadata"]["catalog_candidate"] = {
                "reference": catalog_candidate.reference,
                "score": round(catalog_score, 4),
                "confidence": catalog_details.get("confidence"),
                "token_overlap": catalog_details.get("token_overlap"),
                "exact_mpn_match": catalog_details.get("exact_mpn_match"),
                "manufacturer_match": catalog_details.get("manufacturer_match"),
                "hs4_gate_pass": catalog_hs4_gate_pass,
                "heading_mismatch": catalog_heading_mismatch,
                "target_hs4": catalog_details.get("target_hs4"),
                "actual_hs4": catalog_details.get("actual_hs4"),
                "provider": str((catalog_candidate.metadata or {}).get("provider", "")),
                "mpn": catalog_candidate.mpn,
                "manufacturer": catalog_candidate.manufacturer,
                "description": catalog_candidate.description,
                "product_url": str((catalog_candidate.metadata or {}).get("product_url", "")),
                "block_reason": catalog_block_reason,
                "block_reason_detail": catalog_block_reason_detail,
            }
        if catalog_applied:
            _apply_catalog_variant(enriched, catalog_candidate, catalog_score)

        review_rows.append(
            {
                "id": enriched.get("id"),
                "canonical_id": enriched.get("canonical_id"),
                "hs4_label": enriched.get("hs4_label"),
                "hs6_label": enriched.get("hs6_label"),
                "tier1_description": enriched.get("tier1_description"),
                "tier2_provenance": enriched.get("tier2_provenance"),
                "bol_reference": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("bol") or {}).get("reference")
                ),
                "catalog_reference": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog") or {}).get("reference")
                ),
                "catalog_candidate_reference": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("reference")
                ),
                "bol_score": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("bol_candidate") or {}).get("score")
                ),
                "catalog_score": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("score")
                ),
                "catalog_confidence": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("confidence")
                ),
                "catalog_exact_mpn_match": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("exact_mpn_match")
                ),
                "catalog_manufacturer_match": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("manufacturer_match")
                ),
                "catalog_hs4_gate_pass": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("hs4_gate_pass")
                ),
                "catalog_heading_mismatch": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("heading_mismatch")
                ),
                "catalog_target_hs4": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("target_hs4")
                ),
                "catalog_actual_hs4": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("actual_hs4")
                ),
                "catalog_candidate_provider": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("provider")
                ),
                "catalog_candidate_mpn": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("mpn")
                ),
                "catalog_candidate_manufacturer": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("manufacturer")
                ),
                "catalog_candidate_description": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("description")
                ),
                "catalog_candidate_product_url": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("product_url")
                ),
                "catalog_block_reason": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("block_reason")
                ),
                "catalog_block_reason_detail": (
                    ((enriched.get("auxiliary_match_metadata") or {}).get("catalog_candidate") or {}).get("block_reason_detail")
                ),
                "bol_applied": bol_applied,
                "catalog_applied": catalog_applied,
            }
        )
        enriched_records.append(enriched)

    exact_mpn_review_rows = build_exact_mpn_review_rows(review_rows)
    return {
        "records": enriched_records,
        "review_rows": review_rows,
        "exact_mpn_review_rows": exact_mpn_review_rows,
        "summary": {
            "records_considered": len(enriched_records),
            "bol_candidates": len(filtered_bol_candidates),
            "catalog_candidates": len(filtered_catalog_candidates),
            "tier2_natural_mpn": sum(1 for record in enriched_records if record.get("tier2_provenance") == "natural_mpn"),
            "exact_mpn_review_rows": len(exact_mpn_review_rows),
        },
    }
