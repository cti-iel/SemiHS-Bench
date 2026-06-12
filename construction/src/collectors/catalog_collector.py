"""Catalog importers for natural sparse-input (Tier 2) variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.models import RawAuxiliaryRecord
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.text_utils import normalize_text


DISALLOWED_LIFECYCLE_STATUSES = {"obsolete", "discontinued"}
CATALOG_EXPORT_FIELDS = {
    "manufacturer",
    "manufacturer_part_number",
    "supplier_part_number",
    "description_short",
    "category_path",
    "key_specs",
    "target_hs_family",
}


def _normalize_hs_code(value: Any) -> str:
    return "".join(character for character in str(value or "") if character.isdigit())


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = normalize_text(str(value or ""))
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_bool(value: Any) -> bool:
    return normalize_text(str(value or "")).lower() in {"1", "true", "yes", "y"}


def _parse_int(value: Any) -> int | None:
    text = normalize_text(str(value or ""))
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_category_path(value: Any) -> str:
    parsed = _parse_json_value(value)
    if isinstance(parsed, list):
        parts = [normalize_text(str(item)) for item in parsed if normalize_text(str(item))]
        return " > ".join(parts)
    return normalize_text(str(value or ""))


def _parse_specs(payload: Dict[str, Any]) -> Dict[str, str]:
    specs: Dict[str, str] = {}

    key_specs = _parse_json_value(payload.get("key_specs"))
    if isinstance(key_specs, dict):
        for key, value in key_specs.items():
            key_text = normalize_text(str(key))
            value_text = normalize_text(str(value))
            if key_text and value_text:
                specs[key_text] = value_text

    raw_parameters = _parse_json_value(payload.get("raw_parameters"))
    if isinstance(raw_parameters, list):
        for parameter in raw_parameters:
            if not isinstance(parameter, dict):
                continue
            key_text = normalize_text(str(parameter.get("name") or ""))
            value_text = normalize_text(str(parameter.get("value") or ""))
            if key_text and value_text and key_text not in specs:
                specs[key_text] = value_text

    return specs


def _catalog_snapshot_path(path: str) -> Path | None:
    file_path = Path(path)
    if "catalogs" in file_path.parts:
        return file_path.with_name(file_path.stem + "_normalized.jsonl")
    if "imports" in file_path.parts and "catalog" in file_path.parts:
        return Path("data/raw/catalogs") / (file_path.stem + "_normalized.jsonl")
    return None


def _looks_like_normalized_catalog(payload: Dict[str, Any]) -> bool:
    return {"source", "reference", "description", "metadata"}.issubset(payload)


def _looks_like_catalog_export(payload: Dict[str, Any]) -> bool:
    return CATALOG_EXPORT_FIELDS.issubset(payload)


def _normalize_existing_catalog_payload(payload: Dict[str, Any]) -> RawAuxiliaryRecord:
    metadata = payload.get("metadata")
    return RawAuxiliaryRecord(
        source=normalize_text(str(payload.get("source") or "catalog")) or "catalog",
        reference=normalize_text(str(payload.get("reference") or "")),
        description=normalize_text(str(payload.get("description") or "")),
        manufacturer=normalize_text(str(payload.get("manufacturer") or "")),
        mpn=normalize_text(str(payload.get("mpn") or "")),
        hs_code=_normalize_hs_code(payload.get("hs_code")),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _normalize_generic_catalog_payload(payload: Dict[str, Any]) -> RawAuxiliaryRecord:
    metadata = {
        "provider": payload.get("provider", ""),
        "category_path": payload.get("category_path", ""),
        "specs": payload.get("specs", {}),
        "short_description": payload.get("short_description") or payload.get("description") or "",
    }
    return RawAuxiliaryRecord(
        source="catalog",
        reference=str(payload.get("reference") or payload.get("mpn") or "").strip(),
        description=normalize_text(str(payload.get("short_description") or payload.get("description") or "")),
        manufacturer=normalize_text(str(payload.get("manufacturer") or "")),
        mpn=normalize_text(str(payload.get("mpn") or "")),
        metadata=metadata,
    )


def _normalize_catalog_export_payload(payload: Dict[str, Any]) -> RawAuxiliaryRecord:
    description_short = normalize_text(str(payload.get("description_short") or ""))
    description_detailed = normalize_text(str(payload.get("description_detailed") or ""))
    category_path = _parse_category_path(payload.get("category_path"))
    metadata = {
        "provider": "catalog",
        "record_id": normalize_text(str(payload.get("record_id") or "")),
        "base_product_number": normalize_text(str(payload.get("base_product_number") or "")),
        "short_description": description_short,
        "detailed_description": description_detailed,
        "category_path": category_path,
        "supplier_category_id_path": _parse_json_value(payload.get("supplier_category_id_path")) or [],
        "specs": _parse_specs(payload),
        "classifications": _parse_json_value(payload.get("classifications")) or {},
        "product_url": normalize_text(str(payload.get("product_url") or "")),
        "datasheet_url": normalize_text(str(payload.get("datasheet_url") or "")),
        "photo_url": normalize_text(str(payload.get("photo_url") or "")),
        "target_hs_family": _normalize_hs_code(payload.get("target_hs_family"))[:4],
        "hs_heading": _normalize_hs_code(payload.get("hs_heading"))[:4],
        "hs_code": _normalize_hs_code(payload.get("hs_code")),
        "intended_bucket": normalize_text(str(payload.get("intended_bucket") or "")),
        "intended_hs_prefixes": _parse_json_value(payload.get("intended_hs_prefixes")) or [],
        "source_queries": _parse_json_value(payload.get("source_queries")) or [],
        "source_keywords": _parse_json_value(payload.get("source_keywords")) or [],
        "is_off_target": _parse_bool(payload.get("is_off_target")),
        "is_boundary_case": _parse_bool(payload.get("is_boundary_case")),
        "retrieval_rank": _parse_int(payload.get("retrieval_rank")),
        "quantity_available": _parse_int(payload.get("quantity_available")),
        "pricing_snapshot": _parse_json_value(payload.get("pricing_snapshot")) or {},
        "lifecycle_status": normalize_text(str(payload.get("lifecycle_status") or "")),
    }
    return RawAuxiliaryRecord(
        source="catalog",
        reference=normalize_text(str(payload.get("supplier_part_number") or payload.get("reference") or "")),
        description=description_short or description_detailed,
        manufacturer=normalize_text(str(payload.get("manufacturer") or "")),
        mpn=normalize_text(str(payload.get("manufacturer_part_number") or payload.get("mpn") or "")),
        hs_code=_normalize_hs_code(payload.get("hs_code")),
        metadata=metadata,
    )


def normalize_catalog_payload(payload: Dict[str, Any]) -> RawAuxiliaryRecord:
    if _looks_like_normalized_catalog(payload):
        return _normalize_existing_catalog_payload(payload)
    if _looks_like_catalog_export(payload):
        return _normalize_catalog_export_payload(payload)
    return _normalize_generic_catalog_payload(payload)


def _keep_catalog_record(record: RawAuxiliaryRecord) -> bool:
    lifecycle_status = normalize_text(str((record.metadata or {}).get("lifecycle_status") or "")).lower()
    if bool((record.metadata or {}).get("is_off_target")):
        return False
    if lifecycle_status in DISALLOWED_LIFECYCLE_STATUSES:
        return False
    return bool(record.mpn and record.description)


def _catalog_record_sort_key(record: RawAuxiliaryRecord) -> tuple[int, int, int, str]:
    rank = (record.metadata or {}).get("retrieval_rank")
    rank_value = rank if isinstance(rank, int) else 10**9
    return (
        rank_value,
        -len(record.description),
        -len(record.mpn),
        record.reference,
    )


def _deduplicate_catalog_records(records: Iterable[RawAuxiliaryRecord]) -> List[RawAuxiliaryRecord]:
    deduped: Dict[str, RawAuxiliaryRecord] = {}
    for record in records:
        key = record.mpn.lower()
        existing = deduped.get(key)
        if existing is None or _catalog_record_sort_key(record) < _catalog_record_sort_key(existing):
            deduped[key] = record
    return list(deduped.values())


def _normalize_catalog_rows(rows: Iterable[Dict[str, Any]], *, is_catalog_export: bool) -> List[RawAuxiliaryRecord]:
    normalized = [normalize_catalog_payload(row) for row in rows]
    if not is_catalog_export:
        return normalized
    filtered = [record for record in normalized if _keep_catalog_record(record)]
    return _deduplicate_catalog_records(filtered)


def load_catalog_imports(path: str) -> List[RawAuxiliaryRecord]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(path)
    elif suffix == ".json":
        rows = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("records", [])
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    is_catalog_export = bool(rows) and isinstance(rows[0], dict) and _looks_like_catalog_export(rows[0])
    normalized_rows = _normalize_catalog_rows(rows, is_catalog_export=is_catalog_export)

    snapshot_path = _catalog_snapshot_path(path) if is_catalog_export else None
    if snapshot_path is not None:
        write_jsonl(str(snapshot_path), [record.to_dict() for record in normalized_rows])

    return normalized_rows
