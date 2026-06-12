"""BOL importers and quality filters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.models import RawAuxiliaryRecord
from src.utils.io_utils import read_jsonl
from src.utils.text_utils import clean_bol_description, normalize_text, substantive_word_count


def _first(payload: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def normalize_bol_payload(payload: Dict[str, Any]) -> RawAuxiliaryRecord:
    shipper = _first(payload, "shipper", "supplier_t", "supplier_name")
    consignee = _first(payload, "consignee", "buyer_t", "buyer_name")
    metadata = {
        "shipper": shipper,
        "consignee": consignee,
        "port_origin": _first(payload, "port_origin", "orig_port"),
        "port_dest": _first(payload, "port_dest", "dest_port"),
        "quantity": _first(payload, "quantity"),
        "quantity_unit": _first(payload, "quantity_unit"),
        "arrival_date": _first(payload, "arrival_date", "date_of_arrival", "date"),
        "origin_country": _first(payload, "origin_country", "orig_country"),
        "declared_hs": _first(payload, "declared_hs", "hs_code").replace(".", ""),
    }
    return RawAuxiliaryRecord(
        source="BOL",
        reference=_first(payload, "reference", "bol_reference", "master_bill_no", "sub_bill_no", "id").strip(),
        description=clean_bol_description(_first(payload, "description", "bol_description", "prod_desc")),
        manufacturer=normalize_text(_first(payload, "manufacturer", "shipper", "supplier_t")),
        metadata=metadata,
    )


def load_bol_imports(path: str) -> List[RawAuxiliaryRecord]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(path)
    elif suffix == ".json":
        rows = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("records", [])
    else:
        with open(path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    return [normalize_bol_payload(row) for row in rows]


def is_specific_bol_record(record: RawAuxiliaryRecord, generic_terms: List[str]) -> bool:
    if substantive_word_count(record.description) < 3:
        return False
    lowered = record.description.lower()
    return not any(term in lowered for term in generic_terms)

