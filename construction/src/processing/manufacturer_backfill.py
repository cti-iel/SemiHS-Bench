"""Backfill ``source_metadata.manufacturer_hint`` from in-record fallbacks.

Honest scope: of the 580 records currently missing a ``manufacturer_hint``,
523 are EBTI rulings (which anonymize the importer/holder by EU policy) and
56 are CROSS rulings (which only mention companies in prose). Public
authoritative customs data does not carry a structured manufacturer field
for these records — that's a property of the source, not a pipeline gap.

This module backfills only the records where a *structured* manufacturer
exists somewhere on the record but didn't make it into ``manufacturer_hint``:

* ``tier2_minimal.manufacturer`` (when non-empty) — the most reliable
  fallback. a small number of records.

A backfilled value carries the provenance tag
``manufacturer_hint_source: "backfilled_from_tier2"`` so consumers can
distinguish backfilled hints from authoritative ones.

Records without any structured fallback are left untouched. The audit
report flags the broader gap; the data-quality doc explains why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


BACKFILL_SOURCE_TIER2 = "backfilled_from_tier2"


@dataclass
class BackfillOutcome:
    records: List[Dict[str, Any]]
    backfilled_count: int
    untouched_count: int
    skipped_already_set: int


def _existing_hint(record: Mapping[str, Any]) -> str:
    sm = record.get("source_metadata") or {}
    return str(sm.get("manufacturer_hint") or "").strip()


def _tier2_manufacturer(record: Mapping[str, Any]) -> Optional[str]:
    tier2 = record.get("tier2_minimal") or {}
    manufacturer = tier2.get("manufacturer")
    if isinstance(manufacturer, str) and manufacturer.strip():
        return manufacturer.strip()
    return None


def backfill_manufacturer_hint(
    records: Sequence[Mapping[str, Any]],
) -> BackfillOutcome:
    """Return a copy of ``records`` with ``manufacturer_hint`` filled where
    a structured fallback is available. Records are not mutated in place."""
    out: List[Dict[str, Any]] = []
    backfilled = 0
    skipped_already_set = 0
    untouched = 0

    for record in records:
        new_record = dict(record)
        sm = dict(new_record.get("source_metadata") or {})

        if _existing_hint(record):
            skipped_already_set += 1
            new_record["source_metadata"] = sm
            out.append(new_record)
            continue

        candidate = _tier2_manufacturer(record)
        if candidate is None:
            untouched += 1
            new_record["source_metadata"] = sm
            out.append(new_record)
            continue

        sm["manufacturer_hint"] = candidate
        sm["manufacturer_hint_source"] = BACKFILL_SOURCE_TIER2
        new_record["source_metadata"] = sm
        out.append(new_record)
        backfilled += 1

    return BackfillOutcome(
        records=out,
        backfilled_count=backfilled,
        untouched_count=untouched,
        skipped_already_set=skipped_already_set,
    )
