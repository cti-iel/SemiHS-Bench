"""Tests for the manufacturer_hint backfill."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processing.manufacturer_backfill import (  # noqa: E402
    BACKFILL_SOURCE_TIER2,
    backfill_manufacturer_hint,
)


def _record(*, tier2_mfg: str = "", existing_hint: str = "") -> dict:
    sm = {}
    if existing_hint:
        sm["manufacturer_hint"] = existing_hint
    return {
        "id": "SH-0001",
        "tier2_minimal": {"part_name": "P1", "manufacturer": tier2_mfg},
        "source_metadata": sm,
    }


class BackfillTests(unittest.TestCase):
    def test_fills_from_tier2(self) -> None:
        out = backfill_manufacturer_hint([_record(tier2_mfg="ACME Corp")])
        self.assertEqual(out.backfilled_count, 1)
        sm = out.records[0]["source_metadata"]
        self.assertEqual(sm["manufacturer_hint"], "ACME Corp")
        self.assertEqual(sm["manufacturer_hint_source"], BACKFILL_SOURCE_TIER2)

    def test_does_not_overwrite_existing_hint(self) -> None:
        out = backfill_manufacturer_hint([
            _record(tier2_mfg="From Tier2", existing_hint="Authoritative"),
        ])
        self.assertEqual(out.backfilled_count, 0)
        self.assertEqual(out.skipped_already_set, 1)
        sm = out.records[0]["source_metadata"]
        self.assertEqual(sm["manufacturer_hint"], "Authoritative")
        self.assertNotIn("manufacturer_hint_source", sm)

    def test_untouched_when_no_fallback(self) -> None:
        out = backfill_manufacturer_hint([_record(tier2_mfg="")])
        self.assertEqual(out.backfilled_count, 0)
        self.assertEqual(out.untouched_count, 1)
        self.assertNotIn("manufacturer_hint", out.records[0]["source_metadata"])

    def test_strips_whitespace(self) -> None:
        out = backfill_manufacturer_hint([_record(tier2_mfg="  ACME  ")])
        self.assertEqual(out.records[0]["source_metadata"]["manufacturer_hint"], "ACME")

    def test_does_not_mutate_input(self) -> None:
        original = _record(tier2_mfg="X")
        backfill_manufacturer_hint([original])
        self.assertNotIn("manufacturer_hint", original["source_metadata"])

    def test_aggregate_counts(self) -> None:
        out = backfill_manufacturer_hint([
            _record(tier2_mfg="A"),                                  # backfill
            _record(tier2_mfg="B", existing_hint="kept"),            # already set
            _record(tier2_mfg=""),                                    # untouched
            _record(tier2_mfg="C"),                                  # backfill
        ])
        self.assertEqual(out.backfilled_count, 2)
        self.assertEqual(out.skipped_already_set, 1)
        self.assertEqual(out.untouched_count, 1)


if __name__ == "__main__":
    unittest.main()
