"""Unit tests for the browser-capture normalizer:
scripts/normalize_jp_customs.py

The normalizer is loaded as a module from disk (it lives under scripts/,
not a package). The tests run end-to-end against the JSONL fixture in
tests/fixtures/raw/jp_customs/sample_browser_captures.jsonl, asserting that:

  - in-scope records survive HS6 allowlist filtering
  - out-of-scope records (e.g. HS 8504.40) are dropped
  - records missing an HS code are dropped
  - source / jurisdiction / language fields are set per the design
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures" / "raw"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_jp = _load_script(ROOT / "scripts" / "normalize_jp_customs.py", "normalize_jp_customs")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class JPCustomsNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = FIXTURES / "jp_customs" / "sample_browser_captures.jsonl"
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "new_pulls.jsonl"
        self.report = Path(self.tmp.name) / "_report.json"

    def test_filters_to_hs6_scope_and_drops_missing_codes(self):
        rc = _jp.main([
            "--capture-path", str(self.fixture),
            "--output-path", str(self.out),
            "--report-path", str(self.report),
        ])
        self.assertEqual(rc, 0)
        records = _read_jsonl(self.out)
        # Fixture has 5 rows: 3 in-scope (854110, 854232, 848620),
        # 1 out-of-scope PCB (853400), 1 missing HS6.
        self.assertEqual(len(records), 3)
        hs6s = sorted(r["hs6_label"] for r in records)
        self.assertEqual(hs6s, ["848620", "854110", "854232"])

    def test_records_have_required_corpus_input_fields(self):
        _jp.main([
            "--capture-path", str(self.fixture),
            "--output-path", str(self.out),
            "--report-path", str(self.report),
        ])
        records = _read_jsonl(self.out)
        for rec in records:
            self.assertEqual(rec["source"], "JP_CUSTOMS")
            self.assertEqual(rec["jurisdiction"], "JP")
            self.assertEqual(rec["language"], "ja")
            self.assertTrue(rec["evidence_id_hint"])
            self.assertTrue(rec["hs6_label"])
            self.assertTrue(rec["tier1_text_seed"])
            # rationale_excerpt is allowed to be empty but present:
            self.assertIn("rationale_excerpt", rec)
            self.assertIn("raw_metadata", rec)
            self.assertEqual(
                rec["raw_metadata"]["source_release"],
                "jp_customs_browser_capture",
            )

    def test_no_filter_passes_out_of_scope_through(self):
        _jp.main([
            "--capture-path", str(self.fixture),
            "--output-path", str(self.out),
            "--report-path", str(self.report),
            "--no-filter",
        ])
        records = _read_jsonl(self.out)
        # Still drops the row missing an HS code → 4 records.
        self.assertEqual(len(records), 4)
        hs6s = sorted(r["hs6_label"] for r in records)
        self.assertEqual(hs6s, ["848620", "853400", "854110", "854232"])


class AuthoritativeSourcesRegistrationTests(unittest.TestCase):
    """Guards the registered Tier-1 authoritative ruling sources. The
    reference corpus draws on exactly EBTI, CROSS, and JP_CUSTOMS; a
    regression here would silently demote a source to Tier 2/3 in
    downstream consumers (degrader, candidate selection)."""

    def test_corpus_sources_are_authoritative(self):
        from src.models import AUTHORITATIVE_SOURCES
        self.assertEqual(AUTHORITATIVE_SOURCES, {"EBTI", "CROSS", "JP_CUSTOMS"})


class CorpusBuilderEvidenceIDTests(unittest.TestCase):
    """End-to-end coverage that the corpus builder mints valid evidence IDs
    and jurisdictions for Japan Customs rulings without raising ValueError
    (the original behaviour for unknown sources)."""

    def test_stable_evidence_id_handles_jp_customs(self):
        builder = _load_script(
            ROOT / "scripts" / "build_reference_corpus.py",
            "build_reference_corpus",
        )
        eid = builder._stable_evidence_id({
            "source": "JP_CUSTOMS",
            "evidence_id_hint": "123003574",
        })
        self.assertEqual(eid, "JP_CUSTOMS-123003574")

    def test_resolve_jurisdiction_handles_jp_customs(self):
        builder = _load_script(
            ROOT / "scripts" / "build_reference_corpus.py",
            "build_reference_corpus_juris",
        )
        # Inputs already carry jurisdiction (passthrough).
        self.assertEqual(
            builder._resolve_jurisdiction({"source": "JP_CUSTOMS", "jurisdiction": "JP"}),
            "JP",
        )
        # Inputs missing jurisdiction (fallback by source).
        self.assertEqual(
            builder._resolve_jurisdiction({"source": "JP_CUSTOMS"}),
            "JP",
        )


if __name__ == "__main__":
    unittest.main()
