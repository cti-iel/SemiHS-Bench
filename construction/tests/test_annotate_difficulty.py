"""Tests for scripts/annotate_difficulty.py.

Exercises the in-place refresh of difficulty_tags / boundary_note on a
synthetic data file: old-tag migration, expert-tag preservation, note
composition, sibling-field safety, idempotency, and the canonical
serialization format.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT_PATH = ROOT / "scripts" / "annotate_difficulty.py"
_spec = importlib.util.spec_from_file_location("annotate_difficulty", _SCRIPT_PATH)
annotate_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(annotate_mod)


def _record(**overrides) -> dict:
    base = {
        "id": "SH-0001",
        "hs6_label": "854231",
        "hs4_label": "8542",
        "tier1_description": "Monolithic integrated circuit.",
        "justification_text": "",
        "tier2_classifiable": "partial",
        "difficulty_tags": [],
        "candidate_set": {"codes": ["854231", "854232", "854239", "854110"]},
    }
    base.update(overrides)
    return base


class AnnotateDifficultyTests(unittest.TestCase):
    def _write(self, records) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "data.json"
        tmp.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        return tmp

    def test_old_tag_migration(self) -> None:
        rec = _record(difficulty_tags=["8542.31_vs_8542.32_vs_8542.33_vs_8542.39"])
        path = self._write([rec])
        annotate_mod.main(["--data", str(path)])
        out = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertIn("8542_ic_function", out["difficulty_tags"])
        self.assertNotIn("8542.31_vs_8542.32_vs_8542.33_vs_8542.39", out["difficulty_tags"])

    def test_expert_tag_preserved_when_detector_silent(self) -> None:
        # A tag the slate/keyword rules would not produce on their own is kept.
        rec = _record(
            hs6_label="280410",
            hs4_label="2804",
            tier1_description="Hydrogen gas.",
            candidate_set={"codes": ["280410", "999999", "888888", "777777"]},
            difficulty_tags=["sensor_boundary"],
        )
        path = self._write([rec])
        annotate_mod.main(["--data", str(path)])
        out = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertIn("sensor_boundary", out["difficulty_tags"])

    def test_note_set_and_empty(self) -> None:
        tagged = _record()
        untagged = _record(
            id="SH-0002",
            hs6_label="280530",
            hs4_label="2805",
            candidate_set={"codes": ["280530", "999999", "888888", "777777"]},
        )
        path = self._write([tagged, untagged])
        annotate_mod.main(["--data", str(path)])
        out = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(out[0]["difficulty_tags"])
        self.assertTrue(out[0]["boundary_note"])
        self.assertEqual(out[1]["difficulty_tags"], [])
        self.assertEqual(out[1]["boundary_note"], "")

    def test_sibling_fields_untouched(self) -> None:
        rec = _record(tier2_classifiable="yes")
        path = self._write([rec])
        annotate_mod.main(["--data", str(path)])
        out = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertEqual(out["tier2_classifiable"], "yes")
        self.assertEqual(out["hs6_label"], "854231")

    def test_idempotent_and_canonical_format(self) -> None:
        path = self._write([_record(), _record(id="SH-0002")])
        annotate_mod.main(["--data", str(path)])
        first = path.read_text(encoding="utf-8")
        annotate_mod.main(["--data", str(path)])
        second = path.read_text(encoding="utf-8")
        self.assertEqual(first, second)
        # Canonical: indent 2, sorted keys, trailing newline, no ASCII escaping.
        records = json.loads(first)
        expected = json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        self.assertEqual(first, expected)

    def test_check_mode_exit_code(self) -> None:
        path = self._write([_record()])
        # First pass: stale (no boundary_note yet) -> exit 2.
        self.assertEqual(annotate_mod.main(["--data", str(path), "--check"]), 2)
        # Write, then check is clean -> exit 0.
        annotate_mod.main(["--data", str(path)])
        self.assertEqual(annotate_mod.main(["--data", str(path), "--check"]), 0)


if __name__ == "__main__":
    unittest.main()
