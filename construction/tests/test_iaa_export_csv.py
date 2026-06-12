"""Tests for scripts/iaa_export_csv.py.

Builds a tiny synthetic pool plus an HS6-description CSV in a temp dir, runs
the export, and asserts on rater-B label computation, tier2 pass-through, the
group strata, and the manifest shape.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT_PATH = ROOT / "scripts" / "iaa_export_csv.py"
_spec = importlib.util.spec_from_file_location("iaa_export_csv", _SCRIPT_PATH)
iaa_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(iaa_mod)


def _record(idx, *, hs6, hs4, tags, codes, tier2_classifiable="partial", desc="A device."):
    return {
        "id": f"SH-{idx:04d}",
        "frozen_id": f"v2.0.eval.{idx:04d}",
        "hs6_label": hs6,
        "hs4_label": hs4,
        "tier1_description": desc,
        "justification_text": "",
        "tier2_minimal": {"part_name": "MCU 3.3 V", "manufacturer": "Acme"},
        "tier2_classifiable": tier2_classifiable,
        "difficulty_tags": tags,
        "candidate_set": {"codes": codes},
    }


class IaaExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        # A pool spanning all three strata.
        records = []
        for i in range(1, 31):  # cross_family
            records.append(_record(i, hs6="854110", hs4="8541", tags=["8541_vs_8542"],
                                    codes=["854110", "854231", "854121", "854129"]))
        for i in range(31, 61):  # sibling_split only
            records.append(_record(i, hs6="854231", hs4="8542", tags=["8542_ic_function"],
                                    codes=["854231", "854232", "854239", "854233"]))
        for i in range(61, 91):  # control
            records.append(_record(i, hs6="280530", hs4="2805", tags=[],
                                    codes=["280530", "999999", "888888", "777777"]))
        self.data_path = self.tmp / "pool.json"
        self.data_path.write_text(json.dumps(records), encoding="utf-8")
        self.tax_path = self.tmp / "hs6_descriptions.csv"
        with self.tax_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hs6", "description"])
            for code, d in [
                ("854110", "Diodes"), ("854231", "Processors and controllers"),
                ("854232", "Memories"), ("854233", "Amplifiers"), ("854239", "Other ICs"),
                ("854121", "Transistors low power"), ("854129", "Transistors other"),
                ("280530", "Rare earth metals"),
            ]:
                w.writerow([code, d])

    def _run(self):
        in_csv = self.tmp / "iaa_input.csv"
        rb_csv = self.tmp / "iaa_rater_b.csv"
        manifest = self.tmp / "manifest.json"
        iaa_mod.main([
            "--input", str(self.data_path),
            "--taxonomy", str(self.tax_path),
            "--input-csv", str(in_csv),
            "--rater-b-csv", str(rb_csv),
            "--manifest", str(manifest),
            "--seed", "17",
        ])
        return in_csv, rb_csv, manifest

    def test_strata_and_manifest_shape(self) -> None:
        _, _, manifest_path = self._run()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["sample_size"], 90)  # 30 + 30 + 30
        names = {s["stratum"]: s for s in manifest["strata"]}
        self.assertEqual(set(names), {"cross_family", "sibling_split", "control_no_boundary"})
        self.assertIn("cross_family_tag_coverage", manifest)

    def test_rater_b_labels_computed(self) -> None:
        _, rb_csv, _ = self._run()
        rows = list(csv.DictReader(rb_csv.open(encoding="utf-8")))
        self.assertTrue(rows)
        for row in rows:
            # Every rater-B row has the pipeline labels filled in.
            self.assertNotEqual(row["ambiguity_score"], "")
            self.assertNotEqual(row["classification_driver"], "")
            # tier2_classifiable passed through from the record (never re-inferred).
            self.assertIn(row["tier2_classifiable"], ("yes", "partial", "no"))
        # A cross_family record carries its difficulty_tags as boundary_tags.
        cf = [r for r in rows if r["hs4_label_hint"] == "8541"]
        self.assertTrue(cf)
        self.assertIn("8541_vs_8542", cf[0]["boundary_tags"])

    def test_input_csv_is_blank(self) -> None:
        in_csv, _, _ = self._run()
        rows = list(csv.DictReader(in_csv.open(encoding="utf-8")))
        for row in rows:
            self.assertEqual(row["ambiguity_score"], "")
            self.assertEqual(row["boundary_tags"], "")
            self.assertEqual(row["classification_driver"], "")
            self.assertEqual(row["tier2_classifiable"], "")
            self.assertTrue(row["frozen_id"])


if __name__ == "__main__":
    unittest.main()
