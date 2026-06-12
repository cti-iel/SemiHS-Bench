"""Unit tests for scripts/ingest_bol.py.

Each filter step in the pipeline is exercised by at least one fixture
row that should trip it; one composite end-to-end test asserts the
per-step counters and survivor count.
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

FIXTURES = ROOT / "tests" / "fixtures"
INTAKE_FIXTURE = FIXTURES / "raw" / "bol" / "bol_intake_minimal.csv"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ingest = _load_script(ROOT / "scripts" / "ingest_bol.py", "ingest_bol")


def _row(id_: str = "X", **overrides: str) -> Dict[str, str]:
    base = {
        "id": id_,
        "prod_desc": "Photoresist developer TMAH 2.38% for lithography",
        "hs_code": "37071000",
        "supplier_t": "Tokyo Ohka Kogyo",
        "buyer_t": "TSMC Arizona",
        "quantity": "500",
        "orig_country": "JP",
        "target_hs_family": "3707",
    }
    base.update(overrides)
    return base


class FilterUnitTests(unittest.TestCase):
    """Per-step unit tests for _apply_filters."""

    @classmethod
    def setUpClass(cls) -> None:
        allow_hs4, allow_hs6, scope_tier = _ingest._load_scope(
            ROOT / "configs" / "hs6_scope_tiers.yaml"
        )
        cls.allow_hs4 = allow_hs4
        cls.allow_hs6 = allow_hs6
        cls.scope_tier = scope_tier

    def _filter(self, rows: List[Dict[str, str]]):
        from src.processing.auxiliary_enricher import DEFAULT_GENERIC_BOL_TERMS

        return _ingest._apply_filters(
            [("test.csv", r) for r in rows],
            allow_hs4=self.allow_hs4,
            generic_terms=DEFAULT_GENERIC_BOL_TERMS,
            min_desc_chars=20,
            quantity_max=10000,
        )

    def test_short_description_drops_below_threshold(self) -> None:
        survivors, steps, dropped = self._filter([_row(id_="A", prod_desc="too short")])
        self.assertEqual(len(survivors), 0)
        self.assertEqual(steps["short_description"]["dropped"], 1)
        self.assertEqual(dropped[0]["disposition"], "short_description")

    def test_quantity_cap_drops_above_max(self) -> None:
        survivors, steps, dropped = self._filter(
            [_row(id_="A", quantity="99999999")]
        )
        self.assertEqual(len(survivors), 0)
        self.assertEqual(steps["quantity_cap"]["dropped"], 1)
        self.assertEqual(dropped[0]["disposition"], "quantity_cap")

    def test_freight_forwarder_drops_forwarder_consignee(self) -> None:
        survivors, steps, dropped = self._filter(
            [_row(id_="A", buyer_t="DHL Global Forwarding GmbH")]
        )
        self.assertEqual(len(survivors), 0)
        self.assertEqual(steps["freight_forwarder"]["dropped"], 1)

    def test_off_scope_hs_drops_unknown_hs4(self) -> None:
        survivors, steps, dropped = self._filter(
            [_row(id_="A", hs_code="85076000")]  # battery — not in scope
        )
        self.assertEqual(len(survivors), 0)
        self.assertEqual(steps["off_scope_hs"]["dropped"], 1)

    def test_generic_terms_drops_blocklist_match(self) -> None:
        survivors, steps, dropped = self._filter(
            [_row(id_="A",
                  prod_desc="electronic parts assorted from manufacturer",
                  hs_code="85423100")]
        )
        self.assertEqual(len(survivors), 0)
        self.assertEqual(steps["generic_terms"]["dropped"], 1)

    def test_near_duplicate_drops_one_keeps_longer(self) -> None:
        rows = [
            _row(id_="A",
                 prod_desc="Bare silicon wafer 300mm prime grade Czochralski",
                 hs_code="38180000"),
            _row(id_="B",
                 prod_desc="Bare silicon wafer 300mm prime grade Czochralski grown",
                 hs_code="38180000"),
        ]
        survivors, steps, dropped = self._filter(rows)
        self.assertEqual(len(survivors), 1)
        self.assertEqual(steps["near_duplicate"]["dropped"], 1)
        # Keeper is the longer description.
        _, _, keeper_aux = survivors[0]
        self.assertIn("grown", keeper_aux.description)

    def test_valid_record_passes_all_steps(self) -> None:
        survivors, steps, dropped = self._filter([_row(id_="A")])
        self.assertEqual(len(survivors), 1)
        self.assertEqual(dropped, [])
        for step_name in _ingest.STEP_ORDER:
            self.assertEqual(
                steps[step_name]["dropped"], 0,
                f"step {step_name} unexpectedly dropped the valid row",
            )


class EndToEndIngestTest(unittest.TestCase):
    """Run main() against the minimal intake fixture in a tempdir."""

    def test_full_pipeline_with_minimal_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pool_out = tmp_path / "_candidate_pool_bol.jsonl"
            report_out = tmp_path / "_bol_ingest_report.json"

            rc = _ingest.main([
                "--intake-dir", str(INTAKE_FIXTURE.parent),
                "--intake-files", INTAKE_FIXTURE.name,
                "--candidate-pool-out", str(pool_out),
                "--report-out", str(report_out),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(pool_out.exists())
            self.assertTrue(report_out.exists())

            # Inspect output records.
            records = [
                json.loads(line) for line in pool_out.read_text().splitlines()
                if line.strip()
            ]
            # 11 fixture rows; expected 5 survivors: TS-0001 (3707),
            # TS-0002 (2804), TS-0005 (3818 dedupe keeper), TS-0009 (8486),
            # TS-0010 (9030).
            self.assertEqual(len(records), 5)

            # Every record must have the canonical pre-audit shape.
            for r in records:
                self.assertTrue(r["id"].startswith("SH-bol-"))
                self.assertTrue(r["frozen_id"].startswith("v2.0.dev."))
                self.assertEqual(r["tier1_source"], "BOL")
                self.assertEqual(r["tier2_provenance"], "degraded_manual")
                self.assertEqual(
                    r["label_source"], "BOL_expert_validated_pending_reaudit"
                )
                self.assertEqual(r["adjudication_status"], "pending")
                self.assertEqual(r["confidence_tier"], "pending")
                self.assertEqual(r["cited_evidence_ids"], [])
                self.assertIn(r["scope_tier"], {"core", "supply_chain"})
                self.assertTrue(r["bol_metadata"]["declared_hs"])
                self.assertTrue(r["bol_metadata"]["hs_verified"])
                self.assertIn(
                    r["bol_metadata"]["origin_country"], {"JP", "US"}
                )
                sm = r["source_metadata"]
                self.assertEqual(sm["primary_source"], "BOL")
                self.assertEqual(sm["bol_intake_file"], INTAKE_FIXTURE.name)
                self.assertEqual(sm["bol_data_source"], "commercial_bol_provider")
                # candidate_set present + well-formed.
                cs = r["candidate_set"]
                self.assertGreaterEqual(cs["size"], 2)
                self.assertIn(r["hs6_label"], cs["codes"])

            # Report sanity.
            report = json.loads(report_out.read_text())
            steps = report["filter_pipeline"]["steps"]
            self.assertEqual(steps["short_description"]["dropped"], 1)
            self.assertEqual(steps["quantity_cap"]["dropped"], 1)
            self.assertEqual(steps["freight_forwarder"]["dropped"], 1)
            self.assertEqual(steps["off_scope_hs"]["dropped"], 1)
            self.assertEqual(steps["generic_terms"]["dropped"], 1)
            self.assertEqual(steps["near_duplicate"]["dropped"], 1)
            self.assertEqual(report["yield"]["total_survivors"], 5)
            self.assertIn("3707", report["yield"]["per_hs4"])
            self.assertIn("3818", report["yield"]["per_hs4"])
            self.assertIn("2804", report["yield"]["per_hs4"])
            # 9030 and 8486 also present
            self.assertIn("8486", report["yield"]["per_hs4"])
            self.assertIn("9030", report["yield"]["per_hs4"])

    def test_idempotent_byte_identical_outputs(self) -> None:
        """Running twice on the same inputs should produce byte-identical files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pool_a = tmp_path / "pool_a.jsonl"
            report_a = tmp_path / "report_a.json"
            pool_b = tmp_path / "pool_b.jsonl"
            report_b = tmp_path / "report_b.json"

            args_base = [
                "--intake-dir", str(INTAKE_FIXTURE.parent),
                "--intake-files", INTAKE_FIXTURE.name,
            ]
            self.assertEqual(0, _ingest.main(
                args_base + ["--candidate-pool-out", str(pool_a),
                             "--report-out", str(report_a)]
            ))
            self.assertEqual(0, _ingest.main(
                args_base + ["--candidate-pool-out", str(pool_b),
                             "--report-out", str(report_b)]
            ))
            self.assertEqual(pool_a.read_bytes(), pool_b.read_bytes())
            # Reports include intake paths; just check the records inside.
            ra = json.loads(report_a.read_text())
            rb = json.loads(report_b.read_text())
            self.assertEqual(ra["yield"], rb["yield"])
            self.assertEqual(ra["filter_pipeline"], rb["filter_pipeline"])

    def test_dry_run_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pool_out = tmp_path / "pool.jsonl"
            report_out = tmp_path / "report.json"
            rc = _ingest.main([
                "--intake-dir", str(INTAKE_FIXTURE.parent),
                "--intake-files", INTAKE_FIXTURE.name,
                "--candidate-pool-out", str(pool_out),
                "--report-out", str(report_out),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            self.assertFalse(pool_out.exists())
            self.assertFalse(report_out.exists())


if __name__ == "__main__":
    unittest.main()
