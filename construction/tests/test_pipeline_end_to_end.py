"""End-to-end integration test for the audit pipeline.

Exercises every stage downstream of the candidate-pool builders /
``build_reference_corpus.py`` (whose outputs are already on disk in
``release/working/data/``) and confirms that schemas + acceptance gates
compose cleanly across script boundaries:

    on-disk candidate_pool.jsonl  (133 carryover records)
    on-disk reference_corpus.jsonl (564 entries; ~ evidence-coverage floor)
        │
        ▼
    generate_review_worksheet.py  → temp audit_worksheet.csv
        │
        ▼
    synthetic rater fill (in-test helper)
        │
        ▼
    apply_audit_decisions.py  → temp _candidate_pool_audited.jsonl
                                   temp _audit_report.json
        │
        ▼
    jsonschema validate against record_schema.json
    + assert all 3 acceptance gates pass

The test does NOT rebuild the candidate pool or reference corpus from
scratch (those are pool-builder / build_reference_corpus territory and have
their own unit tests). It DOES catch the cross-stage regressions those
unit tests cannot — e.g., a future change that alters the worksheet
output shape and breaks the apply parser, or a record_schema tightening
that rejects emitted records.

Skipped automatically if the on-disk artifacts are missing — e.g., a
fresh clone before the scaffolding runs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATE_POOL_PATH = ROOT / "release" / "working" / "data" / "_candidate_pool.jsonl"
REFERENCE_CORPUS_PATH = ROOT / "release" / "working" / "data" / "reference_corpus.jsonl"
RECORD_SCHEMA_PATH = ROOT / "release" / "working" / "data" / "record_schema.json"
REFERENCE_CORPUS_SCHEMA_PATH = (
    ROOT / "release" / "working" / "data" / "reference_corpus_schema.json"
)
SCOPE_CONFIG_PATH = ROOT / "configs" / "hs6_scope_tiers.yaml"

# Load the two stage-runner scripts as modules so we can call main(argv=).
_GENERATE_SCRIPT = ROOT / "scripts" / "generate_review_worksheet.py"
_APPLY_SCRIPT = ROOT / "scripts" / "apply_audit_decisions.py"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_generate = _load_script(_GENERATE_SCRIPT, "generate_review_worksheet")
_apply = _load_script(_APPLY_SCRIPT, "apply_audit_decisions")


def _on_disk_artifacts_present() -> bool:
    return all(p.exists() for p in (
        CANDIDATE_POOL_PATH,
        REFERENCE_CORPUS_PATH,
        RECORD_SCHEMA_PATH,
        SCOPE_CONFIG_PATH,
    ))


def _synthetic_fill(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fill the rater columns on every row.

    Mix of decisions to exercise all paths:
      - first row: change → relabel
      - second row: drop with adjudication_status=unresolved_dropped
      - third row: confirm with adjudicated_evidence_resolved + rubric
      - all other rows: confirm with medium confidence + 1 citation
    """
    filled: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        new = dict(row)
        candidates = (row.get("candidate_reference_rulings") or "").split(",")
        if i == 0 and len(candidates) >= 2:
            # Relabel — pick a sibling HS6 in the same HS4 if possible,
            # else fall back to a known supply_chain HS6 (280461) so
            # scope_tier refresh has something to land on.
            current_hs4 = row.get("current_hs4", "")
            sibling = {
                "2804": "280469",
                "3707": "370710",
                "3818": "381800",  # only HS6 in 3818; degenerate but valid
                "8486": "848690",
                "8541": "854190",
                "8542": "854290",
            }.get(current_hs4, "280461")
            new.update({
                "action": "change",
                "new_hs6": sibling,
                "expert_hs6": sibling,
                "confidence_tier": "high",
                "cited_evidence_ids": ",".join(candidates[:2]),
                "rationale_short": "Test-pipeline relabel.",
                "tier2_classifiable": "no",
                "adjudication_status": "single_reviewer",
            })
        elif i == 1:
            new.update({
                "action": "drop",
                "adjudication_status": "unresolved_dropped",
                "notes": "Test-pipeline drop.",
                # Leave action-required fields blank per the parser rules:
                # action=drop + unresolved_dropped exempts confidence/cited/tier3.
                "confidence_tier": "",
                "cited_evidence_ids": "",
                "tier2_classifiable": "",
                "expert_hs6": "",
            })
        elif i == 2 and len(candidates) >= 2:
            new.update({
                "action": "confirm",
                "expert_hs6": row["current_hs6"],
                "confidence_tier": "high",
                "cited_evidence_ids": ",".join(candidates[:2]),
                "rationale_short": "Adjudicated-evidence-resolved test row.",
                "tier2_classifiable": "partial",
                "adjudication_status": "adjudicated_evidence_resolved",
                "adjudication_winning_evidence_id": candidates[0],
                "adjudication_rubric_score": "5",
            })
        else:
            cited = candidates[0] if candidates and candidates[0] else "EBTI-DE-X1"
            new.update({
                "action": "confirm",
                "expert_hs6": row["current_hs6"],
                "confidence_tier": "medium",
                "cited_evidence_ids": cited,
                "rationale_short": "Carryover accepted.",
                "tier2_classifiable": "no",
                "adjudication_status": "single_reviewer",
            })
        filled.append(new)
    return filled


@unittest.skipUnless(
    _on_disk_artifacts_present(),
    "on-disk artifacts missing — run the scaffolding scripts first",
)
class PipelineEndToEnd(unittest.TestCase):
    """One big test: chain the pipeline stages and confirm everything
    composes."""

    @classmethod
    def setUpClass(cls) -> None:
        # Validate that the on-disk reference corpus passes its own schema —
        # if it doesn't, every downstream stage is suspect.
        try:
            import jsonschema  # type: ignore
            cls._jsonschema_available = True
        except ImportError:
            cls._jsonschema_available = False

        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.tmp_dir.name)
        cls.worksheet_path = cls.tmp / "audit_worksheet.csv"
        cls.audited_path = cls.tmp / "_candidate_pool_audited.jsonl"
        cls.corrections_path = cls.tmp / "_audit_corrections.json"
        cls.report_path = cls.tmp / "_audit_report.json"
        cls.candidate_pool_count = sum(
            1 for line in CANDIDATE_POOL_PATH.open() if line.strip()
        )
        cls.reference_corpus_count = sum(
            1 for line in REFERENCE_CORPUS_PATH.open() if line.strip()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()

    def test_01_on_disk_reference_corpus_validates(self) -> None:
        if not self._jsonschema_available:
            self.skipTest("jsonschema not installed")
        import jsonschema

        schema = json.loads(REFERENCE_CORPUS_SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = jsonschema.Draft7Validator(schema)
        errors: List[str] = []
        with REFERENCE_CORPUS_PATH.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for err in validator.iter_errors(entry):
                    errors.append(
                        f"entry[{i}] eid={entry.get('evidence_id', '?')}: "
                        f"{err.absolute_path}: {err.message}"
                    )
                    if len(errors) >= 5:
                        break
                if len(errors) >= 5:
                    break
        self.assertEqual(errors, [], f"reference corpus schema errors: {errors}")

    def test_02_generate_worksheet_stage(self) -> None:
        """Run generate_review_worksheet against the on-disk inputs;
        confirm it emits the expected number of rows with the required worksheet columns."""
        rc = _generate.main([
            "--candidate-pool", str(CANDIDATE_POOL_PATH),
            "--reference-corpus", str(REFERENCE_CORPUS_PATH),
            "--output", str(self.worksheet_path),
            "--force",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(self.worksheet_path.exists())
        rows = list(csv.DictReader(self.worksheet_path.open()))
        self.assertEqual(
            len(rows), self.candidate_pool_count,
            f"worksheet should have one row per candidate pool record "
            f"({self.candidate_pool_count}); got {len(rows)}"
        )
        # required worksheet columns must be present
        for required in (
            "frozen_id", "current_hs6", "scope_tier",
            "candidate_reference_rulings", "confidence_tier",
            "cited_evidence_ids", "adjudication_status",
        ):
            self.assertIn(required, rows[0],
                          f"worksheet missing required column {required}")
        # Pre-filled informational columns must be populated
        first = rows[0]
        self.assertTrue(first["frozen_id"])
        self.assertTrue(first["current_hs6"])
        self.assertTrue(first["scope_tier"] in {"core", "supply_chain"})
        # candidate_reference_rulings should generally be populated
        # (we proved 100% earlier in the build) but allow zero-coverage
        # corner case to be robust:
        n_with_candidates = sum(1 for r in rows if r["candidate_reference_rulings"])
        self.assertGreaterEqual(n_with_candidates, len(rows) - 5)
        # Rater columns must be blank
        for col in ("action", "expert_hs6", "confidence_tier", "cited_evidence_ids"):
            self.assertEqual(first[col], "",
                             f"rater column {col} should be blank in generated worksheet")

    def test_03_fill_and_apply_stage(self) -> None:
        # Read the generated worksheet, fill synthetically, write back.
        rows = list(csv.DictReader(self.worksheet_path.open()))
        filled = _synthetic_fill(rows)
        with self.worksheet_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(filled)

        # Run apply_audit_decisions.
        rc = _apply.main([
            "--worksheet", str(self.worksheet_path),
            "--candidate-pool", str(CANDIDATE_POOL_PATH),
            "--audited-out", str(self.audited_path),
            "--corrections-out", str(self.corrections_path),
            "--report-out", str(self.report_path),
            "--record-schema", str(RECORD_SCHEMA_PATH),
            "--strict-schema",
        ])
        self.assertEqual(rc, 0, "apply returned non-zero (strict-schema gate failed)")
        self.assertTrue(self.audited_path.exists())
        self.assertTrue(self.corrections_path.exists())
        self.assertTrue(self.report_path.exists())

    def test_04_apply_outputs_validate_against_record_schema(self) -> None:
        if not self._jsonschema_available:
            self.skipTest("jsonschema not installed")
        import jsonschema

        schema = json.loads(RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        bad: List[str] = []
        n = 0
        with self.audited_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                record = json.loads(line)
                for err in validator.iter_errors(record):
                    bad.append(
                        f"record[{n}] frozen_id={record.get('frozen_id', '?')}: "
                        f"{err.absolute_path}: {err.message}"
                    )
                    if len(bad) >= 5:
                        break
                if len(bad) >= 5:
                    break
        self.assertEqual(bad, [], f"record schema errors in audited pool: {bad}")
        self.assertGreater(n, 0)

    def test_05_acceptance_gates_pass(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        gates = report["acceptance_gates"]
        # Show all gates in the failure message so a regression is obvious.
        for gate_name, gate_value in gates.items():
            self.assertTrue(gate_value, f"gate {gate_name} failed; gates={gates}")

    def test_06_audited_records_have_no_pending_labels(self) -> None:
        """Invariant: every audited record's label_source ∈ the 3
        released values (no transient pending_reaudit)."""
        released = {"catalog_expert_validated", "BOL_expert_validated", "expert_relabeled"}
        with self.audited_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.assertIn(
                    record["label_source"], released,
                    f"record {record.get('frozen_id', '?')} has "
                    f"unreleased label_source {record['label_source']!r}"
                )

    def test_07_dropped_row_actually_absent(self) -> None:
        """Synthetic fill drops row i=1; the audited pool should be
        candidate_pool_count - 1 records."""
        with self.audited_path.open(encoding="utf-8") as f:
            n_audited = sum(1 for line in f if line.strip())
        self.assertEqual(
            n_audited, self.candidate_pool_count - 1,
            f"expected {self.candidate_pool_count - 1} audited records "
            f"(dropped one); got {n_audited}"
        )

    def test_08_relabel_changes_hs6_and_refreshes_scope_tier(self) -> None:
        """The first synthetic-fill row is a relabel. The audited record
        for that frozen_id should have hs6_label changed and scope_tier
        re-derived from the new HS6."""
        rows = list(csv.DictReader(self.worksheet_path.open()))
        first_row = rows[0]
        new_hs6 = first_row["new_hs6"]
        self.assertTrue(new_hs6, "synthetic-fill row 0 should be a relabel")

        # Load scope_tier_lookup directly so we can predict the expected tier.
        scope_lookup = _apply._scope_tier_lookup()
        expected_tier = scope_lookup.get(new_hs6, "")
        self.assertTrue(expected_tier,
                        f"new_hs6={new_hs6} not in scope lookup")

        relabeled_record = None
        with self.audited_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["frozen_id"] == first_row["frozen_id"]:
                    relabeled_record = record
                    break
        self.assertIsNotNone(relabeled_record,
                             "relabeled record missing from audited pool")
        self.assertEqual(relabeled_record["hs6_label"], new_hs6)
        self.assertEqual(relabeled_record["hs4_label"], new_hs6[:4])
        self.assertEqual(relabeled_record["scope_tier"], expected_tier)
        self.assertEqual(relabeled_record["label_source"], "expert_relabeled")

    def test_09_adjudicated_evidence_resolved_carries_rubric(self) -> None:
        """Row i=2 carries adjudicated_evidence_resolved with rubric_score=5
        and a winning_evidence_id; that must round-trip into the audited
        record."""
        rows = list(csv.DictReader(self.worksheet_path.open()))
        target_row = rows[2]
        self.assertEqual(
            target_row["adjudication_status"], "adjudicated_evidence_resolved"
        )
        target_id = target_row["frozen_id"]
        found = None
        with self.audited_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["frozen_id"] == target_id:
                    found = record
                    break
        self.assertIsNotNone(found, f"frozen_id {target_id} missing in audited pool")
        self.assertEqual(found["adjudication_status"], "adjudicated_evidence_resolved")
        self.assertEqual(found["adjudication_rubric_score"], 5)
        self.assertEqual(
            found["adjudication_winning_evidence_id"],
            target_row["adjudication_winning_evidence_id"],
        )

    def test_10_high_confidence_rows_have_at_least_two_citations(self) -> None:
        """Core-4 evidence invariant verified at the audited-record
        level (worksheet parser already enforces, but we re-check after
        apply for defense-in-depth)."""
        with self.audited_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("confidence_tier") == "high":
                    self.assertGreaterEqual(
                        len(record.get("cited_evidence_ids") or []), 2,
                        f"record {record['frozen_id']} has confidence_tier=high "
                        f"but only {len(record.get('cited_evidence_ids') or [])} "
                        f"citations"
                    )


@unittest.skipUnless(
    _on_disk_artifacts_present(),
    "on-disk artifacts missing — run the scaffolding scripts first",
)
class PipelineWithBolPool(unittest.TestCase):
    """Confirm that the worksheet + apply stages compose carryover + BOL pools.

    Drops a synthetic ``_candidate_pool_bol.jsonl`` in a tempdir alongside
    the on-disk carryover pool, invokes both stages with ``--candidate-pool``
    pointing at the two files, and asserts the worksheet row count =
    carryover count + BOL count.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.tmp_dir.name)
        cls.worksheet_path = cls.tmp / "audit_worksheet.csv"
        cls.bol_pool_path = cls.tmp / "_candidate_pool_bol.jsonl"
        cls.candidate_pool_count = sum(
            1 for line in CANDIDATE_POOL_PATH.open() if line.strip()
        )
        # Synthesize two minimal BOL pool records. Choose HS6s that exist
        # in scope and have boundary-pair coverage so the worksheet's
        # candidate_reference_rulings column populates.
        cls.synthetic_bol = [
            {
                "id": "SH-bol-00001",
                "frozen_id": "v2.0.dev.0001",
                "split": "dev",
                "tier1_description": "Plasma etch chamber for 300mm wafer fab",
                "tier1_source": "BOL",
                "tier2_minimal": {"part_name": "plasma etch", "manufacturer": "Lam Research"},
                "tier2_provenance": "degraded_manual",
                "tier2_classifiable": "no",
                "hs6_label": "848620",
                "hs4_label": "8486",
                "hs2_label": "84",
                "scope_tier": "core",
                "label_source": "BOL_expert_validated_pending_reaudit",
                "confidence_tier": "pending",
                "cited_evidence_ids": [],
                "adjudication_status": "pending",
                "difficulty_tags": [],
                "justification_text": "",
                "bol_metadata": {
                    "shipper": "Lam Research",
                    "consignee": "TSMC Arizona",
                    "port_origin": "san jose",
                    "port_dest": "phoenix",
                    "bol_description": "Plasma etch chamber for 300mm wafer fab",
                    "declared_hs": "84862000",
                    "hs_verified": True,
                    "arrival_date": "2025-03-07",
                    "origin_country": "US",
                },
                "candidate_set": {
                    "size": 4,
                    "codes": ["848610", "848620", "848640", "848690"],
                    "construction": "sibling_heading",
                    "gold_rank_in_candidates": 1,
                },
                "source_metadata": {
                    "primary_source": "BOL",
                    "primary_reference": "TS-0001",
                    "manufacturer_hint": "Lam Research",
                    "manufacturer_hint_source": "backfilled_from_tier2",
                    "bol_intake_file": "synthetic.csv",
                    "bol_target_hs_family": "8486",
                    "bol_data_source": "commercial_bol_provider",
                },
            },
        ]
        with cls.bol_pool_path.open("w", encoding="utf-8") as f:
            for rec in cls.synthetic_bol:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()

    def test_worksheet_combines_carryover_and_bol_pools(self) -> None:
        rc = _generate.main([
            "--candidate-pool", str(CANDIDATE_POOL_PATH), str(self.bol_pool_path),
            "--reference-corpus", str(REFERENCE_CORPUS_PATH),
            "--output", str(self.worksheet_path),
            "--force",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(self.worksheet_path.exists())

        rows = list(csv.DictReader(self.worksheet_path.open()))
        expected = self.candidate_pool_count + len(self.synthetic_bol)
        self.assertEqual(len(rows), expected)

        # The synthetic BOL record should appear with row_kind=bol_audit.
        bol_rows = [r for r in rows if r["row_kind"] == "bol_audit"]
        self.assertEqual(len(bol_rows), len(self.synthetic_bol))
        self.assertEqual(bol_rows[0]["frozen_id"], "v2.0.dev.0001")
        self.assertEqual(bol_rows[0]["current_hs6"], "848620")
        self.assertEqual(bol_rows[0]["scope_tier"], "core")

        # Carryover rows should still carry the carryover_reaudit kind.
        carryover_rows = [r for r in rows if r["row_kind"] == "carryover_reaudit"]
        self.assertEqual(len(carryover_rows), self.candidate_pool_count)


if __name__ == "__main__":
    unittest.main()
