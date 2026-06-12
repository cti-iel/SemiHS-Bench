"""Tests for scripts/apply_audit_decisions.py.

Exercises the carryover-pool apply step end-to-end in a temp dir:

  1. Build a small synthetic candidate pool (BOL + catalog carryover).
  2. Build a worksheet with confirm / change / drop rows.
  3. Run the apply pipeline (parse → apply → scope_tier refresh →
     schema validate → report).
  4. Assert on the post-apply pool + report contents.

Tests are self-contained (do not depend on production data on disk).
"""

from __future__ import annotations

import csv
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

from src.audit.decisions import (  # noqa: E402
    WORKSHEET_COLUMNS,
    apply_corrections,
    parse_worksheet,
)

# Load the script as a module so we can call its helpers directly without
# spawning a subprocess.
_SCRIPT_PATH = ROOT / "scripts" / "apply_audit_decisions.py"
_spec = importlib.util.spec_from_file_location(
    "apply_audit_decisions", _SCRIPT_PATH
)
apply_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(apply_mod)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_CANDIDATE_KEYS = {
    "id", "frozen_id", "split", "tier1_description", "tier1_source",
    "tier2_minimal", "tier2_provenance", "tier2_classifiable",
    "hs6_label", "hs4_label",
    "hs2_label", "label_source", "difficulty_tags", "justification_text",
    "bol_metadata", "candidate_set",
}

_FALLBACK_DISTRACTORS = ["854110", "854231", "854239", "854110", "381800"]


def _candidate_codes_for(hs6: str) -> List[str]:
    """Build a unique 4-code candidate_set including hs6 as the gold."""
    out: List[str] = [hs6]
    for c in _FALLBACK_DISTRACTORS:
        if c == hs6 or c in out:
            continue
        out.append(c)
        if len(out) >= 4:
            break
    while len(out) < 4:
        # Generate a plausible filler that's not in out.
        for candidate in ("854121", "854143", "848620", "903180", "370790"):
            if candidate not in out:
                out.append(candidate)
                break
        else:
            break
    return out[:4]


def _candidate_record(
    *,
    frozen_id: str,
    hs6: str,
    label_source: str = "BOL_expert_validated_pending_reaudit",
) -> Dict[str, Any]:
    """Build a minimal candidate-pool record that the record_schema
    accepts after audit fields are populated."""
    return {
        "id": "SH-" + frozen_id.split(".")[-1].zfill(5),
        "frozen_id": frozen_id,
        "split": "test" if "test" in frozen_id else "dev",
        "tier1_description": "Sample product description for testing.",
        "tier1_source": "BOL" if "BOL" in label_source else "catalog",
        "tier2_minimal": {"part_name": "TEST-PN-001", "manufacturer": "ACME"},
        "tier2_provenance": "natural_mpn",
        "tier2_classifiable": "yes",
        "hs6_label": hs6,
        "hs4_label": hs6[:4],
        "hs2_label": hs6[:2],
        "label_source": label_source,
        "difficulty_tags": [],
        "justification_text": "Test justification.",
        "bol_metadata": (
            {"declared_hs": hs6, "hs_verified": True}
            if "BOL" in label_source else None
        ),
        "candidate_set": {
            "size": 4,
            "codes": _candidate_codes_for(hs6),
            "construction": "sibling_heading",
            "gold_rank_in_candidates": 0,
        },
    }


def _worksheet_row(
    frozen_id: str,
    current_hs6: str,
    action: str,
    *,
    new_hs6: str = "",
    expert_hs6: str = "",
    confidence_tier: str = "medium",
    cited: str = "EBTI-DE-X1",
    rationale: str = "test rationale",
    tier3: str = "no",
    adj_status: str = "single_reviewer",
    adj_winner: str = "",
    adj_score: str = "",
    notes: str = "",
    label_source: str = "BOL_expert_validated_pending_reaudit",
    scope_tier: str = "core",
) -> Dict[str, str]:
    row = {col: "" for col in WORKSHEET_COLUMNS}
    row.update({
        "row_kind": "carryover_reaudit",
        "frozen_id": frozen_id,
        "record_id": "SH-" + frozen_id.split(".")[-1].zfill(5),
        "split": "test" if "test" in frozen_id else "dev",
        "current_hs6": current_hs6,
        "current_hs4": current_hs6[:4],
        "label_source": label_source,
        "tier1_description": "Sample product description.",
        "candidate_reference_rulings": "EBTI-DE-X1,CROSS-N999",
        "manufacturer_hint": "ACME",
        "scope_tier": scope_tier,
        "action": action,
        "new_hs6": new_hs6,
        "expert_hs6": expert_hs6 or (
            new_hs6 if action == "change"
            else current_hs6 if action == "confirm"
            else ""
        ),
        "confidence_tier": confidence_tier,
        "cited_evidence_ids": cited,
        "rationale_short": rationale,
        "tier2_classifiable": tier3,
        "adjudication_status": adj_status,
        "adjudication_winning_evidence_id": adj_winner,
        "adjudication_rubric_score": adj_score,
        "notes": notes,
    })
    return row


def _write_pool(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_worksheet(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(WORKSHEET_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ApplyPipeline(unittest.TestCase):
    """End-to-end apply step in a temp dir."""

    def test_confirm_flips_label_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0001", hs6="854231",
                                  label_source="catalog_expert_validated_pending_reaudit"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0001", "854231", "confirm",
                               cited="EBTI-DE-X1,CROSS-N999",
                               confidence_tier="high",
                               label_source="catalog_expert_validated_pending_reaudit"),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            self.assertEqual(len(audited), 1)
            self.assertEqual(audited[0]["label_source"], "catalog_expert_validated")
            self.assertEqual(audited[0]["confidence_tier"], "high")
            self.assertEqual(len(audited[0]["cited_evidence_ids"]), 2)

    def test_relabel_changes_hs_and_refreshes_scope_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            # Start in core 8541 (single discrete), relabel to supply_chain 280461 silicon.
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0002", hs6="854110",
                                  label_source="BOL_expert_validated_pending_reaudit"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0002", "854110", "change",
                               new_hs6="280461",
                               expert_hs6="280461",
                               confidence_tier="medium",
                               cited="EBTI-DE-X1"),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            audited = apply_mod._refresh_scope_tier(
                audited, apply_mod._scope_tier_lookup()
            )
            self.assertEqual(audited[0]["hs6_label"], "280461")
            self.assertEqual(audited[0]["hs4_label"], "2804")
            self.assertEqual(audited[0]["hs2_label"], "28")
            self.assertEqual(audited[0]["label_source"], "expert_relabeled")
            # 280461 is in supply_chain per configs/hs6_scope_tiers.yaml.
            self.assertEqual(audited[0]["scope_tier"], "supply_chain")

    def test_drop_removes_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0003", hs6="854231"),
                _candidate_record(frozen_id="v2.0.eval.0004", hs6="854239"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0003", "854231", "drop",
                               adj_status="unresolved_dropped",
                               confidence_tier="", cited="", tier3=""),
                _worksheet_row("v2.0.eval.0004", "854239", "confirm",
                               confidence_tier="medium"),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            audited = apply_mod._refresh_scope_tier(
                audited, apply_mod._scope_tier_lookup()
            )
            self.assertEqual(len(audited), 1)
            self.assertEqual(audited[0]["frozen_id"], "v2.0.eval.0004")

    def test_no_pending_labels_after_apply(self) -> None:
        """Sanity check: every record in the audited pool MUST have its
        label_source flipped off the *_pending_reaudit transient values."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0005", hs6="854231",
                                  label_source="catalog_expert_validated_pending_reaudit"),
                _candidate_record(frozen_id="v2.0.eval.0006", hs6="854239",
                                  label_source="BOL_expert_validated_pending_reaudit"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0005", "854231", "confirm",
                               label_source="catalog_expert_validated_pending_reaudit"),
                _worksheet_row("v2.0.eval.0006", "854239", "confirm",
                               label_source="BOL_expert_validated_pending_reaudit"),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            bad = apply_mod._validate_no_pending_labels(audited)
            self.assertEqual(bad, [])

    def test_record_schema_validation_passes(self) -> None:
        """After full apply + scope_tier refresh, audited records should
        validate against release/working/data/record_schema.json."""
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema not installed")
        if not (ROOT / "release" / "working" / "data" / "record_schema.json").exists():
            self.skipTest("non-public release/working working pool not present")

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0007", hs6="854231",
                                  label_source="catalog_expert_validated_pending_reaudit"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0007", "854231", "confirm",
                               confidence_tier="high",
                               cited="EBTI-DE-X1,CROSS-N999",
                               tier3="yes",
                               label_source="catalog_expert_validated_pending_reaudit"),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            audited = apply_mod._refresh_scope_tier(
                audited, apply_mod._scope_tier_lookup()
            )
            schema_path = ROOT / "release" / "working" / "data" / "record_schema.json"
            errors = apply_mod._validate_records_against_schema(audited, schema_path)
            self.assertEqual(errors, [], f"validation errors: {errors}")


class ReportBuilder(unittest.TestCase):
    def test_report_counts_match_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0008", hs6="854231"),
                _candidate_record(frozen_id="v2.0.eval.0009", hs6="854239"),
                _candidate_record(frozen_id="v2.0.eval.0010", hs6="854110"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0008", "854231", "confirm"),
                _worksheet_row("v2.0.eval.0009", "854239", "change",
                               new_hs6="854231"),
                _worksheet_row("v2.0.eval.0010", "854110", "drop",
                               adj_status="unresolved_dropped",
                               confidence_tier="", cited="", tier3=""),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            audited = apply_mod._refresh_scope_tier(
                audited, apply_mod._scope_tier_lookup()
            )
            report = apply_mod._build_report(
                corrections, audited, dropped_frozen_ids=["v2.0.eval.0010"],
                schema_errors=[],
            )
            self.assertEqual(report["counts"]["confirmed"], 1)
            self.assertEqual(report["counts"]["relabeled"], 1)
            self.assertEqual(report["counts"]["dropped"], 1)
            self.assertEqual(report["counts"]["audited_records_written"], 2)

    def test_acceptance_gates_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            pool_path = tmp / "pool.jsonl"
            ws_path = tmp / "ws.csv"
            _write_pool(pool_path, [
                _candidate_record(frozen_id="v2.0.eval.0011", hs6="854231"),
            ])
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0011", "854231", "confirm"),
            ])
            corrections = parse_worksheet(ws_path)
            pool = [json.loads(l) for l in pool_path.open()]
            audited = apply_corrections(pool, corrections)
            audited = apply_mod._refresh_scope_tier(
                audited, apply_mod._scope_tier_lookup()
            )
            report = apply_mod._build_report(corrections, audited,
                                            dropped_frozen_ids=[],
                                            schema_errors=[])
            gates = report["acceptance_gates"]
            self.assertTrue(gates["no_pending_reaudit_labels_remaining"])
            self.assertTrue(gates["no_high_confidence_without_2_citations"])
            self.assertTrue(gates["schema_validation_passed"])


class ErrorPaths(unittest.TestCase):
    def test_malformed_worksheet_raises(self) -> None:
        from src.audit.decisions import DecisionParseError
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ws_path = tmp / "ws.csv"
            # Missing required worksheet fields (confidence_tier, tier2_classifiable, etc.)
            _write_worksheet(ws_path, [
                _worksheet_row("v2.0.eval.0012", "854231", "confirm",
                               confidence_tier="high", cited="EBTI-DE-X1"),
                # high requires ≥2 citations
            ])
            with self.assertRaises(DecisionParseError) as ctx:
                parse_worksheet(ws_path)
            self.assertIn("high requires", str(ctx.exception))


class RelativePathHelper(unittest.TestCase):
    def test_path_inside_root(self) -> None:
        p = ROOT / "release" / "working" / "data" / "record_schema.json"
        rendered = apply_mod._rel(p)
        self.assertFalse(rendered.startswith("/"))
        self.assertIn("release/working", rendered)

    def test_path_outside_root_falls_back_to_absolute(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv") as tf:
            p = Path(tf.name)
            rendered = apply_mod._rel(p)
            # /tmp on macOS may be a symlink to /private/tmp; either is fine.
            self.assertTrue(rendered.startswith("/"), f"got {rendered!r}")


if __name__ == "__main__":
    unittest.main()
