"""Tests for the expert-decision worksheet parser + applier."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.decisions import (  # noqa: E402
    ADJUDICATION_STATUSES,
    CONFIDENCE_TIERS,
    WORKSHEET_COLUMNS,
    DecisionParseError,
    apply_corrections,
    corrections_to_dict,
    parse_worksheet,
)


HEADERS = list(WORKSHEET_COLUMNS)


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            full = {k: "" for k in HEADERS}
            full.update(row)
            writer.writerow(full)


def _row(
    frozen_id,
    current_hs6,
    action,
    *,
    expert_hs6="",
    new_hs6="",
    confidence_tier="medium",
    cited_evidence_ids="EBTI-DE-X1",
    rationale_short="ok",
    tier2_classifiable="no",
    adjudication_status="single_reviewer",
    adjudication_winning_evidence_id="",
    adjudication_rubric_score="",
    notes="",
    row_kind="catalog_audit",
    record_id="SH-0001",
    split="test",
    label_source="BOL_expert_validated_pending_reaudit",
):
    return {
        "row_kind": row_kind,
        "frozen_id": frozen_id,
        "record_id": record_id,
        "split": split,
        "current_hs6": current_hs6,
        "current_hs4": current_hs6[:4],
        "label_source": label_source,
        "tier1_description": "Sample.",
        "candidate_reference_rulings": "EBTI-DE-X1,CROSS-N123",
        "manufacturer_hint": "ACME",
        "scope_tier": "core",
        "action": action,
        "new_hs6": new_hs6,
        "expert_hs6": expert_hs6 or new_hs6 or current_hs6,
        "confidence_tier": confidence_tier,
        "cited_evidence_ids": cited_evidence_ids,
        "rationale_short": rationale_short,
        "tier2_classifiable": tier2_classifiable,
        "adjudication_status": adjudication_status,
        "adjudication_winning_evidence_id": adjudication_winning_evidence_id,
        "adjudication_rubric_score": adjudication_rubric_score,
        "notes": notes,
    }


def _record(*, frozen_id="v2.0.test.0001", hs6="280410",
            label_source="BOL_expert_validated_pending_reaudit") -> dict:
    return {
        "id": "SH-0001",
        "frozen_id": frozen_id,
        "split": "test",
        "hs6_label": hs6,
        "hs4_label": hs6[:4],
        "hs2_label": hs6[:2],
        "label_source": label_source,
        "tier1_description": "Sample.",
        "source_metadata": {"manufacturer_hint": "ACME"},
    }


def parse_worksheet_from_rows(rows):
    """Helper: write rows to a temp CSV and parse them."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ws.csv"
        _write_csv(path, rows)
        return parse_worksheet(path)


# --- parse_worksheet ---------------------------------------------------------


class ParseTests(unittest.TestCase):
    def test_confirm_action(self) -> None:
        corr = parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "confirm")])
        self.assertEqual(corr.confirmed, ["v2.0.test.0001"])
        self.assertEqual(corr.relabeled, {})
        self.assertEqual(corr.dropped, [])

    def test_change_action_requires_new_hs6(self) -> None:
        corr = parse_worksheet_from_rows([
            _row("v2.0.test.0001", "280410", "change", new_hs6="280421"),
        ])
        self.assertEqual(corr.relabeled, {"v2.0.test.0001": "280421"})

    def test_change_without_new_hs6_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "change")])
        self.assertIn("requires a 6-digit new_hs6", str(ctx.exception))

    def test_change_with_invalid_hs6(self) -> None:
        with self.assertRaises(DecisionParseError):
            parse_worksheet_from_rows([
                _row("v2.0.test.0001", "280410", "change", new_hs6="ABC123"),
            ])

    def test_drop_action(self) -> None:
        corr = parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "drop")])
        self.assertEqual(corr.dropped, ["v2.0.test.0001"])

    def test_drop_with_new_hs6_rejected(self) -> None:
        with self.assertRaises(DecisionParseError):
            parse_worksheet_from_rows([
                _row("v2.0.test.0001", "280410", "drop", new_hs6="280421"),
            ])

    def test_blank_action_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "")])
        self.assertIn("blank", str(ctx.exception))

    def test_unknown_action_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "ignore")])
        self.assertIn("unknown action", str(ctx.exception))

    def test_action_aliases_accepted(self) -> None:
        cases = [
            ("correct", "confirm"),
            ("ok", "confirm"),
            ("wrong", "change"),
            ("relabel", "change"),
            ("remove", "drop"),
        ]
        for alias, canonical in cases:
            new_hs6 = "280421" if canonical == "change" else ""
            corr = parse_worksheet_from_rows([
                _row("v2.0.test.0001", "280410", alias, new_hs6=new_hs6),
            ])
            if canonical == "confirm":
                self.assertEqual(corr.confirmed, ["v2.0.test.0001"])
            elif canonical == "change":
                self.assertEqual(corr.relabeled, {"v2.0.test.0001": "280421"})
            elif canonical == "drop":
                self.assertEqual(corr.dropped, ["v2.0.test.0001"])

    def test_duplicate_frozen_id_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([
                _row("v2.0.test.0001", "280410", "confirm"),
                _row("v2.0.test.0001", "280410", "confirm"),
            ])
        self.assertIn("duplicate frozen_id", str(ctx.exception))

    def test_action_case_insensitive(self) -> None:
        corr = parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "CONFIRM")])
        self.assertEqual(corr.confirmed, ["v2.0.test.0001"])


# --- apply_corrections -------------------------------------------------------


class ApplyTests(unittest.TestCase):
    def test_confirm_flips_label_source(self) -> None:
        records = [_record(label_source="BOL_expert_validated_pending_reaudit")]
        corr = parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "confirm")])
        out = apply_corrections(records, corr)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label_source"], "BOL_expert_validated")
        self.assertEqual(out[0]["hs6_label"], "280410")

    def test_catalog_confirm_flips_label_source(self) -> None:
        records = [_record(label_source="catalog_expert_validated_pending_reaudit")]
        corr = parse_worksheet_from_rows([_row("v2.0.test.0001", "280410", "confirm")])
        out = apply_corrections(records, corr)
        self.assertEqual(out[0]["label_source"], "catalog_expert_validated")

    def test_change_updates_hs_codes(self) -> None:
        records = [_record(hs6="280410")]
        corr = parse_worksheet_from_rows([
            _row("v2.0.test.0001", "280410", "change", new_hs6="382490"),
        ])
        out = apply_corrections(records, corr)
        self.assertEqual(out[0]["hs6_label"], "382490")
        self.assertEqual(out[0]["hs4_label"], "3824")
        self.assertEqual(out[0]["hs2_label"], "38")
        self.assertEqual(out[0]["label_source"], "expert_relabeled")

    def test_drop_removes_record(self) -> None:
        records = [_record(frozen_id="v2.0.test.0001"),
                   _record(frozen_id="v2.0.test.0002")]
        corr = parse_worksheet_from_rows([
            _row("v2.0.test.0001", "280410", "drop"),
        ])
        out = apply_corrections(records, corr)
        self.assertEqual([r["frozen_id"] for r in out], ["v2.0.test.0002"])

    def test_records_not_in_corrections_unchanged(self) -> None:
        records = [_record(frozen_id="v2.0.test.0099", hs6="854231",
                           label_source="CROSS")]
        corr = parse_worksheet_from_rows([
            _row("v2.0.test.0001", "280410", "confirm"),
        ])
        out = apply_corrections(records, corr)
        self.assertEqual(out[0]["hs6_label"], "854231")
        self.assertEqual(out[0]["label_source"], "CROSS")

    def test_apply_populates_core4_fields(self) -> None:
        records = [_record(label_source="BOL_expert_validated_pending_reaudit")]
        corr = parse_worksheet_from_rows([_row(
            "v2.0.test.0001", "280410", "confirm",
            confidence_tier="high",
            cited_evidence_ids="EBTI-DE-X1,CROSS-N456",
            tier2_classifiable="partial",
        )])
        out = apply_corrections(records, corr)
        self.assertEqual(out[0]["confidence_tier"], "high")
        self.assertEqual(out[0]["cited_evidence_ids"], ["EBTI-DE-X1", "CROSS-N456"])
        self.assertEqual(out[0]["tier2_classifiable"], "partial")

    def test_corrections_to_dict_serializable(self) -> None:
        corr = parse_worksheet_from_rows([
            _row("v2.0.test.0001", "280410", "confirm"),
            _row("v2.0.test.0002", "280410", "change", new_hs6="382490"),
            _row("v2.0.test.0003", "280410", "drop", notes="duplicate"),
        ])
        payload = corrections_to_dict(corr)
        self.assertEqual(payload["confirmed"], ["v2.0.test.0001"])
        self.assertEqual(payload["relabeled"], {"v2.0.test.0002": "382490"})
        self.assertEqual(payload["dropped"], ["v2.0.test.0003"])
        self.assertEqual(len(payload["decisions"]), 3)


# --- Core-4 evidence binding -------------------------------------------------


class ConfidenceCitationBinding(unittest.TestCase):
    """high → ≥2 citations, medium → ≥1, low → any."""

    def test_high_with_one_citation_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "confirm",
                confidence_tier="high",
                cited_evidence_ids="EBTI-DE-X1",  # only 1
            )])
        self.assertIn("high requires", str(ctx.exception))

    def test_medium_with_zero_citations_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "confirm",
                confidence_tier="medium",
                cited_evidence_ids="",
            )])
        self.assertIn("medium requires", str(ctx.exception))

    def test_low_with_zero_citations_accepted(self) -> None:
        corr = parse_worksheet_from_rows([_row(
            "v2.0.test.0001", "280461", "confirm",
            confidence_tier="low",
            cited_evidence_ids="",
        )])
        self.assertEqual(corr.confirmed, ["v2.0.test.0001"])


class ExpertHS6Binding(unittest.TestCase):
    def test_change_expert_hs6_must_match_new_hs6(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280410", "change",
                new_hs6="280461",
                expert_hs6="382490",  # disagrees with new_hs6
            )])
        self.assertIn("disagrees with new_hs6", str(ctx.exception))

    def test_confirm_expert_hs6_must_match_current_hs6(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280410", "confirm",
                expert_hs6="382490",  # disagrees with current_hs6
            )])
        self.assertIn("disagrees with current_hs6", str(ctx.exception))

    def test_confirm_without_expert_hs6_rejected(self) -> None:
        row = _row("v2.0.test.0001", "280461", "confirm")
        row["expert_hs6"] = ""  # override the default
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([row])
        self.assertIn("expert_hs6", str(ctx.exception))


class AdjudicationStatus(unittest.TestCase):
    def test_evidence_resolved_requires_rubric_score(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "change",
                new_hs6="280469",
                expert_hs6="280469",
                adjudication_status="adjudicated_evidence_resolved",
                # Missing rubric_score + winning_evidence_id
            )])
        msg = str(ctx.exception)
        self.assertIn("adjudication_rubric_score", msg)
        self.assertIn("adjudication_winning_evidence_id", msg)

    def test_evidence_resolved_with_score_and_winner_ok(self) -> None:
        corr = parse_worksheet_from_rows([_row(
            "v2.0.test.0001", "280461", "change",
            new_hs6="280469",
            expert_hs6="280469",
            adjudication_status="adjudicated_evidence_resolved",
            adjudication_winning_evidence_id="EBTI-DE-X1",
            adjudication_rubric_score="5",
        )])
        d = corr.decisions[0]
        self.assertEqual(d.adjudication_status, "adjudicated_evidence_resolved")
        self.assertEqual(d.adjudication_rubric_score, 5)
        self.assertEqual(d.adjudication_winning_evidence_id, "EBTI-DE-X1")

    def test_rubric_score_out_of_range(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "confirm",
                adjudication_status="adjudicated_evidence_resolved",
                adjudication_winning_evidence_id="EBTI-DE-X1",
                adjudication_rubric_score="7",
            )])
        self.assertIn("out of range", str(ctx.exception))

    def test_unresolved_dropped_requires_drop_action(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "confirm",
                adjudication_status="unresolved_dropped",
            )])
        self.assertIn("requires action=drop", str(ctx.exception))

    def test_unresolved_dropped_with_drop_ok(self) -> None:
        corr = parse_worksheet_from_rows([_row(
            "v2.0.test.0001", "280461", "drop",
            adjudication_status="unresolved_dropped",
            confidence_tier="",
            cited_evidence_ids="",
            tier2_classifiable="",
        )])
        self.assertEqual(corr.dropped, ["v2.0.test.0001"])

    def test_pending_status_skips_confidence_requirement(self) -> None:
        """Carryover records can be left with adjudication_status=pending
        and no confidence_tier yet."""
        corr = parse_worksheet_from_rows([_row(
            "v2.0.test.0001", "280461", "confirm",
            adjudication_status="pending",
            confidence_tier="",
            cited_evidence_ids="",
        )])
        self.assertEqual(corr.decisions[0].adjudication_status, "pending")


class Tier2Classifiable(unittest.TestCase):
    def test_value_required_for_confirm(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "confirm",
                tier2_classifiable="",
            )])
        self.assertIn("tier2_classifiable", str(ctx.exception))

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(DecisionParseError) as ctx:
            parse_worksheet_from_rows([_row(
                "v2.0.test.0001", "280461", "confirm",
                tier2_classifiable="maybe",
            )])
        self.assertIn("tier2_classifiable", str(ctx.exception))


class WorksheetColumns(unittest.TestCase):
    def test_columns_include_all_core4_fields(self) -> None:
        cols = set(WORKSHEET_COLUMNS)
        for required in ("expert_hs6", "confidence_tier", "cited_evidence_ids",
                         "rationale_short", "tier2_classifiable",
                         "adjudication_status",
                         "adjudication_winning_evidence_id",
                         "adjudication_rubric_score"):
            self.assertIn(required, cols)


if __name__ == "__main__":
    unittest.main()
