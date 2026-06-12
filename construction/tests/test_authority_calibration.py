"""Tests for src/annotation/authority_calibration.py.

Builds small synthetic fixtures in a temp dir and asserts the calibration
calibration scorer produces the expected metrics.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.annotation.authority_calibration import (
    _normalize_hs6,
    _parse_cited,
    score_calibration,
)


RATER_COLUMNS = [
    "calibration_id",
    "source",
    "jurisdiction",
    "language",
    "tier1_text",
    "tier2_part_name",
    "tier2_manufacturer",
    "tier2_extra_details",
    "tier3_part_name",
    "tier3_manufacturer",
    "subject_terms",
    "rater_hs6",
    "rater_confidence_tier",
    "rater_cited_evidence_ids",
    "rater_rationale_short",
]


def _write_truth(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_rater_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RATER_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({col: r.get(col, "") for col in RATER_COLUMNS})


def _write_ref_corpus(path: Path, entries: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _truth_row(cid: str, hs6: str, **overrides: Any) -> Dict[str, Any]:
    base = {
        "calibration_id": cid,
        "authoritative_hs6": hs6,
        "authoritative_hs4": hs6[:4],
        "evidence_id": f"CROSS-{cid}",
        "source": "CROSS",
        "jurisdiction": "US",
        "language": "en",
        "tier1_redactions_applied": 0,
        "source_rationale_contained_hs6": False,
        "source_url": None,
        "ruling_date": None,
    }
    base.update(overrides)
    return base


def _rater_row(cid: str, proposed_hs6: str, *,
               confidence: str = "medium",
               cited: str = "",
               source: str = "CROSS",
               language: str = "en") -> Dict[str, str]:
    return {
        "calibration_id": cid,
        "source": source,
        "jurisdiction": "US",
        "language": language,
        "tier1_text": "stub",
        "tier2_part_name": "stub",
        "tier2_manufacturer": "",
        "tier2_extra_details": "[]",
        "tier3_part_name": "stub",
        "tier3_manufacturer": "",
        "subject_terms": "[]",
        "rater_hs6": proposed_hs6,
        "rater_confidence_tier": confidence,
        "rater_cited_evidence_ids": cited,
        "rater_rationale_short": "test",
    }


class NormalizeHS6(unittest.TestCase):
    def test_solid_6_digits_ok(self) -> None:
        self.assertEqual(_normalize_hs6("854231"), ("854231", "ok"))

    def test_dotted_form_normalized(self) -> None:
        self.assertEqual(_normalize_hs6("8542.31"), ("854231", "ok"))

    def test_htsus_truncated_to_6(self) -> None:
        self.assertEqual(_normalize_hs6("8542.31.0001"), ("854231", "ok"))

    def test_4_digit_flagged(self) -> None:
        canon, status = _normalize_hs6("8542")
        self.assertEqual(canon, "8542")
        self.assertEqual(status, "malformed_4")

    def test_blank_flagged(self) -> None:
        self.assertEqual(_normalize_hs6(""), ("", "blank"))
        self.assertEqual(_normalize_hs6("   "), ("", "blank"))

    def test_malformed_other(self) -> None:
        canon, status = _normalize_hs6("abc")
        self.assertEqual(canon, "")
        self.assertEqual(status, "malformed_other")


class ParseCited(unittest.TestCase):
    def test_comma_separated(self) -> None:
        self.assertEqual(_parse_cited("EBTI-DE-1,CROSS-N123"),
                         ["EBTI-DE-1", "CROSS-N123"])

    def test_semicolon_separated(self) -> None:
        self.assertEqual(_parse_cited("EBTI-DE-1; CROSS-N123"),
                         ["EBTI-DE-1", "CROSS-N123"])

    def test_blank(self) -> None:
        self.assertEqual(_parse_cited(""), [])
        self.assertEqual(_parse_cited("   "), [])

    def test_strips_whitespace(self) -> None:
        self.assertEqual(_parse_cited(" A , B "), ["A", "B"])


class ScoreCalibration(unittest.TestCase):
    def test_perfect_raters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 11)]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = [_rater_row(t["calibration_id"], t["authoritative_hs6"],
                                  confidence="high")
                       for t in truth]
            rater_b = list(rater_a)
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            self.assertEqual(report["n_joined_records"], 10)
            self.assertEqual(
                report["per_rater_authority_accuracy"]["rater_a"]["accuracy"], 1.0
            )
            self.assertEqual(
                report["joint_authority_accuracy"]["accuracy"], 1.0
            )
            self.assertTrue(report["acceptance"]["passes"])

    def test_below_acceptance_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # 10 records, half correct, half wrong → 0.5 joint accuracy
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 11)]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = []
            rater_b = []
            for i, t in enumerate(truth):
                if i < 5:
                    rater_a.append(_rater_row(t["calibration_id"], "854231"))
                    rater_b.append(_rater_row(t["calibration_id"], "854231"))
                else:
                    rater_a.append(_rater_row(t["calibration_id"], "854239"))
                    rater_b.append(_rater_row(t["calibration_id"], "854239"))
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            self.assertEqual(report["joint_authority_accuracy"]["accuracy"], 0.5)
            self.assertFalse(report["acceptance"]["passes"])

    def test_high_confidence_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 21)]
            _write_truth(tmp / "truth.jsonl", truth)
            # Rater A: 15 high (all correct), 5 low (wrong). Within-high accuracy=1.0.
            rater_a = []
            for i, t in enumerate(truth):
                if i < 15:
                    rater_a.append(_rater_row(t["calibration_id"], "854231",
                                              confidence="high"))
                else:
                    rater_a.append(_rater_row(t["calibration_id"], "854239",
                                              confidence="low"))
            rater_b = list(rater_a)
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            ca = report["confidence_calibration"]["rater_a"]
            self.assertEqual(ca["n_high"], 15)
            self.assertEqual(ca["accuracy_within_high"], 1.0)
            self.assertTrue(ca["meets_target"])

    def test_citation_usefulness_peek_detected(self) -> None:
        """If a rater cites the actual authoritative ruling, the peek-test
        rate should fire."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 6)]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = []
            for i, t in enumerate(truth):
                # All correct; 3 of 5 cite the actual authoritative ruling
                cited = t["evidence_id"] if i < 3 else "OTHER-EID"
                rater_a.append(_rater_row(t["calibration_id"], "854231",
                                          confidence="high", cited=cited))
            rater_b = list(rater_a)
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            # Tiny corpus: every "OTHER-EID" doesn't resolve.
            _write_ref_corpus(tmp / "corpus.jsonl", [
                {"evidence_id": t["evidence_id"], "hs6_label": "854231"}
                for t in truth
            ])
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                ref_corpus_path=tmp / "corpus.jsonl",
                n_boot=200,
            )
            cit = report["citation_usefulness"]["rater_a"]
            self.assertEqual(cit["n_reached_authority"], 5)
            # 3/5 cited the actual authoritative evidence_id.
            self.assertAlmostEqual(cit["cited_authoritative_ruling_rate"], 0.6)

    def test_format_issues_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 5)]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = [
                _rater_row("CAL-001", "854231"),       # ok
                _rater_row("CAL-002", "8542"),          # malformed_4
                _rater_row("CAL-003", ""),               # blank
                _rater_row("CAL-004", "abc"),            # malformed_other
            ]
            rater_b = [_rater_row(t["calibration_id"], "854231") for t in truth]
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            issues = report["format_issues"]["rater_a"]
            self.assertEqual(issues["ok"], 1)
            self.assertEqual(issues["malformed_4_digit"], 1)
            self.assertEqual(issues["blank"], 1)
            self.assertEqual(issues["malformed_other"], 1)

    def test_missing_rater_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 6)]
            _write_truth(tmp / "truth.jsonl", truth)
            # Rater A skips CAL-003 entirely.
            rater_a = [_rater_row(t["calibration_id"], "854231")
                       for t in truth if t["calibration_id"] != "CAL-003"]
            rater_b = [_rater_row(t["calibration_id"], "854231") for t in truth]
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            self.assertEqual(report["n_joined_records"], 4)  # CAL-003 dropped
            self.assertIn("CAL-003",
                          report["missing"]["rater_a_missing_calibration_ids"])

    def test_per_hs4_stratification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [
                _truth_row("CAL-001", "854231"),
                _truth_row("CAL-002", "854110"),
                _truth_row("CAL-003", "854110"),
            ]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = [
                _rater_row("CAL-001", "854231"),  # correct
                _rater_row("CAL-002", "854110"),  # correct
                _rater_row("CAL-003", "854121"),  # wrong (same HS4 different HS6)
            ]
            rater_b = list(rater_a)
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            by_hs4 = report["stratified_accuracy"]["by_hs4"]["rater_a"]
            self.assertEqual(by_hs4["8542"]["accuracy"], 1.0)
            self.assertEqual(by_hs4["8541"]["accuracy"], 0.5)


class AcceptanceCompleteness(unittest.TestCase):
    """Per the calibration protocol, acceptance must verify completeness AND
    accuracy. Partial submissions must not slip past with a high joint
    score on a subset of records."""

    def test_complete_perfect_submission_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 11)]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = [_rater_row(t["calibration_id"], t["authoritative_hs6"])
                       for t in truth]
            rater_b = list(rater_a)
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            accept = report["acceptance"]
            self.assertTrue(accept["accuracy_pass"])
            self.assertTrue(accept["completeness"]["complete"])
            self.assertTrue(accept["passes"])

    def test_partial_submission_blocked_even_if_accuracy_is_1(self) -> None:
        """Reviewer's P1 scenario: rater A submits only half the records
        (the half they're confident about). joint_accuracy = 1.0 on the
        joined subset but the calibration is incomplete — must NOT pass."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 11)]
            _write_truth(tmp / "truth.jsonl", truth)
            # Rater A skips 5 records (CAL-006..CAL-010).
            rater_a = [
                _rater_row(t["calibration_id"], "854231")
                for t in truth if int(t["calibration_id"].split("-")[1]) <= 5
            ]
            rater_b = [_rater_row(t["calibration_id"], "854231") for t in truth]
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            accept = report["acceptance"]
            # Joint accuracy on the joined 5 records is 1.0 — accuracy alone passes.
            self.assertEqual(report["joint_authority_accuracy"]["accuracy"], 1.0)
            self.assertTrue(accept["accuracy_pass"])
            # But completeness FAILS, so the overall gate must be False.
            self.assertFalse(accept["completeness"]["complete"])
            self.assertEqual(accept["completeness"]["rater_a_missing_count"], 5)
            self.assertFalse(accept["passes"])

    def test_blank_hs6_rows_block_acceptance(self) -> None:
        """A rater submitting 60 rows but leaving some hs6 blank should
        not pass the gate, even when joint accuracy on the parseable
        ones is high."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 6)]
            _write_truth(tmp / "truth.jsonl", truth)
            rater_a = [
                _rater_row("CAL-001", "854231"),
                _rater_row("CAL-002", "854231"),
                _rater_row("CAL-003", ""),  # blank
                _rater_row("CAL-004", "854231"),
                _rater_row("CAL-005", "854231"),
            ]
            rater_b = [_rater_row(t["calibration_id"], "854231") for t in truth]
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            accept = report["acceptance"]
            self.assertFalse(accept["completeness"]["complete"])
            self.assertEqual(accept["completeness"]["rater_a_blank_hs6_count"], 1)
            self.assertFalse(accept["passes"])

    def test_low_accuracy_blocks_even_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = [_truth_row(f"CAL-{i:03d}", "854231") for i in range(1, 11)]
            _write_truth(tmp / "truth.jsonl", truth)
            # Both raters wrong on 5 of 10 → joint accuracy = 0.5 (below 0.70).
            rater_a = [
                _rater_row(t["calibration_id"],
                            "854231" if i < 5 else "854239")
                for i, t in enumerate(truth)
            ]
            rater_b = list(rater_a)
            _write_rater_csv(tmp / "a.csv", rater_a)
            _write_rater_csv(tmp / "b.csv", rater_b)
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            accept = report["acceptance"]
            self.assertTrue(accept["completeness"]["complete"])
            self.assertFalse(accept["accuracy_pass"])
            self.assertFalse(accept["passes"])

    def test_empty_truth_blocks_acceptance(self) -> None:
        """Degenerate case — empty truth file should not be treated as a
        pass even though n_joined == n_truth == 0."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_truth(tmp / "truth.jsonl", [])
            _write_rater_csv(tmp / "a.csv", [])
            _write_rater_csv(tmp / "b.csv", [])
            report = score_calibration(
                tmp / "truth.jsonl", tmp / "a.csv", tmp / "b.csv",
                n_boot=200,
            )
            accept = report["acceptance"]
            self.assertFalse(accept["completeness"]["complete"])
            self.assertFalse(accept["passes"])


if __name__ == "__main__":
    unittest.main()
