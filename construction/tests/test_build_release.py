"""Tests for scripts/build_release.py.

Exercises the release-gate functions and supporting helpers with
synthetic benchmark records, so the tests don't depend on the actual
benchmark records existing on disk yet. The on-disk pipeline integration test
(``test_pipeline_end_to_end``) covers the cross-stage data path
separately.
"""

from __future__ import annotations

import hashlib
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

_spec = importlib.util.spec_from_file_location(
    "build_release", ROOT / "scripts" / "build_release.py"
)
build = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(build)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _good_record(
    *,
    frozen_id: str = "v2.0.test.0001",
    hs6: str = "854231",
    source: str = "catalog",
    manufacturer: str = "Texas Instruments",
    label_source: str = "catalog_expert_validated",
    confidence_tier: str = "high",
    cited: List[str] | None = None,
    split: str = "test",
    candidate_codes: List[str] | None = None,
    boundary_tag: bool = False,
) -> Dict[str, Any]:
    if cited is None:
        cited = ["EBTI-DE-BTI-2024-1234", "CROSS-N567890"]
    if candidate_codes is None:
        candidate_codes = [hs6, "854110", "854239", "854290"]
    return {
        "id": "SH-" + frozen_id.split(".")[-1].zfill(5),
        "frozen_id": frozen_id,
        "split": split,
        "tier1_description": "Test product description for HS classification test.",
        "tier1_source": source,
        "tier2_minimal": {
            "part_name": f"PN-{frozen_id.split('.')[-1]}",
            "manufacturer": manufacturer,
        },
        "tier2_provenance": "natural_mpn",
        "tier2_classifiable": "yes",
        "hs6_label": hs6,
        "hs4_label": hs6[:4],
        "hs2_label": hs6[:2],
        "scope_tier": "core",
        "label_source": label_source,
        "confidence_tier": confidence_tier,
        "cited_evidence_ids": list(cited),
        "rationale_short": "Test rationale.",
        "adjudication_status": "single_reviewer",
        "adjudication_winning_evidence_id": None,
        "adjudication_rubric_score": None,
        "difficulty_tags": (["8541.10_vs_8542.31"] if boundary_tag else []),
        "justification_text": "Test justification.",
        "bol_metadata": (
            {"declared_hs": hs6, "hs_verified": True} if source == "BOL" else None
        ),
        "candidate_set": {
            "size": 4,
            "codes": candidate_codes,
            "construction": "sibling_heading",
            "gold_rank_in_candidates": 0,
        },
        "source_metadata": {"manufacturer_hint": manufacturer},
    }


def _good_reference_entry(
    *, evidence_id: str = "EBTI-DE-X1", hs6: str = "854231",
    source: str = "EBTI", jurisdiction: str = "EU-DE",
) -> Dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source": source,
        "jurisdiction": jurisdiction,
        "hs6_label": hs6,
        "ruling_date": "2024-01-01",
        "url": None,
        "tier1_text": "Sample reference-corpus tier1 text describing the ruling subject.",
        "tier2_minimal": {"part_name": "REF PART", "manufacturer": ""},
        "subject_terms": [],
        "rationale_excerpt": "",
        "wco_en_section": None,
    }


# ---------------------------------------------------------------------------
# Methodology compliance gate
# ---------------------------------------------------------------------------

# Synthetic reference-corpus jurisdiction lookup used by the evidence-coverage gate.
# Includes the citations seeded by _good_record() (EBTI-DE-BTI-2024-1234 +
# CROSS-N567890) so the default clean-records test passes resolution.
_CLEAN_LOOKUP = {
    "EBTI-DE-BTI-2024-1234": "EU-DE",
    "CROSS-N567890": "US",
    "EBTI-DE-X1": "EU-DE",
    "EBTI-DE-X2": "EU-DE",
    "EBTI-FR-X1": "EU-FR",
    "CROSS-N123": "US",
}


class MethodologyComplianceGate(unittest.TestCase):
    def test_clean_records_pass(self) -> None:
        records = [_good_record(frozen_id=f"v2.0.test.{i:04d}") for i in range(5)]
        result = build._gate_methodology_compliance(
            records, evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertTrue(result["no_pending_reaudit_labels"])
        self.assertTrue(result["no_legacy_label_sources"])
        self.assertTrue(result["every_record_has_evidence_id"])
        self.assertTrue(result["confidence_evidence_binding_holds"])
        self.assertTrue(result["all_cited_evidence_ids_resolve"])
        self.assertTrue(result["no_legacy_bronze_fields"])

    def test_pending_reaudit_label_caught(self) -> None:
        bad = _good_record(label_source="BOL_expert_validated_pending_reaudit")
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["no_pending_reaudit_labels"])
        self.assertEqual(result["_diagnostics"]["pending_remaining"], 1)

    def test_legacy_label_source_caught(self) -> None:
        bad = _good_record(label_source="EBTI")
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["no_legacy_label_sources"])

    def test_high_confidence_with_one_citation_caught(self) -> None:
        bad = _good_record(confidence_tier="high", cited=["EBTI-DE-X1"])
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["confidence_evidence_binding_holds"])

    def test_medium_confidence_with_zero_citations_caught(self) -> None:
        bad = _good_record(confidence_tier="medium", cited=[])
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["confidence_evidence_binding_holds"])

    def test_bronze_field_caught(self) -> None:
        bad = _good_record()
        bad["classification_driver"] = "function"
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["no_legacy_bronze_fields"])

    # ---------- PR #8 reviewer's P1.2: jurisdiction + ID resolution ----------

    def test_high_confidence_with_two_same_jurisdiction_citations_caught(self) -> None:
        """Two EU-DE citations satisfy the count rule but NOT the
        jurisdictional-diversity rule of the Core-4 evidence binding."""
        bad = _good_record(
            confidence_tier="high",
            cited=["EBTI-DE-X1", "EBTI-DE-X2"],  # both EU-DE
        )
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["confidence_evidence_binding_holds"])
        violations = result["_diagnostics"]["confidence_violations_sample"]
        self.assertTrue(any("≥2 distinct jurisdictions" in v["reason"]
                            for v in violations))

    def test_high_confidence_with_eu_and_us_citations_passes(self) -> None:
        good = _good_record(
            confidence_tier="high",
            cited=["EBTI-DE-X1", "CROSS-N123"],  # EU-DE + US
        )
        result = build._gate_methodology_compliance(
            [good], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertTrue(result["confidence_evidence_binding_holds"])

    def test_high_confidence_with_eu_and_eu_other_jurisdiction_passes(self) -> None:
        """Two distinct EU jurisdictions (DE + FR) count as cross-jurisdictional
        per the Core-4 evidence wording — the gate accepts these too."""
        good = _good_record(
            confidence_tier="high",
            cited=["EBTI-DE-X1", "EBTI-FR-X1"],  # EU-DE + EU-FR
        )
        result = build._gate_methodology_compliance(
            [good], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertTrue(result["confidence_evidence_binding_holds"])

    def test_unresolved_citation_caught(self) -> None:
        """A cited evidence_id that doesn't exist in the reference corpus
        must block the release. Previously the gate only counted citations
        and accepted nonsense IDs like 'EBTI-XX-FAKE'."""
        bad = _good_record(
            confidence_tier="high",
            cited=["EBTI-DE-X1", "FAKE-NONEXISTENT-1"],
        )
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["all_cited_evidence_ids_resolve"])
        self.assertFalse(result["confidence_evidence_binding_holds"])
        unresolved = result["_diagnostics"]["unresolved_citations_sample"]
        self.assertEqual(len(unresolved), 1)
        self.assertIn("FAKE-NONEXISTENT-1", unresolved[0]["unresolved"])

    def test_medium_confidence_with_unresolved_citation_caught(self) -> None:
        bad = _good_record(
            confidence_tier="medium",
            cited=["FAKE-NONEXISTENT-1"],
        )
        result = build._gate_methodology_compliance(
            [bad], evidence_jurisdiction=_CLEAN_LOOKUP,
        )
        self.assertFalse(result["confidence_evidence_binding_holds"])
        self.assertFalse(result["all_cited_evidence_ids_resolve"])

    def test_empty_evidence_jurisdiction_blocks_release(self) -> None:
        """Caller passes {} (no reference corpus loaded). Every cited id
        fails to resolve → release blocked. This is the safe default for
        a build run before the reference corpus is in place."""
        records = [_good_record(frozen_id=f"v2.0.test.{i:04d}") for i in range(2)]
        result = build._gate_methodology_compliance(
            records, evidence_jurisdiction={},
        )
        self.assertFalse(result["all_cited_evidence_ids_resolve"])
        self.assertFalse(result["confidence_evidence_binding_holds"])


# ---------------------------------------------------------------------------
# Scope gate
# ---------------------------------------------------------------------------

class ScopeGate(unittest.TestCase):
    def setUp(self) -> None:
        self.allow_hs6 = set(build._scope_lookup().keys())

    def test_in_scope_pass(self) -> None:
        result = build._gate_scope([_good_record(hs6="854231")], self.allow_hs6)
        self.assertTrue(result["no_dropped_hs4"])
        self.assertTrue(result["all_hs4_in_allow_set"])
        self.assertTrue(result["every_record_has_scope_tier"])

    def test_dropped_hs4_8543_caught(self) -> None:
        bad = _good_record()
        bad["hs4_label"] = "8543"
        bad["hs6_label"] = "854370"
        bad["hs2_label"] = "85"
        result = build._gate_scope([bad], self.allow_hs6)
        self.assertFalse(result["no_dropped_hs4"])

    def test_dropped_hs4_8534_caught(self) -> None:
        bad = _good_record()
        bad["hs4_label"] = "8534"
        bad["hs6_label"] = "853400"
        result = build._gate_scope([bad], self.allow_hs6)
        self.assertFalse(result["no_dropped_hs4"])

    def test_missing_scope_tier_caught(self) -> None:
        bad = _good_record()
        bad.pop("scope_tier", None)
        result = build._gate_scope([bad], self.allow_hs6)
        self.assertFalse(result["every_record_has_scope_tier"])


# ---------------------------------------------------------------------------
# HS rebalance gate
# ---------------------------------------------------------------------------

class HSRebalanceGate(unittest.TestCase):
    def test_no_caps_exceeded(self) -> None:
        records = [
            _good_record(frozen_id=f"v2.0.test.{i:04d}", hs6="854239")
            for i in range(60)  # exactly at cap
        ]
        result = build._gate_hs_rebalance(records)
        self.assertTrue(result["hs6_caps_respected"])

    def test_cap_violation_caught(self) -> None:
        records = [
            _good_record(frozen_id=f"v2.0.test.{i:04d}", hs6="854239")
            for i in range(61)  # 1 over cap
        ]
        result = build._gate_hs_rebalance(records)
        self.assertFalse(result["hs6_caps_respected"])
        violations = result["_diagnostics"]["cap_violations"]
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["hs6"], "854239")
        self.assertEqual(violations[0]["count"], 61)


# ---------------------------------------------------------------------------
# Splits & leakage gate
# ---------------------------------------------------------------------------

class SplitsLeakageGate(unittest.TestCase):
    def test_clean_split_passes(self) -> None:
        records = [
            _good_record(frozen_id="v2.0.dev.0001", split="dev",
                          boundary_tag=True),
            _good_record(frozen_id="v2.0.dev.0002", split="dev",
                          manufacturer="STMicroelectronics",
                          boundary_tag=True),
            _good_record(frozen_id="v2.0.test.0001", split="test",
                          manufacturer="Infineon Technologies",
                          boundary_tag=False),
        ]
        # Boundary share: 2/3 = 0.67 (out of range, but we test other gates here)
        # Override tier3 part_name to be split-distinct.
        records[0]["tier2_minimal"]["part_name"] = "PN-DEV-1"
        records[1]["tier2_minimal"]["part_name"] = "PN-DEV-2"
        records[2]["tier2_minimal"]["part_name"] = "PN-TEST-1"
        result = build._gate_splits_and_leakage(records)
        self.assertTrue(result["no_exact_mpn_in_both_splits"])
        self.assertTrue(result["no_manufacturer_product_family_leakage"])

    def test_mpn_leakage_caught(self) -> None:
        records = [
            _good_record(frozen_id="v2.0.dev.0001", split="dev"),
            _good_record(frozen_id="v2.0.test.0001", split="test",
                          manufacturer="STMicroelectronics"),
        ]
        # Same tier3.part_name in both splits → leakage
        records[0]["tier2_minimal"]["part_name"] = "SHARED-PN"
        records[1]["tier2_minimal"]["part_name"] = "SHARED-PN"
        result = build._gate_splits_and_leakage(records)
        self.assertFalse(result["no_exact_mpn_in_both_splits"])

    def test_manufacturer_family_leakage_caught(self) -> None:
        records = [
            _good_record(frozen_id="v2.0.dev.0001", split="dev", hs6="854231",
                          manufacturer="ACME"),
            _good_record(frozen_id="v2.0.test.0001", split="test", hs6="854239",
                          manufacturer="ACME"),
        ]
        # tier3 part_name distinct, but (manufacturer, hs4) overlaps → leakage
        records[1]["hs4_label"] = "8542"
        records[1]["hs6_label"] = "854239"
        records[0]["tier2_minimal"]["part_name"] = "PN-A"
        records[1]["tier2_minimal"]["part_name"] = "PN-B"
        result = build._gate_splits_and_leakage(records)
        self.assertFalse(result["no_manufacturer_product_family_leakage"])

    def test_boundary_share_in_range(self) -> None:
        # 40% boundary tagging → in [0.38, 0.45]
        records = []
        for i in range(10):
            r = _good_record(frozen_id=f"v2.0.test.{i:04d}",
                              boundary_tag=(i < 4))
            r["tier2_minimal"]["part_name"] = f"PN-{i}"
            records.append(r)
        result = build._gate_splits_and_leakage(records)
        self.assertTrue(result["boundary_share_in_range"])

    def test_boundary_share_out_of_range_caught(self) -> None:
        # 10% boundary tagging → below 0.38
        records = []
        for i in range(10):
            r = _good_record(frozen_id=f"v2.0.test.{i:04d}",
                              boundary_tag=(i < 1))
            r["tier2_minimal"]["part_name"] = f"PN-{i}"
            records.append(r)
        result = build._gate_splits_and_leakage(records)
        self.assertFalse(result["boundary_share_in_range"])


# ---------------------------------------------------------------------------
# Reference corpus coverage gate
# ---------------------------------------------------------------------------

class ReferenceCorpusCoverageGate(unittest.TestCase):
    def test_every_hs6_covered_passes(self) -> None:
        benchmark = [_good_record(hs6="854231")]
        ref = [
            _good_reference_entry(evidence_id=f"EBTI-DE-X{i}", hs6="854231")
            for i in range(3)
        ]
        result = build._gate_reference_corpus_coverage(benchmark, ref)
        self.assertTrue(result["every_benchmark_hs6_has_3_reference_entries"])

    def test_under_covered_hs6_caught(self) -> None:
        benchmark = [_good_record(hs6="854231")]
        ref = [_good_reference_entry(hs6="854231")]  # only 1
        result = build._gate_reference_corpus_coverage(benchmark, ref)
        self.assertFalse(result["every_benchmark_hs6_has_3_reference_entries"])
        self.assertEqual(result["_diagnostics"]["under_covered_count"], 1)


# ---------------------------------------------------------------------------
# Source mix gate
# ---------------------------------------------------------------------------

class SourceMixGate(unittest.TestCase):
    def test_target_mix_passes(self) -> None:
        # 100 records: 46% catalog, 54% BOL
        records = []
        for i in range(46):
            records.append(_good_record(
                frozen_id=f"v2.0.test.{i:04d}", source="catalog"))
        for i in range(54):
            records.append(_good_record(
                frozen_id=f"v2.0.test.{i+100:04d}", source="BOL"))
        result = build._gate_source_mix(records)
        self.assertTrue(result["source_mix_within_targets"])
        self.assertTrue(result["no_ebti_cross_records"])

    def test_ebti_cross_records_caught(self) -> None:
        bad = _good_record()
        bad["tier1_source"] = "EBTI"
        result = build._gate_source_mix([bad])
        self.assertFalse(result["no_ebti_cross_records"])


# ---------------------------------------------------------------------------
# Helpers + tree hash
# ---------------------------------------------------------------------------

class HelpersAndTreeHash(unittest.TestCase):
    def test_file_sha256_matches_shasum(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("hello world")
            path = Path(tf.name)
        try:
            got = build._file_sha256(path)
            expected = hashlib.sha256(b"hello world").hexdigest()
            self.assertEqual(got, expected)
        finally:
            path.unlink()

    def test_tree_hash_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a")
            (root / "b.txt").write_text("b")
            h1 = build.compute_tree_hash(root)
            h2 = build.compute_tree_hash(root)
            self.assertEqual(h1, h2)

    def test_tree_hash_changes_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a")
            h1 = build.compute_tree_hash(root)
            (root / "a.txt").write_text("a-modified")
            h2 = build.compute_tree_hash(root)
            self.assertNotEqual(h1, h2)

    def test_tree_hash_skips_dsstore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a")
            h_before = build.compute_tree_hash(root)
            (root / ".DS_Store").write_text("noise")
            h_after = build.compute_tree_hash(root)
            self.assertEqual(h_before, h_after)

    def test_distributions_empty_input(self) -> None:
        self.assertEqual(build._distributions([]), {})

    def test_distributions_populated(self) -> None:
        records = [_good_record(hs6="854231"), _good_record(hs6="854239")]
        dist = build._distributions(records)
        self.assertEqual(dist["hs6"], {"854231": 1, "854239": 1})
        self.assertIn("confidence_tier", dist)
        self.assertEqual(dist["confidence_tier"], {"high": 2})


# ---------------------------------------------------------------------------
# All-gates-pass aggregator
# ---------------------------------------------------------------------------

class AllGatesPass(unittest.TestCase):
    def test_skipped_gate_blocks_release(self) -> None:
        gates = {
            "scope": {"status": "skipped", "reason": "no records"},
            "release_artifacts": {"all_required_artifacts_present": True},
        }
        self.assertFalse(build._all_gates_pass(gates))

    def test_false_value_blocks_release(self) -> None:
        gates = {
            "scope": {"no_dropped_hs4": True},
            "calibration": {"calibration_report_present": False},
        }
        self.assertFalse(build._all_gates_pass(gates))

    def test_all_true_passes(self) -> None:
        gates = {
            "scope": {
                "no_dropped_hs4": True,
                "all_hs4_in_allow_set": True,
                "every_record_has_scope_tier": True,
                "_diagnostics": {"any": "value"},
            },
        }
        self.assertTrue(build._all_gates_pass(gates))


# ---------------------------------------------------------------------------
# CLI: --check exits 0 regardless of state
# ---------------------------------------------------------------------------

class CLI(unittest.TestCase):
    def test_check_mode_exits_zero(self) -> None:
        """--check is a survey; should never fail (even if everything is
        missing, the user gets an inventory they can act on)."""
        rc = build.main(["--check"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
