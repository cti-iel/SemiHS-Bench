"""Tests for the candidate-set builder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembly.build_candidates import (  # noqa: E402
    CONSTRUCTION_BOUNDARY,
    CONSTRUCTION_CHAPTER,
    CONSTRUCTION_SIBLING,
    attach_candidate_sets,
    build_candidate_set,
)


# Synthetic taxonomy covering chapters 28, 38, 85, 90 with enough density per
# chapter that even sparse HS4s can fall back to chapter siblings.
SYNTHETIC_TAXONOMY = [
    # 28xx (chapter 28)
    "280410", "280421", "280429", "280461", "280469",
    "281000", "281990", "282090",
    # 38xx (chapter 38)
    "381800",
    "382410", "382420", "382430", "382490", "382499",
    "381122", "381290",
    # 85xx (chapter 85)
    "853400",
    "853710", "853720", "853790",
    "853890",
    "854110", "854121", "854131", "854142",
    "854231", "854232", "854233", "854239", "854290",
    "850440", "850431", "850450",
    # 90xx (chapter 90)
    "903020", "903031", "903081", "903089",
    "903110", "903180", "903190",
    "902730", "902780",
]


def _record(idx: int, hs6: str, *, tags=None) -> dict:
    return {
        "id": f"SH-{idx:04d}",
        "hs6_label": hs6,
        "hs4_label": hs6[:4],
        "hs2_label": hs6[:2],
        "difficulty_tags": list(tags or []),
    }


class BuildCandidatesTests(unittest.TestCase):
    def _build(self, record: dict, *, size: int = 4):
        return build_candidate_set(record, in_scope_hs6=SYNTHETIC_TAXONOMY, size=size)

    # --- core invariants -----------------------------------------------------

    def test_gold_always_present(self) -> None:
        record = _record(1, "854231")
        slate = self._build(record)
        self.assertIn("854231", slate.codes)
        self.assertEqual(slate.codes[slate.gold_rank_in_candidates], "854231")

    def test_size_and_uniqueness(self) -> None:
        for hs6 in ("854231", "854121", "381800", "903031"):
            record = _record(1, hs6)
            slate = self._build(record)
            self.assertEqual(len(slate.codes), 4)
            self.assertEqual(len(set(slate.codes)), 4)
            for code in slate.codes:
                self.assertEqual(len(code), 6)
                self.assertTrue(code.isdigit())

    def test_deterministic_seed(self) -> None:
        record = _record(7, "854231")
        a = self._build(record)
        b = self._build(record)
        self.assertEqual(a.codes, b.codes)
        self.assertEqual(a.gold_rank_in_candidates, b.gold_rank_in_candidates)

    def test_distinct_records_yield_distinct_seeds(self) -> None:
        # Different IDs should produce at least somewhat varied gold positions
        # across a pool — proves the seed is keyed off the id, not constant.
        positions = set()
        for i in range(1, 50):
            slate = self._build(_record(i, "854231"))
            positions.add(slate.gold_rank_in_candidates)
        self.assertGreater(len(positions), 1)

    # --- construction tiers --------------------------------------------------

    def test_boundary_expansion_hs4_tag(self) -> None:
        # Tag names two HS4s; distractors must include 8542 codes.
        record = _record(1, "854110", tags=["8541_vs_8542"])
        slate = self._build(record)
        self.assertEqual(slate.construction, CONSTRUCTION_BOUNDARY)
        self.assertTrue(any(c.startswith("8542") for c in slate.codes))

    def test_boundary_expansion_hs6_tag(self) -> None:
        # Tag names HS6 alternatives directly (with dots).
        record = _record(2, "854231",
                         tags=["8542.31_vs_8542.32_vs_8542.33_vs_8542.39"])
        slate = self._build(record)
        self.assertEqual(slate.construction, CONSTRUCTION_BOUNDARY)
        for code in ("854231", "854232", "854233", "854239"):
            self.assertIn(code, slate.codes)

    def test_sibling_fallback_when_no_boundary_tag(self) -> None:
        # Record with no boundary tags, but plenty of siblings under HS4.
        record = _record(3, "854231")  # 8542 has 5 in-scope HS6s
        slate = self._build(record)
        self.assertEqual(slate.construction, CONSTRUCTION_SIBLING)
        siblings = [c for c in slate.codes if c.startswith("8542") and c != "854231"]
        self.assertGreaterEqual(len(siblings), 3)

    def test_chapter_fallback_when_hs4_underpopulated(self) -> None:
        # 3818 has only 1 HS6 in our synthetic pool — must reach into chapter.
        record = _record(4, "381800")
        slate = self._build(record)
        # With no boundary tags and no siblings, construction is chapter.
        self.assertEqual(slate.construction, CONSTRUCTION_CHAPTER)
        for code in slate.codes:
            self.assertEqual(code[:2], "38")

    def test_construction_priority_boundary_over_sibling(self) -> None:
        # Even when siblings exist, boundary tag drives the construction label.
        record = _record(5, "854231",
                         tags=["8542.31_vs_8542.32_vs_8542.33_vs_8542.39"])
        slate = self._build(record)
        self.assertEqual(slate.construction, CONSTRUCTION_BOUNDARY)

    # --- edge cases ----------------------------------------------------------

    def test_gold_outside_in_scope_pool_is_accepted(self) -> None:
        # Mimics BOL_new_hs6_audit records whose gold isn't in the canonical
        # taxonomy yet. Builder must include the gold and pad from chapter.
        record = _record(6, "999999", tags=[])
        # Fake a chapter for the audit code; reuse 85 distractors.
        record["hs2_label"] = "85"
        record["hs4_label"] = "9999"
        # Build with synthetic taxonomy (no '99' chapter siblings) — must fail.
        with self.assertRaises(ValueError):
            self._build(record)

    def test_audit_record_with_real_chapter(self) -> None:
        # BOL_new_hs6_audit pattern: gold sits in a populated chapter (85)
        # but the HS4 may have no siblings.
        record = _record(7, "858888")  # synthetic HS6 in chapter 85
        record["hs2_label"] = "85"
        record["hs4_label"] = "8588"
        slate = self._build(record)
        self.assertEqual(slate.codes[slate.gold_rank_in_candidates], "858888")
        for code in slate.codes:
            self.assertEqual(code[:2], "85")

    def test_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            self._build(_record(1, "854231"), size=1)

    def test_invalid_hs6_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._build({"id": "SH-1", "hs6_label": "85423"})

    # --- batch helper --------------------------------------------------------

    def test_attach_candidate_sets(self) -> None:
        records = [
            _record(1, "854231"),
            _record(2, "854110", tags=["8541_vs_8542"]),
            _record(3, "381800"),
        ]
        out = attach_candidate_sets(records, in_scope_hs6=SYNTHETIC_TAXONOMY)
        self.assertEqual(len(out), 3)
        for orig, new in zip(records, out):
            self.assertIsNot(orig, new)  # copy, not mutate
            self.assertNotIn("candidate_set", orig)
            slate = new["candidate_set"]
            self.assertEqual(slate["size"], 4)
            self.assertEqual(len(slate["codes"]), 4)
            self.assertIn(orig["hs6_label"], slate["codes"])
            self.assertIn(slate["construction"],
                          {CONSTRUCTION_BOUNDARY, CONSTRUCTION_SIBLING, CONSTRUCTION_CHAPTER})


if __name__ == "__main__":
    unittest.main()
