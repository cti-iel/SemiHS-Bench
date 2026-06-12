from pathlib import Path
import tempfile
import unittest
from typing import List, Optional

from src.annotation.boundary_detector import (
    BOUNDARY_TAGS,
    compose_boundary_note,
    detect_boundaries,
    load_boundary_tags,
)
from src.annotation.difficulty_tagger import (
    annotate_records,
    build_annotation_report,
    infer_tier2_classifiable,
)
from src.collectors.hts_taxonomy import load_taxonomy_csv


def degraded_record(
    *,
    description: str,
    hs6_label: str,
    hs4_label: str,
    justification_text: str = "",
    keywords: Optional[List[str]] = None,
    tier3_part_name: str = "",
) -> dict:
    return {
        "id": "SH-009999",
        "canonical_id": "CP-009999",
        "tier1_description": description,
        "tier1_source": "CROSS",
        "tier2_minimal": {"part_name": tier3_part_name, "manufacturer": ""},
        "tier2_provenance": "degraded_manual",
        "hs6_label": hs6_label,
        "hs4_label": hs4_label,
        "hs2_label": hs6_label[:2],
        "label_source": "CROSS",
        "difficulty_tags": [],
        "justification_text": justification_text,
        "keywords": keywords or [],
        "bol_metadata": None,
        "source_reference": "R-1",
        "source_metadata": {},
        "degradation_metadata": {"mpn_extracted": False},
    }


class AnnotationTests(unittest.TestCase):
    def _taxonomy(self) -> object:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "taxonomy.csv"
            path.write_text(
                "\n".join(
                    [
                        "hs6,description",
                        "854231,Processors and controllers combined with memories",
                        "854232,Electronic integrated circuits memories",
                        "854233,Electronic integrated circuits amplifiers",
                        "854239,Other electronic integrated circuits",
                        "903082,Instruments for measuring or checking semiconductor wafers or devices",
                        "903149,Optical instruments for measuring or checking",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return load_taxonomy_csv(str(path))

    def test_annotation_uses_taxonomy_overlap_for_ambiguity(self) -> None:
        taxonomy = self._taxonomy()
        record = degraded_record(
            description="Integrated processor controller with memory and amplifier functions.",
            hs6_label="854231",
            hs4_label="8542",
            justification_text="Classification under integrated circuit provisions.",
            tier3_part_name="processor controller 3.3 V",
        )
        annotated = annotate_records([record], taxonomy)[0]
        self.assertEqual(annotated["ambiguity_score"], "3+")

    def test_classification_driver_prefers_justification_text(self) -> None:
        taxonomy = self._taxonomy()
        record = degraded_record(
            description="Controller module used in telecom equipment.",
            hs6_label="854239",
            hs4_label="8542",
            justification_text="The goods are made of ceramic and copper materials.",
            tier3_part_name="controller",
        )
        annotated = annotate_records([record], taxonomy)[0]
        self.assertEqual(annotated["classification_driver"], "material")

    def test_multiple_boundary_tags_can_coexist(self) -> None:
        taxonomy = self._taxonomy()
        record = degraded_record(
            description="Monolithic integrated circuit with processor logic alongside discrete transistor and diode functions.",
            hs6_label="854239",
            hs4_label="8542",
            tier3_part_name="module",
        )
        # Slate distractor 854231 sits in the gold's own 8542 IC-function cluster
        # (sibling split); 854110 is an 8541 discrete device (cross-family 8541 vs 8542).
        record["candidate_set"] = {"codes": ["854239", "854231", "854110", "854141"]}
        annotated = annotate_records([record], taxonomy)[0]
        self.assertIn("8542_ic_function", annotated["boundary_tags"])
        self.assertIn("8541_vs_8542", annotated["boundary_tags"])

    def test_tier3_classifiability_rules(self) -> None:
        mpn_record = degraded_record(description="x", hs6_label="854231", hs4_label="8542", tier3_part_name="STM32F407VGT6")
        partial_record = degraded_record(description="x", hs6_label="854231", hs4_label="8542", tier3_part_name="MCU controller")
        yes_record = degraded_record(description="x", hs6_label="854231", hs4_label="8542", tier3_part_name="MCU 3.3 V")

        self.assertEqual(infer_tier2_classifiable(mpn_record), "no")
        self.assertEqual(infer_tier2_classifiable(partial_record), "partial")
        self.assertEqual(infer_tier2_classifiable(yes_record), "yes")

    def test_annotation_report_counts_boundary_tags(self) -> None:
        taxonomy = self._taxonomy()
        rec_ic = degraded_record(
            description="Monolithic integrated circuit with processor logic.",
            hs6_label="854239",
            hs4_label="8542",
            tier3_part_name="module",
        )
        rec_ic["candidate_set"] = {"codes": ["854239", "854231"]}
        rec_meas = degraded_record(
            description="Device for measuring electrical optical semiconductor wafers.",
            hs6_label="903082",
            hs4_label="9030",
            tier3_part_name="sensor 5 V",
        )
        rec_meas["candidate_set"] = {"codes": ["903082", "903090"]}
        records = annotate_records([rec_ic, rec_meas], taxonomy)
        report = build_annotation_report(records)
        self.assertEqual(report["total_records"], 2)
        self.assertGreaterEqual(report["boundary_case_counts"]["8542_ic_function"], 1)


class BoundaryDetectorTests(unittest.TestCase):
    def _record(self, hs6, hs4, codes, *, text="", justification=""):
        return {
            "hs6_label": hs6,
            "hs4_label": hs4,
            "tier1_description": text,
            "justification_text": justification,
            "candidate_set": {"codes": codes},
        }

    def test_config_ids_match_module_constant_and_notes(self) -> None:
        specs = load_boundary_tags()
        ids = tuple(spec.tag_id for spec in specs)
        self.assertEqual(ids, BOUNDARY_TAGS)
        self.assertEqual(len(ids), 25)
        for spec in specs:
            self.assertIn(spec.group, ("sibling_split", "cross_family"))
            self.assertTrue(spec.note, f"{spec.tag_id} has no note")
            self.assertTrue(all(spec.sides), f"{spec.tag_id} has an empty side")

    def test_sibling_split_needs_a_distractor_in_the_cluster(self) -> None:
        # Gold + a sibling distractor in the same 8542 IC cluster -> fires.
        rec = self._record("854231", "8542", ["854231", "854232", "854110", "854141"])
        self.assertIn("8542_ic_function", detect_boundaries(rec))
        # Gold alone in its cluster (no sibling distractor) -> does not fire.
        rec_no = self._record("854231", "8542", ["854231", "854110", "854141", "854160"])
        self.assertNotIn("8542_ic_function", detect_boundaries(rec_no))

    def test_cross_family_fires_via_slate(self) -> None:
        # Gold 8541 discrete + an 8542 IC distractor on the opposing side.
        rec = self._record("854110", "8541", ["854110", "854231", "854121", "854129"])
        self.assertIn("8541_vs_8542", detect_boundaries(rec))

    def test_cross_family_fires_via_keywords(self) -> None:
        # No opposing-side distractor on the slate, but the text carries
        # evidence for both sides of the discrete-vs-IC frontier.
        rec = self._record(
            "854110",
            "8541",
            ["854110", "854121", "854129", "854160"],
            text="Rectifier diode module incorporating a monolithic integrated circuit controller.",
        )
        self.assertIn("8541_vs_8542", detect_boundaries(rec))

    def test_compose_boundary_note(self) -> None:
        self.assertEqual(compose_boundary_note([]), "")
        single = compose_boundary_note(["2804_gas_purity"])
        self.assertIn("2804", single)
        multi = compose_boundary_note(["2804_gas_purity", "doped_vs_undoped"])
        # Order preserved, both notes present, joined with a space.
        self.assertTrue(multi.startswith(single))
        self.assertIn("Doped vs undoped", multi)
        with self.assertRaises(ValueError):
            compose_boundary_note(["not_a_tag"])


if __name__ == "__main__":
    unittest.main()
