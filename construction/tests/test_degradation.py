from pathlib import Path
import unittest
from typing import List, Optional

from src.annotation.difficulty_tagger import annotate_records
from src.collectors.hts_taxonomy import load_taxonomy_csv
from src.processing.degrader import generate_tiers


FIXTURES = Path(__file__).parent / "fixtures"


def canonical_product(
    canonical_id: str,
    description: str,
    hs6_label: str = "854231",
    primary_source: str = "CROSS",
    keywords: Optional[List[str]] = None,
    justification_text: str = "Classification note.",
    manufacturer_hint: str = "",
) -> dict:
    return {
        "canonical_id": canonical_id,
        "canonical_description": description,
        "hs6_label": hs6_label,
        "hs4_label": hs6_label[:4],
        "hs2_label": hs6_label[:2],
        "primary_source": primary_source,
        "primary_reference": canonical_id,
        "language": "en",
        "justification_text": justification_text,
        "keywords": keywords or [],
        "source_evidence": [
            {
                "source": primary_source,
                "reference": canonical_id,
                "description": description,
                "keywords": keywords or [],
                "source_metadata": {},
            }
        ],
        "manufacturer_hint": manufacturer_hint,
        "manufacturer_hint_source": "authoritative_text" if manufacturer_hint else "",
        "merge_confidence": 1.0,
        "selection_score": 50.0,
        "sampling_metadata": {
            "boundary_tags": [],
            "sample_priority": {},
            "sample_reason": "test",
            "source_count": 1,
        },
    }


class DegradationTests(unittest.TestCase):
    def test_generate_tiers_extracts_specs_and_real_mpn(self) -> None:
        canonical_records = [
            canonical_product(
                canonical_id="CP-000001",
                description=(
                    "Monolithic microcontroller integrated circuit designed for industrial automation "
                    "consisting of STM32F407VGT6 in LQFP 100 pins 168 MHz 1 MB 3.3 V 10 x 10 mm 2 g."
                ),
                keywords=["STM32F407VGT6", "microcontroller", "LQFP"],
            )
        ]

        outputs = generate_tiers(
            canonical_records,
            abbreviations_path="configs/abbreviations.csv",
            rules_path="configs/degradation_rules.yaml",
        )

        record = outputs["records"][0]
        tier2 = record["tier2_minimal"]

        self.assertEqual(record["id"], "SH-000001")
        self.assertEqual(record["label_source"], "CROSS")
        self.assertLessEqual(len(tier2["part_name"]), 40)
        self.assertEqual(tier2["part_name"], "STM32F407VGT6")
        self.assertEqual(record["tier2_provenance"], "natural_mpn")
        self.assertFalse(record["degradation_metadata"]["tier2_equals_tier1"])
        self.assertTrue(record["degradation_metadata"]["mpn_extracted"])

    def test_generate_tiers_uses_manufacturer_hint_when_available(self) -> None:
        outputs = generate_tiers(
            [
                canonical_product(
                    canonical_id="CP-000001A",
                    description="Operational amplifier LM358DR in SOIC 8 package.",
                    hs6_label="854233",
                    manufacturer_hint="Texas Instruments",
                    keywords=["LM358DR", "operational amplifier"],
                )
            ],
            abbreviations_path="configs/abbreviations.csv",
            rules_path="configs/degradation_rules.yaml",
        )
        self.assertEqual(outputs["records"][0]["tier2_minimal"]["manufacturer"], "Texas Instruments")

    def test_generate_tiers_regression_for_known_unchanged_weak_patterns(self) -> None:
        records = [
            canonical_product(
                canonical_id="CP-000010",
                description="VERY LOW PROFILE ABSOLUTE POSITION ROTARY ELECTRIC ENCODER",
                hs6_label="903180",
            ),
            canonical_product(
                canonical_id="CP-000011",
                description="SORBITAN MONEPALMITATE CAS 26266-57-9 APPEARANCE: SOLID",
                hs6_label="382499",
            ),
            canonical_product(
                canonical_id="CP-000012",
                description="4N HYDROGEN CHLORIDE IN DIOXANE PHYSICAL PROPERTIES: LIQUID",
                hs6_label="382499",
            ),
        ]

        outputs = generate_tiers(
            records,
            abbreviations_path="configs/abbreviations.csv",
            rules_path="configs/degradation_rules.yaml",
        )
        self.assertEqual(outputs["summary"]["weak_record_count"], 0)
        for record in outputs["records"]:
            self.assertFalse(record["degradation_metadata"]["weak_degradation_reasons"])

    def test_generate_tiers_rejects_false_positive_mpn_matches(self) -> None:
        canonical_records = [
            canonical_product(
                canonical_id="CP-000002",
                description=(
                    "Silicon powder CAS No 7440-21-3 not less than 99.99 percent purity by weight, "
                    "classified under heading 280461 and packed in 5 kg drums."
                ),
                hs6_label="280461",
                primary_source="EBTI",
                keywords=["7440-21-3", "280461", "5 kg"],
            )
        ]

        outputs = generate_tiers(
            canonical_records,
            abbreviations_path="configs/abbreviations.csv",
            rules_path="configs/degradation_rules.yaml",
        )

        record = outputs["records"][0]
        self.assertFalse(record["degradation_metadata"]["mpn_extracted"])
        self.assertNotEqual(record["tier2_minimal"]["part_name"], "7440-21-3")
        self.assertNotEqual(record["tier2_minimal"]["part_name"], "280461")
        self.assertLessEqual(len(record["tier2_minimal"]["part_name"]), 40)

    def test_annotated_degraded_records_build_final_schema_shape(self) -> None:
        canonical_records = [
            canonical_product(
                canonical_id="CP-000004",
                description="Semiconductor pressure sensor array with analog amplifier and 5 V supply.",
                hs6_label="854239",
                primary_source="CROSS",
                keywords=["pressure sensor", "5 V"],
            )
        ]
        outputs = generate_tiers(
            canonical_records,
            abbreviations_path="configs/abbreviations.csv",
            rules_path="configs/degradation_rules.yaml",
        )
        taxonomy = load_taxonomy_csv(str(FIXTURES / "taxonomy" / "hts.csv"))
        annotated = annotate_records(outputs["records"], taxonomy)
        record = annotated[0]
        self.assertIn("classification_driver", record)
        self.assertIn("ambiguity_score", record)
        self.assertIn("tier2_classifiable", record)


if __name__ == "__main__":
    unittest.main()
