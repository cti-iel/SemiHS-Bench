import csv
from pathlib import Path
import tempfile
import unittest

from src.collectors.bol_collector import load_bol_imports
from src.collectors.catalog_collector import load_catalog_imports
from src.models import RawAuxiliaryRecord
from src.processing.auxiliary_enricher import enrich_records_with_auxiliary
from src.processing.mpn_resolver import best_match_details


FIXTURES = Path(__file__).parent / "fixtures"


def degraded_record() -> dict:
    return {
        "id": "SH-0001",
        "canonical_id": "CP-0001",
        "tier1_description": "Doped silicon wafer 200mm for semiconductor production.",
        "tier1_source": "CROSS",
        "tier2_minimal": {"part_name": "SI WAFER 200MM", "manufacturer": ""},
        "tier2_provenance": "degraded_manual",
        "hs6_label": "381800",
        "hs4_label": "3818",
        "hs2_label": "38",
        "label_source": "CROSS",
        "difficulty_tags": [],
        "ambiguity_score": 1,
        "classification_driver": "material",
        "tier2_classifiable": "partial",
        "justification_text": "Classification note.",
        "keywords": ["doped silicon wafer 200mm", "WaferWorks", "STM32F407VGT6"],
        "bol_metadata": None,
        "source_reference": "R-1",
        "source_metadata": {},
        "degradation_metadata": {"mpn_extracted": False},
    }


class AuxiliaryEnricherTests(unittest.TestCase):
    def test_enrichment_prefers_bol_for_tier2_and_catalog_mpn_for_tier3(self) -> None:
        record = degraded_record()
        bol_records = load_bol_imports(str(FIXTURES / "raw" / "bol" / "bol_sample.csv"))
        catalog_records = load_catalog_imports(str(FIXTURES / "raw" / "catalog" / "catalog_sample.jsonl"))

        outputs = enrich_records_with_auxiliary([record], bol_records, catalog_records, catalog_min_score=0.1)
        enriched = outputs["records"][0]

        self.assertEqual(enriched["bol_metadata"]["declared_hs"], "381800")
        self.assertTrue(enriched["bol_metadata"]["hs_verified"])
        self.assertEqual(enriched["bol_metadata"]["consignee"], "Example Consignee")
        self.assertIn("arrival_date", enriched["bol_metadata"])
        self.assertEqual(enriched["tier2_provenance"], "natural_mpn")
        self.assertEqual(enriched["tier2_minimal"]["part_name"], "STM32F407VGT6")

    def test_generic_bol_is_filtered_and_catalog_can_supply_tier2(self) -> None:
        record = {
            **degraded_record(),
            "tier1_description": "Microcontroller integrated circuit STM32F407VGT6 in LQFP 100 pins 168 MHz.",
            "hs6_label": "854231",
            "hs4_label": "8542",
            "hs2_label": "85",
            "keywords": ["STM32F407VGT6", "microcontroller"],
        }
        catalog_records = load_catalog_imports(str(FIXTURES / "raw" / "catalog" / "catalog_sample.jsonl"))

        with tempfile.TemporaryDirectory() as temp_dir:
            bol_path = Path(temp_dir) / "generic_bol.csv"
            bol_path.write_text(
                "\n".join(
                    [
                        "reference,description,shipper,consignee,port_origin,port_dest,quantity,declared_hs",
                        "BOL-0009,ELECTRONIC PARTS,Generic Shipper,Example,Busan,Long Beach,10,854231",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bol_records = load_bol_imports(str(bol_path))

        outputs = enrich_records_with_auxiliary([record], bol_records, catalog_records, catalog_min_score=0.1)
        enriched = outputs["records"][0]

        self.assertEqual(enriched["tier2_provenance"], "natural_mpn")
        self.assertIsNone(enriched.get("bol_metadata"))

    def test_exact_mpn_match_beats_generic_token_overlap(self) -> None:
        record = {
            **degraded_record(),
            "tier1_description": "Microcontroller integrated circuit STM32F407VGT6 in LQFP package.",
            "hs6_label": "854231",
            "hs4_label": "8542",
            "hs2_label": "85",
            "keywords": ["STM32F407VGT6", "microcontroller", "STMicroelectronics"],
        }
        generic_candidate = RawAuxiliaryRecord(
            source="catalog",
            reference="GENERIC-1",
            description="Microcontroller integrated circuit ARM Cortex controller",
            manufacturer="STMicroelectronics",
            mpn="",
            metadata={"provider": "catalog", "target_hs_family": "8542", "hs_heading": "8542"},
        )
        exact_candidate = RawAuxiliaryRecord(
            source="catalog",
            reference="EXACT-1",
            description="IC MCU 32BIT ARM CORTEX-M4 168MHZ LQFP100",
            manufacturer="STMicroelectronics",
            mpn="STM32F407VGT6",
            metadata={"provider": "catalog", "target_hs_family": "8542", "hs_heading": "8542"},
        )

        candidate, score, details = best_match_details(record, [generic_candidate, exact_candidate])

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.reference, "EXACT-1")
        self.assertGreaterEqual(score, 0.7)
        self.assertEqual(details["confidence"], "exact_mpn")

    def test_catalog_heading_mismatch_routes_to_review_without_overlay(self) -> None:
        record = {
            **degraded_record(),
            "tier1_description": "Wireless system on chip ESP32-WROOM-32E module.",
            "hs6_label": "854231",
            "hs4_label": "8542",
            "hs2_label": "85",
            "keywords": ["ESP32-WROOM-32E", "Espressif", "wireless module"],
        }
        mismatch_candidate = RawAuxiliaryRecord(
            source="catalog",
            reference="1965-ESP32-WROOM-32E-ND",
            description="RF TXRX MODULE WIFI TRACE U.FL",
            manufacturer="Espressif",
            mpn="ESP32-WROOM-32E",
            metadata={
                "provider": "catalog",
                "target_hs_family": "8542",
                "hs_heading": "8517",
                "is_boundary_case": True,
            },
        )

        outputs = enrich_records_with_auxiliary([record], [], [mismatch_candidate], catalog_min_score=0.1)
        enriched = outputs["records"][0]
        review_row = outputs["review_rows"][0]

        self.assertEqual(enriched["tier2_provenance"], "degraded_manual")
        self.assertFalse(review_row["catalog_applied"])
        self.assertTrue(review_row["catalog_heading_mismatch"])
        self.assertEqual(review_row["catalog_confidence"], "exact_mpn")
        self.assertEqual(review_row["catalog_block_reason"], "heading_mismatch")
        self.assertEqual(len(outputs["exact_mpn_review_rows"]), 1)

    def test_exact_mpn_review_csv_includes_reviewer_columns(self) -> None:
        record = {
            **degraded_record(),
            "tier1_description": "Wireless system on chip ESP32-WROOM-32E module.",
            "hs6_label": "854231",
            "hs4_label": "8542",
            "hs2_label": "85",
            "keywords": ["ESP32-WROOM-32E", "Espressif", "wireless module"],
        }
        mismatch_candidate = RawAuxiliaryRecord(
            source="catalog",
            reference="1965-ESP32-WROOM-32E-ND",
            description="RF TXRX MODULE WIFI TRACE U.FL",
            manufacturer="Espressif",
            mpn="ESP32-WROOM-32E",
            metadata={
                "provider": "catalog",
                "target_hs_family": "8542",
                "hs_heading": "8517",
                "product_url": "https://example.com/esp32",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            exact_mpn_review_path = Path(temp_dir) / "exact_mpn_review.csv"

            outputs = enrich_records_with_auxiliary([record], [], [mismatch_candidate], catalog_min_score=0.1)
            from src.processing.auxiliary_enricher import write_exact_mpn_review_csv

            write_exact_mpn_review_csv(str(exact_mpn_review_path), outputs["exact_mpn_review_rows"])
            with exact_mpn_review_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["catalog_candidate_reference"], "1965-ESP32-WROOM-32E-ND")
        self.assertEqual(rows[0]["catalog_block_reason"], "heading_mismatch")
        self.assertEqual(rows[0]["reviewer_decision"], "")


if __name__ == "__main__":
    unittest.main()
