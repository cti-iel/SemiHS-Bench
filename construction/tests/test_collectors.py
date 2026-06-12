import csv
from pathlib import Path
import tempfile
import unittest

from src.collectors.bol_collector import load_bol_imports
from src.collectors.catalog_collector import load_catalog_imports
from src.utils.io_utils import read_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


class CollectorTests(unittest.TestCase):
    def test_import_adapters(self) -> None:
        catalog_records = load_catalog_imports(str(FIXTURES / "raw" / "catalog" / "catalog_sample.jsonl"))
        bol_records = load_bol_imports(str(FIXTURES / "raw" / "bol" / "bol_sample.csv"))

        self.assertEqual(len(catalog_records), 1)
        self.assertEqual(len(bol_records), 1)
        self.assertEqual(catalog_records[0].mpn, "STM32F407VGT6")
        self.assertEqual(bol_records[0].metadata["declared_hs"], "381800")

    def test_bol_normalizer_handles_provider_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "hs_2804.csv"
            csv_path.write_text(
                "prod_desc,hs_code,date,supplier_t,buyer_t,orig_port,dest_port,quantity,quantity_unit,master_bill_no\n"
                "SILICON METAL GRADE 553,28046900,2025-03-15 00:00:00,Mitsui Bussan,Acme Wafer Co,HAMBURG,NEW YORK,24000,KGS,MBL-12345\n",
                encoding="utf-8",
            )
            records = load_bol_imports(str(csv_path))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.reference, "MBL-12345")
        self.assertEqual(record.description, "SILICON METAL GRADE 553")
        self.assertEqual(record.metadata["shipper"], "Mitsui Bussan")
        self.assertEqual(record.metadata["consignee"], "Acme Wafer Co")
        self.assertEqual(record.metadata["port_origin"], "HAMBURG")
        self.assertEqual(record.metadata["port_dest"], "NEW YORK")
        self.assertEqual(record.metadata["arrival_date"], "2025-03-15 00:00:00")
        self.assertEqual(record.metadata["quantity_unit"], "KGS")
        self.assertEqual(record.metadata["declared_hs"], "28046900")

    def test_bol_normalizer_cleans_provider_garbage_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "bol_dirty.csv"
            csv_path.write_text(
                "prod_desc,hs_code,master_bill_no\n"
                '"TH#&Liquid Nitrogen, purity 99.999%, 100% new#&VN",28043000,MBL-1\n'
                '"5474B008AA VARIOPRINT 6000 TONER PCR (6 BOTTLES) 5474B008AA VARIOPRINT 6000 TONER PCR (6 BOTTLES)",37079000,MBL-2\n'
                '"P-No:8974938303 QTY:60PCS PRODUCTION PARTS",84099900,MBL-3\n',
                encoding="utf-8",
            )
            records = load_bol_imports(str(csv_path))
        descriptions = [r.description for r in records]
        self.assertNotIn("#", descriptions[0])
        self.assertIn("Liquid Nitrogen", descriptions[0])
        self.assertEqual(descriptions[1], "5474B008AA VARIOPRINT 6000 TONER PCR")
        self.assertNotIn("60PCS", descriptions[2])

    def test_load_catalog_export_csv_filters_deduplicates_and_writes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "data" / "raw" / "catalogs"
            base.mkdir(parents=True, exist_ok=True)
            csv_path = base / "catalog_response.csv"
            fieldnames = [
                "record_id",
                "manufacturer",
                "manufacturer_part_number",
                "supplier_part_number",
                "base_product_number",
                "description_short",
                "description_detailed",
                "product_url",
                "datasheet_url",
                "photo_url",
                "category_path",
                "supplier_category_id_path",
                "key_specs",
                "raw_parameters",
                "classifications",
                "hs_code",
                "hs_heading",
                "intended_bucket",
                "target_hs_family",
                "intended_hs_prefixes",
                "source_queries",
                "source_keywords",
                "is_off_target",
                "is_boundary_case",
                "retrieval_rank",
                "quantity_available",
                "pricing_snapshot",
                "lifecycle_status",
            ]
            rows = [
                {
                    "record_id": "ti::1",
                    "manufacturer": "Texas Instruments",
                    "manufacturer_part_number": "STM32F407VGT6",
                    "supplier_part_number": "497-16525-ND",
                    "base_product_number": "STM32F407",
                    "description_short": "IC MCU 32BIT ARM CORTEX-M4 168MHZ LQFP100",
                    "description_detailed": "32-Bit Microcontroller IC with 1MB Flash",
                    "product_url": "https://example.com/p1",
                    "datasheet_url": "https://example.com/d1",
                    "photo_url": "",
                    "category_path": "[\"Integrated Circuits (ICs)\", \"Embedded\", \"Microcontrollers\"]",
                    "supplier_category_id_path": "[32, 44, 55]",
                    "key_specs": "{\"package\": \"LQFP-100\", \"flash\": \"1MB\"}",
                    "raw_parameters": "[]",
                    "classifications": "{\"HtsusCode\": \"8542.31.0000\"}",
                    "hs_code": "8542.31.0000",
                    "hs_heading": "8542",
                    "intended_bucket": "mcu",
                    "target_hs_family": "8542",
                    "intended_hs_prefixes": "[\"854231\"]",
                    "source_queries": "[1]",
                    "source_keywords": "[\"microcontroller\"]",
                    "is_off_target": "False",
                    "is_boundary_case": "False",
                    "retrieval_rank": "1",
                    "quantity_available": "100",
                    "pricing_snapshot": "{\"unit_price\": 4.2}",
                    "lifecycle_status": "Active",
                },
                {
                    "record_id": "ti::2",
                    "manufacturer": "Texas Instruments",
                    "manufacturer_part_number": "STM32F407VGT6",
                    "supplier_part_number": "497-16525-ALT-ND",
                    "base_product_number": "STM32F407",
                    "description_short": "IC MCU DUPLICATE",
                    "description_detailed": "Lower priority duplicate",
                    "product_url": "https://example.com/p2",
                    "datasheet_url": "",
                    "photo_url": "",
                    "category_path": "[\"Integrated Circuits (ICs)\", \"Embedded\", \"Microcontrollers\"]",
                    "supplier_category_id_path": "[32, 44, 55]",
                    "key_specs": "{\"package\": \"LQFP-100\"}",
                    "raw_parameters": "[]",
                    "classifications": "{\"HtsusCode\": \"8542.31.0000\"}",
                    "hs_code": "8542.31.0000",
                    "hs_heading": "8542",
                    "intended_bucket": "mcu",
                    "target_hs_family": "8542",
                    "intended_hs_prefixes": "[\"854231\"]",
                    "source_queries": "[2]",
                    "source_keywords": "[\"microcontroller\"]",
                    "is_off_target": "False",
                    "is_boundary_case": "False",
                    "retrieval_rank": "8",
                    "quantity_available": "20",
                    "pricing_snapshot": "{\"unit_price\": 4.8}",
                    "lifecycle_status": "Active",
                },
                {
                    "record_id": "bad::offtarget",
                    "manufacturer": "Acme",
                    "manufacturer_part_number": "OFF-1",
                    "supplier_part_number": "OFF-1-ND",
                    "base_product_number": "OFF-1",
                    "description_short": "Off target catalog result",
                    "description_detailed": "Off target",
                    "product_url": "",
                    "datasheet_url": "",
                    "photo_url": "",
                    "category_path": "[\"Wireless\"]",
                    "supplier_category_id_path": "[1]",
                    "key_specs": "{}",
                    "raw_parameters": "[]",
                    "classifications": "{}",
                    "hs_code": "8517.62.0000",
                    "hs_heading": "8517",
                    "intended_bucket": "off",
                    "target_hs_family": "8542",
                    "intended_hs_prefixes": "[\"854231\"]",
                    "source_queries": "[3]",
                    "source_keywords": "[\"radio\"]",
                    "is_off_target": "True",
                    "is_boundary_case": "False",
                    "retrieval_rank": "3",
                    "quantity_available": "5",
                    "pricing_snapshot": "{}",
                    "lifecycle_status": "Active",
                },
                {
                    "record_id": "bad::obsolete",
                    "manufacturer": "Acme",
                    "manufacturer_part_number": "OBS-1",
                    "supplier_part_number": "OBS-1-ND",
                    "base_product_number": "OBS-1",
                    "description_short": "Obsolete integrated circuit",
                    "description_detailed": "Obsolete",
                    "product_url": "",
                    "datasheet_url": "",
                    "photo_url": "",
                    "category_path": "[\"Integrated Circuits (ICs)\"]",
                    "supplier_category_id_path": "[32]",
                    "key_specs": "{}",
                    "raw_parameters": "[]",
                    "classifications": "{}",
                    "hs_code": "8542.39.0000",
                    "hs_heading": "8542",
                    "intended_bucket": "old",
                    "target_hs_family": "8542",
                    "intended_hs_prefixes": "[\"854239\"]",
                    "source_queries": "[4]",
                    "source_keywords": "[\"obsolete\"]",
                    "is_off_target": "False",
                    "is_boundary_case": "False",
                    "retrieval_rank": "4",
                    "quantity_available": "0",
                    "pricing_snapshot": "{}",
                    "lifecycle_status": "Obsolete",
                },
                {
                    "record_id": "mismatch::1",
                    "manufacturer": "Espressif Systems",
                    "manufacturer_part_number": "ESP32-WROOM-32E",
                    "supplier_part_number": "1965-ESP32-WROOM-32E-ND",
                    "base_product_number": "ESP32-WROOM",
                    "description_short": "RF TXRX MODULE WIFI TRACE U.FL",
                    "description_detailed": "WiFi and Bluetooth module",
                    "product_url": "",
                    "datasheet_url": "",
                    "photo_url": "",
                    "category_path": "[\"RF and Wireless\", \"RF Transceiver Modules\"]",
                    "supplier_category_id_path": "[77, 88]",
                    "key_specs": "{\"protocol\": \"WiFi\"}",
                    "raw_parameters": "[]",
                    "classifications": "{\"HtsusCode\": \"8517.62.0090\"}",
                    "hs_code": "8517.62.0090",
                    "hs_heading": "8517",
                    "intended_bucket": "soc",
                    "target_hs_family": "8542",
                    "intended_hs_prefixes": "[\"854231\"]",
                    "source_queries": "[5]",
                    "source_keywords": "[\"soc wifi\"]",
                    "is_off_target": "False",
                    "is_boundary_case": "True",
                    "retrieval_rank": "2",
                    "quantity_available": "40",
                    "pricing_snapshot": "{\"unit_price\": 3.1}",
                    "lifecycle_status": "Active",
                },
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            catalog_records = load_catalog_imports(str(csv_path))

            self.assertEqual(len(catalog_records), 2)
            self.assertEqual(catalog_records[0].reference, "497-16525-ND")
            self.assertEqual(catalog_records[0].mpn, "STM32F407VGT6")
            self.assertEqual(catalog_records[0].metadata["provider"], "catalog")
            self.assertEqual(
                catalog_records[0].metadata["category_path"],
                "Integrated Circuits (ICs) > Embedded > Microcontrollers",
            )
            self.assertEqual(catalog_records[0].metadata["specs"]["package"], "LQFP-100")
            self.assertEqual(catalog_records[1].metadata["hs_heading"], "8517")

            snapshot_path = csv_path.with_name("catalog_response_normalized.jsonl")
            snapshot_rows = read_jsonl(str(snapshot_path))
            self.assertEqual(len(snapshot_rows), 2)
            self.assertEqual(snapshot_rows[0]["metadata"]["provider"], "catalog")


if __name__ == "__main__":
    unittest.main()
