import unittest

from src.utils.text_utils import clean_bol_description, normalize_text


class CleanBolDescriptionTests(unittest.TestCase):
    def test_strips_hash_separator_and_country_suffix(self) -> None:
        self.assertEqual(
            clean_bol_description(
                "TH#&Liquid Nitrogen (Liquid Nitrogen-LN2), purity 99.999%, "
                "used to cool heat treatment furnaces, 100% new"
            ),
            "Liquid Nitrogen (Liquid Nitrogen-LN2), purity 99.999%, "
            "used to cool heat treatment furnaces, 100% new",
        )

    def test_drops_leading_dot_segment_and_trailing_country_code(self) -> None:
        self.assertEqual(
            clean_bol_description(".#&Photosensitive powder 00520, 100% new#&VN"),
            "Photosensitive powder 00520, 100% new",
        )

    def test_picks_one_product_name_when_concatenated(self) -> None:
        cleaned = clean_bol_description(
            "5474B008AA VARIOPRINT 6000 TONER PCR (6 BOTTLES) "
            "5474B008AA VARIOPRINT 6000 TONER PCR (6 BOTTLES)"
        )
        self.assertEqual(cleaned, "5474B008AA VARIOPRINT 6000 TONER PCR")

    def test_picks_one_product_name_when_truncated_dup(self) -> None:
        cleaned = clean_bol_description(
            "MFD SPARES - Bizhub C224/284/364/454/554 DV512C Developer 210g/Bag"
            "MFD SPARES - Bizhub C224/284/364/454/554 DV512C"
        )
        self.assertEqual(
            cleaned,
            "MFD SPARES - Bizhub C224/284/364/454/554 DV512C Developer 210g/Bag",
        )

    def test_strips_quantity_tokens(self) -> None:
        self.assertNotIn("60PCS", clean_bol_description("QTY:60PCS PRODUCTION PARTS"))
        self.assertNotIn("2 PALLETS", clean_bol_description("CO2 CYLINDER ON 2 PALLETS HS CODE 281121"))
        self.assertNotIn("15 BOXES", clean_bol_description("N2 CYLINDER 15 BOXES UN 1013"))
        self.assertNotIn("48 BAG", clean_bol_description("UNIVERSAL TONER (48 BAG) (BULK)"))
        self.assertNotIn("6 BOTTLES", clean_bol_description("VARIOPRINT TONER (6 BOTTLES)"))

    def test_preserves_embedded_specs(self) -> None:
        for spec in (
            "Solution 99.999% purity, 32L tank",
            "Material grade 1100g per CRTG/CTN bag",
            "Wafer 183.75mm*183.75mm 200KGS load",
            "Capacitor 15kg roll grade A+",
        ):
            cleaned = clean_bol_description(spec)
            for token in ("99.999%", "32L", "1100g", "200KGS", "15kg", "183.75mm"):
                if token in spec:
                    self.assertIn(token, cleaned, f"{token!r} should survive in {cleaned!r}")

    def test_empty_input(self) -> None:
        self.assertEqual(clean_bol_description(""), "")

    def test_plain_description_untouched(self) -> None:
        plain = "Microcontroller IC 32-bit ARM Cortex-M4 168MHz LQFP-100"
        self.assertEqual(clean_bol_description(plain), plain)

    def test_normalize_text_unchanged_for_clean_input(self) -> None:
        # Sanity: existing normalize_text behavior unaffected.
        self.assertEqual(normalize_text("hello   world"), "hello world")


if __name__ == "__main__":
    unittest.main()
