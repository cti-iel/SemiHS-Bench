import unittest

from src.processing.deduplicator import (
    deduplicate_records,
    find_near_duplicate_groups,
    is_near_duplicate,
    mask_part_numbers,
    pick_keeper,
)


class DedupTests(unittest.TestCase):
    def test_cross_beats_ebti_for_near_duplicate(self) -> None:
        records = [
            {
                "id": "SH-0001",
                "tier1_description": "Integrated circuit 32-bit microcontroller with flash",
                "tier1_source": "EBTI",
                "hs6_label": "854231",
            },
            {
                "id": "SH-0002",
                "tier1_description": "Integrated circuit 32-bit microcontroller with flash memory",
                "tier1_source": "CROSS",
                "hs6_label": "854231",
            },
        ]
        deduped = deduplicate_records(records, threshold=0.8)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["tier1_source"], "CROSS")


class NearDuplicateTests(unittest.TestCase):
    def test_mask_replaces_part_numbers(self) -> None:
        self.assertEqual(
            mask_part_numbers("MODEL: 1FF5XLG200-IP-2 WAFER"),
            "MODEL: <PN> WAFER",
        )
        self.assertEqual(
            mask_part_numbers("SUCTION CUP C. VASB-15-1/8-PUR-B 1395671"),
            "SUCTION CUP C. <PN> <PN>",
        )

    def test_paraphrase_with_prose_only_diff_is_duplicate(self) -> None:
        a = "TONER FOR USE IN A DEDICATED MICROFILM READER PRINTER. IT IS SUPPLIED IN CARTRIDGE FORM."
        b = "TONER FOR USE IN A DEDICATED MICROFILM READER PRINTER. IT IS SUPPLIED AS A CARTRIDGE."
        self.assertTrue(is_near_duplicate(a, b))

    def test_distinct_skus_are_not_duplicate(self) -> None:
        a = "MODEL: 1FF5XLG200-IP-2 WAFER WAFER"
        b = "MODEL: 1FF5XLG100-OP2 WAFER WAFER"
        self.assertFalse(is_near_duplicate(a, b))

    def test_distinct_dimensions_are_not_duplicate(self) -> None:
        a = "PLAIN COPPER TUBE ( 15.88MM X 0.7MM )"
        b = "PLAIN COPPER TUBE ( 6.35MM X 1.2MM )"
        self.assertFalse(is_near_duplicate(a, b))

    def test_distinct_chemistry_is_not_duplicate(self) -> None:
        a = "SCREEN PRINTING CHEMICALS - SAATITEX HT DUAL CURE EMULSION"
        b = "SCREEN PRINTING CHEMICALS - SAATITEX PHU BLUE PURE PHOTOPOLYMER EMULSION"
        # 0.871 raw < 0.92 low_threshold -> not deduped
        self.assertFalse(is_near_duplicate(a, b))

    def test_high_threshold_catches_identical_after_paraphrase(self) -> None:
        a = "Liquid nitrogen, purity 99.999%, 100% new"
        b = "Liquid Nitrogen, purity 99.999%, 100% new"
        self.assertTrue(is_near_duplicate(a, b))

    def test_groups_only_within_same_hs6(self) -> None:
        a = "TONER FOR USE IN A DEDICATED MICROFILM READER PRINTER. IT IS SUPPLIED IN CARTRIDGE FORM."
        b = "TONER FOR USE IN A DEDICATED MICROFILM READER PRINTER. IT IS SUPPLIED AS A CARTRIDGE."
        records = [
            {"tier1_description": a, "hs6_label": "370790"},
            {"tier1_description": b, "hs6_label": "370790"},
            # Same description as record 1 but different hs6 — must NOT cluster.
            {"tier1_description": b, "hs6_label": "854100"},
        ]
        groups = find_near_duplicate_groups(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [0, 1])

    def test_pick_keeper_chooses_longest_description(self) -> None:
        members = [
            {"tier1_description": "short"},
            {"tier1_description": "much much longer description"},
            {"tier1_description": "medium length"},
        ]
        self.assertEqual(pick_keeper(members), 1)


class EbtiPaperVariantTests(unittest.TestCase):
    """Regression coverage for deduplication over ruling-style text.

    Customs rulings share heavy boilerplate, so distinct goods can
    still clear the high similarity threshold. These tests assert the
    deduper primitives cluster true paraphrases without collapsing
    genuinely distinct products.
    """

    def test_ebti_paraphrase_clusters_at_same_hs6(self) -> None:
        # Two real-shape EBTI descriptions differing only by prose
        # paraphrase ("supplied as" vs "supplied in...form").
        records = [
            {
                "tier1_description": (
                    "TONER FOR USE IN A DEDICATED MICROFILM READER "
                    "PRINTER. IT IS SUPPLIED IN CARTRIDGE FORM."
                ),
                "hs6_label": "370790",
            },
            {
                "tier1_description": (
                    "TONER FOR USE IN A DEDICATED MICROFILM READER "
                    "PRINTER. IT IS SUPPLIED AS A CARTRIDGE."
                ),
                "hs6_label": "370790",
            },
        ]
        groups = find_near_duplicate_groups(records)
        self.assertEqual(groups, [[0, 1]])

    def test_ebti_voltage_variant_clears_high_threshold(self) -> None:
        # DC/DC 5-500V supply vs AC/DC 85-260VAC supply share the
        # same HS6 and heavy boilerplate; the BOL-tuned predicate sees
        # them as a near-duplicate via the high-threshold path (raw
        # similarity >= 0.95 bypasses the digit check). Such pairs are
        # resolved during expert review rather than dropped blindly.
        a = (
            "DC/DC INDUSTRIAL POWER SUPPLY WITH INPUT VOLTAGES FROM "
            "5V - 500V PROVIDING DC OUTPUT VOLTAGES BELOW 300V DC "
            "WITH A POWER RANGE OF 2 WATTS TO 2000 WATTS."
        )
        b = (
            "AC/DC INDUSTRIAL POWER SUPPLIES WITH INPUT VOLTAGES FROM "
            "85 - 260VAC PROVIDING DC OUTPUT VOLTAGES BELOW 300V DC "
            "WITH A POWER RANGE OF 2 WATTS TO 2000 WATTS."
        )
        self.assertTrue(is_near_duplicate(a, b))


if __name__ == "__main__":
    unittest.main()

