# DATASHEET - SemiHS-Bench

**Status:** balanced benchmark of 1,800 records in matched **900 train + 900 eval**
splits with identical HS6 coverage (`data/train.json`, `data/eval.json`). All
reported experiments use the eval split; train is provided for reuse.

- **Task:** single-label HS6 tariff classification, semiconductor supply chain.
- **Sources:** real manufacturer-catalog records and bill-of-lading line items,
  expert-validated. Eval split by source: catalog 582, BOL 318.
- **Labels:** gold HS6 per record, assigned and validated by domain experts.
  Eval split by confidence tier: high 525, medium 372, low 3.
- **Balance:** water-fill across 38 HS4 families, then HS6 within each.
- **Anonymization:** no brand/manufacturer in descriptions (structured field only);
  no supplier name or SKUs; common nouns lowercased.
- **Limitations:** gold labels are expert-validated against the HS2022 nomenclature;
  the reference ruling corpus is provided as supporting domain evidence. A long tail of
  small HS4 families is retained intentionally.
- **License / citation:** inherit from the parent SemiHS-Bench release.
