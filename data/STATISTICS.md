# STATISTICS - eval split (900 records)

### By source (tier1_source)

| key | count |
|---|---|
| BOL | 318 |
| catalog | 582 |

### By HS4

| key | count |
|---|---|
| 2804 | 24 |
| 2805 | 4 |
| 2826 | 16 |
| 2844 | 1 |
| 2850 | 28 |
| 3707 | 29 |
| 3818 | 57 |
| 7104 | 5 |
| 7320 | 1 |
| 7326 | 1 |
| 7616 | 8 |
| 8443 | 1 |
| 8471 | 16 |
| 8473 | 68 |
| 8486 | 81 |
| 8504 | 81 |
| 8513 | 2 |
| 8514 | 22 |
| 8516 | 1 |
| 8517 | 11 |
| 8518 | 1 |
| 8523 | 34 |
| 8524 | 3 |
| 8526 | 2 |
| 8536 | 38 |
| 8538 | 3 |
| 8541 | 81 |
| 8542 | 81 |
| 8543 | 7 |
| 8544 | 12 |
| 9001 | 2 |
| 9013 | 1 |
| 9025 | 7 |
| 9027 | 6 |
| 9029 | 1 |
| 9030 | 81 |
| 9031 | 81 |
| 9405 | 2 |

### By confidence tier

| key | count |
|---|---|
| high | 525 |
| low | 3 |
| medium | 372 |

### By segment

| key | count |
|---|---|
| material | 164 |
| equipment | 113 |
| metrology | 176 |
| component | 413 |
| end_product | 34 |

### By difficulty tag

718 of 900 eval records (79.8 %) sit on at least one confusable HS boundary;
84 carry more than one tag. A record is counted under every tag it carries, so
the per-tag counts below sum to more than 718. Tags come in two groups
(within-family sibling splits and cross-family frontiers); see
`construction/configs/boundary_tags.yaml` for the deciding criteria, and the
`boundary_note` field on each record for the human-readable rationale.

**Group A — within-family sibling splits**

| tag | count |
|---|---|
| 8541_siblings | 81 |
| 8542_ic_function | 81 |
| 8486_process_stage | 81 |
| 8504_power_splits | 81 |
| 9030_measurement_splits | 81 |
| 9031_inspection_splits | 81 |
| 8536_connection_splits | 38 |
| 3707_photochemical | 29 |
| 2804_gas_purity | 24 |
| 8471_adp_splits | 16 |
| 9027_analysis_splits | 6 |

**Group B — cross-family frontiers**

| tag | count |
|---|---|
| 8541_vs_8542 | 55 |
| storage_boundary | 45 |
| furnace_boundary | 20 |
| doped_vs_undoped | 18 |
| crystal_substrate_boundary | 16 |
| sensor_boundary | 15 |
| parts_attribution | 13 |
| amplifier_boundary | 7 |
| cable_vs_connector | 6 |
| machine_with_function | 4 |
| led_device_vs_luminaire | 3 |
| display_module_boundary | 2 |
| populated_board_boundary | 0 |
| process_vs_metrology | 0 |

Two Group-B frontiers (`populated_board_boundary`, `process_vs_metrology`)
have no eval records; both appear in the train split.
