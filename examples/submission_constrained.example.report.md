# SemiHS-Bench - Submission Report: oracle-constrained-tier1.example

- **Model**: `oracle`
- **Mode**: `constrained` · **Tier**: 1
- **Schema version**: 2.0.0
- **Notes**: Example - gold-first ranking on the first 5 eval records (eval split).

## Overall metrics

| slice | n | top-1 (HS6) | top-3 (HS6) | top-1 (HS4) | top-1 (HS2) | MRR (HS6) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Hierarchical-distance distribution** (top-1 prediction vs. gold):

- HS6 exact match:        5/5 (100.0%)
- HS4 match (HS6 wrong):  0/5 (0.0%)
- HS2 match (HS4 wrong):  0/5 (0.0%)
- No match:               0/5 (0.0%)
- Mean hierarchical distance: 0.000

## Per-HS2 chapter breakdown

| chapter | n | top-1 (HS6) | top-3 (HS6) | top-1 (HS4) | top-1 (HS2) | MRR (HS6) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Difficulty breakdown

| slice | n | top-1 (HS6) | top-3 (HS6) | top-1 (HS4) | top-1 (HS2) | MRR (HS6) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| boundary | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| non_boundary | 0 | - | - | - | - | - |
| group: sibling_split | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2804_gas_purity | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

_See `INTERPRETING_RESULTS.md` for guidance on reading these numbers._
