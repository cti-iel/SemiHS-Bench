> **Why this directory exists.**
> This is the data-construction pipeline for SemiHS-Bench, shared to document
> the data-processing methodology described in the paper — Section 2,
> *Benchmark Construction* — in full detail, from raw industrial observation
> to the released records. It is published for transparency and provenance,
> **not as a turnkey product.** The raw source dumps it reads from
> (`data/raw/`, `data/intermediate/`) and the internal `release/` working
> pools are **intentionally not included** in the public repository: they
> contain third-party catalog / bill-of-lading source data that is not ours
> to redistribute. As a result most stages will not run end-to-end from a
> public clone — see [What runs and what does not](#what-runs-and-what-does-not).
> The finished, redistributable benchmark lives at the repository root under
> [`../data/`](../data); see the root [`../README.md`](../README.md) for the
> canonical layout and the quickstart.

---

# SemiHS-Bench — construction pipeline

The pipeline that produced the released benchmark: **1,800 records in
matched 900 train + 900 eval splits** with identical HS6 coverage —
**73 HS6 codes / 38 HS4 families** across 10 HS chapters (28, 37, 38, 71,
73, 76, 84, 85, 90, 94), built on **HS2022**. All reported experiments use
the eval split (`../data/eval.json`); the train split
(`../data/train.json`) is provided for reuse.

Each record starts as a real industrial observation — a manufacturer
catalog row or a bill-of-lading (BOL) line item — and flows through:

```
ingest & normalize  →  derive two input tiers  →  evidence-grounded
annotation against the customs-ruling corpus  →  adjudication +
hardness tags  →  fixed 4-code candidate slates  →  balanced splits
```

The product, gold label, and candidate slate stay fixed across the two
input tiers, so measured degradation between tiers isolates input
sparsity rather than a change in product distribution.

## What runs and what does not

This code documents *how* the benchmark was built; it is not a one-command
rebuild. From a public clone:

**Works** (reads only the released `../data/` files or self-contained fixtures):

- `make test` — the full unit + integration suite, over synthetic fixtures
  in `tests/fixtures/`. No network, no API keys, no private inputs.
- [`scripts/annotate_difficulty.py`](scripts/annotate_difficulty.py) —
  recomputes the `difficulty_tags` / `boundary_note` fields on the released
  splits from [`configs/boundary_tags.yaml`](configs/boundary_tags.yaml)
  (reads and rewrites `../data/eval.json` + `../data/train.json`).
- The `src/` library — normalization, candidate-slate construction, boundary
  detection, IAA scoring, etc. — can be imported and exercised directly
  (this is exactly what the tests do).

**Does not run end-to-end** (needs the non-public inputs):

- **Ingest** — `ingest_catalog.py`, `ingest_bol.py`, `normalize_jp_customs.py`
  read raw catalog / BOL / customs-ruling dumps under `data/raw/`, which are
  third-party and not redistributed.
- **Corpus → audit → slate → split → release** — `build_reference_corpus.py`,
  `generate_review_worksheet.py`, `prefill_core4_catalog.py`, the `apply_*`
  scripts, `build_part1_release.py`, `build_eval.py`, `build_release.py`,
  `build_hs6_dossiers.py`, and `build_calibration_set.py` read the internal
  `release/` working pools (reviewed candidate pools, expert-audit worksheets,
  the working corpus copy). Those directories are absent from the public repo,
  so each script exits with a missing-input error rather than fabricating data.

In short: the **transformations and methodology are fully readable here**, the
**outputs are shipped under `../data/`**, and only the **raw / intermediate
working data is withheld** because it is not ours to redistribute.

## 1. Source observations

Every record originates from one of two operational sources:

- **Manufacturer catalog rows** — product names, part numbers, functional
  descriptions, specifications. Ingested by
  [`scripts/ingest_catalog.py`](scripts/ingest_catalog.py)
  (lifecycle/scope filtering, MPN dedup, near-duplicate removal).
- **Bill-of-lading line items** — terse goods-description fields and
  logistics terminology. Ingested by
  [`scripts/ingest_bol.py`](scripts/ingest_bol.py)
  (generic-term and freight-forwarder filtering, declared-HS sanity
  checks, near-duplicate removal).

The scope covers the semiconductor supply chain — materials and
chemicals, electronic components, metrology instruments, manufacturing
equipment, and specialized subassemblies — defined by the in-scope HS6
allowlist in [`configs/hs6_scope_tiers.yaml`](configs/hs6_scope_tiers.yaml).
Each released record carries a `segment` field
(`material, equipment, metrology, component, end_product`) mapped per
HS6 in [`../data/taxonomy.csv`](../data/taxonomy.csv).

Eval-split composition: catalog 582 / BOL 318; by segment: component 413,
metrology 176, material 164, equipment 113, end_product 34.

## 2. Input tiers

Each product is represented at two input tiers derived from the same
observation:

- **Tier 1** (`tier1_description`) — a natural-language product
  description. An LLM drafts a concise, label-free description from the
  source observation; domain experts then review every description —
  correcting unsupported statements, removing HS-code leakage, and
  keeping only information justified by the source. The expert-review
  pass is applied by
  [`scripts/apply_tier1_review_catalog.py`](scripts/apply_tier1_review_catalog.py).
- **Tier 2** (`tier2_minimal`) — the sparse ERP-style input, taken
  directly from the originating observation (not truncated from Tier 1):
  the manufacturer name plus part number for catalog records, or the raw
  goods-description line for BOL records. Part-number-alone
  classification is intentionally hard for semiconductors.

Text normalization shared by records and the ruling corpus lives in
[`src/processing/degrader.py`](src/processing/degrader.py) and
[`src/utils/text_utils.py`](src/utils/text_utils.py), driven by
[`configs/degradation_rules.yaml`](configs/degradation_rules.yaml) and
[`configs/abbreviations.csv`](configs/abbreviations.csv).

**Anonymization:** the manufacturer name lives only in the structured
`manufacturer` field (never in description text); the catalog supplier is
never named, and supplier SKUs / internal part numbers are excluded
(scrubbing implemented in [`scripts/build_eval.py`](scripts/build_eval.py)).

## 3. Evidence-grounded annotation

Gold HS6 labels are assigned through an evidence-grounded expert process.
The annotation evidence is a held-out corpus of **889 public customs
rulings** ([`../data/reference_corpus.jsonl`](../data/reference_corpus.jsonl)):
400 from the U.S. CROSS system, 389 from the EU EBTI database, and 100
from Japan Customs advance rulings — issued from 2013 onward and
normalized to HS2022. The corpus is annotation evidence only; it is never
provided to evaluated models.

- [`scripts/normalize_jp_customs.py`](scripts/normalize_jp_customs.py)
  — normalizes Japan Customs rulings into corpus inputs.
- [`scripts/build_reference_corpus.py`](scripts/build_reference_corpus.py)
  — unifies CROSS/EBTI/JP inputs into the corpus (stable evidence IDs,
  jurisdiction tags, shared text normalization).
- [`scripts/build_hs6_dossiers.py`](scripts/build_hs6_dossiers.py)
  — per-HS6 evidence dossiers that surface the relevant rulings to
  annotators during review.
- [`scripts/generate_review_worksheet.py`](scripts/generate_review_worksheet.py)
  / [`scripts/prefill_core4_catalog.py`](scripts/prefill_core4_catalog.py)
  — build the expert-audit worksheets, pre-filling candidate rulings.
- [`scripts/apply_audit_decisions.py`](scripts/apply_audit_decisions.py)
  + [`src/audit/decisions.py`](src/audit/decisions.py)
  — parse the completed worksheets and apply the expert verdicts
  (citation-required labels; evidence-count → confidence-tier binding).

Disagreements between raters are resolved by comparing the cited rulings
against the product's determinative attributes —
[`docs/ADJUDICATION_PROTOCOL.md`](docs/ADJUDICATION_PROTOCOL.md). Rater
reliability is measured both rater-vs-rater and rater-vs-authority
([`docs/IAA_PROTOCOL.md`](docs/IAA_PROTOCOL.md), with
[`scripts/build_calibration_set.py`](scripts/build_calibration_set.py),
[`scripts/iaa_export_csv.py`](scripts/iaa_export_csv.py), and
[`src/annotation/`](src/annotation)).

Every released record carries a `difficulty_tags` list and a `boundary_note`.
The tags are drawn from a closed 25-tag vocabulary
([`configs/boundary_tags.yaml`](configs/boundary_tags.yaml)) in two groups:
**within-family sibling splits** (the subheading decision inside one HS4
family, e.g. processor vs memory IC) and **cross-family frontiers** (the
harder boundaries between HS families, e.g. discrete device vs integrated
circuit). A tag is assigned when the gold code and a candidate-slate
distractor share a cluster (sibling split) or sit on opposing sides with
supporting keyword evidence (cross-family);
[`src/annotation/boundary_detector.py`](src/annotation/boundary_detector.py)
implements the detection and composes `boundary_note`, the human-readable
deciding criterion for each tag.
[`scripts/annotate_difficulty.py`](scripts/annotate_difficulty.py) writes both
fields onto the released splits and is the one annotation stage that runs from
a public clone.

## 4. Candidate slates

For constrained evaluation every record carries a fixed four-code slate:
the gold HS6 plus three hard distractors, drawn (in priority order) from
boundary-pair expansions, sibling HS6 codes under the same HS4, and
same-chapter codes — deterministic per record
([`src/assembly/build_candidates.py`](src/assembly/build_candidates.py)).
The slate is identical across input tiers, so random top-1 is 25% in
constrained mode; open mode ranks the 73 in-scope codes (random top-1
≈ 1.4%).

## 5. Splits and release packaging

- [`scripts/build_part1_release.py`](scripts/build_part1_release.py)
  — unions the reviewed catalog and BOL pools.
- [`scripts/build_eval.py`](scripts/build_eval.py) — selects the
  balanced 900-record eval split by water-filling across every HS4 in the
  reviewed pools, then HS6 within each HS4; applies the final
  anonymization pass. The train split mirrors the same procedure with
  identical HS6 coverage.
- [`scripts/build_release.py`](scripts/build_release.py) — runs the
  release gates (schema validation, evidence-coverage floor, source-mix
  and boundary-share checks, manufacturer caps) and emits the manifest.

Label provenance on the eval split: `catalog_expert_validated` 582,
`BOL_expert_validated` 318 — 0 records with unverified gold labels.
By confidence tier: high 525 / medium 372 / low 3.

## File-by-file reference

### `scripts/` — pipeline stages (roughly in run order)

| script | what it does | public clone |
| --- | --- | --- |
| `ingest_catalog.py` | Ingest manufacturer-catalog rows: lifecycle/scope filtering, MPN dedup, near-duplicate removal → candidate pool. | needs `data/raw/` |
| `ingest_bol.py` | Ingest bill-of-lading line items: generic-term + freight-forwarder filtering, declared-HS checks, dedup → candidate pool. | needs `data/raw/` |
| `normalize_jp_customs.py` | Normalize Japan Customs advance rulings into reference-corpus inputs. | needs `data/raw/` |
| `build_reference_corpus.py` | Unify CROSS / EBTI / JP ruling inputs into the 889-ruling corpus (stable evidence IDs, jurisdiction tags, shared normalization). | needs `release/` pool |
| `build_hs6_dossiers.py` | Per-HS6 evidence dossiers that surface relevant rulings to annotators. | needs `release/` pool |
| `generate_review_worksheet.py` | Build the expert-audit worksheets from the candidate pools. | needs `release/` pool |
| `prefill_core4_catalog.py` | Pre-fill candidate rulings into the catalog audit worksheet. | needs `release/` pool |
| `apply_tier1_review_catalog.py` | Apply the expert Tier-1 description review to catalog records. | needs `release/` pool |
| `apply_audit_decisions.py` | Parse a completed audit worksheet and apply the expert verdicts to the pool. | needs `release/` pool |
| `build_part1_release.py` | Union the reviewed catalog and BOL pools into one set. | needs `release/` pool |
| `build_eval.py` | Water-fill the balanced 900-record eval split (HS4, then HS6 within HS4); final anonymization pass. | needs `release/` pool |
| `build_release.py` | Run the release gates (schema, evidence-coverage floor, source-mix / boundary-share, manufacturer caps) and emit the manifest. | needs `release/` pool |
| `build_calibration_set.py` | Sample the blind rater-vs-authority calibration set. | needs `release/` pool |
| `annotate_difficulty.py` | Recompute `difficulty_tags` / `boundary_note` on the released splits from the boundary vocabulary. | **yes** |
| `iaa_export_csv.py` | Pool the released splits into an inter-annotator-agreement worksheet for a second rater. | reads `../data/` |
| `_release_utils.py` | Shared helpers (SHA-256, git commit, distributions, corpus-subset writer) imported by the release-packaging scripts. | library |

### `src/` — library code

**`collectors/`** — raw-source parsers
- `catalog_collector.py` — normalize a catalog export payload; lifecycle filtering + dedup.
- `bol_collector.py` — normalize a BOL payload; record-specificity checks.
- `hts_taxonomy.py` — load the HS taxonomy CSV; HS4 / HS6 helpers.

**`processing/`** — record transformations
- `degrader.py` — derive the two input tiers (text normalization, MPN extraction) from a canonical record.
- `auxiliary_enricher.py` — overlay natural BOL / catalog variants onto records; emit the exact-MPN review CSV.
- `deduplicator.py` — near-duplicate detection and quality filtering.
- `manufacturer_backfill.py` — backfill `manufacturer_hint` from in-record fallbacks.
- `mpn_resolver.py` — match catalog / BOL variants by manufacturer part number.

**`annotation/`** — label quality
- `difficulty_tagger.py` — assign hardness tags, ambiguity score, and Tier-2 classifiability.
- `boundary_detector.py` — load the boundary-tag vocabulary and detect which tags a record triggers.
- `authority_calibration.py` — rater-vs-authority calibration scoring.
- `iaa_report.py` — inter-annotator agreement metrics (κ, set-Jaccard).

**`assembly/`**
- `build_candidates.py` — construct the fixed 4-code candidate slate (gold + 3 hard distractors), deterministic per record.

**`audit/`**
- `decisions.py` — parse the expert-audit worksheet and apply corrections / verdicts.

**`utils/`**
- `text_utils.py` — text normalization shared by records and the corpus.
- `abbreviation_engine.py` — abbreviation expansion.
- `config_loader.py` — load YAML / JSON configs.
- `io_utils.py` — JSON / JSONL read/write helpers.

`models.py` — shared dataclasses and the authoritative-source constants.

### `configs/`
- `hs6_scope_tiers.yaml` — the in-scope HS6 allowlist (the 73 codes) plus per-HS4 sub-allow rules.
- `boundary_tags.yaml` — the closed boundary-tag vocabulary (sibling-split / cross-family) and the human-readable note per tag.
- `hs_chapters.yaml` — HS chapter numbering / family boundaries.
- `manufacturer_caps.yaml` — per-manufacturer record caps enforced at release packaging.
- `degradation_rules.yaml` + `abbreviations.csv` — the text-normalization and MPN-extraction rule set used during tier derivation.

### `docs/`
- `ADJUDICATION_PROTOCOL.md` — evidence-binding rubric for resolving rater disagreements.
- `IAA_PROTOCOL.md` — inter-annotator agreement + rater-vs-authority calibration protocol.

### `tests/`
Unit + integration tests over synthetic fixtures in `tests/fixtures/`.
Integration tests that need the non-public `release/` pools skip
automatically when those artifacts are absent.

## Running the tests

```bash
pip install jsonschema openpyxl pandas PyYAML pytest
make test
```

The suite is self-contained (synthetic fixtures under `tests/fixtures/`);
tests that exercise the non-public `release/` working pools skip when
those artifacts are absent.
