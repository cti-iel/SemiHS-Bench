# IAA Protocol — SemiHS-Bench

Scope: measure inter-annotator agreement on the four judgment-based annotation
fields produced by [src/annotation/difficulty_tagger.py](../src/annotation/difficulty_tagger.py)
and [src/annotation/boundary_detector.py](../src/annotation/boundary_detector.py).
`hs6_label` itself is authoritative (CROSS / EBTI) and is **not** re-annotated.

## 1. Sampling frame

- **Pool**: both released splits pooled —
  [`../../data/eval.json`](../../data/eval.json) +
  [`../../data/train.json`](../../data/train.json) (1,800 records).
- **Size**: 150 records.
- **Strata** (produced by [scripts/iaa_export_csv.py](../scripts/iaa_export_csv.py)),
  stratified by boundary group (§3.2):
  - **60** records carrying at least one `cross_family` tag, with a per-tag
    floor of 3 (rarest frontier first) so the thin frontiers
    (`process_vs_metrology`, `machine_with_function`, `display_module_boundary`,
    …) are represented rather than swamped by the common ones.
  - **60** records carrying at least one `sibling_split` tag (and no
    `cross_family` tag).
  - **30** records with **no** boundary tag (control group — prevents κ from
    being dominated by hard cases).
- **Stratum-level cap**: if a stratum has fewer eligible records, fill the
  deficit from the leftover pool; log the fill in the export manifest. With
  the current data all three strata fill exactly (no deficit), and the export
  manifest also records per-tag coverage for the cross-family floor.

## 2. Rater workflow

1. Rater receives `iaa_input.csv` with columns: `frozen_id`, `id`,
   `hs4_label_hint`, `tier1_description`, `tier2_part_name`,
   `tier2_manufacturer`, and the four blank annotation columns below.
   (`frozen_id` disambiguates the two splits, whose `id` values overlap;
   `tier2_*` are the minimal-form fields — short part-name plus manufacturer.)
2. For each record rater fills:
   - `ambiguity_score` — integer 1, 2, or the literal string `3+`.
   - `boundary_tags` — semicolon-separated list drawn from the closed vocabulary
     in §3.2. Empty string = no tags.
   - `classification_driver` — one of `material`, `function`, `use`,
     `combination`.
   - `tier2_classifiable` — one of `no`, `partial`, `yes`.
3. Rater returns `iaa_annotated_{rater_id}.csv`. File is converted to JSONL for
   `src/annotation/iaa_report.py` via a small wrapper (see §5).

## 3. Codebook

### Released vs. internal fields

Two of the four judgment fields below are **shipped on the released records**;
the other two are computed only for this IAA exercise and are never released:

| Field | On released records? | Notes |
|---|---|---|
| `difficulty_tags` | **yes** | The boundary tags of §3.2, refreshed by [scripts/annotate_difficulty.py](../scripts/annotate_difficulty.py). The IAA export reads these directly as rater B's `boundary_tags`. |
| `boundary_note` | **yes** | Human-readable comment stating the deciding criterion for each tag (the `note` text in [configs/boundary_tags.yaml](../configs/boundary_tags.yaml)); composed automatically from `difficulty_tags`. Not re-annotated here. |
| `tier2_classifiable` | **yes** | Expert value carried on the record; rater B's column is the stored value passed through unchanged. |
| `ambiguity_score` | no | Pipeline label inferred at export time by the difficulty-tagger heuristics; never shipped. |
| `classification_driver` | no | Pipeline label inferred at export time; never shipped. |

Rater A annotates all four judgment fields blind; rater B's labels are the
pipeline values described above, emitted automatically by the export script.

### 3.1 `ambiguity_score` (ordinal)

**Definition (taxonomy-counting, not description-reading).**
Count the HS6 subheadings within the same HS4 that a reasonable classifier
could defend *after reading the Tier-1 text to exhaustion*. "Could defend"
means the classifier could write a non-trivial justification citing product
attributes. Do **not** tally every HS6 sibling regardless of relevance.

- `1` — exactly one HS6 sibling is plausible; the description is unambiguous.
- `2` — two HS6 siblings are plausible; rater can narrow with minor assumptions.
- `3+` — three or more HS6 siblings plausible, or the record is under-specified.

**Worked examples**:

- "P-channel MOSFET, 30V, SOT-23" in HS4 8541 — only 854129 (transistors) is
  plausible. Score = `1`.
- "Silicon carbide wafer, 150mm diameter, doped" — plausible under 381800
  (doped substrates) and arguably 280461 (silicon, unwrought). Score = `2`.
- "Semiconductor device, surface mount, for automotive use" — could be
  854129 / 854131 / 854229 and several others. Score = `3+`.

### 3.2 `boundary_tags` (multilabel, closed vocabulary)

The closed vocabulary has 25 tags in two groups, defined in
[configs/boundary_tags.yaml](../configs/boundary_tags.yaml). Apply every tag
whose competing readings are both plausible for the record. **Group A**
(sibling splits) are the subheading decisions inside one HS4 family; **Group B**
(cross-family frontiers) are the harder boundaries between HS families where the
`boundary_note` comment earns its keep. On the released records, a tag is
assigned automatically when the gold code and a candidate-slate distractor sit
in the same cluster (Group A) or on opposing sides plus keyword evidence
(Group B); raters apply the same criteria by judgment.

**Group A — within-family sibling splits**

| Tag | Codes in play | Deciding criterion |
|---|---|---|
| `8541_siblings` | 854110/21/29/30/41/42/49/51/60/90 | device type, the 1 W dissipation threshold (854121 vs 854129), LED vs photovoltaic vs other photosensitive, the HS2022 854151 transducers |
| `8542_ic_function` | 854231/32/33/39 | processor/controller vs memory vs amplifier vs other IC |
| `8486_process_stage` | 848610/20/40/90 | boule/wafer vs device fabrication vs Note 11(C) machines vs parts |
| `8504_power_splits` | 850431/40/50/90 | transformer vs static converter vs inductor vs parts |
| `8536_connection_splits` | 853610/50/69 | fuse vs switch vs connector |
| `9030_measurement_splits` | 903033/39/40/82/84/90 | with/without recording, telecom-specific, semiconductor-specific (903082), parts |
| `9031_inspection_splits` | 903141/49/80/90 | optical for semiconductors vs other optical vs non-optical vs parts |
| `2804_gas_purity` | 280410/40/61 | element identity plus the 99.99 % silicon purity threshold |
| `3707_photochemical` | 370710/90 | sensitised emulsion vs other photographic preparation |
| `9027_analysis_splits` | 902710/50 | gas/smoke analysis vs other optical-radiation instruments |
| `8471_adp_splits` | 847130/50 | portable complete machine vs processing unit |

**Group B — cross-family frontiers**

| Tag | Codes in play | Deciding criterion |
|---|---|---|
| `8541_vs_8542` | 8541x vs 8542x | discrete device vs integrated circuit; multichip and hybrid modules sit on this line |
| `populated_board_boundary` | 847330 vs 854231/32/39 vs 853890 | populated PCB as ADP part vs IC vs control-apparatus part; Section XVI Note 2 |
| `storage_boundary` | 852351 vs 854232 vs 847330 | solid-state drive vs memory IC vs ADP part |
| `doped_vs_undoped` | 381800 vs 280461 vs 285000 | doped for use in electronics vs high-purity element vs nitride/silicide compound (GaN, silicides) |
| `process_vs_metrology` | 8486x vs 903082/903141/903180 | manufacturing function vs measuring/inspection function for in-line tools |
| `furnace_boundary` | 851419 vs 848620 | general industrial furnace vs Chapter 84 Note 11 semiconductor processing (diffusion, RTP, anneal) |
| `machine_with_function` | 854370 vs 8486x vs 8516x | electrical machine with individual function vs semiconductor-specific equipment |
| `parts_attribution` | 848690/853890/903190/903090 vs 732090/732690/761699 | identifiable part of a machine vs general metal article; Section XV vs XVI notes |
| `led_device_vs_luminaire` | 854141 vs 940542 | LED as semiconductor device or module vs finished light fitting |
| `display_module_boundary` | 852491 vs 901380 vs 847130 | HS2022 flat-panel display module vs other optical appliance vs complete ADP machine |
| `amplifier_boundary` | 851840 vs 854233 vs 854370 | audio amplifier apparatus vs amplifier IC vs other electrical machine |
| `cable_vs_connector` | 854442 vs 853669 | cable assembly fitted with connectors vs the connector itself |
| `sensor_boundary` | 854151 vs 902519 vs 903x | HS2022 semiconductor transducer vs thermometer/instrument; sensor packaged alone vs as instrument |
| `crystal_substrate_boundary` | 710499 vs 381800 vs 900190 | worked synthetic crystal vs doped electronic element vs optical-fibre/optical element |

### 3.3 `classification_driver` (categorical, single choice)

Which HS Rule of Interpretation most naturally drives this classification?

- `material` — heading chosen by what the item **is made of** (e.g. silicon
  wafer classified to 3818 by substrate material).
- `function` — heading chosen by what the item **does** (e.g. LM7805 voltage
  regulator classified to 854231 by its integrated-circuit function).
- `use` — heading chosen by the application or end-use (e.g. "for automotive
  ECUs" classified to 8537 because the end-use determines the heading, not
  the component's intrinsic nature).
- `combination` — General Rule of Interpretation 3 applies: two or more
  characteristics carry the classification jointly, or the justification text
  explicitly cites GRI 3.

**Function vs. combination — common confusion**:

Choose `function` when a *single* characteristic (the product's primary
operation) drives the classification, even if other specs are mentioned.
Choose `combination` only when two characteristics are load-bearing together,
such as: "power semiconductor module with integrated heatsink" — the
heatsink (material) and the semiconductor (function) jointly determine
whether 8541/8542 applies. If only one attribute is load-bearing, pick that
attribute's category instead.

### 3.4 `tier2_classifiable` (ordinal)

This field is **shipped on the released records** and rater B's column is the
stored expert value passed through unchanged; only rater A annotates it blind.
Given only the minimal Tier-2 form (typically an MPN or short code plus the
manufacturer), could an expert recover the HS6?

- `yes` — the identifier is an MPN-style token **and** includes a recognizable
  product-type or spec hint.
- `partial` — a product-type token is present but no MPN or spec hint; HS4 is
  inferable, HS6 is not.
- `no` — neither MPN nor product-type token; classification requires the full
  Tier-1 description.

## 4. Agreement metrics ([src/annotation/iaa_report.py](../src/annotation/iaa_report.py))

| Field | Metric | Rationale |
|---|---|---|
| `ambiguity_score` | Krippendorff's α (ordinal) | Ordinal scale with known distance; handles missing values. |
| `boundary_tags` | set-Jaccard κ (chance-corrected) | Multilabel set agreement over the 25-tag vocabulary (§3.2); empty-vs-empty counts as perfect. |
| `classification_driver` | Cohen's κ (unweighted) | Nominal 4-way. |
| `tier2_classifiable` | Weighted Cohen's κ (linear) | Ordinal 3-way; `yes ↔ no` disagreement weighted 2×. |

All metrics reported with 95 % non-parametric bootstrap CIs (n = 1 000).

### Acceptance targets

- Cohen's κ on `classification_driver` ≥ 0.70.
- Weighted κ on `tier2_classifiable` ≥ 0.60.
- Set-Jaccard κ on `boundary_tags` ≥ 0.60.
- Krippendorff's α on `ambiguity_score` ≥ 0.60.

Lower values are reported unredacted and discussed in the paper's limitations
section; they do not block release.

## 5. File layout

```
../../data/eval.json                 # pooled with train.json as the sampling frame
../../data/train.json
../../data/hs6_descriptions.csv         # HS6 descriptions for the ambiguity heuristic
data/intermediate/review/
  iaa_input.csv                         # produced by scripts/iaa_export_csv.py
  iaa_annotated_rater_a.csv             # rater A submission
  iaa_annotated_rater_b.csv             # rater B submission (baseline = pipeline labels)
  iaa_manifest.json                     # stratum draws + cross-family tag coverage
  iaa_report.json                       # produced by src/annotation/iaa_report.py
docs/IAA_PROTOCOL.md                    # this file
```

Rater B is the existing pipeline — its labels are computed at export time from
the released records (`boundary_tags` = `difficulty_tags`; `tier2_classifiable`
= the stored expert value; `ambiguity_score` and `classification_driver` are
inferred by the difficulty-tagger heuristics), so `iaa_annotated_rater_b.csv`
is generated automatically by the export script.

## 6. Re-run

```
python scripts/iaa_export_csv.py --seed 17
# ... rater A returns iaa_annotated_rater_a.csv ...
python -m src.annotation.iaa_report \
    --rater-a data/intermediate/review/iaa_annotated_rater_a.csv \
    --rater-b data/intermediate/review/iaa_annotated_rater_b.csv \
    --out data/intermediate/review/iaa_report.json
```

`iaa_export_csv.py` pools `../data/eval.json` + `../data/train.json`
and reads HS6 descriptions from `../data/hs6_descriptions.csv` by default;
override with `--input` (repeatable) and `--taxonomy`.

## 7. Authority calibration stratum (Core 4)

Adds a **rater-vs-authority** layer on top of the rater-vs-rater
κ machinery in §§1–6. The motivation is direct: κ on free-text judgment
fields under-measures label quality on a skewed HS6 distribution, where
chance-agreement corrections are unstable. Rater-vs-authority *accuracy* —
scoring blind rater annotations against legally binding customs rulings —
is the more defensible reliability statistic, and this protocol adopts
it as the headline measure.

### 7.1 Sampling frame

- **Pool:** ``../../data/reference_corpus.jsonl`` — the released 889-ruling
  corpus (CROSS 400 + EBTI 389 + JP 100). Calibration draws only the EBTI and
  CROSS rulings (789), whose authoritative HS6 is legally binding; the JP
  rulings are excluded by the source filter.
- **Size:** 60 records (40 EBTI + 20 CROSS).
- **Strata:** stratified by HS4 across the in-scope allow-set
  (`configs/hs6_scope_tiers.yaml`). Per-HS4 floor: 1 record per HS4
  represented; remainder distributed proportionally.
- **Deterministic seed:** `20260514` (fixed integer for reproducibility; no external significance).
- **Generated by:** `scripts/build_calibration_set.py`.

### 7.2 Label-strip protocol

The rater-facing CSV omits the authoritative `hs6_label` column. Raters
receive only the same two-tier inputs they would see for a benchmark
record (tier1_text, tier2_minimal) plus the same audit
worksheet schema (`expert_hs6`, `confidence_tier`, `cited_evidence_ids`,
`rationale_short`). The truth table lives at
`data/intermediate/_calibration_truth.jsonl` and is **read only after**
both raters submit.

### 7.3 Rater workflow

1. Two raters receive `data/intermediate/calibration_input.csv`,
   completed independently (no cross-communication).
2. Each rater submits
   `data/intermediate/calibration_annotated_rater_{a,b}.csv` with the
   same column schema plus their proposed `rater_hs6`,
   `rater_confidence_tier`, `rater_cited_evidence_ids`,
   `rater_rationale_short`.
3. `src/annotation/authority_calibration.py` (new) joins each rater's
   submission against `_calibration_truth.jsonl` and computes the metrics
   in §7.4.

### 7.4 Metrics

| Metric | Definition | Acceptance target |
|---|---|---|
| Per-rater authority accuracy | fraction of records where `rater_hs6 == authoritative_hs6` | ≥ 0.75 |
| Joint authority accuracy | fraction of records where **both** raters reach authority | ≥ 0.70 |
| Authority accuracy by HS4 | as above, stratified by HS4 | report unredacted; flag HS4s < 0.50 for codebook review |
| Inter-rater κ | continuity with §4 — Cohen's κ on `rater_hs6` | reported; no target — secondary to authority accuracy |
| Confidence calibration | among `rater_confidence_tier=high` records, authority accuracy should be ≥ 0.85 | yes/no |
| Citation usefulness | among records where the rater reached authority, fraction whose `rater_cited_evidence_ids` includes the actual authoritative ruling | reported; informs adjudication-rubric tuning |

All metrics ship with non-parametric bootstrap 95% CIs (n=1000) using the
same machinery as §4.

### 7.5 Failure handling

If joint authority accuracy < 0.70 on first pass:

1. Inspect by-HS4 breakdown to localize failure modes (boundary confusables,
   thin-HS4 categories, BOL-style description peculiarities).
2. Update the codebook in §3 to address the failure pattern. Note the
   change in `data/intermediate/_calibration_codebook_history.md`.
3. Re-run with a **fresh** 60-record sample (different seed:
   `20260514 + iteration_index`). Do NOT re-rate the same records — that
   contaminates the calibration.
4. The benchmark audit pass proceeds only after a calibration run achieves
   ≥ 0.70 joint authority accuracy OR three iterations have been attempted
   and the cause is documented as out of scope.

### 7.6 Files

| File | Role |
|---|---|
| `scripts/build_calibration_set.py` | Sampler (label-stripping). |
| `data/intermediate/calibration_input.csv` | Rater-facing CSV. |
| `data/intermediate/calibration_annotated_rater_{a,b}.csv` | Rater submissions. |
| `data/intermediate/_calibration_truth.jsonl` | Held-out authoritative labels. |
| `data/intermediate/_calibration_sample_report.json` | Sampling manifest. |
| `src/annotation/authority_calibration.py` | Scorer (computes §7.4 metrics). |
| `data/intermediate/calibration_report.json` | Output: per-rater authority accuracy + joint accuracy + breakdowns. |

### 7.7 Re-run

```
python scripts/build_calibration_set.py
# ... raters return calibration_annotated_rater_{a,b}.csv ...
python -m src.annotation.authority_calibration \
    --rater-a data/intermediate/calibration_annotated_rater_a.csv \
    --rater-b data/intermediate/calibration_annotated_rater_b.csv \
    --truth data/intermediate/_calibration_truth.jsonl \
    --out data/intermediate/calibration_report.json
```

The `calibration_report.json` is read by `scripts/build_release.py` during
release packaging — see the release-gate checklist in that script.
