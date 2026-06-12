# Adjudication Protocol — SemiHS-Bench

Scope: governs how disagreements between paired raters are resolved during
the expert audit pass. This protocol uses an **evidence-binding rubric**
rather than third-reviewer majority votes: the win goes to the
rater whose cited EBTI/CROSS evidence more tightly matches the disputed
record.

This document operationalizes the evidence-binding step of the expert audit.

## 1. When this protocol applies

- Any benchmark record whose two raters assigned **different** `expert_hs6`.
- Any benchmark record with `difficulty_tags ≠ []` (boundary case) — these
  are routed through second-rater adjudication even if both raters agree, to
  document the citation chain.
- Any BOL-sourced record — BOL records always carry second-rater confirmation
  given the higher rate of declared-HS error in BOL data.

## 2. Rubric

The third reviewer (adjudicator) scores **each rater independently** on the
four dimensions below, blind to which rater cited which evidence (see §3
procedure). The rater with the higher total score wins.

| Dimension | Weight | Scoring |
|---|---:|---|
| Jurisdiction match | 1 | Record source (BOL country / catalog market) matches the ruling jurisdiction → +1. Mismatch → 0. |
| HS6 exactness | 2 | Ruling's `hs6_label` == rater's proposed HS6 → +2. Ruling's HS4 matches but HS6 differs → +1. Otherwise → 0. |
| Technical-spec overlap | 2 | ≥3 spec tokens shared between the record's structured-intermediate `key_specs` (the BOM-style form computed internally during construction) and ruling's `subject_terms` → +2. 1–2 tokens shared → +1. None → 0. |
| Ruling recency | 1 | Ruling issued within ≤5 years of record's source date → +1. Older → 0. |

**Maximum score per rater: 6.** Ties go to a fourth-step direct
adjudication (see §4).

### 2.1 Worked example

A record describes "200mm 4H-SiC wafer, p-type, boron-doped, 5000 PCS,
consignee TSMC Arizona". BOL date: 2024-06-15.

- **Rater A** proposes `hs6=381800` (doped wafer), cites
  `EBTI-DE-BTI-2024-456` (ruling on a 150mm SiC wafer, hs6=381800, dated
  2024-02-01, subject_terms includes ["SiC wafer", "doped", "p-type"]).
- **Rater B** proposes `hs6=854190` (parts of semiconductor devices),
  cites `CROSS-N234567` (ruling on a partially-processed SiC die with
  metallization, hs6=854190, dated 2021-08-12,
  subject_terms includes ["SiC die", "processed wafer", "metal layer"]).

Scoring:

| Dimension | Rater A | Rater B |
|---|---:|---:|
| Jurisdiction match | 0 (US record, EU ruling) | +1 (US record, US ruling) |
| HS6 exactness | +2 (ruling hs6=381800 matches A's proposed 381800) | +2 (ruling hs6=854190 matches B's proposed 854190) |
| Technical-spec overlap | +2 (3 tokens: "SiC", "wafer", "doped") | +1 (1 token: "SiC") |
| Ruling recency | +1 (2024) | 0 (2021) |
| **Total** | **5** | **4** |

Rater A wins. The record's `expert_hs6` is set to 381800, with
`adjudication_winning_evidence_id = "EBTI-DE-BTI-2024-456"` and
`adjudication_rubric_score = 5`.

## 3. Procedure (blind scoring)

The adjudicator UI / worksheet enforces blind scoring:

1. The system surfaces the record + both raters' citations **without** rater
   identity attached. Rater A's citation appears as "Citation 1"; Rater B's
   appears as "Citation 2" (or vice versa — random per record).
2. The adjudicator scores both citations on all four rubric dimensions
   without knowing which rater chose which.
3. After scoring is submitted, the system reveals rater identity and applies
   the higher score → wins the proposed HS6.
4. The adjudicator may **not** edit scores after the reveal. Disagreements
   with the scoring outcome go to direct adjudication (§4).

This blinding prevents adjudicator drift toward whichever rater they trust
more, and centers the resolution on evidence quality.

## 4. Direct adjudication (tie-breaker)

When both raters tie at the rubric, the adjudicator may either:

1. **Pick a side** based on independent evaluation of the record. Document
   in `adjudication_rationale` (free text, ≤200 chars). Status:
   `adjudicated_direct`.
2. **Drop the record** as unresolvable. Status: `unresolved_dropped`. The
   record is removed from the benchmark — never released in an auxiliary
   subset.

The expectation is ≤5% of adjudicated records require §4 direct adjudication
(most disagreements have a clear citation winner). If §4 invocations exceed
10%, this is a signal that the rubric weights need recalibration; flag for
review before continuing the audit pass.

## 5. Schema fields

These are **internal audit-trail columns**, recorded during the expert audit
and kept in the working pool for provenance. They are **stripped during release
packaging** and do not appear on the released records or in the released
schemas — only the resolved `hs6_label`, `difficulty_tags`, and `boundary_note`
survive to release. Each audited record carries:

| Field | Type | Meaning |
|---|---|---|
| `adjudication_status` | enum | `single_reviewer` (no disagreement) / `adjudicated_consensus` (both raters agreed at outset) / `adjudicated_evidence_resolved` (§2 rubric resolved disagreement) / `adjudicated_direct` (§4 tie-break) / `unresolved_dropped` (record removed) |
| `adjudication_winning_evidence_id` | str \| null | The `evidence_id` from `reference_corpus.jsonl` that the winning citation pointed at. `null` if status ∈ {single_reviewer, adjudicated_consensus} or §4 direct. |
| `adjudication_rubric_score` | int \| null | Winning citation's total rubric score (0–6). `null` for non-§2 outcomes. |
| `adjudication_rationale` | str \| null | Free text, ≤200 chars. Required for §4 direct adjudication; optional otherwise. |

These columns are populated by the audit worksheet
(`src/audit/decisions.py`) and validated there during the audit pass. They are
not part of the released record schema (`../../data/record_schema.json`) or the
released submission schema (`../../eval/submission_schema.json`).

## 6. Inter-adjudicator agreement (optional)

For a small sample (~30 records), have two adjudicators independently score
the same disagreements. Compare:

- κ on dimension scores (per dimension).
- Agreement on the winning rater.
- Agreement on the final HS6.

Report in the reliability section. Not a release blocker — this is
a robustness check, not a gate.

## 7. Files

| File | Role |
|---|---|
| `docs/ADJUDICATION_PROTOCOL.md` | This document (rubric, procedure). |
| `src/audit/decisions.py` | Worksheet schema + validator. |
| `release/working/data/_adjudication_report.json` | Per-record adjudication audit trail (internal working pool) emitted by `scripts/build_release.py`. |
| `configs/boundary_tags.yaml` | Boundary tags whose records are always routed through adjudication. |
| `../../data/reference_corpus.jsonl` | Evidence pool the rater citations point at. |

## 8. Re-run

Adjudication is part of the main audit pass — there is no separate re-run
script. To re-score a subset (e.g., after a rubric calibration update):

```
python scripts/apply_audit_decisions.py \
    --rescore-from data/intermediate/audit/decisions.csv \
    --rubric-version 2.0 \
    --out release/working/data/_adjudication_report.json
```

A rubric version bump (`--rubric-version`) is required when rescoring with
modified weights; the release MANIFEST records the rubric version used so
re-scored releases are not confused with original ones.
