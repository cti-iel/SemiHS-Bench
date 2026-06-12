#!/usr/bin/env python3
"""Sample the 60-record rater-vs-authority calibration set.

Implements the authority-calibration stratum (docs/IAA_PROTOCOL.md §7):
construct a calibration stratum from EBTI/CROSS rulings whose authoritative
HS6 label is *stripped before annotation*. Two raters annotate blind, then
their answers are scored against the original authoritative HS6 to compute
rater-vs-authority accuracy (the headline reliability statistic for the
annotation protocol).

Inputs:

* ``../data/reference_corpus.jsonl`` — the released 889-ruling corpus
  (CROSS 400 + EBTI 389 + JP 100). Only the EBTI and CROSS rulings are
  drawn for calibration; the JP rulings are excluded by the source filter.

Sampling rule (stratified):

* **60 records total** = 40 EBTI + 20 CROSS (the calibration stratum
  spec).
* Stratified by HS4 across the in-scope allow-set (``configs/hs6_scope_tiers.yaml``).
* Per-HS4 floor: at least 1 record per HS4 represented; remainder
  distributed proportionally to HS4 share in the source pool.
* Deterministic: seed = ``20260514`` (fixed integer for reproducibility).

Blinding (the part that makes this a *calibration* not a peek):

* The authoritative ``hs6_label`` field is **stripped** from rater-facing CSVs.
* HS-code patterns ARE REDACTED in ``tier1_text`` (rare but present in
  some CROSS rulings that quote their own HTSUS code, ~1.8% of records).
* ``rationale_excerpt`` is NOT included in rater-facing CSVs at all — that
  field is the authority's reasoning about WHY the HS6 was chosen, and
  contains explicit HS-code references in ~25% of records. Showing it
  would defeat the blind-annotation design.

Outputs:

* ``data/intermediate/calibration_input.csv`` — combined rater-facing CSV.
* ``data/intermediate/calibration_input_rater_a.csv`` — rater A's copy
  (identical content to the combined CSV; named for downstream pairing).
* ``data/intermediate/calibration_input_rater_b.csv`` — rater B's copy.
* ``data/intermediate/CALIBRATION_README.md`` — rater instructions.
* ``data/intermediate/_calibration_truth.jsonl`` — held-out truth table
  mapping ``calibration_id`` → authoritative HS6, source, jurisdiction,
  language, and a ``leakage_flag`` for records where redaction caught a
  reveal (so the scorer can drop or weight those).
* ``data/intermediate/_calibration_sample_report.json`` — sampling
  manifest (per-source, per-HS4 counts, seed, redaction stats).
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPO_DATA = ROOT.parent / "data"
INTERMEDIATE = ROOT / "data" / "intermediate"

REF_CORPUS_PATH = REPO_DATA / "reference_corpus.jsonl"
CALIBRATION_INPUT_PATH = INTERMEDIATE / "calibration_input.csv"
CALIBRATION_RATER_A_PATH = INTERMEDIATE / "calibration_input_rater_a.csv"
CALIBRATION_RATER_B_PATH = INTERMEDIATE / "calibration_input_rater_b.csv"
CALIBRATION_README_PATH = INTERMEDIATE / "CALIBRATION_README.md"
CALIBRATION_TRUTH_PATH = INTERMEDIATE / "_calibration_truth.jsonl"
SAMPLE_REPORT_PATH = INTERMEDIATE / "_calibration_sample_report.json"

_SAMPLE_SEED = 20260514  # fixed integer; no external significance
_TARGET_TOTAL = 60
_TARGET_EBTI = 40
_TARGET_CROSS = 20

# Rater CSV columns (order matters — set once for both rater files).
_RATER_FIELDS = [
    "calibration_id",
    "source",
    "jurisdiction",
    "language",
    "tier1_text",
    "tier2_part_name",
    "tier2_manufacturer",
    "subject_terms",         # JSON-encoded list
    "rater_hs6",             # rater fills (6 digits)
    "rater_confidence_tier", # rater fills (high|medium|low)
    "rater_cited_evidence_ids",  # rater fills (comma-separated)
    "rater_rationale_short", # rater fills (free text, ≤200 chars)
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratify(records: List[Mapping[str, Any]], rng: random.Random,
              target: int) -> List[Dict[str, Any]]:
    """Sample ``target`` records stratified by HS4.

    Per-HS4 quota = floor of 1 per HS4 present + proportional remainder
    (largest-fractional-remainder distribution).
    """
    by_hs4: Dict[str, List[Mapping[str, Any]]] = {}
    for r in records:
        hs6 = str(r.get("hs6_label") or "")
        if not hs6:
            continue
        by_hs4.setdefault(hs6[:4], []).append(r)
    total = sum(len(v) for v in by_hs4.values())
    if total == 0:
        return []

    floor = {hs4: 1 for hs4 in by_hs4}
    if sum(floor.values()) > target:
        # More HS4s than target → keep the largest HS4s.
        ranked = sorted(by_hs4, key=lambda h: -len(by_hs4[h]))[:target]
        floor = {h: 1 for h in ranked}
        remainder = 0
    else:
        remainder = target - sum(floor.values())

    if remainder > 0:
        weights = {hs4: len(by_hs4[hs4]) / total for hs4 in by_hs4}
        ideal = {hs4: weights[hs4] * remainder for hs4 in by_hs4}
        whole = {hs4: int(ideal[hs4]) for hs4 in by_hs4}
        leftover = remainder - sum(whole.values())
        if leftover > 0:
            fractional = sorted(
                by_hs4, key=lambda h: -(ideal[h] - whole[h])
            )
            for hs4 in fractional[:leftover]:
                whole[hs4] += 1
        for hs4 in by_hs4:
            floor[hs4] += whole.get(hs4, 0)

    sampled: List[Dict[str, Any]] = []
    for hs4 in floor:
        pool = list(by_hs4[hs4])
        rng.shuffle(pool)
        for r in pool[: min(floor[hs4], len(pool))]:
            sampled.append(dict(r))
    return sampled


# ---------------------------------------------------------------------------
# HS-code redaction (prevent leakage of authoritative HS6 in tier1_text)
# ---------------------------------------------------------------------------

_REDACT_TOKEN = "[REDACTED-HS]"


def _redact_hs_mentions(text: str, hs6: str) -> Tuple[str, int]:
    """Redact authoritative HS6 / HS6-dotted / HTSUS-pattern mentions in
    ``text``. Returns the redacted text plus the number of substitutions
    made. We do NOT redact HS4-only mentions because raters need to be
    able to reason about HS4 family in their rationale.

    The redaction targets:
      - "854110" (six solid digits)
      - "8541.10" (4.2 dotted form)
      - "8541.10.00", "8541.10.0000" (HTSUS, 8 or 10 digits with dots)
    """
    if not text or not hs6 or len(hs6) != 6:
        return text, 0
    hs4 = hs6[:4]
    hs2 = hs6[4:6]
    htsus_re = re.compile(rf"\b{hs4}\.{hs2}\.\d{{2,6}}\b")
    n = 0
    new_text, m = htsus_re.subn(_REDACT_TOKEN, text)
    n += m
    new_text, m = re.subn(rf"\b{hs4}\.{hs2}\b", _REDACT_TOKEN, new_text)
    n += m
    new_text, m = re.subn(rf"\b{hs6}\b", _REDACT_TOKEN, new_text)
    n += m
    return new_text, n


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def _build_rater_row(
    calibration_id: str,
    record: Mapping[str, Any],
) -> Tuple[Dict[str, str], int]:
    """Return (rater_row_dict, redaction_count)."""
    hs6 = str(record.get("hs6_label") or "")
    tier1 = str(record.get("tier1_text") or "")
    tier1_redacted, redactions = _redact_hs_mentions(tier1, hs6)

    tier2 = record.get("tier2_minimal") or {}
    return (
        {
            "calibration_id": calibration_id,
            "source": str(record.get("source") or ""),
            "jurisdiction": str(record.get("jurisdiction") or ""),
            "language": str(record.get("language") or "")
                        or _guess_lang_from_jurisdiction(record.get("jurisdiction")),
            "tier1_text": tier1_redacted,
            "tier2_part_name": str(tier2.get("part_name") or ""),
            "tier2_manufacturer": str(tier2.get("manufacturer") or ""),
            "subject_terms": json.dumps(
                record.get("subject_terms") or [], ensure_ascii=False
            ),
            "rater_hs6": "",
            "rater_confidence_tier": "",
            "rater_cited_evidence_ids": "",
            "rater_rationale_short": "",
        },
        redactions,
    )


def _guess_lang_from_jurisdiction(jur: Any) -> str:
    """When ``language`` is empty on a carryover record, fall back to
    the jurisdiction code's language (best-effort hint for the rater)."""
    if not jur:
        return ""
    j = str(jur)
    if j == "US":
        return "en"
    if j == "EU-EN":
        return "en"
    if j.startswith("EU-"):
        return j[3:].lower()
    return ""


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

_README = """# SemiHS-Bench — Calibration Set Instructions

## What this is

You are looking at **60 customs-ruling records** whose authoritative
HS6 classification has been hidden from you. Your job is to read each
record like you would a benchmark record and propose your own HS6 — we
then score your answer against the customs authority's actual ruling
to measure rater-vs-authority accuracy.

This calibration step is described in
[IAA_PROTOCOL.md §7](../../../docs/IAA_PROTOCOL.md#7-authority-calibration-stratum).
Its purpose is to measure rater reliability directly against customs
authority decisions.

## Files

- **calibration_input.csv** — combined input. Either you and rater B
  both edit this file (in two copies) or you each use the per-rater file
  below.
- **calibration_input_rater_a.csv** — your copy if you are rater A.
- **calibration_input_rater_b.csv** — rater B's copy.

Edit your assigned file, fill the four `rater_*` columns, and return.

## Columns to fill (per record)

| Column | What you put | Format |
|---|---|---|
| `rater_hs6` | Your proposed HS6 | exactly 6 digits, e.g. `854231` |
| `rater_confidence_tier` | How confident you are | `high` / `medium` / `low` |
| `rater_cited_evidence_ids` | EBTI/CROSS ruling IDs that support your call | comma-separated; can be empty for `low` |
| `rater_rationale_short` | One-sentence rationale | free text, ≤ 200 chars |

Use confidence tiers per the Core-4 evidence rule:
- `high` — you cite ≥ 2 supporting EBTI/CROSS rulings from ≥ 2
  jurisdictions (EU + US).
- `medium` — you cite ≥ 1 supporting ruling.
- `low` — you cannot cite a supporting ruling (still propose an HS6,
  flagged as uncertain).

## Columns NOT to edit

- `calibration_id` — opaque ID; do not change.
- `source`, `jurisdiction`, `language` — informational.
- `tier1_text`, `tier2_*`, `subject_terms` — the record's
  content. Some HS-code patterns may appear as `[REDACTED-HS]` — the
  authority's HS6 has been masked there so you can't peek.

## Working rules

1. **No cross-talk** between raters while filling. The whole point is
   independent annotation.
2. Reference the **reference corpus** (`../data/reference_corpus.jsonl`)
   to find supporting rulings. **You may NOT look at the calibration
   record's own evidence_id** — that ID is stripped from the rater
   CSV; if you stumble across the original ruling in the corpus while
   searching, skip it and use a different one.
3. Take 5–10 min per record. If you genuinely can't reach an HS6 with
   confidence ≥ `low`, leave the row blank and flag in
   `rater_rationale_short`.
4. Submit your file as
   `data/intermediate/calibration_annotated_rater_<your_id>.csv`.

## Scoring

After both raters submit:

```
python -m src.annotation.authority_calibration \\
    --rater-a data/intermediate/calibration_annotated_rater_a.csv \\
    --rater-b data/intermediate/calibration_annotated_rater_b.csv \\
    --truth data/intermediate/_calibration_truth.jsonl \\
    --out data/intermediate/calibration_report.json
```

Acceptance gate: joint authority accuracy ≥ 70%.
Below that, the codebook is refined and a fresh 60-record sample is
drawn (see IAA_PROTOCOL.md §7.5).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not REF_CORPUS_PATH.exists():
        print(
            f"ERROR: {REF_CORPUS_PATH} missing. "
            "Run scripts/build_reference_corpus.py first.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(_SAMPLE_SEED)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    corpus = _read_jsonl(REF_CORPUS_PATH)
    ebti = [r for r in corpus if r.get("source") == "EBTI"]
    cross = [r for r in corpus if r.get("source") == "CROSS"]

    print(f"reference corpus loaded: {len(corpus)} entries "
          f"({len(ebti)} EBTI + {len(cross)} CROSS)")

    if len(ebti) < _TARGET_EBTI:
        print(f"WARN: only {len(ebti)} EBTI available; target {_TARGET_EBTI}",
              file=sys.stderr)
    if len(cross) < _TARGET_CROSS:
        print(f"WARN: only {len(cross)} CROSS available; target {_TARGET_CROSS}",
              file=sys.stderr)

    sampled_ebti = _stratify(ebti, rng, _TARGET_EBTI)
    sampled_cross = _stratify(cross, rng, _TARGET_CROSS)
    sampled = sampled_ebti + sampled_cross
    rng.shuffle(sampled)

    print(f"sampled {len(sampled)}: {len(sampled_ebti)} EBTI + "
          f"{len(sampled_cross)} CROSS")

    # Build rater + truth rows.
    rater_rows: List[Dict[str, str]] = []
    truth_rows: List[Dict[str, Any]] = []
    redaction_counter: Counter = Counter()
    leakage_flags: List[bool] = []
    for i, record in enumerate(sampled):
        cid = f"CAL-{i+1:03d}"
        row, redactions = _build_rater_row(cid, record)
        rater_rows.append(row)
        redaction_counter["total_substitutions"] += redactions
        if redactions > 0:
            redaction_counter["records_with_redactions"] += 1
        # Truth row also tags whether the source rationale_excerpt
        # contained the authoritative HS6 (informational; we don't ship
        # rationale to raters but the scorer can use this).
        rationale = str(record.get("rationale_excerpt") or "")
        rat_leak = bool(rationale) and (
            str(record.get("hs6_label") or "") in rationale
            or f"{record.get('hs6_label', '')[:4]}.{record.get('hs6_label', '')[4:6]}" in rationale
        )
        leakage_flags.append(rat_leak)
        truth_rows.append({
            "calibration_id": cid,
            "evidence_id": record.get("evidence_id"),
            "authoritative_hs6": record.get("hs6_label"),
            "authoritative_hs4": str(record.get("hs6_label") or "")[:4],
            "source": record.get("source"),
            "jurisdiction": record.get("jurisdiction"),
            "language": record.get("language") or "",
            "tier1_redactions_applied": redactions,
            "source_rationale_contained_hs6": rat_leak,
            "source_url": record.get("url"),
            "ruling_date": record.get("ruling_date"),
        })

    # Write rater-facing CSVs (one combined + one per rater).
    for out_path in (CALIBRATION_INPUT_PATH, CALIBRATION_RATER_A_PATH,
                     CALIBRATION_RATER_B_PATH):
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_RATER_FIELDS)
            writer.writeheader()
            writer.writerows(rater_rows)
    print(f"wrote {CALIBRATION_INPUT_PATH.relative_to(ROOT)}")
    print(f"wrote {CALIBRATION_RATER_A_PATH.relative_to(ROOT)}")
    print(f"wrote {CALIBRATION_RATER_B_PATH.relative_to(ROOT)}")

    # Write README.
    CALIBRATION_README_PATH.write_text(_README, encoding="utf-8")
    print(f"wrote {CALIBRATION_README_PATH.relative_to(ROOT)}")

    # Write truth table.
    with CALIBRATION_TRUTH_PATH.open("w", encoding="utf-8") as f:
        for row in truth_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {CALIBRATION_TRUTH_PATH.relative_to(ROOT)}")

    # Per-HS4 / per-source / per-language stats for the report.
    per_hs4: Counter = Counter()
    per_hs4_by_source: Dict[str, Counter] = {"EBTI": Counter(), "CROSS": Counter()}
    per_jurisdiction: Counter = Counter()
    per_language: Counter = Counter()
    for r in sampled:
        h4 = str(r.get("hs6_label") or "")[:4]
        per_hs4[h4] += 1
        per_hs4_by_source[str(r.get("source") or "")][h4] += 1
        per_jurisdiction[str(r.get("jurisdiction") or "")] += 1
        lang = str(r.get("language") or "") or _guess_lang_from_jurisdiction(r.get("jurisdiction"))
        per_language[lang or "unknown"] += 1

    # In-scope HS4 coverage check.
    in_scope_hs4 = {"2804", "3707", "3818", "8486", "8541", "8542", "9030", "9031"}
    missing_hs4 = sorted(in_scope_hs4 - set(per_hs4))

    report = {
        "release": "working",
        "seed": _SAMPLE_SEED,
        "target": {
            "total": _TARGET_TOTAL,
            "ebti": _TARGET_EBTI,
            "cross": _TARGET_CROSS,
        },
        "sampled": {
            "total": len(sampled),
            "ebti": len(sampled_ebti),
            "cross": len(sampled_cross),
        },
        "per_hs4": dict(sorted(per_hs4.items())),
        "per_hs4_by_source": {
            src: dict(sorted(counter.items()))
            for src, counter in per_hs4_by_source.items()
        },
        "per_jurisdiction": dict(sorted(per_jurisdiction.items())),
        "per_language": dict(sorted(per_language.items())),
        "missing_in_scope_hs4": missing_hs4,
        "redactions": {
            "total_substitutions": redaction_counter["total_substitutions"],
            "records_with_tier1_redactions": redaction_counter["records_with_redactions"],
            "records_with_source_rationale_hs6_mention": sum(leakage_flags),
            "note": (
                "tier1_text redactions replace the authoritative HS6 / "
                "HS6-dotted / HTSUS patterns with '[REDACTED-HS]'. "
                "rationale_excerpt is NOT shipped to raters (it contains "
                "the authority's reasoning about why HS6=X). The "
                "source_rationale_contained_hs6 truth-table flag is "
                "informational only."
            ),
        },
        "outputs": {
            "rater_csv": str(CALIBRATION_INPUT_PATH.relative_to(ROOT)),
            "rater_a_csv": str(CALIBRATION_RATER_A_PATH.relative_to(ROOT)),
            "rater_b_csv": str(CALIBRATION_RATER_B_PATH.relative_to(ROOT)),
            "rater_readme": str(CALIBRATION_README_PATH.relative_to(ROOT)),
            "truth_jsonl": str(CALIBRATION_TRUTH_PATH.relative_to(ROOT)),
        },
        "downstream": (
            "After two raters submit annotated CSVs (e.g., "
            "data/intermediate/calibration_annotated_rater_a.csv), run "
            "src/annotation/authority_calibration.py to score against "
            "_calibration_truth.jsonl."
        ),
    }
    SAMPLE_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {SAMPLE_REPORT_PATH.relative_to(ROOT)}")

    print()
    print(f"Sampling summary:")
    print(f"  per HS4: {dict(sorted(per_hs4.items()))}")
    print(f"  per language: {dict(sorted(per_language.items()))}")
    print(f"  tier1 redactions: {redaction_counter['records_with_redactions']} "
          f"records, {redaction_counter['total_substitutions']} substitutions")
    print(f"  rationale-leakage in source data: {sum(leakage_flags)} "
          f"(not shipped to raters; flag preserved in truth table)")
    if missing_hs4:
        print(f"  ⚠ in-scope HS4 missing from calibration: {missing_hs4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
