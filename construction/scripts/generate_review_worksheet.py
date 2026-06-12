#!/usr/bin/env python3
"""Generate the expert-audit worksheet from one or more candidate pools.

Implements the operational unlock for the expert audit pass: seeds the
records from ``--candidate-pool`` (default: the carryover pool plus the
net-new BOL pool emitted by ``scripts/ingest_bol.py``) into a
ready-to-fill CSV worksheet that uses
the Core-4 protocol column set (``WORKSHEET_COLUMNS`` from
``src/audit/decisions.py``).

Inputs:
  - ``release/working/data/_candidate_pool.jsonl`` — carryover records
    tagged with ``label_source = *_pending_reaudit`` and
    ``adjudication_status = pending``.
  - ``release/working/data/_candidate_pool_bol.jsonl`` (when present) —
    net-new BOL records from ``scripts/ingest_bol.py``.
  - ``release/working/data/reference_corpus.jsonl`` — EBTI/CROSS entries used
    to surface up to 5 same-HS4 candidate reference rulings per record.
  - ``configs/hs6_scope_tiers.yaml`` — to derive ``scope_tier`` ∈
    {core, supply_chain} scope partition.
  - ``configs/manufacturer_caps.yaml`` (optional) — surfaced in the
    informational ``manufacturer_hint`` column when source metadata has it.

Outputs:
  - ``data/intermediate/audit_worksheet.csv`` — the worksheet.
    Pre-fills 8 informational columns; leaves 11 rater columns blank.
  - ``data/intermediate/AUDIT_README.md`` — rater instructions
    (column meanings, action/confidence-tier/adjudication-status enums,
    evidence-count → confidence binding, submission steps).
  - ``data/intermediate/_audit_worksheet_report.json`` — manifest
    (per-HS4 row counts, per-source breakdown, reference-corpus
    coverage for each row).

Idempotent (deterministic ordering). Re-running overwrites the worksheet
only when ``--force`` is set — protects in-progress rater edits.

The completed worksheet is consumed by
``src/audit/decisions.parse_worksheet(path)`` and then
``apply_corrections()`` to flip the candidate-pool records into the
final benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.decisions import WORKSHEET_COLUMNS  # noqa: E402


def _rel(path: Path) -> str:
    """Render ``path`` relative to ROOT when possible, else absolute.
    Tests pass output paths from /tmp/..., which is outside ROOT and
    would crash Path.relative_to(). Matches the same helper in
    apply_audit_decisions.py."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


WORKING_DATA = ROOT / "release" / "working" / "data"
CANDIDATE_POOL_PATH = WORKING_DATA / "_candidate_pool.jsonl"
BOL_POOL_PATH = WORKING_DATA / "_candidate_pool_bol.jsonl"
CATALOG_POOL_PATH = WORKING_DATA / "_candidate_pool_catalog.jsonl"
DEFAULT_CANDIDATE_POOLS = [CANDIDATE_POOL_PATH, BOL_POOL_PATH, CATALOG_POOL_PATH]
REF_CORPUS_PATH = WORKING_DATA / "reference_corpus.jsonl"
SCOPE_PATH = ROOT / "configs" / "hs6_scope_tiers.yaml"

INTERMEDIATE = ROOT / "data" / "intermediate"
OUTPUT_PATH = INTERMEDIATE / "audit_worksheet.csv"
README_PATH = INTERMEDIATE / "AUDIT_README.md"
REPORT_PATH = INTERMEDIATE / "_audit_worksheet_report.json"

_MAX_CANDIDATE_RULINGS_PER_ROW = 5

# Source-family filter values for --filter-source. Mapped to per-source
# output paths and row-builder behaviour below.
FILTER_SOURCE_VALUES = ("all", "catalog", "bol", "carryover")

# Per-source output filename stems. Joined to the --output directory.
PER_SOURCE_PATHS: Dict[str, Dict[str, str]] = {
    "all": {
        "worksheet": "audit_worksheet.csv",
        "readme": "AUDIT_README.md",
        "report": "_audit_worksheet_report.json",
    },
    "catalog": {
        "worksheet": "audit_worksheet_catalog.csv",
        "readme": "AUDIT_README_catalog.md",
        "report": "_audit_worksheet_catalog_report.json",
    },
    "bol": {
        "worksheet": "audit_worksheet_bol.csv",
        "readme": "AUDIT_README_bol.md",
        "report": "_audit_worksheet_bol_report.json",
    },
    "carryover": {
        "worksheet": "audit_worksheet_carryover.csv",
        "readme": "AUDIT_README_carryover.md",
        "report": "_audit_worksheet_carryover_report.json",
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return json.loads(content)


def _scope_tier_lookup() -> Dict[str, str]:
    """Return HS6 → scope_tier mapping from configs/hs6_scope_tiers.yaml."""
    cfg = _load_yaml_or_json(SCOPE_PATH)
    out: Dict[str, str] = {}
    for h6 in cfg.get("core") or []:
        out[str(h6)] = "core"
    for h6 in cfg.get("supply_chain") or []:
        out[str(h6)] = "supply_chain"
    return out


def _ref_corpus_by_hs4(entries: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Index reference-corpus entries by HS4 for same-HS4 candidate retrieval.

    Within each HS4 bucket, entries are sorted so that:
      1. Same-HS6-as-record entries come first (preferred matches).
      2. Then by jurisdiction diversity (alternates EU vs US).
      3. Then by recency.

    The candidate-row builder takes the top-N from this list filtered to
    the row's exact HS6 first, then falls back to same-HS4 if fewer than
    N matches.
    """
    by_hs4: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for e in entries:
        hs6 = e.get("hs6_label") or ""
        if not hs6:
            continue
        by_hs4[hs6[:4]].append(e)
    return by_hs4


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set:
    """Lowercased alphanumeric tokens, length ≥ 2 (drops noise tokens like 'a'
    and digits split out by punctuation)."""
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2}


def _record_tokens(record: Mapping[str, Any]) -> set:
    """Token set for a catalog record's description + key_specs values.

    Used by similarity-boosted ruling ranking to prioritize EBTI/CROSS
    candidates whose description shares vocabulary with the catalog row.
    """
    parts: List[str] = [str(record.get("tier1_description") or "")]
    tier2 = record.get("tier2_minimal") or {}
    parts.append(str(tier2.get("part_name") or ""))
    sm = record.get("source_metadata") or {}
    parts.append(str(sm.get("manufacturer_hint") or ""))
    return _tokens(" ".join(parts))


def _score_catalog_ruling_similarity(
    record_tokens: set, ruling: Mapping[str, Any]
) -> float:
    """Jaccard overlap between catalog record tokens and a candidate ruling's
    description/keywords. Returns 0.0 when either side is empty."""
    if not record_tokens:
        return 0.0
    ruling_text_parts: List[str] = []
    for key in ("description", "product_name", "summary"):
        v = ruling.get(key)
        if v:
            ruling_text_parts.append(str(v))
    for kw in (ruling.get("keywords") or []):
        ruling_text_parts.append(str(kw))
    ruling_tokens = _tokens(" ".join(ruling_text_parts))
    if not ruling_tokens:
        return 0.0
    inter = len(record_tokens & ruling_tokens)
    union = len(record_tokens | ruling_tokens)
    return inter / union if union else 0.0


def _candidate_rulings_for(
    record: Mapping[str, Any],
    by_hs4: Mapping[str, List[Mapping[str, Any]]],
    limit: int = _MAX_CANDIDATE_RULINGS_PER_ROW,
    *,
    similarity_boost: bool = False,
) -> List[str]:
    """Return up to ``limit`` evidence_ids to surface as candidate
    reference rulings for the audit row, preferring same-HS6 matches and
    then jurisdictional diversity within the same HS4.

    When ``similarity_boost`` is True (catalog rows), high-Jaccard rulings
    within the same-HS6 partition are sorted to the top — gives the rater
    a content-matched citation first.
    """
    hs6 = str(record.get("hs6_label") or "")
    hs4 = hs6[:4]
    pool = list(by_hs4.get(hs4) or [])
    if not pool:
        return []

    same_hs6 = [e for e in pool if e.get("hs6_label") == hs6]
    other_hs4 = [e for e in pool if e.get("hs6_label") != hs6]

    def _interleave(records: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Round-robin the candidate pool across jurisdictions, recency-sorted
        within each bucket. EU and US lead because they have the deepest
        coverage; other jurisdictions follow so non-US/EU sources also surface
        on each row. Anything else falls into a generic bucket so no record
        is dropped from the result list."""
        def _bucket(predicate):
            return sorted(
                [r for r in records if predicate(r)],
                key=lambda r: (r.get("ruling_date") or "1900-01-01"),
                reverse=True,
            )

        eu = _bucket(lambda r: str(r.get("jurisdiction") or "").startswith("EU-"))
        us = _bucket(lambda r: r.get("jurisdiction") == "US")
        jp = _bucket(lambda r: r.get("jurisdiction") == "JP")
        ca = _bucket(lambda r: r.get("jurisdiction") == "CA")
        intl = _bucket(lambda r: r.get("jurisdiction") == "INT")
        known_juris = {"US", "JP", "CA", "INT"}
        other = _bucket(
            lambda r: not str(r.get("jurisdiction") or "").startswith("EU-")
            and r.get("jurisdiction") not in known_juris
        )

        buckets = [eu, us, jp, ca, intl, other]
        out: List[Mapping[str, Any]] = []
        total = len(records)
        while any(buckets) and len(out) < total:
            for b in buckets:
                if b and len(out) < total:
                    out.append(b.pop(0))
        return out

    if similarity_boost and same_hs6:
        # Pre-rank same_hs6 by Jaccard similarity (descending), with ties
        # broken by recency. Rulings scoring ≥ 0.3 are surfaced first; the
        # rest fall back to the default EU/US interleave.
        record_tokens = _record_tokens(record)
        scored = [
            (_score_catalog_ruling_similarity(record_tokens, r), r) for r in same_hs6
        ]
        high = [r for s, r in sorted(scored, key=lambda x: -x[0]) if s >= 0.3]
        seen_ids = {id(r) for r in high}
        rest_same_hs6 = [r for _, r in scored if id(r) not in seen_ids]
        ranked = high + _interleave(rest_same_hs6) + _interleave(other_hs4)
    else:
        ranked = _interleave(same_hs6) + _interleave(other_hs4)

    return [str(r.get("evidence_id") or "") for r in ranked[:limit] if r.get("evidence_id")]


# ---------------------------------------------------------------------------
# Source-family detection
# ---------------------------------------------------------------------------

def _classify_source(record: Mapping[str, Any]) -> str:
    """Return 'carryover' | 'catalog' | 'bol' | 'unknown' for a pool record.

    Carryover is detected by the presence of a carryover_origin block (a
    data-format constant of the existing candidate pools; set when the
    carryover pool was built). Net-new records are classified by
    label_source.
    """
    if record.get("carryover_origin") is not None:
        return "carryover"
    label_source = str(record.get("label_source") or "")
    if label_source == "catalog_expert_validated_pending_reaudit":
        return "catalog"
    if label_source == "BOL_expert_validated_pending_reaudit":
        return "bol"
    return "unknown"


def _carryover_origin_source(record: Mapping[str, Any]) -> str:
    """Recover the carryover record's original label_source (BOL_verified /
    expert_verified_catalog) from its carryover-origin metadata block."""
    origin = record.get("carryover_origin") or {}
    return str(origin.get("original_label_source") or "")


def _mfr_asserted_hs_for(record: Mapping[str, Any], source: str) -> str:
    """Pull the manufacturer-asserted HTSUS string surfaced to the rater.

    For net-new catalog rows: from source_metadata.catalog_hs_hint.
    For carryover rows whose original source was catalog: from the
      carryover-origin block's source_metadata when present.
    Blank for BOL and unknown.
    """
    sm = record.get("source_metadata") or {}
    if source == "catalog":
        return str(sm.get("catalog_hs_hint") or "")
    if source == "carryover":
        origin = record.get("carryover_origin") or {}
        original_source = str(origin.get("original_label_source") or "")
        if "catalog" in original_source.lower():
            origin_sm = origin.get("source_metadata") or {}
            return str(origin_sm.get("catalog_hs_hint") or origin_sm.get("hs_code") or "")
    return ""


def _has_mpn_and_manufacturer(record: Mapping[str, Any]) -> bool:
    """True iff both MPN and manufacturer are present somewhere on the
    record. Used to pre-fill tier2_classifiable=yes for catalog rows."""
    sm = record.get("source_metadata") or {}
    mpn = str(sm.get("catalog_part_number") or "").strip()
    if not mpn:
        tier2 = record.get("tier2_minimal") or {}
        mpn = str(tier2.get("part_name") or "").strip()
    manufacturer = str(sm.get("manufacturer_hint") or "").strip()
    if not manufacturer:
        tier2 = record.get("tier2_minimal") or {}
        manufacturer = str(tier2.get("manufacturer") or "").strip()
    return bool(mpn and manufacturer)


def _row_kind_for(source: str) -> str:
    if source == "carryover":
        return "carryover_reaudit"
    if source == "catalog":
        return "catalog_audit"
    if source == "bol":
        return "bol_audit"
    return "unknown_audit"


def _build_row(
    record: Mapping[str, Any],
    by_hs4: Mapping[str, List[Mapping[str, Any]]],
    scope_lookup: Mapping[str, str],
    *,
    source: Optional[str] = None,
) -> Dict[str, str]:
    """Build one worksheet row.

    ``source`` controls catalog-specific ergonomics (similarity-boosted
    ruling ranking, tier2_classifiable pre-fill, mfr_asserted_hs surfacing,
    row_kind). When None, falls back to _classify_source(record) — keeps
    the function callable in tests without threading the flag through.
    """
    if source is None:
        source = _classify_source(record)

    hs6 = str(record.get("hs6_label") or "")
    sm = record.get("source_metadata") or {}
    manufacturer_hint = str(sm.get("manufacturer_hint") or "")
    if not manufacturer_hint:
        tier2 = record.get("tier2_minimal") or {}
        manufacturer_hint = str(tier2.get("manufacturer") or "")

    # Manufacturer part number, surfaced informationally. The clean MPN lives in
    # tier2_minimal.part_name (e.g. "AH49ENTR-G1", "CM0116"); fall back to the
    # Catalog/source reference. Blank for BOL rows.
    mpn = str((record.get("tier2_minimal") or {}).get("part_name")
              or sm.get("catalog_part_number")
              or record.get("source_reference") or "")

    candidate_rulings = _candidate_rulings_for(
        record, by_hs4, similarity_boost=(source == "catalog"),
    )
    scope_tier = scope_lookup.get(hs6, "")
    mfr_asserted_hs = _mfr_asserted_hs_for(record, source)

    # Pre-fill tier2_classifiable=yes for catalog rows that have MPN+mfr.
    # Saves rater clicks; rater can override to partial/no.
    tier2_prefill = ""
    if source == "catalog" and _has_mpn_and_manufacturer(record):
        tier2_prefill = "yes"

    row = {col: "" for col in WORKSHEET_COLUMNS}
    row.update({
        "row_kind": _row_kind_for(source),
        "frozen_id": str(record.get("frozen_id") or ""),
        "record_id": str(record.get("id") or ""),
        "split": str(record.get("split") or ""),
        "current_hs6": hs6,
        "current_hs4": hs6[:4],
        "label_source": str(record.get("label_source") or ""),
        "tier1_description": (record.get("tier1_description") or "")[:300],
        "candidate_reference_rulings": ",".join(candidate_rulings),
        "manufacturer_hint": manufacturer_hint,
        "manufacturer_part_number": mpn,
        "scope_tier": scope_tier,
        "mfr_asserted_hs": mfr_asserted_hs,
        # Rater fills these (mostly blank; catalog rows get tier2 pre-fill):
        "action": "",
        "new_hs6": "",
        "expert_hs6": "",
        "confidence_tier": "",
        "cited_evidence_ids": "",
        "rationale_short": "",
        "tier2_classifiable": tier2_prefill,
        "adjudication_status": "",
        "adjudication_winning_evidence_id": "",
        "adjudication_rubric_score": "",
        "notes": "",
    })
    return row


def _sort_key_for(source: str):
    """Return a callable suitable for sorted(..., key=...) per source family.

    Catalog: (hs6, manufacturer_hint, frozen_id) — lets a vendor-savvy
    reviewer chunk their pass per vendor within an HS6.
    Others: (hs4, hs6, frozen_id) — preserves existing BOL/carryover sort.
    """
    if source == "catalog":
        return lambda r: (
            r.get("current_hs6", ""),
            (r.get("manufacturer_hint") or "").lower(),
            r.get("frozen_id", ""),
        )
    return lambda r: (
        r.get("current_hs4", ""),
        r.get("current_hs6", ""),
        r.get("frozen_id", ""),
    )


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

_SOURCE_HEADERS = {
    "all": (
        "Records come from up to three pre-audit pools:\n\n"
        "  * `release/working/data/_candidate_pool.jsonl` — carryover (re-audit pass).\n"
        "  * `release/working/data/_candidate_pool_bol.jsonl` — net-new BOL ingest.\n"
        "  * `release/working/data/_candidate_pool_catalog.jsonl` — net-new catalog ingest.\n"
    ),
    "catalog": (
        "**This worksheet contains only net-new catalog records** "
        "(`release/working/data/_candidate_pool_catalog.jsonl`). Two ergonomic "
        "tweaks for catalog rows:\n\n"
        "  * Rows are sorted by `manufacturer_hint` within each HS6 so you can "
        "review one vendor at a time.\n"
        "  * `tier2_classifiable` is pre-filled with `yes` when both MPN and "
        "manufacturer are present (override to `partial`/`no` if needed).\n"
        "  * The `mfr_asserted_hs` column shows the catalog's manufacturer-asserted "
        "HTSUS code (e.g. `8541.10.0080`) as a hint. **NOT authoritative** — "
        "verify against EBTI/CROSS rulings as usual.\n"
    ),
    "bol": (
        "**This worksheet contains only net-new BOL records** "
        "(`release/working/data/_candidate_pool_bol.jsonl`, output of "
        "`scripts/ingest_bol.py`). Standard audit conventions apply.\n"
    ),
    "carryover": (
        "**This worksheet contains only carryover records** "
        "(`release/working/data/_candidate_pool.jsonl`, output of "
        "re-tagged for re-audit). These were previously expert-validated "
        "under the earlier protocol and are being re-audited under the Core-4 protocol. The "
        "`carryover_origin` block on each record (visible in the pool JSONL) preserves "
        "the original `label_source` and source metadata for context.\n"
    ),
}

_README_TEMPLATE = """# SemiHS-Bench — Expert Audit Worksheet Instructions

## What this is

You are looking at the **expert-audit worksheet** (filter-source = `{source}`).
Each row is one candidate-pool record that needs Core-4 expert audit under
the Core-4 protocol (see `docs/ADJUDICATION_PROTOCOL.md`)
before it can be released as part of the benchmark.

{source_header}

The pre-filled informational columns surface everything you need to make a
decision; you then fill the 11 rater columns described below.

## Columns

### Pre-filled (do NOT edit)

| Column | Meaning |
|---|---|
| `row_kind` | `carryover_reaudit` (carryover re-audit), `bol_audit` (net-new BOL), or `catalog_audit` (net-new catalog). |
| `frozen_id` | Stable record ID (e.g. `v2.0.eval.0042`). |
| `record_id` | Internal record ID (e.g. `SH-0042`). |
| `split` | `dev` or `test`. |
| `current_hs6` / `current_hs4` | The HS6/HS4 label currently on the record. For catalog rows this comes from the catalog's manufacturer-asserted HTSUS — verify against rulings. |
| `label_source` | One of `BOL_expert_validated_pending_reaudit` or `catalog_expert_validated_pending_reaudit`. |
| `tier1_description` | The product description (≤ 300 chars; full text in the source candidate-pool JSONL). |
| `candidate_reference_rulings` | Up to 5 EBTI/CROSS evidence_ids in the same HS4 family that you should consider citing. Comma-separated. For catalog rows, the top entries are content-similarity ranked. |
| `manufacturer_hint` | Manufacturer name, if known. |
| `scope_tier` | `core` or `supply_chain` per `configs/hs6_scope_tiers.yaml`. |
| `mfr_asserted_hs` | (Catalog rows only) The catalog's manufacturer-asserted HTSUS code, e.g. `8541.10.0080`. **Informational only — verify against rulings.** Blank for BOL and most carryover rows. |

### Rater-filled (you fill these)

| Column | Format | What to put |
|---|---|---|
| `action` | `confirm` / `change` / `drop` | Your decision on the current HS6. Aliases accepted (e.g. `ok`, `wrong`, `relabel`, `remove`). |
| `new_hs6` | 6 digits | Required when `action=change`; the HS6 you propose instead. |
| `expert_hs6` | 6 digits | The HS6 you assign as gold. For `confirm`: must equal `current_hs6`. For `change`: must equal `new_hs6`. Optional but recommended for `drop`. |
| `confidence_tier` | `high` / `medium` / `low` | How confident you are in `expert_hs6`. See the binding below. |
| `cited_evidence_ids` | comma-separated | The EBTI/CROSS evidence_ids that support `expert_hs6`. Pull from the `candidate_reference_rulings` cell or search `release/working/data/reference_corpus.jsonl` yourself. |
| `rationale_short` | ≤ 200 chars | One-sentence justification. |
| `tier2_classifiable` | `yes` / `partial` / `no` | Can the gold HS6 be recovered from the MPN + manufacturer alone? Required for non-drop rows. |
| `adjudication_status` | enum (see below) | The outcome of this audit row. For first-pass single-rater work, use `single_reviewer`. |
| `adjudication_winning_evidence_id` | EBTI-…/CROSS-… ID | Required only when `adjudication_status = adjudicated_evidence_resolved` (rubric-resolved disagreement). |
| `adjudication_rubric_score` | integer 0–6 | Required only when `adjudication_status = adjudicated_evidence_resolved`. |
| `notes` | free text | Anything else. |

## Evidence-count → confidence-tier binding

This is enforced by the parser; non-conforming rows fail with a clear error:

| `confidence_tier` | Requires |
|---|---|
| `high` | ≥ 2 cited evidence IDs, from ≥ 2 jurisdictions (EU + US) wherever both jurisdictions have ruled. |
| `medium` | ≥ 1 cited evidence ID. |
| `low` | Any (including 0). Flag in `notes` why you couldn't cite anything. |

## Adjudication status values

| Status | When to use |
|---|---|
| `single_reviewer` | First-pass audit by one rater; no disagreement to resolve. **Most common in this worksheet.** |
| `adjudicated_consensus` | Two raters reviewed and agreed at outset. |
| `adjudicated_evidence_resolved` | Two raters disagreed; a third reviewer scored citations on the rubric (see `docs/ADJUDICATION_PROTOCOL.md`). Requires `adjudication_winning_evidence_id` + `adjudication_rubric_score`. |
| `adjudicated_direct` | Two raters disagreed; rubric tied; third reviewer adjudicated directly. |
| `unresolved_dropped` | Two raters disagreed; rubric tied; no third-reviewer call possible. **Requires `action=drop`.** |
| `pending` | Carryover not yet re-audited. **Should NOT appear in a completed worksheet.** |

## What to do

1. Open `audit_worksheet.csv` in Excel / Numbers / Google Sheets.
2. Sort by `current_hs4` if you want to review a heading at a time.
3. Fill the 11 rater columns per row. Take 5–10 min/row.
4. Save as CSV at the same path.
5. Validate with:
   ```bash
   python3 -c "from pathlib import Path; from src.audit.decisions import parse_worksheet; \\
       print(parse_worksheet(Path('data/intermediate/audit_worksheet.csv')))"
   ```
   The parser will surface every malformed row with a clear error message.
6. Hand to `scripts/apply_audit_decisions.py` to
   apply the decisions into the candidate pool, flipping records into
   the released label_source enum.

## Row counts (this worksheet)

The per-HS4 and per-source breakdown for this run is recorded in the
report JSON alongside the worksheet — open that file to see the exact
totals for the candidate pools you're auditing.
"""


def _render_readme(source: str) -> str:
    """Build the README body for a given --filter-source."""
    header = _SOURCE_HEADERS.get(source, _SOURCE_HEADERS["all"])
    return _README_TEMPLATE.format(source=source, source_header=header)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _filter_records_by_source(
    records: Sequence[Mapping[str, Any]], source: str
) -> List[Mapping[str, Any]]:
    """Filter candidate-pool records to a single source family.

    `source == "all"` passes everything through unfiltered (preserves the
    combined-worksheet behavior for backwards compatibility).
    """
    if source == "all":
        return list(records)
    return [r for r in records if _classify_source(r) == source]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        nargs="+",
        default=DEFAULT_CANDIDATE_POOLS,
        help="One or more candidate pool JSONL files. Default: the carryover "
             "pool plus the net-new BOL pool and net-new catalog pool. Missing "
             "files are skipped with a warning so this script still runs before "
             "the BOL or catalog ingest has produced a pool.",
    )
    parser.add_argument(
        "--reference-corpus",
        type=Path,
        default=REF_CORPUS_PATH,
        help="Reference corpus JSONL (default: release data).",
    )
    parser.add_argument(
        "--filter-source",
        choices=FILTER_SOURCE_VALUES,
        default="all",
        help="Restrict the worksheet to one source family: 'catalog' (net-new "
             "catalog), 'bol' (net-new BOL), 'carryover' (re-audit), or "
             "'all' (default; combined worksheet preserving today's behavior). "
             "When set to anything other than 'all', --output defaults to a "
             "source-suffixed path so the three worksheets co-exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Worksheet CSV output path. Default depends on --filter-source: "
             "see PER_SOURCE_PATHS at the top of this script.",
    )
    parser.add_argument(
        "--readme-out",
        type=Path,
        default=None,
        help="Override README output path. Default: <output dir>/<source-suffixed name>.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Override report manifest path. Default: <output dir>/<source-suffixed name>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing worksheet (will lose any in-progress edits).",
    )
    args = parser.parse_args(argv)

    # Resolve per-source default paths.
    source = args.filter_source
    defaults = PER_SOURCE_PATHS[source]
    output_path = args.output or (INTERMEDIATE / defaults["worksheet"])
    output_dir = output_path.parent
    readme_path = args.readme_out or (output_dir / defaults["readme"])
    report_path = args.report_out or (output_dir / defaults["report"])

    existing_pools = [p for p in args.candidate_pool if p.exists()]
    missing_pools = [p for p in args.candidate_pool if not p.exists()]
    for missing in missing_pools:
        print(f"WARN: candidate pool missing, skipping: {_rel(missing)}",
              file=sys.stderr)
    if not existing_pools:
        raise SystemExit(
            f"no candidate pool files found among {[_rel(p) for p in args.candidate_pool]}; "
            f"run the ingest scripts to (re)create the candidate pools "
            f"and/or scripts/ingest_catalog.py first."
        )
    if not args.reference_corpus.exists():
        raise SystemExit(
            f"reference corpus missing: {args.reference_corpus}; "
            f"run scripts/build_reference_corpus.py first."
        )
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {output_path}; pass --force "
            f"to discard in-progress rater edits."
        )

    candidates: List[Dict[str, Any]] = []
    for pool_path in existing_pools:
        print(f"reading candidate pool from {_rel(pool_path)} …")
        rows = _read_jsonl(pool_path)
        print(f"  {len(rows)} records")
        candidates.extend(rows)
    print(f"  total candidate records (all sources loaded): {len(candidates)}")

    filtered = _filter_records_by_source(candidates, source)
    if source != "all":
        print(f"  after --filter-source={source}: {len(filtered)} records")
        if not filtered:
            print(f"WARN: no records match --filter-source={source}. "
                  f"Did the corresponding ingest script run?",
                  file=sys.stderr)

    print(f"reading reference corpus from {_rel(args.reference_corpus)} …")
    corpus = _read_jsonl(args.reference_corpus)
    print(f"  {len(corpus)} entries")

    print(f"loading scope tiers from {_rel(SCOPE_PATH)} …")
    scope_lookup = _scope_tier_lookup()
    print(f"  {len(scope_lookup)} in-scope HS6 codes")

    by_hs4 = _ref_corpus_by_hs4(corpus)

    # Build rows. When source == 'all', let _build_row classify each record
    # individually so catalog rows still get their pre-fill / similarity boost
    # within the combined worksheet.
    if source == "all":
        rows: List[Dict[str, str]] = [
            _build_row(rec, by_hs4, scope_lookup) for rec in filtered
        ]
    else:
        rows = [
            _build_row(rec, by_hs4, scope_lookup, source=source) for rec in filtered
        ]

    # Sort. Catalog-only worksheet uses manufacturer-within-HS6; combined
    # and other source-only worksheets keep the historical HS4→HS6→frozen_id
    # ordering.
    rows.sort(key=_sort_key_for(source))

    # Write CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(WORKSHEET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {_rel(output_path)} ({len(rows)} rows)")

    # README — source-tailored.
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(_render_readme(source), encoding="utf-8")
    print(f"wrote {_rel(readme_path)}")

    # Manifest.
    per_hs4: Counter = Counter()
    per_source: Counter = Counter()
    per_scope: Counter = Counter()
    per_row_kind: Counter = Counter()
    rows_without_candidates: int = 0
    tier2_prefilled = 0
    mfr_hs_populated = 0
    for row in rows:
        per_hs4[row["current_hs4"]] += 1
        per_source[row["label_source"]] += 1
        per_scope[row["scope_tier"] or "unknown"] += 1
        per_row_kind[row["row_kind"]] += 1
        if not row["candidate_reference_rulings"]:
            rows_without_candidates += 1
        if row.get("tier2_classifiable"):
            tier2_prefilled += 1
        if row.get("mfr_asserted_hs"):
            mfr_hs_populated += 1
    report = {
        "release": "working",
        "filter_source": source,
        "inputs": {
            "candidate_pools": [str(_rel(p)) for p in existing_pools],
            "candidate_pools_missing": [str(_rel(p)) for p in missing_pools],
            "reference_corpus": str(_rel(args.reference_corpus)),
            "scope_config": str(_rel(SCOPE_PATH)),
        },
        "row_counts": {
            "total": len(rows),
            "per_hs4": dict(sorted(per_hs4.items())),
            "per_label_source": dict(sorted(per_source.items())),
            "per_scope_tier": dict(sorted(per_scope.items())),
            "per_row_kind": dict(sorted(per_row_kind.items())),
            "rows_with_no_candidate_rulings": rows_without_candidates,
            "tier2_classifiable_prefilled": tier2_prefilled,
            "mfr_asserted_hs_populated": mfr_hs_populated,
        },
        "outputs": {
            "worksheet": str(_rel(output_path)),
            "readme": str(_rel(readme_path)),
        },
        "downstream": (
            "Hand the worksheet to expert reviewers. After completion, run "
            "src.audit.decisions.parse_worksheet(path) to "
            "validate per the Core-4 evidence rule, then apply via "
            "src.audit.decisions.apply_corrections() to flip the candidate "
            "pool into the final release pool."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {_rel(report_path)}")

    print()
    print(f"Summary (filter-source={source}):")
    print(f"  per HS4: {dict(sorted(per_hs4.items()))}")
    print(f"  per row_kind: {dict(sorted(per_row_kind.items()))}")
    print(f"  per scope_tier: {dict(sorted(per_scope.items()))}")
    if source == "catalog":
        print(f"  tier2_classifiable pre-filled: {tier2_prefilled} / {len(rows)}")
        print(f"  mfr_asserted_hs populated: {mfr_hs_populated} / {len(rows)}")
    if rows_without_candidates:
        print(f"  ⚠ {rows_without_candidates} rows have NO candidate "
              f"reference rulings (HS4 has no entries in reference corpus)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
