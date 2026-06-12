#!/usr/bin/env python3
"""Ingest the net-new manufacturer-catalog intake CSV into the candidate pool.

Companion to ``scripts/ingest_bol.py`` (net-new BOL). This script turns
net-new catalog rows into pre-audit records.

Output is a SEPARATE pool file so the BOL and carryover pools stay byte-
identical:

* ``release/working/data/_candidate_pool.jsonl`` — carryover (untouched).
* ``release/working/data/_candidate_pool_bol.jsonl`` — net-new BOL (untouched).
* ``release/working/data/_candidate_pool_catalog.jsonl`` — net-new catalog
  survivors (THIS script's output).

Downstream, ``scripts/generate_review_worksheet.py`` accepts all three
pools via ``--candidate-pool`` and optionally slices by source via the
``--filter-source`` flag.

Filter pipeline (ordered, each step counted in the report):

  1. ``lifecycle_status`` — drop Obsolete / Discontinued /
                            Last Time Buy / Not For New Designs.
  2. ``mpn_required``     — drop rows with no manufacturer_part_number.
  3. ``short_description``— description_short empty or shorter than
                            --min-desc-chars (falls back to description_detailed).
  4. ``off_scope_hs``     — target_hs_family HS4 not in scope allow-set.
                            target_hs_family is the trusted prefix (query-design
                            controlled); hs_code is mfr-asserted and informational
                            only.
  5. ``mpn_dedupe``       — hard MPN-based dedup (case-insensitive); keep
                            best retrieval_rank. Reuses
                            ``src/collectors/catalog_collector._deduplicate_catalog_records``.
  6. ``near_duplicate``   — per-HS6 fuzzy near-dup at the 0.92 / 0.95
                            thresholds; keeper = longest description.

Skipped (BOL-only) filters: ``quantity_cap``, ``freight_forwarder``,
``generic_terms``, ``hs_verification`` — none apply to catalog data.

The script is deterministic and idempotent: re-running on the same intake
file produces byte-identical outputs.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembly.build_candidates import (  # noqa: E402
    CANDIDATE_SIZE,
    CONSTRUCTION_CHAPTER,
    CandidateSet,
    _record_seed,
    build_candidate_set,
)
from src.collectors.catalog_collector import (  # noqa: E402
    DISALLOWED_LIFECYCLE_STATUSES,
    _deduplicate_catalog_records,
    _normalize_catalog_export_payload,
)
from src.models import RawAuxiliaryRecord  # noqa: E402
from src.processing.auxiliary_enricher import _apply_catalog_variant  # noqa: E402
from src.processing.degrader import generate_tiers  # noqa: E402
from src.processing.deduplicator import (  # noqa: E402
    find_near_duplicate_groups,
    pick_keeper,
)
from src.processing.manufacturer_backfill import backfill_manufacturer_hint  # noqa: E402


WORKING_DATA = ROOT / "release" / "working" / "data"
DEFAULT_INTAKE_DIR = ROOT / "data" / "raw" / "imports" / "catalog"
DEFAULT_INTAKE_FILES: Tuple[str, ...] = ("hs_benchmark_products.csv",)
DEFAULT_POOL_OUT = WORKING_DATA / "_candidate_pool_catalog.jsonl"
DEFAULT_REPORT_OUT = WORKING_DATA / "_catalog_ingest_report.json"
DEFAULT_SCOPE = ROOT / "configs" / "hs6_scope_tiers.yaml"
DEFAULT_RULES = ROOT / "configs" / "degradation_rules.yaml"
DEFAULT_ABBREV = ROOT / "configs" / "abbreviations.csv"
DEFAULT_REF_CORPUS = WORKING_DATA / "reference_corpus.jsonl"

NEAR_DUPLICATE_LOW_THRESHOLD = 0.92
NEAR_DUPLICATE_HIGH_THRESHOLD = 0.95
PENDING_LABEL = "catalog_expert_validated_pending_reaudit"

# Filter step names — keep in sync with the report doc above.
STEP_LIFECYCLE = "lifecycle_status"
STEP_MPN_REQUIRED = "mpn_required"
STEP_SHORT_DESCRIPTION = "short_description"
STEP_OFF_SCOPE_HS = "off_scope_hs"
STEP_MPN_DEDUPE = "mpn_dedupe"
STEP_NEAR_DUPLICATE = "near_duplicate"
STEP_ORDER: Tuple[str, ...] = (
    STEP_LIFECYCLE,
    STEP_MPN_REQUIRED,
    STEP_SHORT_DESCRIPTION,
    STEP_OFF_SCOPE_HS,
    STEP_MPN_DEDUPE,
    STEP_NEAR_DUPLICATE,
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return json.loads(content)


def _load_scope(path: Path) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Return (allow_hs4, allow_hs6, hs6→scope_tier)."""
    cfg = _load_yaml_or_json(path)
    core = [str(c) for c in (cfg.get("core") or [])]
    supply = [str(c) for c in (cfg.get("supply_chain") or [])]
    allow_hs6 = set(core) | set(supply)
    if not allow_hs6:
        raise RuntimeError(
            f"hs6_scope_tiers.yaml at {path} has empty core and supply_chain — "
            "refusing to ingest."
        )
    allow_hs4 = {h6[:4] for h6 in allow_hs6}
    scope_tier: Dict[str, str] = {}
    for h6 in core:
        scope_tier[h6] = "core"
    for h6 in supply:
        scope_tier[h6] = "supply_chain"
    return allow_hs4, allow_hs6, scope_tier


def _load_reference_corpus_hs6_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            hs6 = str(entry.get("hs6_label") or "").strip()
            if hs6:
                counts[hs6] += 1
    return counts


# ---------------------------------------------------------------------------
# Intake loading
# ---------------------------------------------------------------------------

def _resolve_intake_paths(
    intake_dir: Path, intake_files: Sequence[str]
) -> List[Path]:
    resolved: List[Path] = []
    for name in intake_files:
        p = Path(name)
        if not p.is_absolute():
            p = intake_dir / name
        resolved.append(p)
    return resolved


def _load_intakes(paths: Sequence[Path]) -> List[Tuple[str, Dict[str, str]]]:
    """Yield (intake_basename, raw_row_dict) over all intake CSVs."""
    rows: List[Tuple[str, Dict[str, str]]] = []
    for path in paths:
        if not path.exists():
            print(f"WARN: intake file missing, skipping: {_rel(path)}",
                  file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                rows.append((path.name, dict(raw)))
    return rows


def _row_id(raw: Mapping[str, str]) -> str:
    """Catalog row id, preferring record_id then the supplier part number."""
    for key in ("record_id", "supplier_part_number", "manufacturer_part_number"):
        v = raw.get(key)
        if v:
            return str(v).strip()
    return ""


def _union_dedupe_by_id(
    rows: Sequence[Tuple[str, Dict[str, str]]],
) -> List[Tuple[str, Dict[str, str]]]:
    """Keep first-occurrence of each row id across all intake files."""
    seen: Set[str] = set()
    out: List[Tuple[str, Dict[str, str]]] = []
    for intake_name, raw in rows:
        rid = _row_id(raw)
        if not rid:
            rid = f"{intake_name}::row::{len(out)}"
        if rid in seen:
            continue
        seen.add(rid)
        out.append((intake_name, raw))
    return out


# ---------------------------------------------------------------------------
# Field accessors (raw-CSV level; before normalization)
# ---------------------------------------------------------------------------

def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _target_hs_family(raw: Mapping[str, str]) -> str:
    """Return the HS4 prefix derived from target_hs_family (trusted prefix);
    fall back to hs_heading, then hs_code. Never falls back to mfr-asserted
    fields silently — the off_scope filter relies on this being honest."""
    for key in ("target_hs_family", "hs_heading", "hs_code"):
        v = raw.get(key)
        if v:
            return _digits(v)[:4]
    return ""


def _mfr_asserted_hs6(raw: Mapping[str, str]) -> str:
    """Manufacturer-asserted HS6 from hs_code (HTSUS). Informational only —
    expert audit determines final hs6_label."""
    return _digits(raw.get("hs_code") or "")[:6]


def _mfr_asserted_hs_hint(raw: Mapping[str, str]) -> str:
    """Full manufacturer-asserted HTSUS string for the worksheet hint
    column (e.g. '8541.10.0080'). Preserved verbatim for expert reference."""
    return str(raw.get("hs_code") or "").strip()


def _description(raw: Mapping[str, str]) -> str:
    """description_short with description_detailed fallback."""
    for key in ("description_short", "description_detailed"):
        v = raw.get(key)
        if v:
            return str(v).strip()
    return ""


def _lifecycle_disallowed(raw: Mapping[str, str]) -> Optional[str]:
    """Return the offending status string if the row should be dropped."""
    v = (raw.get("lifecycle_status") or "").strip().lower()
    if v in DISALLOWED_LIFECYCLE_STATUSES:
        return v
    # Additional lifecycle statuses excluded at intake but not in the
    # catalog_collector DISALLOWED set.
    for token in ("last time buy", "not for new designs"):
        if token in v:
            return v
    return ""  # empty string means allowed (truthy check distinguishes from None)


def _has_mpn(raw: Mapping[str, str]) -> bool:
    return bool((raw.get("manufacturer_part_number") or "").strip())


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

def _apply_filters(
    rows: Sequence[Tuple[str, Dict[str, str]]],
    *,
    allow_hs4: Set[str],
    min_desc_chars: int,
    near_dedup: bool = True,
) -> Tuple[
    List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]],
    Dict[str, Dict[str, int]],
    List[Dict[str, Any]],
    Dict[str, int],
]:
    """Return (survivors, step_counts, dropped_records, mismatch_hint_count).

    ``survivors`` is a list of (intake_basename, raw_row, RawAuxiliaryRecord).
    ``mismatch_hint_count`` reports rows whose hs_code HS4 disagrees with
    target_hs_family — informational only, not used for filtering.
    """
    step_counts: Dict[str, Dict[str, int]] = {
        name: {"in": 0, "dropped": 0, "out": 0} for name in STEP_ORDER
    }
    dropped: List[Dict[str, Any]] = []
    mismatch_count = 0

    def _drop(intake: str, raw: Mapping[str, str], step: str, reason: str) -> None:
        dropped.append({
            "raw_id": _row_id(raw),
            "target_hs_family": _target_hs_family(raw),
            "mfr_asserted_hs": _mfr_asserted_hs_hint(raw),
            "intake_file": intake,
            "disposition": step,
            "reason": reason,
        })

    # Linear steps 1..4 (cheap; pre-normalization).
    pre_normalized: List[Tuple[str, Dict[str, str]]] = []
    for intake, raw in rows:
        # Step 1: lifecycle_status
        step_counts[STEP_LIFECYCLE]["in"] += 1
        offending = _lifecycle_disallowed(raw)
        if offending:
            step_counts[STEP_LIFECYCLE]["dropped"] += 1
            _drop(intake, raw, STEP_LIFECYCLE,
                  f"lifecycle_status {offending!r} in disallowed set")
            continue
        step_counts[STEP_LIFECYCLE]["out"] += 1

        # Step 2: mpn_required
        step_counts[STEP_MPN_REQUIRED]["in"] += 1
        if not _has_mpn(raw):
            step_counts[STEP_MPN_REQUIRED]["dropped"] += 1
            _drop(intake, raw, STEP_MPN_REQUIRED,
                  "manufacturer_part_number is empty")
            continue
        step_counts[STEP_MPN_REQUIRED]["out"] += 1

        # Step 3: short_description
        step_counts[STEP_SHORT_DESCRIPTION]["in"] += 1
        desc = _description(raw)
        if not desc or len(desc) < min_desc_chars:
            step_counts[STEP_SHORT_DESCRIPTION]["dropped"] += 1
            _drop(intake, raw, STEP_SHORT_DESCRIPTION,
                  f"description length {len(desc)} < {min_desc_chars}")
            continue
        step_counts[STEP_SHORT_DESCRIPTION]["out"] += 1

        # Step 4: off_scope_hs (uses target_hs_family — trusted prefix)
        step_counts[STEP_OFF_SCOPE_HS]["in"] += 1
        hs4 = _target_hs_family(raw)
        if not hs4 or hs4 not in allow_hs4:
            step_counts[STEP_OFF_SCOPE_HS]["dropped"] += 1
            _drop(intake, raw, STEP_OFF_SCOPE_HS,
                  f"target_hs_family HS4 {hs4!r} not in scope allow-set")
            continue
        step_counts[STEP_OFF_SCOPE_HS]["out"] += 1

        # Track hs_code vs target_hs_family disagreement (informational).
        mfr_hs4 = _digits(raw.get("hs_code") or "")[:4]
        if mfr_hs4 and mfr_hs4 != hs4:
            mismatch_count += 1

        pre_normalized.append((intake, raw))

    # Normalize survivors into RawAuxiliaryRecord (parses key_specs JSON,
    # category_path, classifications, etc.). normalize_catalog_export_payload is
    # what catalog_collector uses internally.
    normalized: List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]] = []
    for intake, raw in pre_normalized:
        aux = _normalize_catalog_export_payload(raw)
        normalized.append((intake, raw, aux))

    # Step 5: mpn_dedupe (hard dedup on MPN, retrieval_rank tie-break).
    # Reuses catalog_collector._deduplicate_catalog_records on the
    # RawAuxiliaryRecord list, then we map back to (intake, raw, aux)
    # tuples using the MPN as the join key.
    step_counts[STEP_MPN_DEDUPE]["in"] = len(normalized)
    if normalized:
        # Build aux→bucket map. Keep first occurrence of each MPN as the
        # canonical entry; record the rest as dropped.
        by_mpn: Dict[str, List[int]] = {}
        for i, (_, _, aux) in enumerate(normalized):
            key = (aux.mpn or "").lower()
            by_mpn.setdefault(key, []).append(i)

        # Use _deduplicate_catalog_records to pick the keeper indices.
        # We call it on the aux list then re-derive the kept indices by
        # matching reference back. _deduplicate's tie-break sorts by
        # retrieval_rank ASC, longer description, longer mpn, reference;
        # we replicate by calling it directly on a per-MPN slice.
        keep_indices: Set[int] = set()
        for mpn_key, idxs in by_mpn.items():
            if len(idxs) == 1:
                keep_indices.add(idxs[0])
                continue
            sub = [normalized[i][2] for i in idxs]
            kept = _deduplicate_catalog_records(sub)
            if not kept:
                continue
            kept_ref = kept[0].reference
            for i in idxs:
                if normalized[i][2].reference == kept_ref:
                    keep_indices.add(i)
                    break

        post_mpn: List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]] = []
        for i, (intake, raw, aux) in enumerate(normalized):
            if i in keep_indices:
                step_counts[STEP_MPN_DEDUPE]["out"] += 1
                post_mpn.append((intake, raw, aux))
            else:
                step_counts[STEP_MPN_DEDUPE]["dropped"] += 1
                _drop(intake, raw, STEP_MPN_DEDUPE,
                      f"duplicate MPN {aux.mpn!r} (kept higher-rank twin)")
    else:
        post_mpn = []

    # Step 6: near_duplicate (per-HS6 fuzzy on description).
    # When near_dedup is False, every MPN-distinct survivor is kept (used to
    # expose near-duplicate-description rows for a downstream audit worksheet);
    # the rows are flagged elsewhere so a reviewer can drop trivial twins.
    survivors: List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]] = []
    step_counts[STEP_NEAR_DUPLICATE]["in"] = len(post_mpn)
    if post_mpn and not near_dedup:
        step_counts[STEP_NEAR_DUPLICATE]["out"] = len(post_mpn)
        survivors = list(post_mpn)
    elif post_mpn:
        dedup_input = [
            {
                "hs6_label": _mfr_asserted_hs6(raw) or (_target_hs_family(raw) + "00"),
                "tier1_description": aux.description,
            }
            for _, raw, aux in post_mpn
        ]
        groups = find_near_duplicate_groups(
            dedup_input,
            high_threshold=NEAR_DUPLICATE_HIGH_THRESHOLD,
            low_threshold=NEAR_DUPLICATE_LOW_THRESHOLD,
        )
        drop_indices: Set[int] = set()
        for group in groups:
            members = [dedup_input[i] for i in group]
            keeper_local = pick_keeper(members)
            keeper_global = group[keeper_local]
            for idx in group:
                if idx == keeper_global:
                    continue
                drop_indices.add(idx)

        for i, (intake, raw, aux) in enumerate(post_mpn):
            if i in drop_indices:
                step_counts[STEP_NEAR_DUPLICATE]["dropped"] += 1
                hs6_for_msg = _mfr_asserted_hs6(raw) or _target_hs_family(raw)
                _drop(intake, raw, STEP_NEAR_DUPLICATE,
                      f"near-duplicate of another survivor in HS6 {hs6_for_msg!r}")
                continue
            step_counts[STEP_NEAR_DUPLICATE]["out"] += 1
            survivors.append((intake, raw, aux))

    mismatch_hint = {"target_vs_hs_code_hs4_mismatch": mismatch_count}
    return survivors, step_counts, dropped, mismatch_hint


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _canonical_dict(
    aux: RawAuxiliaryRecord,
    raw: Mapping[str, str],
    ordinal: int,
) -> Dict[str, Any]:
    """Build the canonical-product-shaped dict consumed by
    canonical_to_benchmark_record + generate_tiers."""
    # Use mfr-asserted HS6 as the initial hs6_label (best hint we have);
    # the expert audit will confirm/change during Core-4 review.
    hs6 = _mfr_asserted_hs6(raw)
    if not hs6:
        # Fall back to target_hs_family — pad to 6 digits with "00" so
        # downstream HS4 / HS2 derivations work. Expert MUST relabel.
        hs6 = (_target_hs_family(raw) + "00")[:6]

    canonical_id = f"CP-catalog-{ordinal:05d}"
    return {
        "canonical_id": canonical_id,
        "canonical_description": aux.description,
        "primary_source": "catalog",
        "primary_reference": aux.reference or _row_id(raw),
        "hs6_label": hs6,
        "hs4_label": hs6[:4],
        "hs2_label": hs6[:2],
        # MUST be "catalog_classified" so the degrader sets the minimal-tier
        # tier2_provenance (natural_mpn / degraded_manual) automatically.
        "label_source": "catalog_classified",
        "keywords": [],
        "justification_text": "",
        "manufacturer_hint": aux.manufacturer,
        "manufacturer_hint_source": "catalog_csv" if aux.manufacturer else "",
        "sampling_metadata": {},
        "source_evidence": [],
        "merge_confidence": 1.0,
    }


def _build_records(
    survivors: Sequence[Tuple[str, Dict[str, str], RawAuxiliaryRecord]],
    *,
    allow_hs6: Set[str],
    scope_tier: Mapping[str, str],
    rules_path: Path,
    abbrev_path: Path,
) -> List[Dict[str, Any]]:
    """Run the degrader + catalog-variant overlay over all survivors and
    return pre-audit records (ordinals provisional; reassigned
    after sort)."""
    if not survivors:
        return []

    canonicals: List[Dict[str, Any]] = []
    aux_by_canonical: Dict[str, RawAuxiliaryRecord] = {}
    raw_by_canonical: Dict[str, Dict[str, str]] = {}
    intake_by_canonical: Dict[str, str] = {}
    for i, (intake, raw, aux) in enumerate(survivors):
        canon = _canonical_dict(aux, raw, ordinal=i)
        canonicals.append(canon)
        aux_by_canonical[canon["canonical_id"]] = aux
        raw_by_canonical[canon["canonical_id"]] = raw
        intake_by_canonical[canon["canonical_id"]] = intake

    # generate_tiers does canonical_to_benchmark_record + rule-based
    # tier2_minimal. Because label_source=="catalog_classified" it sets the
    # minimal-tier tier2_provenance (natural_mpn / degraded_manual).
    result = generate_tiers(canonicals, str(abbrev_path), str(rules_path))
    records: List[Dict[str, Any]] = result["records"]

    # Overlay the natural-catalog variant: refreshes tier2_minimal to the
    # MPN-based form from the catalog description + category + specs. Mirrors
    # what enrich_records_with_auxiliary would do but unconditionally (every
    # record here IS a catalog record).
    out: List[Dict[str, Any]] = []
    for rec in records:
        canon_id = rec["canonical_id"]
        aux = aux_by_canonical[canon_id]
        raw = raw_by_canonical[canon_id]
        intake = intake_by_canonical[canon_id]

        rec["auxiliary_match_metadata"] = {}
        _apply_catalog_variant(rec, aux, score=1.0)

        # Carryover-pool convention for pre-audit fields.
        rec["label_source"] = PENDING_LABEL
        rec["tier1_source"] = "catalog"
        rec["confidence_tier"] = "pending"
        rec["cited_evidence_ids"] = []
        rec["adjudication_status"] = "pending"
        rec["difficulty_tags"] = rec.get("difficulty_tags") or []
        rec["scope_tier"] = scope_tier.get(rec.get("hs6_label", ""), "")
        rec["split"] = "dev"  # provisional

        # source_metadata: keep what canonical_to_benchmark_record set, add
        # catalog-specific provenance keys. record_schema.json does not
        # constrain source_metadata's shape, so adding keys is safe.
        sm = dict(rec.get("source_metadata") or {})
        sm["primary_source"] = "catalog"
        sm["primary_reference"] = aux.reference or _row_id(raw)
        sm["catalog_intake_file"] = intake
        sm["catalog_target_hs_family"] = _target_hs_family(raw)
        sm["catalog_data_source"] = "manufacturer_catalog"
        sm["catalog_hs_hint"] = _mfr_asserted_hs_hint(raw)
        sm["catalog_part_number"] = str(raw.get("supplier_part_number") or "").strip()
        sm["catalog_lifecycle_status"] = str(raw.get("lifecycle_status") or "").strip()
        # Ensure manufacturer_hint is preserved (degrader may have cleared it).
        if aux.manufacturer and not sm.get("manufacturer_hint"):
            sm["manufacturer_hint"] = aux.manufacturer
            sm["manufacturer_hint_source"] = "catalog_csv"
        rec["source_metadata"] = sm

        # Drop transient bookkeeping that's not part of the released record.
        rec.pop("degradation_metadata", None)
        rec.pop("auxiliary_match_metadata", None)

        out.append(rec)

    # Backfill manufacturer_hint from tier2_minimal.manufacturer where
    # source_metadata.manufacturer_hint is empty. Mirrors the BOL flow.
    backfill = backfill_manufacturer_hint(out)
    out = backfill.records

    return out


def _finalize_ids_and_candidates(
    records: List[Dict[str, Any]],
    *,
    allow_hs6: Set[str],
) -> List[Dict[str, Any]]:
    """Sort records, reassign ordinals so output is byte-identical on rerun,
    then attach candidate_set."""
    records.sort(key=lambda r: (
        str(r.get("hs6_label") or ""),
        str((r.get("source_metadata") or {}).get("canonical_id") or ""),
    ))

    finalized: List[Dict[str, Any]] = []
    for ordinal, rec in enumerate(records, start=1):
        new_rec = dict(rec)
        new_rec["id"] = f"SH-catalog-{ordinal:05d}"
        new_rec["frozen_id"] = f"v2.0.dev.{ordinal:04d}"
        new_rec["source_reference"] = str(new_rec.get("source_reference") or "")
        finalized.append(new_rec)

    in_scope_list = sorted(allow_hs6)
    attached: List[Dict[str, Any]] = []
    for rec in finalized:
        slate = _build_candidate_set_with_fallback(rec, in_scope_list)
        new_rec = dict(rec)
        new_rec["candidate_set"] = slate.to_dict()
        attached.append(new_rec)
    return attached


def _build_candidate_set_with_fallback(
    record: Mapping[str, Any], in_scope_list: Sequence[str]
) -> CandidateSet:
    """Return a 4-code candidate slate, falling back to cross-chapter
    in-scope codes when the gold HS6's chapter is too sparse to fill the
    slate. Mirrors ingest_bol._build_candidate_set_with_fallback."""
    import random

    try:
        return build_candidate_set(record, in_scope_hs6=in_scope_list)
    except ValueError:
        pass

    gold_hs6 = str(record["hs6_label"])
    rng = random.Random(_record_seed(str(record["id"])))
    pool = [code for code in in_scope_list if code != gold_hs6]
    same_hs4 = sorted(c for c in pool if c[:4] == gold_hs6[:4])
    same_hs2 = sorted(c for c in pool if c[:2] == gold_hs6[:2] and c not in same_hs4)
    rest = sorted(c for c in pool if c not in same_hs4 and c not in same_hs2)
    rng.shuffle(same_hs4)
    rng.shuffle(same_hs2)
    rng.shuffle(rest)
    distractors: List[str] = []
    for c in same_hs4 + same_hs2 + rest:
        if len(distractors) >= CANDIDATE_SIZE - 1:
            break
        distractors.append(c)
    if len(distractors) < CANDIDATE_SIZE - 1:
        raise ValueError(
            f"record {record.get('id')!r}: even after cross-chapter fallback, "
            f"only {len(distractors)} distractors available from in-scope pool "
            f"of size {len(in_scope_list)}"
        )
    slate = [gold_hs6] + distractors
    rng.shuffle(slate)
    return CandidateSet(
        codes=tuple(slate),
        construction=CONSTRUCTION_CHAPTER,
        gold_rank_in_candidates=slate.index(gold_hs6),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _build_yield_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_hs4: Counter = Counter()
    per_hs6: Counter = Counter()
    per_scope: Counter = Counter()
    by_data_source: Counter = Counter()
    per_target_family: Counter = Counter()
    for rec in records:
        per_hs4[str(rec.get("hs4_label") or "")] += 1
        per_hs6[str(rec.get("hs6_label") or "")] += 1
        per_scope[str(rec.get("scope_tier") or "unknown")] += 1
        sm = rec.get("source_metadata") or {}
        by_data_source[str(sm.get("catalog_data_source") or "unknown")] += 1
        per_target_family[str(sm.get("catalog_target_hs_family") or "unknown")] += 1
    return {
        "total_survivors": len(records),
        "per_hs4": dict(sorted(per_hs4.items())),
        "per_hs6": dict(sorted(per_hs6.items())),
        "per_scope_tier": dict(sorted(per_scope.items())),
        "by_data_source": dict(sorted(by_data_source.items())),
        "per_target_hs_family": dict(sorted(per_target_family.items())),
    }


def _build_manufacturer_coverage(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with_hint = 0
    without_hint = 0
    by_source: Counter = Counter()
    top_vendors: Counter = Counter()
    for rec in records:
        sm = rec.get("source_metadata") or {}
        hint = str(sm.get("manufacturer_hint") or "").strip()
        if hint:
            with_hint += 1
            by_source[str(sm.get("manufacturer_hint_source") or "unknown")] += 1
            top_vendors[hint.lower()] += 1
        else:
            without_hint += 1
    return {
        "with_hint": with_hint,
        "without_hint": without_hint,
        "by_source": dict(sorted(by_source.items())),
        "top_15_vendors": dict(top_vendors.most_common(15)),
    }


def _build_reference_corpus_check(
    records: Sequence[Mapping[str, Any]],
    ref_counts: Counter,
) -> Dict[str, List[str]]:
    survivor_hs6 = {str(r.get("hs6_label") or "") for r in records}
    survivor_hs6.discard("")
    zero: List[str] = sorted(h6 for h6 in survivor_hs6 if ref_counts.get(h6, 0) == 0)
    below_floor: List[str] = sorted(
        h6 for h6 in survivor_hs6 if 0 < ref_counts.get(h6, 0) < 3
    )
    return {
        "hs6_with_zero_reference_rulings": zero,
        "hs6_below_floor_3": below_floor,
    }


def _build_report(
    *,
    intake_dir: Path,
    intake_paths: Sequence[Path],
    scope_path: Path,
    min_desc_chars: int,
    step_counts: Mapping[str, Mapping[str, int]],
    dropped: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    ref_counts: Counter,
    mismatch_hint: Mapping[str, int],
) -> Dict[str, Any]:
    samples = [copy.deepcopy(r) for r in records[:5]]
    return {
        "release": "working",
        "source": "catalog net-new",
        "inputs": {
            "intake_dir": _rel(intake_dir),
            "intake_files": [_rel(p) for p in intake_paths],
            "scope_config": _rel(scope_path),
        },
        "policy": {
            "min_desc_chars": min_desc_chars,
            "disallowed_lifecycle": sorted(DISALLOWED_LIFECYCLE_STATUSES) + [
                "last time buy",
                "not for new designs",
            ],
            "near_duplicate_low_threshold": NEAR_DUPLICATE_LOW_THRESHOLD,
            "near_duplicate_high_threshold": NEAR_DUPLICATE_HIGH_THRESHOLD,
            "label_source_pre_audit": PENDING_LABEL,
            "hs_prefix_source": "target_hs_family (trusted); hs_code is mfr-asserted hint only",
        },
        "filter_pipeline": {
            "steps": {name: dict(step_counts[name]) for name in STEP_ORDER},
        },
        "informational_warnings": dict(mismatch_hint),
        "yield": _build_yield_summary(records),
        "manufacturer_hint_coverage": _build_manufacturer_coverage(records),
        "reference_corpus_coverage_check": _build_reference_corpus_check(
            records, ref_counts
        ),
        "samples": samples,
        "dropped_records": sorted(
            (dict(d) for d in dropped),
            key=lambda d: (str(d.get("disposition") or ""),
                           str(d.get("target_hs_family") or ""),
                           str(d.get("raw_id") or "")),
        ),
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intake-dir",
        type=Path,
        default=DEFAULT_INTAKE_DIR,
        help=f"Catalog intake directory (default: {_rel(DEFAULT_INTAKE_DIR)}).",
    )
    parser.add_argument(
        "--intake-files",
        nargs="+",
        default=list(DEFAULT_INTAKE_FILES),
        help="Intake CSV filenames or absolute paths. Relative names are "
             "joined to --intake-dir.",
    )
    parser.add_argument(
        "--candidate-pool-out",
        type=Path,
        default=DEFAULT_POOL_OUT,
        help=f"Output JSONL (default: {_rel(DEFAULT_POOL_OUT)}).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_REPORT_OUT,
        help=f"Output report JSON (default: {_rel(DEFAULT_REPORT_OUT)}).",
    )
    parser.add_argument(
        "--scope-config",
        type=Path,
        default=DEFAULT_SCOPE,
        help=f"HS6 scope config (default: {_rel(DEFAULT_SCOPE)}).",
    )
    parser.add_argument(
        "--degradation-rules",
        type=Path,
        default=DEFAULT_RULES,
        help=f"Degradation rules (default: {_rel(DEFAULT_RULES)}).",
    )
    parser.add_argument(
        "--abbreviations",
        type=Path,
        default=DEFAULT_ABBREV,
        help=f"Abbreviation table (default: {_rel(DEFAULT_ABBREV)}).",
    )
    parser.add_argument(
        "--reference-corpus",
        type=Path,
        default=DEFAULT_REF_CORPUS,
        help="Reference corpus JSONL (read-only; used only to populate the "
             "report's reference_corpus_coverage_check section).",
    )
    parser.add_argument("--min-desc-chars", type=int, default=20)
    parser.add_argument("--no-near-dedup", action="store_true",
                        help="Skip the per-HS6 near-duplicate-description filter, "
                             "keeping every MPN-distinct survivor (exposes near-dup "
                             "rows for a downstream audit worksheet).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute counters only; do not write outputs.")
    args = parser.parse_args(argv)

    allow_hs4, allow_hs6, scope_tier = _load_scope(args.scope_config)
    ref_counts = _load_reference_corpus_hs6_counts(args.reference_corpus)

    intake_paths = _resolve_intake_paths(args.intake_dir, args.intake_files)
    raw_rows = _load_intakes(intake_paths)
    print(f"loaded {len(raw_rows)} raw rows from "
          f"{sum(1 for p in intake_paths if p.exists())} intake file(s)")

    deduped = _union_dedupe_by_id(raw_rows)
    print(f"  after union+dedupe-by-id: {len(deduped)} unique rows")

    survivors, step_counts, dropped, mismatch_hint = _apply_filters(
        deduped,
        allow_hs4=allow_hs4,
        min_desc_chars=args.min_desc_chars,
        near_dedup=not args.no_near_dedup,
    )
    print(f"  after filter pipeline: {len(survivors)} survivors")
    for step in STEP_ORDER:
        c = step_counts[step]
        print(f"    {step:24s} in={c['in']:6d} dropped={c['dropped']:6d} out={c['out']:6d}")
    if mismatch_hint.get("target_vs_hs_code_hs4_mismatch"):
        print(f"  ℹ {mismatch_hint['target_vs_hs_code_hs4_mismatch']} rows have "
              f"hs_code HS4 ≠ target_hs_family HS4 (informational; not filtered)")

    if not survivors:
        print("no survivors — emitting empty pool and report.")

    records = _build_records(
        survivors,
        allow_hs6=allow_hs6,
        scope_tier=scope_tier,
        rules_path=args.degradation_rules,
        abbrev_path=args.abbreviations,
    )
    records = _finalize_ids_and_candidates(records, allow_hs6=allow_hs6)

    report = _build_report(
        intake_dir=args.intake_dir,
        intake_paths=intake_paths,
        scope_path=args.scope_config,
        min_desc_chars=args.min_desc_chars,
        step_counts=step_counts,
        dropped=dropped,
        records=records,
        ref_counts=ref_counts,
        mismatch_hint=mismatch_hint,
    )

    if args.dry_run:
        print()
        print("DRY RUN — not writing outputs.")
        print(f"  would write {len(records)} records to {_rel(args.candidate_pool_out)}")
        print(f"  would write report to {_rel(args.report_out)}")
        return 0

    _write_jsonl(args.candidate_pool_out, records)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"wrote {_rel(args.candidate_pool_out)} ({len(records)} records)")
    print(f"wrote {_rel(args.report_out)}")

    rc_check = report["reference_corpus_coverage_check"]
    if rc_check["hs6_with_zero_reference_rulings"]:
        print(f"  ⚠ {len(rc_check['hs6_with_zero_reference_rulings'])} HS6(s) "
              f"have ZERO reference rulings: "
              f"{rc_check['hs6_with_zero_reference_rulings']}")
    if rc_check["hs6_below_floor_3"]:
        print(f"  ⚠ {len(rc_check['hs6_below_floor_3'])} HS6(s) "
              f"have <3 reference rulings: "
              f"{rc_check['hs6_below_floor_3']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
