#!/usr/bin/env python3
"""Ingest net-new BOL intake CSVs into the candidate pool.

Companion to ``scripts/ingest_catalog.py`` (net-new catalog). This
script handles **net-new** BOL records and turns them into
pre-audit records.

Output goes to a SEPARATE pool file so the carryover pool is preserved
byte-identical:

* ``release/working/data/_candidate_pool.jsonl`` — carryover (untouched).
* ``release/working/data/_candidate_pool_bol.jsonl`` — net-new BOL survivors
  (THIS script's output).

Downstream, ``scripts/generate_review_worksheet.py`` and
``scripts/apply_audit_decisions.py`` both accept ``--candidate-pool``
as a multi-value option, so the combined audit pass reads both files.

Filter pipeline (ordered, each step counted in the report):

  1. ``short_description``  — desc empty or shorter than --min-desc-chars
  2. ``quantity_cap``       — quantity > --quantity-max
  3. ``freight_forwarder``  — consignee matches inline denylist
  4. ``off_scope_hs``       — declared HS4 not in scope allow-set
  5. ``generic_terms``      — description trips the generic-terms blocklist
                              (auxiliary_enricher.DEFAULT_GENERIC_BOL_TERMS)
                              or has <3 substantive words
  6. ``hs_verification``    — declared_hs HS4 prefix mismatch (sanity check;
                              redundant after step 4 but catches malformed
                              codes like ``"3818"`` vs ``"381800"``)
  7. ``near_duplicate``     — per-HS6 near-dup grouping at the 0.92 / 0.95
                              thresholds; keeper = longest description.

The script is deterministic and idempotent: re-running on the same intake
files produces byte-identical outputs (sort by HS6 + canonical_id, ordinals
reassigned post-sort, json.dumps with sort_keys=True).

"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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
from src.collectors.bol_collector import (  # noqa: E402
    is_specific_bol_record,
    normalize_bol_payload,
)
from src.models import RawAuxiliaryRecord  # noqa: E402
from src.processing.auxiliary_enricher import (  # noqa: E402
    DEFAULT_GENERIC_BOL_TERMS,
    _apply_bol_variant,
    _bol_hs_verified,
)
from src.processing.degrader import canonical_to_benchmark_record, generate_tiers  # noqa: E402
from src.processing.deduplicator import find_near_duplicate_groups, pick_keeper  # noqa: E402
from src.processing.manufacturer_backfill import backfill_manufacturer_hint  # noqa: E402


WORKING_DATA = ROOT / "release" / "working" / "data"
DEFAULT_INTAKE_DIR = ROOT / "data" / "raw" / "imports" / "bol"
DEFAULT_INTAKE_FILES = (
    "bol_intake.csv",
    "bol_intake_tier2.csv",
    "bol_intake_tier3.csv",
)
DEFAULT_POOL_OUT = WORKING_DATA / "_candidate_pool_bol.jsonl"
DEFAULT_REPORT_OUT = WORKING_DATA / "_bol_ingest_report.json"
DEFAULT_SCOPE = ROOT / "configs" / "hs6_scope_tiers.yaml"
DEFAULT_RULES = ROOT / "configs" / "degradation_rules.yaml"
DEFAULT_ABBREV = ROOT / "configs" / "abbreviations.csv"
DEFAULT_REF_CORPUS = WORKING_DATA / "reference_corpus.jsonl"

# Freight-forwarder denylist (names, not goods). Case-insensitive
# substring match against the consignee field — a forwarder consignee means
# the BOL doesn't identify the actual buyer, so the record is unusable for
# manufacturer-grounded labelling.
FREIGHT_FORWARDER_DENYLIST: Tuple[str, ...] = (
    "DHL",
    "Kuehne+Nagel",
    "Kuehne + Nagel",
    "Expeditors",
    "DB Schenker",
    "Yusen",
    "Nippon Express",
    "CEVA",
    "DSV",
    "Geodis",
    "Kintetsu",
    "Sankyu",
    "Nittsu",
    "Hellmann",
    "Maersk Logistics",
    "Bolloré",
    "Bollore",
    "Panalpina",
    "UPS Supply Chain",
    "FedEx Trade Networks",
)

NEAR_DUPLICATE_LOW_THRESHOLD = 0.92
NEAR_DUPLICATE_HIGH_THRESHOLD = 0.95
PENDING_LABEL = "BOL_expert_validated_pending_reaudit"

# Filter step names — keep in sync with report doc above.
STEP_SHORT_DESCRIPTION = "short_description"
STEP_QUANTITY_CAP = "quantity_cap"
STEP_FREIGHT_FORWARDER = "freight_forwarder"
STEP_OFF_SCOPE_HS = "off_scope_hs"
STEP_GENERIC_TERMS = "generic_terms"
STEP_HS_VERIFICATION = "hs_verification"
STEP_NEAR_DUPLICATE = "near_duplicate"
STEP_ORDER: Tuple[str, ...] = (
    STEP_SHORT_DESCRIPTION,
    STEP_QUANTITY_CAP,
    STEP_FREIGHT_FORWARDER,
    STEP_OFF_SCOPE_HS,
    STEP_GENERIC_TERMS,
    STEP_HS_VERIFICATION,
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
    """Yield (intake_basename, raw_row_dict) over all intake CSVs.

    Missing files are warned and skipped so a partial pull still ingests.
    """
    rows: List[Tuple[str, Dict[str, str]]] = []
    for path in paths:
        if not path.exists():
            print(f"WARN: intake file missing, skipping: {_rel(path)}",
                  file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                rows.append((path.name, dict(raw)))
    return rows


def _row_id(raw: Mapping[str, str]) -> str:
    """BOL-provider row id, preferring the CSV's ``id`` column then aliases."""
    for key in ("id", "record_id", "reference", "master_bill_no", "sub_bill_no"):
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
            # Synthesize a deterministic id so we don't drop rows with no id.
            rid = f"{intake_name}::row::{len(out)}"
        if rid in seen:
            continue
        seen.add(rid)
        out.append((intake_name, raw))
    return out


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

def _declared_hs(raw: Mapping[str, str]) -> str:
    for key in ("declared_hs", "hs_code"):
        v = raw.get(key)
        if v:
            return re.sub(r"\D", "", str(v))
    return ""


def _quantity_int(raw: Mapping[str, str]) -> Optional[int]:
    v = raw.get("quantity")
    if v in (None, ""):
        return None
    cleaned = re.sub(r"[^\d.-]", "", str(v))
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _consignee(raw: Mapping[str, str]) -> str:
    for key in ("consignee", "buyer_t", "buyer_name"):
        v = raw.get(key)
        if v:
            return str(v).strip()
    return ""


def _description(raw: Mapping[str, str]) -> str:
    """Raw (uncleaned) description string, before normalize_bol_payload."""
    for key in ("description", "bol_description", "prod_desc"):
        v = raw.get(key)
        if v:
            return str(v).strip()
    return ""


def _is_freight_forwarder(consignee: str) -> Optional[str]:
    if not consignee:
        return None
    lowered = consignee.lower()
    for token in FREIGHT_FORWARDER_DENYLIST:
        if token.lower() in lowered:
            return token
    return None


def _apply_filters(
    rows: Sequence[Tuple[str, Dict[str, str]]],
    *,
    allow_hs4: Set[str],
    generic_terms: Sequence[str],
    min_desc_chars: int,
    quantity_max: int,
) -> Tuple[
    List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]],
    Dict[str, Dict[str, int]],
    List[Dict[str, Any]],
]:
    """Return (survivors, step_counts, dropped_records).

    ``survivors`` is a list of (intake_basename, raw_row, RawAuxiliaryRecord).
    """
    step_counts: Dict[str, Dict[str, int]] = {
        name: {"in": 0, "dropped": 0, "out": 0} for name in STEP_ORDER
    }
    dropped: List[Dict[str, Any]] = []

    def _drop(intake: str, raw: Mapping[str, str], step: str, reason: str) -> None:
        dropped.append({
            "raw_id": _row_id(raw),
            "declared_hs": _declared_hs(raw),
            "intake_file": intake,
            "disposition": step,
            "reason": reason,
        })

    # ----- linear steps 1..6 -----
    intermediate: List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]] = []
    for intake, raw in rows:
        # Step 1: short_description
        step_counts[STEP_SHORT_DESCRIPTION]["in"] += 1
        desc_raw = _description(raw)
        if not desc_raw or len(desc_raw) < min_desc_chars:
            step_counts[STEP_SHORT_DESCRIPTION]["dropped"] += 1
            _drop(intake, raw, STEP_SHORT_DESCRIPTION,
                  f"description length {len(desc_raw)} < {min_desc_chars}")
            continue
        step_counts[STEP_SHORT_DESCRIPTION]["out"] += 1

        # Step 2: quantity_cap
        step_counts[STEP_QUANTITY_CAP]["in"] += 1
        qty = _quantity_int(raw)
        if qty is not None and qty > quantity_max:
            step_counts[STEP_QUANTITY_CAP]["dropped"] += 1
            _drop(intake, raw, STEP_QUANTITY_CAP,
                  f"quantity {qty} > {quantity_max}")
            continue
        step_counts[STEP_QUANTITY_CAP]["out"] += 1

        # Step 3: freight_forwarder
        step_counts[STEP_FREIGHT_FORWARDER]["in"] += 1
        consignee = _consignee(raw)
        ff_hit = _is_freight_forwarder(consignee)
        if ff_hit is not None:
            step_counts[STEP_FREIGHT_FORWARDER]["dropped"] += 1
            _drop(intake, raw, STEP_FREIGHT_FORWARDER,
                  f"consignee matches forwarder token {ff_hit!r}")
            continue
        step_counts[STEP_FREIGHT_FORWARDER]["out"] += 1

        # Step 4: off_scope_hs
        step_counts[STEP_OFF_SCOPE_HS]["in"] += 1
        hs = _declared_hs(raw)
        hs4 = hs[:4]
        if not hs4 or hs4 not in allow_hs4:
            step_counts[STEP_OFF_SCOPE_HS]["dropped"] += 1
            _drop(intake, raw, STEP_OFF_SCOPE_HS,
                  f"declared HS4 {hs4!r} not in scope allow-set")
            continue
        step_counts[STEP_OFF_SCOPE_HS]["out"] += 1

        # Normalize into RawAuxiliaryRecord for the next two steps (and
        # downstream BOL variant application). normalize_bol_payload cleans
        # the description and reads through the alias chain.
        aux = normalize_bol_payload(raw)

        # Step 5: generic_terms (uses cleaned description from aux).
        step_counts[STEP_GENERIC_TERMS]["in"] += 1
        if not is_specific_bol_record(aux, list(generic_terms)):
            step_counts[STEP_GENERIC_TERMS]["dropped"] += 1
            _drop(intake, raw, STEP_GENERIC_TERMS,
                  "description matches generic-terms blocklist or <3 substantive words")
            continue
        step_counts[STEP_GENERIC_TERMS]["out"] += 1

        # Step 6: hs_verification — sanity check that declared_hs HS4 prefix
        # equals the HS4 we'll assign. Redundant after step 4 for valid
        # codes; catches truncated/malformed declared_hs values.
        step_counts[STEP_HS_VERIFICATION]["in"] += 1
        synthetic_target = {"hs4_label": hs4}
        if not _bol_hs_verified(synthetic_target, aux):
            step_counts[STEP_HS_VERIFICATION]["dropped"] += 1
            _drop(intake, raw, STEP_HS_VERIFICATION,
                  f"declared_hs {aux.metadata.get('declared_hs')!r} does not "
                  f"prefix-match HS4 {hs4!r}")
            continue
        step_counts[STEP_HS_VERIFICATION]["out"] += 1

        intermediate.append((intake, raw, aux))

    # ----- step 7: near_duplicate (group across all survivors) -----
    survivors_after_dedupe: List[Tuple[str, Dict[str, str], RawAuxiliaryRecord]] = []
    step_counts[STEP_NEAR_DUPLICATE]["in"] = len(intermediate)
    if intermediate:
        # Build minimal dicts for find_near_duplicate_groups.
        dedup_input = [
            {
                "hs6_label": _declared_hs(raw)[:6],
                "tier1_description": aux.description,
            }
            for _, raw, aux in intermediate
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

        for i, (intake, raw, aux) in enumerate(intermediate):
            if i in drop_indices:
                step_counts[STEP_NEAR_DUPLICATE]["dropped"] += 1
                _drop(intake, raw, STEP_NEAR_DUPLICATE,
                      f"near-duplicate of another survivor in HS6 "
                      f"{_declared_hs(raw)[:6]}")
                continue
            step_counts[STEP_NEAR_DUPLICATE]["out"] += 1
            survivors_after_dedupe.append((intake, raw, aux))

    return survivors_after_dedupe, step_counts, dropped


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _canonical_dict(
    aux: RawAuxiliaryRecord,
    raw: Mapping[str, str],
    ordinal: int,
) -> Dict[str, Any]:
    """Build the canonical-product-shaped dict consumed by
    canonical_to_benchmark_record + generate_tiers' per-row logic."""
    hs = _declared_hs(raw)
    canonical_id = f"CP-bol-{ordinal:05d}"
    return {
        "canonical_id": canonical_id,
        "canonical_description": aux.description,
        "primary_source": "BOL",
        "primary_reference": aux.reference or _row_id(raw),
        "hs6_label": hs[:6],
        "hs4_label": hs[:4],
        "hs2_label": hs[:2],
        # MUST be "BOL_verified" so the degrader sets the minimal-tier
        # tier2_provenance (natural_mpn / degraded_manual) automatically.
        "label_source": "BOL_verified",
        "keywords": [],
        "justification_text": "",
        "manufacturer_hint": "",
        "manufacturer_hint_source": "",
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
    """Run the degrader + BOL-variant overlay over all survivors and
    return pre-audit records (ordinals provisional; reassigned
    after sort)."""
    if not survivors:
        return []

    # Provisional canonical_ids so generate_tiers has stable input.
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
    # tier2_minimal + sets the minimal-tier tier2_provenance
    # (natural_mpn / degraded_manual) (because label_source=="BOL_verified").
    result = generate_tiers(canonicals, str(abbrev_path), str(rules_path))
    records: List[Dict[str, Any]] = result["records"]

    # Overlay the natural-BOL variant: bol_metadata populated. Mirrors what
    # enrich_records_with_auxiliary would do but unconditionally (every
    # record here IS a BOL record).
    out: List[Dict[str, Any]] = []
    for rec in records:
        canon_id = rec["canonical_id"]
        aux = aux_by_canonical[canon_id]
        raw = raw_by_canonical[canon_id]
        intake = intake_by_canonical[canon_id]

        # _apply_bol_variant expects auxiliary_match_metadata to exist.
        rec["auxiliary_match_metadata"] = {}
        _apply_bol_variant(rec, aux, score=1.0)

        # _apply_bol_variant does not set origin_country, but
        # record_schema.json lists it. Patch from raw row.
        origin_country = (
            (raw.get("origin_country") or raw.get("orig_country") or "").strip()
        )
        if isinstance(rec.get("bol_metadata"), dict):
            rec["bol_metadata"]["origin_country"] = origin_country

        # Carryover-pool convention for pre-audit fields.
        rec["label_source"] = PENDING_LABEL
        rec["confidence_tier"] = "pending"
        rec["cited_evidence_ids"] = []
        rec["adjudication_status"] = "pending"
        rec["difficulty_tags"] = rec.get("difficulty_tags") or []
        rec["scope_tier"] = scope_tier.get(rec.get("hs6_label", ""), "")
        rec["split"] = "dev"  # provisional

        # source_metadata: keep what canonical_to_benchmark_record set, add
        # BOL-specific provenance keys. record_schema.json does not
        # constrain source_metadata's shape, so adding keys is safe.
        sm = dict(rec.get("source_metadata") or {})
        sm["primary_source"] = "BOL"
        sm["primary_reference"] = aux.reference or _row_id(raw)
        sm["bol_intake_file"] = intake
        sm["bol_target_hs_family"] = (raw.get("target_hs_family") or "").strip()
        sm["bol_data_source"] = "commercial_bol_provider"
        rec["source_metadata"] = sm

        # Drop transient bookkeeping that's not part of the released record.
        rec.pop("degradation_metadata", None)
        rec.pop("auxiliary_match_metadata", None)

        out.append(rec)

    # Backfill manufacturer_hint from tier2_minimal.manufacturer where
    # source_metadata.manufacturer_hint is empty. Provenance:
    # "backfilled_from_tier2".
    backfill = backfill_manufacturer_hint(out)
    out = backfill.records

    return out


def _finalize_ids_and_candidates(
    records: List[Dict[str, Any]],
    *,
    allow_hs6: Set[str],
) -> List[Dict[str, Any]]:
    """Sort records, reassign ordinals so output is byte-identical on rerun,
    then attach candidate_set (the candidate_set RNG seeds on record id, so
    it must run AFTER ordinal assignment)."""
    records.sort(key=lambda r: (
        str(r.get("hs6_label") or ""),
        str((r.get("source_metadata") or {}).get("canonical_id") or ""),
    ))

    finalized: List[Dict[str, Any]] = []
    for ordinal, rec in enumerate(records, start=1):
        new_rec = dict(rec)
        new_rec["id"] = f"SH-bol-{ordinal:05d}"
        new_rec["frozen_id"] = f"v2.0.dev.{ordinal:04d}"
        # Also normalize source_reference (carryover convention).
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
    slate (e.g., HS4 3818 has only one in-scope HS6 in scope, so
    build_candidate_set's chapter-restricted step 3 cannot find enough
    distractors). The fallback mirrors how earlier scope-repaired records
    drew distractors across chapters.
    """
    import random

    try:
        return build_candidate_set(record, in_scope_hs6=in_scope_list)
    except ValueError:
        pass

    gold_hs6 = str(record["hs6_label"])
    rng = random.Random(_record_seed(str(record["id"])))
    pool = [code for code in in_scope_list if code != gold_hs6]
    # Prefer same-HS4, then same-HS2, then anything in-scope.
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
    for rec in records:
        per_hs4[str(rec.get("hs4_label") or "")] += 1
        per_hs6[str(rec.get("hs6_label") or "")] += 1
        per_scope[str(rec.get("scope_tier") or "unknown")] += 1
        by_data_source[
            str((rec.get("source_metadata") or {}).get("bol_data_source") or "unknown")
        ] += 1
    return {
        "total_survivors": len(records),
        "per_hs4": dict(sorted(per_hs4.items())),
        "per_hs6": dict(sorted(per_hs6.items())),
        "per_scope_tier": dict(sorted(per_scope.items())),
        "by_data_source": dict(sorted(by_data_source.items())),
    }


def _build_manufacturer_coverage(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with_hint = 0
    without_hint = 0
    by_source: Counter = Counter()
    for rec in records:
        sm = rec.get("source_metadata") or {}
        hint = str(sm.get("manufacturer_hint") or "").strip()
        if hint:
            with_hint += 1
            by_source[str(sm.get("manufacturer_hint_source") or "unknown")] += 1
        else:
            without_hint += 1
    return {
        "with_hint": with_hint,
        "without_hint": without_hint,
        "by_source": dict(sorted(by_source.items())),
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
    quantity_max: int,
    min_desc_chars: int,
    step_counts: Mapping[str, Mapping[str, int]],
    dropped: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    ref_counts: Counter,
) -> Dict[str, Any]:
    samples = [copy.deepcopy(r) for r in records[:5]]
    return {
        "release": "working",
        "source": "BOL net-new",
        "inputs": {
            "intake_dir": _rel(intake_dir),
            "intake_files": [_rel(p) for p in intake_paths],
            "scope_config": _rel(scope_path),
        },
        "policy": {
            "quantity_max": quantity_max,
            "min_desc_chars": min_desc_chars,
            "freight_forwarder_denylist": list(FREIGHT_FORWARDER_DENYLIST),
            "near_duplicate_low_threshold": NEAR_DUPLICATE_LOW_THRESHOLD,
            "near_duplicate_high_threshold": NEAR_DUPLICATE_HIGH_THRESHOLD,
            "label_source_pre_audit": PENDING_LABEL,
        },
        "filter_pipeline": {
            "steps": {name: dict(step_counts[name]) for name in STEP_ORDER},
        },
        "yield": _build_yield_summary(records),
        "manufacturer_hint_coverage": _build_manufacturer_coverage(records),
        "reference_corpus_coverage_check": _build_reference_corpus_check(
            records, ref_counts
        ),
        "samples": samples,
        "dropped_records": sorted(
            (dict(d) for d in dropped),
            key=lambda d: (str(d.get("disposition") or ""),
                           str(d.get("declared_hs") or ""),
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
        help=f"BOL intake directory (default: {_rel(DEFAULT_INTAKE_DIR)}).",
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
    parser.add_argument("--quantity-max", type=int, default=10000)
    parser.add_argument("--min-desc-chars", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute counters only; do not write outputs.")
    args = parser.parse_args(argv)

    # Load configs.
    allow_hs4, allow_hs6, scope_tier = _load_scope(args.scope_config)
    ref_counts = _load_reference_corpus_hs6_counts(args.reference_corpus)

    # Resolve and load intakes.
    intake_paths = _resolve_intake_paths(args.intake_dir, args.intake_files)
    raw_rows = _load_intakes(intake_paths)
    print(f"loaded {len(raw_rows)} raw rows from "
          f"{sum(1 for p in intake_paths if p.exists())} intake file(s)")

    deduped = _union_dedupe_by_id(raw_rows)
    print(f"  after union+dedupe-by-id: {len(deduped)} unique rows")

    survivors, step_counts, dropped = _apply_filters(
        deduped,
        allow_hs4=allow_hs4,
        generic_terms=DEFAULT_GENERIC_BOL_TERMS,
        min_desc_chars=args.min_desc_chars,
        quantity_max=args.quantity_max,
    )
    print(f"  after filter pipeline: {len(survivors)} survivors")
    for step in STEP_ORDER:
        c = step_counts[step]
        print(f"    {step:24s} in={c['in']:6d} dropped={c['dropped']:6d} out={c['out']:6d}")

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
        quantity_max=args.quantity_max,
        min_desc_chars=args.min_desc_chars,
        step_counts=step_counts,
        dropped=dropped,
        records=records,
        ref_counts=ref_counts,
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
