#!/usr/bin/env python3
"""Apply expert audit decisions to the carryover candidate pool.

Closes the audit loop:

  1. The candidate pool is generated upstream from non-public inputs
     (records tagged ``label_source = *_pending_reaudit``).
  2. ``scripts/generate_review_worksheet.py`` emits the worksheet.
  3. <experts fill the worksheet>
  4. **THIS SCRIPT** parses the completed worksheet, applies decisions
     to the candidate pool, and writes an upgraded pool with the
     Core-4 fields populated and label_source flipped to the released
     enum values.

Note this is the **carryover-pool** apply step, NOT the final release
build. Net-new records from the catalog/BOL pulls are ingested
separately; the release builder unions them with the audited carryover
pool, dedupes, splits, and freezes.

Inputs:

  - ``data/intermediate/audit_worksheet.csv`` — completed worksheet.
  - ``release/working/data/_candidate_pool.jsonl`` — 133 carryover records.
  - ``configs/hs6_scope_tiers.yaml`` — for scope_tier refresh when the
    rater changes hs6_label.
  - ``release/working/data/record_schema.json`` (optional) — for output
    validation; soft-skipped when ``jsonschema`` is not installed.

Outputs:

  - ``release/working/data/_candidate_pool_audited.jsonl`` — upgraded
    records, label_source flipped to released enum (per
    src.audit.decisions.apply_corrections), fields populated.
  - ``release/working/data/_audit_corrections.json`` — structured
    corrections payload for archival (an audit_corrections.json
    but with the Core-4 evidence fields).
  - ``release/working/data/_audit_report.json`` — per-action counts,
    per-HS4 breakdown, label_source flip summary, schema validation
    status, acceptance gates.

Exits 0 on success, 1 on worksheet parse error, 2 if schema validation
finds errors (so CI / build_release.py can gate on it).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.decisions import (  # noqa: E402
    Corrections,
    DecisionParseError,
    apply_corrections,
    corrections_to_dict,
    parse_worksheet,
)


def _rel(path: Path) -> str:
    """Render ``path`` relative to ROOT when possible, else absolute.
    Tests + ad-hoc smoke tests pass worksheets from /tmp/..., which is
    outside ROOT and would crash Path.relative_to()."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


WORKING_DATA = ROOT / "release" / "working" / "data"
WORKSHEET_PATH = ROOT / "data" / "intermediate" / "audit_worksheet.csv"
CANDIDATE_POOL_PATH = WORKING_DATA / "_candidate_pool.jsonl"
BOL_POOL_PATH = WORKING_DATA / "_candidate_pool_bol.jsonl"
DEFAULT_CANDIDATE_POOLS = [CANDIDATE_POOL_PATH, BOL_POOL_PATH]
SCOPE_PATH = ROOT / "configs" / "hs6_scope_tiers.yaml"
RECORD_SCHEMA_PATH = WORKING_DATA / "record_schema.json"
AUDITED_POOL_PATH = WORKING_DATA / "_candidate_pool_audited.jsonl"
CORRECTIONS_PATH = WORKING_DATA / "_audit_corrections.json"
REPORT_PATH = WORKING_DATA / "_audit_report.json"

# Labels that MUST NOT survive the apply step (must all be flipped or dropped).
_PENDING_LABELS = frozenset({
    "BOL_expert_validated_pending_reaudit",
    "catalog_expert_validated_pending_reaudit",
})
# Released label values.
_RELEASED_LABELS = frozenset({
    "catalog_expert_validated",
    "BOL_expert_validated",
    "expert_relabeled",
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_mapping(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return json.loads(content)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, records: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def _scope_tier_lookup() -> Dict[str, str]:
    cfg = _load_mapping(SCOPE_PATH)
    out: Dict[str, str] = {}
    for h6 in cfg.get("core") or []:
        out[str(h6)] = "core"
    for h6 in cfg.get("supply_chain") or []:
        out[str(h6)] = "supply_chain"
    return out


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _refresh_scope_tier(
    records: List[Mapping[str, Any]],
    scope_lookup: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """For every record, set ``scope_tier`` from the current ``hs6_label``.

    src.audit.decisions.apply_corrections does NOT do this — scope_tier
    is a configuration-derived field, not a worksheet field. The audit
    worksheet pre-fills it for the rater's information; we re-derive
    here so any rater-relabeled record gets the correct tier."""
    out: List[Dict[str, Any]] = []
    for r in records:
        new_record = dict(r)
        hs6 = str(new_record.get("hs6_label") or "")
        tier = scope_lookup.get(hs6, "")
        if tier:
            new_record["scope_tier"] = tier
        out.append(new_record)
    return out


def _validate_no_pending_labels(records: List[Mapping[str, Any]]) -> List[str]:
    """Sanity check: after apply, no record should retain a *_pending_reaudit
    label_source. Returns a list of frozen_ids that still carry one (empty
    list = OK)."""
    bad: List[str] = []
    for r in records:
        if str(r.get("label_source") or "") in _PENDING_LABELS:
            bad.append(str(r.get("frozen_id") or "?"))
    return bad


def _validate_records_against_schema(
    records: List[Mapping[str, Any]], schema_path: Path
) -> List[str]:
    """Return human-readable schema-validation errors (empty list on
    success). Soft-fails with a single 'jsonschema not installed' message
    when the package isn't available."""
    if not schema_path.exists():
        return [f"schema {schema_path} missing; validation skipped"]
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return ["jsonschema not installed; schema validation skipped"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors: List[str] = []
    for i, r in enumerate(records):
        for err in validator.iter_errors(r):
            loc = ".".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(
                f"record[{i}] frozen_id={r.get('frozen_id', '?')}: "
                f"{loc}: {err.message}"
            )
            if len(errors) >= 50:
                errors.append("… (truncated; >50 errors)")
                return errors
    return errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summarize(corrections: Corrections) -> str:
    return (
        f"  confirmed (pending_reaudit → released):  {len(corrections.confirmed)}\n"
        f"  relabeled (HS6 changed by expert):       {len(corrections.relabeled)}\n"
        f"  dropped (removed from pool):             {len(corrections.dropped)}\n"
        f"  total decisions reviewed:                {len(corrections.decisions)}"
    )


def _build_report(
    corrections: Corrections,
    audited: List[Mapping[str, Any]],
    dropped_frozen_ids: List[str],
    schema_errors: List[str],
) -> Dict[str, Any]:
    label_source_counts: Counter = Counter()
    hs4_counts: Counter = Counter()
    scope_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    adjudication_counts: Counter = Counter()
    citation_counts: Counter = Counter()
    for r in audited:
        label_source_counts[str(r.get("label_source") or "")] += 1
        hs4_counts[str(r.get("hs4_label") or "")] += 1
        scope_counts[str(r.get("scope_tier") or "unknown")] += 1
        confidence_counts[str(r.get("confidence_tier") or "unset")] += 1
        adjudication_counts[str(r.get("adjudication_status") or "unset")] += 1
        n = len(r.get("cited_evidence_ids") or [])
        citation_counts[n] += 1

    # Citation distribution as a sorted dict.
    citation_distribution = dict(sorted(citation_counts.items()))

    pending_remaining = sum(
        1 for r in audited if str(r.get("label_source") or "") in _PENDING_LABELS
    )

    high_with_under_2_citations = sum(
        1 for r in audited
        if r.get("confidence_tier") == "high"
        and len(r.get("cited_evidence_ids") or []) < 2
    )

    return {
        "release": "working",
        "counts": {
            "decisions_reviewed": len(corrections.decisions),
            "confirmed": len(corrections.confirmed),
            "relabeled": len(corrections.relabeled),
            "dropped": len(corrections.dropped),
            "audited_records_written": len(audited),
        },
        "per_label_source": dict(sorted(label_source_counts.items())),
        "per_hs4": dict(sorted(hs4_counts.items())),
        "per_scope_tier": dict(sorted(scope_counts.items())),
        "per_confidence_tier": dict(sorted(confidence_counts.items())),
        "per_adjudication_status": dict(sorted(adjudication_counts.items())),
        "citation_distribution_per_record": citation_distribution,
        "acceptance_gates": {
            "no_pending_reaudit_labels_remaining": pending_remaining == 0,
            "no_high_confidence_without_2_citations": (
                high_with_under_2_citations == 0
            ),
            "schema_validation_passed": (
                len(schema_errors) == 0
                or (len(schema_errors) == 1
                    and "jsonschema not installed" in schema_errors[0])
                or (len(schema_errors) == 1 and "validation skipped" in schema_errors[0])
            ),
        },
        "dropped_frozen_ids": sorted(dropped_frozen_ids),
        "schema_validation_errors": schema_errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worksheet",
        type=Path,
        default=WORKSHEET_PATH,
        help="Completed audit worksheet CSV.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        nargs="+",
        default=DEFAULT_CANDIDATE_POOLS,
        help="One or more candidate pool JSONL files. Default: the carryover "
             "pool plus the net-new BOL pool emitted by scripts/ingest_bol.py. "
             "Missing files are skipped with a warning.",
    )
    parser.add_argument(
        "--audited-out",
        type=Path,
        default=AUDITED_POOL_PATH,
        help="Output JSONL for the audited (post-apply) candidate pool.",
    )
    parser.add_argument(
        "--corrections-out",
        type=Path,
        default=CORRECTIONS_PATH,
        help="Structured corrections payload (audit trail).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=REPORT_PATH,
        help="Per-action / per-HS4 / per-tier audit report JSON.",
    )
    parser.add_argument(
        "--record-schema",
        type=Path,
        default=RECORD_SCHEMA_PATH,
        help="JSON Schema to validate the audited pool against.",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Exit with status 2 if any output record fails record_schema validation. "
             "Without this flag, errors are still reported but exit is 0 on parse success.",
    )
    args = parser.parse_args(argv)

    if not args.worksheet.exists():
        print(f"ERROR: worksheet not found: {args.worksheet}", file=sys.stderr)
        return 1
    existing_pools = [p for p in args.candidate_pool if p.exists()]
    missing_pools = [p for p in args.candidate_pool if not p.exists()]
    for missing in missing_pools:
        print(f"WARN: candidate pool missing, skipping: {_rel(missing)}",
              file=sys.stderr)
    if not existing_pools:
        print(
            f"ERROR: no candidate pool files found among "
            f"{[_rel(p) for p in args.candidate_pool]}; "
            f"run the ingest scripts to (re)create the candidate pools.",
            file=sys.stderr,
        )
        return 1

    # Parse worksheet.
    try:
        corrections = parse_worksheet(args.worksheet)
    except DecisionParseError as exc:
        print(f"ERROR parsing worksheet:\n{exc}", file=sys.stderr)
        return 1

    print(f"parsed {_rel(args.worksheet)}:")
    print(_summarize(corrections))

    # Persist corrections payload for audit-trail continuity.
    args.corrections_out.parent.mkdir(parents=True, exist_ok=True)
    args.corrections_out.write_text(
        json.dumps(corrections_to_dict(corrections), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {_rel(args.corrections_out)}")

    # Load candidate pool(s) + apply. apply_corrections builds a
    # frozen_id → record index internally, so concatenating multiple
    # pools is safe as long as frozen_ids are globally unique (the
    # carryover pool uses earlier-generation IDs; ingest_bol uses v2.0.* IDs).
    pool: List[Dict[str, Any]] = []
    for pool_path in existing_pools:
        rows = _read_jsonl(pool_path)
        print(f"loaded {_rel(pool_path)}: {len(rows)} records")
        pool.extend(rows)
    print(f"  total candidate records: {len(pool)}")

    audited = apply_corrections(pool, corrections)

    # Refresh scope_tier from the (possibly new) hs6_label using the
    # config — apply_corrections doesn't do this since it's external to
    # the worksheet schema.
    scope_lookup = _scope_tier_lookup()
    audited = _refresh_scope_tier(audited, scope_lookup)

    # Sanity check: no record should retain a *_pending_reaudit label.
    bad_labels = _validate_no_pending_labels(audited)
    if bad_labels:
        print(
            f"WARN: {len(bad_labels)} records still carry "
            f"*_pending_reaudit labels after apply: {bad_labels[:5]}…",
            file=sys.stderr,
        )

    # Identify dropped frozen_ids (records absent from audited that were
    # present in pool).
    audited_ids: Set[str] = {str(r.get("frozen_id") or "") for r in audited}
    pool_ids: Set[str] = {str(r.get("frozen_id") or "") for r in pool}
    dropped_ids = sorted(pool_ids - audited_ids)

    # Schema validation.
    schema_errors = _validate_records_against_schema(audited, args.record_schema)
    if schema_errors:
        soft = (
            len(schema_errors) == 1
            and ("not installed" in schema_errors[0]
                 or "validation skipped" in schema_errors[0])
        )
        if soft:
            print(f"  ⚠ {schema_errors[0]}")
        else:
            print(f"  ✗ {len(schema_errors)} schema validation errors")
    else:
        print(f"  ✓ all {len(audited)} audited records pass record_schema validation")

    # Write audited pool.
    _write_jsonl(args.audited_out, audited)
    print(f"wrote {_rel(args.audited_out)} ({len(audited)} records)")

    # Build + write the report.
    report = _build_report(corrections, audited, dropped_ids, schema_errors)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {_rel(args.report_out)}")

    # Console summary of acceptance gates.
    gates = report["acceptance_gates"]
    print()
    print("Acceptance gates:")
    for k, v in gates.items():
        symbol = "✓" if v else "✗"
        print(f"  {symbol} {k}: {v}")

    # Decide exit code.
    all_gates_pass = all(report["acceptance_gates"].values())
    has_schema_errors = (
        len(schema_errors) > 0
        and not any(
            "not installed" in e or "validation skipped" in e for e in schema_errors
        )
    )
    if args.strict_schema and has_schema_errors:
        print(
            f"\nFAIL: --strict-schema set and {len(schema_errors)} validation errors.",
            file=sys.stderr,
        )
        return 2
    if not all_gates_pass and args.strict_schema:
        print("\nFAIL: --strict-schema set and an acceptance gate failed.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
