#!/usr/bin/env python3
"""Build ``release/working/data/reference_corpus.jsonl`` — the customs-ruling
evidence corpus (CROSS, EBTI, Japan Customs) used during annotation.

Inputs (combined into one pipeline pass):

* ``release/working/data/_reference_corpus_inputs.jsonl`` — pre-sourced
  in-scope EBTI/CROSS ruling inputs (non-public working pool).
* ``data/raw/ebti/multilingual_pulls.jsonl`` — multilingual EBTI
  rulings (stratified + gap-coverage), pre-sourced under ``data/raw/``.
* ``data/raw/cross/new_pulls.jsonl`` — CROSS records, pre-sourced
  under ``data/raw/``.
* ``data/raw/jp_customs/new_pulls.jsonl`` — Japan Customs advance
  rulings normalized by ``scripts/normalize_jp_customs.py``.

Outputs:

* ``release/working/data/reference_corpus.jsonl`` — entries conforming
  to ``release/working/data/reference_corpus_schema.json``. Each entry
  carries tier1/tier2 produced by
  ``src/processing/degrader.py`` ``generate_tiers()`` — the SAME
  function called on benchmark records, so corpus and benchmark share
  one normalization pipeline.
* ``release/working/data/_reference_corpus_build_report.json`` — per-HS6
  coverage report, jurisdictional split, schema-validation status.

Reuses without modification:

* ``src/processing/degrader.py`` ``generate_tiers()``
* ``src/processing/degrader.py`` ``canonical_to_benchmark_record()``
* ``configs/degradation_rules.yaml`` + ``configs/abbreviations.csv``

Idempotent. Deterministic ordering by evidence_id.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
WORKING_DATA = ROOT / "release" / "working" / "data"
SCHEMA_PATH = WORKING_DATA / "reference_corpus_schema.json"

REF_INPUTS_PATH = WORKING_DATA / "_reference_corpus_inputs.jsonl"
EBTI_NEW_PATH = ROOT / "data" / "raw" / "ebti" / "multilingual_pulls.jsonl"
CROSS_NEW_PATH = ROOT / "data" / "raw" / "cross" / "new_pulls.jsonl"
JP_CUSTOMS_NEW_PATH = ROOT / "data" / "raw" / "jp_customs" / "new_pulls.jsonl"

ABBREVIATIONS_PATH = ROOT / "configs" / "abbreviations.csv"
RULES_PATH = ROOT / "configs" / "degradation_rules.yaml"

OUTPUT_PATH = WORKING_DATA / "reference_corpus.jsonl"
REPORT_PATH = WORKING_DATA / "_reference_corpus_build_report.json"

# Ensure project root is on sys.path so `from src...` works regardless of cwd.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processing.degrader import generate_tiers  # noqa: E402

# Language → EBTI jurisdiction code. CROSS jurisdiction is always US.
_LANG_TO_JURISDICTION: Dict[str, str] = {
    "de": "EU-DE", "fr": "EU-FR", "nl": "EU-NL", "it": "EU-IT",
    "pl": "EU-PL", "cs": "EU-CZ", "es": "EU-ES", "hu": "EU-HU",
    "sk": "EU-SK", "sv": "EU-SE", "fi": "EU-FI", "da": "EU-DK",
    "ro": "EU-RO", "sl": "EU-SI", "lv": "EU-LV", "bg": "EU-BG",
    "hr": "EU-HR", "el": "EU-GR", "pt": "EU-PT", "ga": "EU-IE",
    "en": "EU-EN",
}


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


def _sanitize_id_segment(s: str) -> str:
    """Sanitize a ruling-number segment for use in evidence_id.

    Whitespace → underscore. Other characters (letters, digits, ``_./-``)
    are preserved verbatim because the reference_corpus_schema.json
    pattern allows them."""
    return "_".join(s.split())  # collapse any whitespace run to single _


def _stable_evidence_id(record: Mapping[str, Any]) -> str:
    """Compute a stable evidence_id from source + jurisdiction + hint.

    EBTI: ``EBTI-<jurisdiction-code>-<bti-number>``
    CROSS: ``CROSS-<ruling-id>``
    JP_CUSTOMS: ``JP_CUSTOMS-<registration-number>``

    Sanitizes the ruling-number segment so the schema pattern accepts
    it (whitespace → underscore)."""
    source = record.get("source") or ""
    hint = _sanitize_id_segment(str(record.get("evidence_id_hint") or "").strip())
    if source == "EBTI":
        juris = (record.get("jurisdiction") or "EU-XX")
        juris_short = juris.replace("EU-", "")
        # Normalize to a 2-letter uppercase code (schema requires ^[A-Z]{2}).
        # Fallbacks "EN" and "XX" are 2-letter already; other inputs may
        # come in lowercase from the language map.
        juris_short = juris_short.upper()[:2] or "XX"
        return f"EBTI-{juris_short}-{hint}" if hint else f"EBTI-{juris_short}-NOREF"
    if source == "CROSS":
        return f"CROSS-{hint}" if hint else "CROSS-NOREF"
    if source == "JP_CUSTOMS":
        return f"JP_CUSTOMS-{hint}" if hint else "JP_CUSTOMS-NOREF"
    raise ValueError(f"unknown source {source!r} in record {record!r}")


_SOURCE_DEFAULT_JURISDICTION: Dict[str, str] = {
    "CROSS": "US",
    "JP_CUSTOMS": "JP",
}


def _resolve_jurisdiction(record: Mapping[str, Any]) -> str:
    """Resolve final jurisdiction string. Inputs already carry this for
    new pulls; carryover inputs may lack it — fall back to source default
    or language map for EBTI."""
    j = record.get("jurisdiction")
    if j:
        return str(j)
    src = record.get("source")
    if src in _SOURCE_DEFAULT_JURISDICTION:
        return _SOURCE_DEFAULT_JURISDICTION[src]
    lang = (record.get("language") or "").lower()
    return _LANG_TO_JURISDICTION.get(lang, "EU-EN")


def _resolve_url(record: Mapping[str, Any]) -> Optional[str]:
    url = record.get("url")
    if url:
        return str(url)
    md = record.get("raw_metadata") or {}
    return md.get("detail_url") or md.get("DETAIL_URL")


def _resolve_date(record: Mapping[str, Any]) -> Optional[str]:
    d = record.get("ruling_date")
    if d:
        return str(d)
    md = record.get("raw_metadata") or {}
    return (
        md.get("date") or md.get("start_date") or md.get("START_DATE_OF_VALIDITY")
        or None
    )


def _to_degrader_input(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Reshape a reference-corpus input into the schema expected by
    ``canonical_to_benchmark_record()`` (degrader.py:86) so we can call
    ``generate_tiers()`` uniformly across all sources.

    The reference-corpus tier1 input is the first paragraph of the
    ruling (``tier1_text_seed``). The degrader applies
    ``normalize_text()`` (NOT ``clean_bol_description()``) because we set
    ``primary_source`` to the ruling source name, not the literal "BOL"."""
    hs6 = str(record.get("hs6_label") or "")
    hint = str(record.get("evidence_id_hint") or "").strip()
    if hint:
        canonical_id = f"REF-{record.get('source', 'UNK')}-{hint}"
    else:
        canonical_id = f"REF-{record.get('source', 'UNK')}-{id(record):x}"
    keywords = list(record.get("subject_terms") or [])
    return {
        "canonical_id": canonical_id,
        "primary_source": str(record.get("source") or "REF"),
        "canonical_description": str(record.get("tier1_text_seed") or ""),
        "hs6_label": hs6,
        "hs4_label": hs6[:4],
        "hs2_label": hs6[:2],
        "label_source": str(record.get("source") or ""),
        "keywords": keywords,
        "justification_text": str(record.get("rationale_excerpt") or ""),
        "primary_reference": hint,
        "manufacturer_hint": "",
        "manufacturer_hint_source": "",
        "sampling_metadata": {
            "jurisdiction": _resolve_jurisdiction(record),
            "language": str(record.get("language") or ""),
        },
        "source_evidence": [],
        "merge_confidence": None,
    }


def _to_corpus_entry(
    degraded_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Combine a degrader output with reference-corpus metadata into the
    schema declared in reference_corpus_schema.json."""
    tier2 = dict(degraded_record.get("tier2_minimal") or {})
    tier2.setdefault("part_name", "")
    tier2.setdefault("manufacturer", "")

    return {
        "evidence_id": _stable_evidence_id(source_record),
        "source": str(source_record.get("source") or ""),
        "jurisdiction": _resolve_jurisdiction(source_record),
        "hs6_label": str(source_record.get("hs6_label") or ""),
        "ruling_date": _resolve_date(source_record),
        "url": _resolve_url(source_record),
        "tier1_text": str(degraded_record.get("tier1_description") or ""),
        "tier2_minimal": tier2,
        "subject_terms": list(source_record.get("subject_terms") or []),
        "rationale_excerpt": str(source_record.get("rationale_excerpt") or ""),
    }


def _validate_against_schema(
    entries: List[Mapping[str, Any]], schema_path: Path
) -> List[str]:
    """Run JSON Schema validation. Returns human-readable error strings
    (empty if all entries validate). Soft check — missing jsonschema is a
    warning, not a failure."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return ["jsonschema not installed; skipping schema validation"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    errors: List[str] = []
    for i, entry in enumerate(entries):
        for err in validator.iter_errors(entry):
            loc = ".".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(
                f"entry[{i}] eid={entry.get('evidence_id', '?')}: {loc}: {err.message}"
            )
            if len(errors) >= 50:
                errors.append("… (truncated; >50 errors)")
                return errors
    return errors


def _build_report(
    entries: List[Mapping[str, Any]],
    schema_errors: List[str],
    counts: Mapping[str, int],
) -> Dict[str, Any]:
    per_hs6: Dict[str, Counter] = defaultdict(Counter)
    per_jurisdiction: Counter = Counter()
    for e in entries:
        hs6 = e.get("hs6_label") or ""
        if hs6:
            per_hs6[hs6][e.get("source", "")] += 1
        per_jurisdiction[e.get("jurisdiction", "")] += 1

    coverage_summary = {"meets_floor": 0, "below_floor": []}
    for hs6 in sorted(per_hs6):
        total = sum(per_hs6[hs6].values())
        if total >= 3:
            coverage_summary["meets_floor"] += 1
        else:
            coverage_summary["below_floor"].append(
                {"hs6": hs6, "total": total, "by_source": dict(per_hs6[hs6])}
            )

    return {
        "release": "working",
        "schema": str(SCHEMA_PATH.relative_to(ROOT)),
        "inputs": dict(counts),
        "outputs": {
            "entries": len(entries),
            "path": str(OUTPUT_PATH.relative_to(ROOT)),
        },
        "coverage": {
            "per_hs6": {h6: dict(per_hs6[h6]) for h6 in sorted(per_hs6)},
            "per_jurisdiction": dict(sorted(per_jurisdiction.items())),
            "coverage_floor_summary": coverage_summary,
        },
        "schema_validation": {
            "errors": schema_errors,
            "ok": (len(schema_errors) == 0) or (
                len(schema_errors) == 1
                and "jsonschema not installed" in schema_errors[0]
            ),
        },
    }


def main() -> int:
    if not REF_INPUTS_PATH.exists():
        print(
            f"ERROR: missing non-public input: {REF_INPUTS_PATH}",
            file=sys.stderr,
        )
        return 1
    if not RULES_PATH.exists() or not ABBREVIATIONS_PATH.exists():
        print(
            f"ERROR: missing rules/abbreviations config "
            f"({RULES_PATH}, {ABBREVIATIONS_PATH})",
            file=sys.stderr,
        )
        return 1

    carryover_inputs = _read_jsonl(REF_INPUTS_PATH)
    ebti_new = _read_jsonl(EBTI_NEW_PATH)
    cross_new = _read_jsonl(CROSS_NEW_PATH)
    jp_customs_new = _read_jsonl(JP_CUSTOMS_NEW_PATH)

    print(f"pre-sourced ruling inputs: {len(carryover_inputs)}")
    print(f"new EBTI multilingual:     {len(ebti_new)}")
    print(f"new CROSS pulls:           {len(cross_new)}")
    print(f"new JP_CUSTOMS pulls:      {len(jp_customs_new)}")

    all_inputs: List[Mapping[str, Any]] = (
        carryover_inputs + ebti_new + cross_new + jp_customs_new
    )

    # Dedup by evidence_id.
    seen_ids: Dict[str, Mapping[str, Any]] = {}
    for record in all_inputs:
        try:
            eid = _stable_evidence_id(record)
        except ValueError as exc:
            print(f"WARN skip record (no source): {exc}", file=sys.stderr)
            continue
        if eid in seen_ids:
            continue
        seen_ids[eid] = record
    deduped = list(seen_ids.values())
    print(f"unique evidence_ids (after dedup): {len(deduped)}")

    degrader_inputs = [_to_degrader_input(r) for r in deduped]

    print(f"calling generate_tiers() on {len(degrader_inputs)} records …")
    output = generate_tiers(
        degrader_inputs,
        abbreviations_path=str(ABBREVIATIONS_PATH),
        rules_path=str(RULES_PATH),
    )
    degraded_records: List[Mapping[str, Any]] = list(output.get("records") or [])
    if len(degraded_records) != len(deduped):
        print(
            f"WARN degrader returned {len(degraded_records)} records but "
            f"received {len(deduped)} inputs",
            file=sys.stderr,
        )

    entries: List[Dict[str, Any]] = []
    for degraded, source in zip(degraded_records, deduped):
        entries.append(_to_corpus_entry(degraded, source))

    entries.sort(key=lambda e: e["evidence_id"])

    WORKING_DATA.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")

    print("validating against reference_corpus_schema.json …")
    schema_errors = _validate_against_schema(entries, SCHEMA_PATH)
    if not schema_errors:
        print("  ✓ all entries valid")
    elif len(schema_errors) == 1 and "jsonschema not installed" in schema_errors[0]:
        print(f"  ⚠ {schema_errors[0]}")
    else:
        print(f"  ⚠ {len(schema_errors)} schema errors (see report)")

    counts = {
        "carryover": len(carryover_inputs),
        "ebti_multilingual": len(ebti_new),
        "cross_new": len(cross_new),
        "jp_customs_new": len(jp_customs_new),
        "deduped_inputs": len(deduped),
    }
    report = _build_report(entries, schema_errors, counts)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)} ({len(entries)} entries)")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    cov = report["coverage"]["coverage_floor_summary"]
    print(f"evidence coverage: {cov['meets_floor']} HS6 ≥3 entries; "
          f"{len(cov['below_floor'])} below floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
