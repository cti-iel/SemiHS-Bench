#!/usr/bin/env python3
"""Orchestrate the release packaging.

Runs the release-gate checklist, generates the MANIFEST,
and emits per-release artifacts. The script is **safe to run before the
full audit pass completes**: stages that depend on the not-yet-built
dev.json / test_frozen.json skip with a clear "skipped (artifact
missing)" status and the release is marked as ``not_ready_for_dist``.

Stages:

  1. verify_inputs       — what's on disk?
  2. validate_schemas    — reference_corpus.jsonl + records vs JSON Schema
  3. run_release_gates   — release-gate checklist (every line)
  4. build_manifest      — MANIFEST.json with hs_version, SHA-256s, dists,
                            reference-corpus stats, calibration-report stats
  5. generate_statistics — STATISTICS.md (only if dev/test exist)
  6. generate_example_submission — only if test_frozen.json exists
  7. compute_tree_hash   — sha256 over (path, contents) for everything
                            currently in release/working/

CLI:

  python3 scripts/build_release.py            # full build
  python3 scripts/build_release.py --check    # what's present? no writes
  python3 scripts/build_release.py --gates-only  # run the gates, no writes

Exit codes:
  0   release built (and ready_for_dist if all gates pass)
  1   missing prerequisite (e.g., reference_corpus.jsonl absent)
  2   release built but one or more gates failed; not ready for distribution
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RELEASE = ROOT / "release" / "working"
RELEASE_DATA = RELEASE / "data"
RELEASE_DOSSIERS = RELEASE / "dossiers"
RELEASE_EVAL = RELEASE / "eval"
RELEASE_PROMPTS = RELEASE / "prompts"
RELEASE_EXAMPLES = RELEASE / "examples"

SCOPE_CONFIG_PATH = ROOT / "configs" / "hs6_scope_tiers.yaml"
BOUNDARY_TAGS_PATH = ROOT / "configs" / "boundary_tags.yaml"
MANUFACTURER_CAPS_PATH = ROOT / "configs" / "manufacturer_caps.yaml"

# Optional companion artifact (authority-calibration scorer output).
CALIBRATION_REPORT_PATH = RELEASE_DATA / "calibration_report.json"

# Released label_source enum (transient *_pending_reaudit must not survive).
_RELEASED_LABEL_SOURCES = frozenset({
    "catalog_expert_validated",
    "BOL_expert_validated",
    "expert_relabeled",
})
_PENDING_LABEL_SOURCES = frozenset({
    "BOL_expert_validated_pending_reaudit",
    "catalog_expert_validated_pending_reaudit",
})

# Per-HS6 caps (HS rebalance rules).
_HS6_CAPS = {
    "854239": 60,
    "854231": 80,
    "854233": 60,
    "854141": 80,
}

# Core-4 evidence-coverage floor.
_COVERAGE_FLOOR = 3
# Boundary-case share window.
_BOUNDARY_SHARE_MIN = 0.38
_BOUNDARY_SHARE_MAX = 0.45
# Manufacturer floors.
_MANUFACTURER_DISTINCT_FLOOR = 100
_MANUFACTURER_TOP50_COVERAGE = 5
# Hint coverage.
_MANUFACTURER_HINT_FLOOR = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scope_lookup() -> Dict[str, str]:
    cfg = _load_mapping(SCOPE_CONFIG_PATH)
    out: Dict[str, str] = {}
    for h6 in cfg.get("core") or []:
        out[str(h6)] = "core"
    for h6 in cfg.get("supply_chain") or []:
        out[str(h6)] = "supply_chain"
    return out


# ---------------------------------------------------------------------------
# Stage 1: verify_inputs
# ---------------------------------------------------------------------------

def verify_inputs() -> Dict[str, bool]:
    """Survey what's on disk. Returns a presence map; never raises."""
    presence = {
        "dev.json": (RELEASE_DATA / "dev.json").exists(),
        "test_frozen.json": (RELEASE_DATA / "test_frozen.json").exists(),
        "reference_corpus.jsonl": (RELEASE_DATA / "reference_corpus.jsonl").exists(),
        "reference_corpus_schema.json": (RELEASE_DATA / "reference_corpus_schema.json").exists(),
        "record_schema.json": (RELEASE_DATA / "record_schema.json").exists(),
        "_candidate_pool.jsonl": (RELEASE_DATA / "_candidate_pool.jsonl").exists(),
        "submission_schema.json": (RELEASE_EVAL / "submission_schema.json").exists(),
        "dossiers/index.json": (RELEASE_DOSSIERS / "index.json").exists(),
        "calibration_report.json": CALIBRATION_REPORT_PATH.exists(),
        "scope_config": SCOPE_CONFIG_PATH.exists(),
        "boundary_tags": BOUNDARY_TAGS_PATH.exists(),
        "manufacturer_caps": MANUFACTURER_CAPS_PATH.exists(),
    }
    return presence


# ---------------------------------------------------------------------------
# Stage 2: validate_schemas
# ---------------------------------------------------------------------------

def _validate_against_schema(
    records: Iterable[Mapping[str, Any]], schema_path: Path, max_errors: int = 25
) -> List[str]:
    if not schema_path.exists():
        return [f"schema {_rel(schema_path)} missing"]
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return ["jsonschema not installed; validation skipped"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors: List[str] = []
    for i, record in enumerate(records):
        for err in validator.iter_errors(record):
            loc = ".".join(str(p) for p in err.absolute_path) or "(root)"
            ident = record.get("frozen_id") or record.get("evidence_id") or f"row[{i}]"
            errors.append(f"{ident}: {loc}: {err.message}")
            if len(errors) >= max_errors:
                errors.append(f"… (truncated; >{max_errors} errors)")
                return errors
    return errors


def validate_schemas() -> Dict[str, Any]:
    """Validate reference corpus + (when present) dev/test records."""
    out: Dict[str, Any] = {
        "reference_corpus": {"status": "skipped", "errors": []},
        "dev_records": {"status": "skipped", "errors": []},
        "test_records": {"status": "skipped", "errors": []},
    }
    ref_path = RELEASE_DATA / "reference_corpus.jsonl"
    ref_schema = RELEASE_DATA / "reference_corpus_schema.json"
    if ref_path.exists() and ref_schema.exists():
        errors = _validate_against_schema(_read_jsonl(ref_path), ref_schema)
        out["reference_corpus"] = {
            "status": "ok" if not errors else (
                "ok" if errors and "not installed" in errors[0] else "fail"
            ),
            "errors": errors,
        }
    record_schema = RELEASE_DATA / "record_schema.json"
    for name, path in (
        ("dev_records", RELEASE_DATA / "dev.json"),
        ("test_records", RELEASE_DATA / "test_frozen.json"),
    ):
        if path.exists() and record_schema.exists():
            errors = _validate_against_schema(_read_json_list(path), record_schema)
            out[name] = {
                "status": "ok" if not errors else (
                    "ok" if errors and "not installed" in errors[0] else "fail"
                ),
                "errors": errors,
            }
    return out


# ---------------------------------------------------------------------------
# Stage 3: run_release_gates (release-gate checklist)
# ---------------------------------------------------------------------------

def _gate_methodology_compliance(
    records: List[Mapping[str, Any]],
    *,
    evidence_jurisdiction: Mapping[str, str],
) -> Dict[str, Any]:
    """Methodology compliance block for the release gates.

    ``evidence_jurisdiction`` maps ``evidence_id → jurisdiction`` (loaded
    from ``reference_corpus.jsonl``). Required so the gate can enforce
    the Core-4 evidence rule fully: a ``confidence_tier=high`` record needs ≥2
    cited evidence_ids, **all of which resolve in the reference corpus**,
    and **spanning ≥2 distinct jurisdictions**. A ``medium`` record
    needs ≥1 cited evidence_id that resolves. Bare count checks (the
    previous behaviour) accepted nonsense evidence_ids like
    ``"EBTI-XX-FAKE"`` and would have accepted two citations both from
    ``EU-DE`` as cross-jurisdictional. Callers without a corpus
    must pass ``{}``, which correctly fails the gate.
    """
    pending_remaining = sum(
        1 for r in records if str(r.get("label_source") or "") in _PENDING_LABEL_SOURCES
    )
    legacy_labels = sum(
        1 for r in records
        if str(r.get("label_source") or "") in {"EBTI", "CROSS", "BOL_verified"}
    )
    no_evidence = sum(1 for r in records if not (r.get("cited_evidence_ids") or []))
    legacy_bronze_fields = sum(
        1 for r in records
        if "classification_driver" in r or "ambiguity_score" in r
    )

    # ---------- evidence-binding checks ----------
    confidence_violations: List[Dict[str, Any]] = []
    unresolved_citations: List[Dict[str, Any]] = []
    for r in records:
        tier = r.get("confidence_tier")
        cited = list(r.get("cited_evidence_ids") or [])
        fid = r.get("frozen_id") or r.get("id") or "?"

        # Resolve every cited id against the reference corpus.
        resolved_jurisdictions: List[str] = []
        unresolved: List[str] = []
        for cid in cited:
            j = evidence_jurisdiction.get(str(cid))
            if j is None:
                unresolved.append(str(cid))
            else:
                resolved_jurisdictions.append(j)

        if unresolved:
            unresolved_citations.append(
                {"frozen_id": fid, "unresolved": unresolved}
            )

        distinct_jurisdictions = set(resolved_jurisdictions)

        # Tier-specific binding (Core-4 evidence rule):
        # - high   : ≥2 cited, all resolve, ≥2 distinct jurisdictions
        # - medium : ≥1 cited, all resolve
        # - low    : no requirement (still flagged at release time
        #            as a confidence_tier=low population summary, but
        #            not blocked here)
        if tier == "high":
            if len(cited) < 2:
                confidence_violations.append({
                    "frozen_id": fid, "tier": "high",
                    "reason": f"high requires ≥2 citations, got {len(cited)}",
                })
            elif unresolved:
                confidence_violations.append({
                    "frozen_id": fid, "tier": "high",
                    "reason": (
                        "high requires every cited evidence_id to resolve "
                        "in reference_corpus.jsonl; unresolved: "
                        f"{unresolved}"
                    ),
                })
            elif len(distinct_jurisdictions) < 2:
                confidence_violations.append({
                    "frozen_id": fid, "tier": "high",
                    "reason": (
                        "high requires ≥2 distinct jurisdictions across "
                        "citations (Core-4 evidence rule); got "
                        f"{sorted(distinct_jurisdictions)}"
                    ),
                })
        elif tier == "medium":
            if len(cited) < 1:
                confidence_violations.append({
                    "frozen_id": fid, "tier": "medium",
                    "reason": "medium requires ≥1 citation",
                })
            elif unresolved:
                confidence_violations.append({
                    "frozen_id": fid, "tier": "medium",
                    "reason": (
                        "medium requires its citation to resolve in "
                        "reference_corpus.jsonl; unresolved: "
                        f"{unresolved}"
                    ),
                })

    return {
        "no_pending_reaudit_labels": pending_remaining == 0,
        "no_legacy_label_sources": legacy_labels == 0,
        "every_record_has_evidence_id": no_evidence == 0,
        "confidence_evidence_binding_holds": len(confidence_violations) == 0,
        "all_cited_evidence_ids_resolve": len(unresolved_citations) == 0,
        "no_legacy_bronze_fields": legacy_bronze_fields == 0,
        "_diagnostics": {
            "pending_remaining": pending_remaining,
            "legacy_labels": legacy_labels,
            "records_without_evidence": no_evidence,
            "confidence_violation_count": len(confidence_violations),
            "confidence_violations_sample": confidence_violations[:10],
            "unresolved_citation_count": len(unresolved_citations),
            "unresolved_citations_sample": unresolved_citations[:10],
            "records_with_legacy_bronze_fields": legacy_bronze_fields,
        },
    }


def _gate_scope(records: List[Mapping[str, Any]], allow_hs6: set) -> Dict[str, Any]:
    out_of_scope_hs4 = {"8543", "8534"}
    in_scope_hs4 = {h[:4] for h in allow_hs6}
    bad_hs4 = sum(
        1 for r in records if str(r.get("hs4_label") or "") in out_of_scope_hs4
    )
    out_of_allow = sum(
        1 for r in records if str(r.get("hs4_label") or "") not in in_scope_hs4
    )
    no_scope_tier = sum(
        1 for r in records if r.get("scope_tier") not in ("core", "supply_chain")
    )
    return {
        "no_dropped_hs4": bad_hs4 == 0,
        "all_hs4_in_allow_set": out_of_allow == 0,
        "every_record_has_scope_tier": no_scope_tier == 0,
        "_diagnostics": {
            "records_with_dropped_hs4": bad_hs4,
            "records_outside_allow_set": out_of_allow,
            "records_missing_scope_tier": no_scope_tier,
        },
    }


def _gate_source_mix(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(r.get("tier1_source") or "") for r in records)
    total = sum(counts.values())
    catalog = counts.get("catalog", 0)
    bol = counts.get("BOL", 0)
    ebti_cross = sum(
        1 for r in records if r.get("tier1_source") in ("EBTI", "CROSS")
    )
    if total == 0:
        return {
            "source_mix_within_targets": False,
            "no_ebti_cross_records": True,
            "total_records": 0,
            "_diagnostics": {"empty_record_set": True},
        }
    catalog_share = catalog / total
    bol_share = bol / total
    return {
        "source_mix_within_targets": (
            0.45 <= catalog_share <= 0.47 and 0.53 <= bol_share <= 0.55
        ),
        "no_ebti_cross_records": ebti_cross == 0,
        "total_records": total,
        "_diagnostics": {
            "catalog_share": round(catalog_share, 4),
            "bol_share": round(bol_share, 4),
            "ebti_cross_count": ebti_cross,
            "target_total": 1800,
            "target_total_tolerance": 10,
            "within_size_tolerance": abs(total - 1800) <= 10,
        },
    }


def _gate_manufacturer(
    records: List[Mapping[str, Any]], caps_cfg: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    mfg_per_record: List[str] = []
    n_with_hint = 0
    for r in records:
        sm = r.get("source_metadata") or {}
        tier2 = r.get("tier2_minimal") or {}
        m = sm.get("manufacturer_hint") or tier2.get("manufacturer") or ""
        if m:
            n_with_hint += 1
            mfg_per_record.append(str(m).strip().lower())
    distinct = len(set(mfg_per_record))
    counter = Counter(mfg_per_record)
    top50 = counter.most_common(50)
    top50_min_records = top50[-1][1] if len(top50) >= 50 else 0
    # Cap check.
    cap_violations: List[Dict[str, Any]] = []
    if caps_cfg:
        cap_map: Dict[str, int] = {}
        for tier_block in (caps_cfg.get("tiers") or {}).values():
            cap = int(tier_block.get("cap", 0))
            for mfg in (tier_block.get("members") or []):
                cap_map[str(mfg).strip().lower()] = cap
        long_tail_cap = int(
            ((caps_cfg.get("tiers") or {}).get("long_tail") or {}).get("cap", 10)
        )
        for mfg, n in counter.items():
            cap = cap_map.get(mfg, long_tail_cap)
            if n > cap:
                cap_violations.append({"manufacturer": mfg, "count": n, "cap": cap})
    n = len(records)
    return {
        "manufacturer_hint_coverage_floor": (
            n > 0 and (n_with_hint / n) >= _MANUFACTURER_HINT_FLOOR
        ),
        "distinct_manufacturers_floor": distinct >= _MANUFACTURER_DISTINCT_FLOOR,
        "top50_have_at_least_5_records": top50_min_records >= _MANUFACTURER_TOP50_COVERAGE,
        "no_cap_violations": not cap_violations,
        "_diagnostics": {
            "n_with_hint": n_with_hint,
            "hint_coverage": (n_with_hint / n) if n else 0,
            "distinct_count": distinct,
            "top50_min_records": top50_min_records,
            "cap_violations": cap_violations[:10],
        },
    }


def _gate_splits_and_leakage(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[Mapping[str, Any]]] = {"dev": [], "test": []}
    for r in records:
        s = str(r.get("split") or "")
        if s in by_split:
            by_split[s].append(r)
    # MPN leakage.
    def _mpn(r: Mapping[str, Any]) -> Optional[str]:
        tier2 = r.get("tier2_minimal") or {}
        return tier2.get("part_name") or None
    dev_mpns = {_mpn(r) for r in by_split["dev"] if _mpn(r)}
    test_mpns = {_mpn(r) for r in by_split["test"] if _mpn(r)}
    overlap_mpns = dev_mpns & test_mpns
    # Per-(mfg, hs4) leakage (rough product_family_hash proxy).
    def _family_key(r: Mapping[str, Any]) -> Tuple[str, str]:
        tier2 = r.get("tier2_minimal") or {}
        m = (tier2.get("manufacturer") or "").strip().lower()
        return (m, str(r.get("hs4_label") or ""))
    dev_families = {_family_key(r) for r in by_split["dev"]}
    test_families = {_family_key(r) for r in by_split["test"]}
    family_overlap = (dev_families & test_families) - {("", "")}
    # Boundary share.
    n = len(records)
    with_tag = sum(1 for r in records if r.get("difficulty_tags"))
    boundary_share = (with_tag / n) if n else 0
    return {
        "no_exact_mpn_in_both_splits": not overlap_mpns,
        "no_manufacturer_product_family_leakage": not family_overlap,
        "boundary_share_in_range": (
            _BOUNDARY_SHARE_MIN <= boundary_share <= _BOUNDARY_SHARE_MAX
        ),
        "_diagnostics": {
            "overlapping_mpns": sorted(overlap_mpns)[:20],
            "overlapping_families": sorted(family_overlap)[:20],
            "boundary_share": round(boundary_share, 4),
            "boundary_target": [_BOUNDARY_SHARE_MIN, _BOUNDARY_SHARE_MAX],
        },
    }


def _gate_hs_rebalance(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    hs6_counts = Counter(str(r.get("hs6_label") or "") for r in records)
    violations: List[Dict[str, Any]] = []
    for hs6, cap in _HS6_CAPS.items():
        if hs6_counts.get(hs6, 0) > cap:
            violations.append(
                {"hs6": hs6, "count": hs6_counts[hs6], "cap": cap}
            )
    return {
        "hs6_caps_respected": not violations,
        "_diagnostics": {"cap_violations": violations},
    }


def _gate_reference_corpus_coverage(
    benchmark_records: List[Mapping[str, Any]],
    reference_corpus: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Every HS6 with ≥1 benchmark record has ≥3 reference entries."""
    bench_hs6 = {str(r.get("hs6_label") or "") for r in benchmark_records}
    bench_hs6.discard("")
    coverage: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "EU": 0, "US": 0}
    )
    for r in reference_corpus:
        h6 = str(r.get("hs6_label") or "")
        if not h6:
            continue
        coverage[h6]["total"] += 1
        j = str(r.get("jurisdiction") or "")
        if j.startswith("EU-"):
            coverage[h6]["EU"] += 1
        elif j == "US":
            coverage[h6]["US"] += 1
    under = [
        {"hs6": h, "coverage": dict(coverage[h])}
        for h in sorted(bench_hs6)
        if coverage[h]["total"] < _COVERAGE_FLOOR
    ]
    return {
        "every_benchmark_hs6_has_3_reference_entries": not under,
        "_diagnostics": {
            "under_covered_count": len(under),
            "under_covered": under[:10],
        },
    }


def _gate_release_artifacts() -> Dict[str, Any]:
    expected = {
        "HS_INVENTORY.md": RELEASE / "HS_INVENTORY.md",
        "DATASHEET.md": RELEASE / "DATASHEET.md",
        "README.md": RELEASE / "README.md",
        "MANIFEST.json": RELEASE_DATA / "MANIFEST.json",
        "reference_corpus.jsonl": RELEASE_DATA / "reference_corpus.jsonl",
        "dossiers_index": RELEASE_DOSSIERS / "index.json",
    }
    present = {name: path.exists() for name, path in expected.items()}
    return {
        "all_required_artifacts_present": all(present.values()),
        "_diagnostics": {"presence": present},
    }


def _gate_dossier_synthesis() -> Dict[str, Any]:
    idx_path = RELEASE_DOSSIERS / "index.json"
    if not idx_path.exists():
        return {
            "every_in_scope_hs6_has_synthesis_filled": False,
            "_diagnostics": {"error": "dossiers/index.json missing"},
        }
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    n_pending = idx.get("n_synthesis_pending", 0)
    return {
        "every_in_scope_hs6_has_synthesis_filled": n_pending == 0,
        "_diagnostics": {
            "dossier_count": idx.get("dossier_count", 0),
            "n_synthesis_pending": n_pending,
            "n_below_floor": idx.get("n_below_floor", 0),
        },
    }


def _gate_calibration() -> Dict[str, Any]:
    if not CALIBRATION_REPORT_PATH.exists():
        return {
            "calibration_report_present": False,
            "_diagnostics": {"path": _rel(CALIBRATION_REPORT_PATH)},
        }
    report = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))
    accepts = report.get("acceptance", {})
    return {
        "calibration_report_present": True,
        "calibration_release_gate_passes": bool(accepts.get("passes")),
        "_diagnostics": {
            "joint_accuracy": (
                report.get("joint_authority_accuracy", {}).get("accuracy")
            ),
            "threshold": accepts.get("threshold"),
            "passes": accepts.get("passes"),
        },
    }


def run_release_gates() -> Dict[str, Any]:
    """Run the release-gate verification checklist, returning a dict of
    gate_name → {passed, diagnostics}. Skips gates whose inputs aren't on
    disk yet (e.g., dev.json absent) and marks them ``skipped``."""
    presence = verify_inputs()
    allow_hs6 = set(_scope_lookup().keys())

    dev_records: List[Mapping[str, Any]] = []
    test_records: List[Mapping[str, Any]] = []
    benchmark_records: List[Mapping[str, Any]] = []
    if presence["dev.json"]:
        dev_records = _read_json_list(RELEASE_DATA / "dev.json")
    if presence["test_frozen.json"]:
        test_records = _read_json_list(RELEASE_DATA / "test_frozen.json")
    benchmark_records = list(dev_records) + list(test_records)

    reference_corpus = _read_jsonl(RELEASE_DATA / "reference_corpus.jsonl")
    # evidence_id → jurisdiction lookup used by the evidence-coverage gate to
    # verify resolution + jurisdictional diversity of every cited id.
    evidence_jurisdiction: Dict[str, str] = {}
    for entry in reference_corpus:
        eid = entry.get("evidence_id")
        j = entry.get("jurisdiction")
        if eid and j:
            evidence_jurisdiction[str(eid)] = str(j)

    caps_cfg = _load_mapping(MANUFACTURER_CAPS_PATH) if presence["manufacturer_caps"] else None

    gates: Dict[str, Any] = {}

    if benchmark_records:
        gates["methodology_compliance"] = _gate_methodology_compliance(
            benchmark_records,
            evidence_jurisdiction=evidence_jurisdiction,
        )
        gates["scope"] = _gate_scope(benchmark_records, allow_hs6)
        gates["source_mix"] = _gate_source_mix(benchmark_records)
        gates["manufacturer"] = _gate_manufacturer(benchmark_records, caps_cfg)
        gates["splits_and_leakage"] = _gate_splits_and_leakage(benchmark_records)
        gates["hs_rebalance"] = _gate_hs_rebalance(benchmark_records)
        gates["reference_corpus_coverage"] = _gate_reference_corpus_coverage(
            benchmark_records, reference_corpus
        )
    else:
        for k in (
            "methodology_compliance", "scope", "source_mix", "manufacturer",
            "splits_and_leakage", "hs_rebalance", "reference_corpus_coverage",
        ):
            gates[k] = {
                "status": "skipped",
                "reason": "benchmark records (dev.json + test_frozen.json) not yet built",
            }

    gates["release_artifacts"] = _gate_release_artifacts()
    gates["dossier_synthesis"] = _gate_dossier_synthesis()
    gates["calibration"] = _gate_calibration()

    return gates


def _all_gates_pass(gates: Mapping[str, Any]) -> bool:
    for name, payload in gates.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "skipped":
            return False  # if a gate is skipped, release is not ready
        for k, v in payload.items():
            if k == "_diagnostics" or k == "status" or k == "reason":
                continue
            if v is False or v is None:
                return False
    return True


# ---------------------------------------------------------------------------
# Stage 4: build_manifest
# ---------------------------------------------------------------------------

def _distributions(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    if not records:
        return {}
    return {
        "hs6": dict(sorted(Counter(r.get("hs6_label") for r in records).items())),
        "hs4": dict(sorted(Counter(r.get("hs4_label") for r in records).items())),
        "hs2": dict(sorted(Counter(r.get("hs2_label") for r in records).items())),
        "tier1_source": dict(sorted(Counter(r.get("tier1_source") for r in records).items())),
        "label_source": dict(sorted(Counter(r.get("label_source") for r in records).items())),
        "scope_tier": dict(sorted(Counter(r.get("scope_tier") for r in records).items())),
        "confidence_tier": dict(sorted(Counter(r.get("confidence_tier") for r in records).items())),
        "adjudication_status": dict(sorted(
            Counter(r.get("adjudication_status") for r in records).items()
        )),
        "boundary_tags_count": sum(
            1 for r in records if r.get("difficulty_tags")
        ),
    }


def build_manifest(gates: Mapping[str, Any]) -> Dict[str, Any]:
    """Build (and return) the MANIFEST payload — release fields:
       - hs_version: HS2022
       - reference_corpus stats
       - calibration_report status
       - gate results
       - SHA-256s over the on-disk artifacts
    """
    presence = verify_inputs()
    dev = _read_json_list(RELEASE_DATA / "dev.json")
    test = _read_json_list(RELEASE_DATA / "test_frozen.json")
    ref = _read_jsonl(RELEASE_DATA / "reference_corpus.jsonl")

    hashes: Dict[str, str] = {}
    for name, path in (
        ("dev_sha256", RELEASE_DATA / "dev.json"),
        ("test_frozen_sha256", RELEASE_DATA / "test_frozen.json"),
        ("reference_corpus_sha256", RELEASE_DATA / "reference_corpus.jsonl"),
        ("record_schema_sha256", RELEASE_DATA / "record_schema.json"),
        ("reference_corpus_schema_sha256", RELEASE_DATA / "reference_corpus_schema.json"),
        ("submission_schema_sha256", RELEASE_EVAL / "submission_schema.json"),
    ):
        if path.exists():
            hashes[name] = _file_sha256(path)

    ref_by_source: Counter = Counter(str(e.get("source") or "") for e in ref)
    ref_by_juris: Counter = Counter(str(e.get("jurisdiction") or "") for e in ref)
    ref_by_hs6: Counter = Counter(str(e.get("hs6_label") or "") for e in ref)

    manifest: Dict[str, Any] = {
        "release": "working",
        "schema_version": "2.0.0",
        "hs_version": "HS2022",
        "release_built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ready_for_distribution": _all_gates_pass(gates),
        "record_count": {
            "dev": len(dev),
            "test": len(test),
            "total": len(dev) + len(test),
        },
        "reference_corpus": {
            "entries": len(ref),
            "by_source": dict(sorted(ref_by_source.items())),
            "by_jurisdiction": dict(sorted(ref_by_juris.items())),
            "hs6_codes_covered": len([h for h in ref_by_hs6 if h]),
        },
        "dev_distributions": _distributions(dev),
        "test_distributions": _distributions(test),
        "hashes": {"file_sha256": hashes},
        "presence": presence,
        "gates": gates,
    }
    return manifest


# ---------------------------------------------------------------------------
# Stage 5: generate_statistics
# ---------------------------------------------------------------------------

def generate_statistics(manifest: Mapping[str, Any]) -> Optional[Path]:
    """Generate STATISTICS.md when dev + test exist. Otherwise None."""
    dev_n = manifest["record_count"]["dev"]
    test_n = manifest["record_count"]["test"]
    if dev_n + test_n == 0:
        return None

    lines: List[str] = [
        "# SemiHS-Bench — Statistics",
        "",
        f"Total records: **{dev_n + test_n}**  (test: **{test_n}**, dev: **{dev_n}**)",
        "",
        f"HS nomenclature: **{manifest['hs_version']}**",
        "",
        "Generated by `scripts/build_release.py`. Re-run any time the "
        "benchmark changes.",
        "",
    ]

    dist = manifest.get("test_distributions") or {}
    if dist.get("label_source"):
        lines.append("## Label sources (post-expert-validation)")
        lines.append("")
        lines.append("| label_source | n |")
        lines.append("| --- | ---: |")
        for k, n in dist["label_source"].items():
            lines.append(f"| `{k}` | {n} |")
        lines.append("")

    if dist.get("hs2"):
        lines.append("## HS2 chapter coverage (test split)")
        lines.append("")
        lines.append("| chapter | records |")
        lines.append("| --- | ---: |")
        for k, n in sorted(dist["hs2"].items()):
            lines.append(f"| `{k}` | {n} |")
        lines.append("")

    if dist.get("hs4"):
        lines.append("## HS4 distribution (test split)")
        lines.append("")
        lines.append("| hs4 | records |")
        lines.append("| --- | ---: |")
        for k, n in sorted(dist["hs4"].items()):
            lines.append(f"| `{k}` | {n} |")
        lines.append("")

    if dist.get("confidence_tier"):
        lines.append("## Confidence tier distribution (test split)")
        lines.append("")
        lines.append("| confidence_tier | n |")
        lines.append("| --- | ---: |")
        for k, n in dist["confidence_tier"].items():
            lines.append(f"| `{k}` | {n} |")
        lines.append("")

    if manifest.get("reference_corpus"):
        rc = manifest["reference_corpus"]
        lines.append("## Reference corpus")
        lines.append("")
        lines.append(f"- entries: **{rc['entries']}**")
        lines.append(f"- HS6 codes covered: **{rc['hs6_codes_covered']}**")
        lines.append(f"- by source: {rc['by_source']}")
        lines.append("")

    calib = (manifest.get("gates") or {}).get("calibration", {})
    diag = calib.get("_diagnostics") or {}
    if diag.get("joint_accuracy") is not None:
        lines.append("## Rater-vs-authority calibration")
        lines.append("")
        lines.append(f"- joint authority accuracy: **{diag['joint_accuracy']:.3f}**")
        lines.append(f"- gate threshold: ≥ {diag.get('threshold')}")
        lines.append(f"- passes: {diag.get('passes')}")
        lines.append("")

    out_path = RELEASE / "STATISTICS.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Stage 6: example submission
# ---------------------------------------------------------------------------

def generate_example_submission() -> Optional[Path]:
    test_path = RELEASE_DATA / "test_frozen.json"
    if not test_path.exists():
        return None
    test = _read_json_list(test_path)[:5]
    if not test:
        return None

    predictions: List[Dict[str, Any]] = []
    for record in test:
        codes = list((record.get("candidate_set") or {}).get("codes") or [])
        gold = str(record["hs6_label"])
        if gold in codes:
            ordered = [gold] + [c for c in codes if c != gold]
        else:
            ordered = codes
        predictions.append({
            "frozen_id": record["frozen_id"],
            "ranked_codes": ordered,
        })
    submission = {
        "submission": {
            "name": "oracle-constrained-tier1.example",
            "model_id": "oracle",
            "mode": "constrained",
            "tier": 1,
            "schema_version": "2.0.0",
            "notes": "Example submission — gold-first ranking on the first 5 test records.",
        },
        "predictions": predictions,
    }
    RELEASE_EXAMPLES.mkdir(parents=True, exist_ok=True)
    out_path = RELEASE_EXAMPLES / "submission_constrained.example.json"
    out_path.write_text(
        json.dumps(submission, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


# ---------------------------------------------------------------------------
# Stage 7: tree hash
# ---------------------------------------------------------------------------

def compute_tree_hash(root: Path = RELEASE) -> str:
    """Stable SHA-256 over (rel-path, contents) of every file in root.
    Excludes __pycache__ and any .DS_Store artifacts."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name == ".DS_Store":
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel + b"\n")
        h.update(path.read_bytes())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_presence(presence: Mapping[str, bool]) -> None:
    print("Inputs:")
    for k, v in presence.items():
        mark = "✓" if v else "✗"
        print(f"  {mark} {k}")


def _print_gates(gates: Mapping[str, Any]) -> None:
    print("Release gates:")
    for name, payload in gates.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "skipped":
            print(f"  ⊘ {name}: skipped — {payload.get('reason', 'unspecified')}")
            continue
        for k, v in payload.items():
            if k == "_diagnostics" or k == "status" or k == "reason":
                continue
            mark = "✓" if v else "✗"
            print(f"  {mark} {name}.{k}: {v}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Survey what's present; no writes.",
    )
    parser.add_argument(
        "--gates-only", action="store_true",
        help="Run the release gates and exit; no writes.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    presence = verify_inputs()
    _print_presence(presence)

    if args.check:
        # Don't run validation or gates in check mode — purely inventory.
        return 0

    if not presence["reference_corpus.jsonl"] or not presence["record_schema.json"]:
        print(
            "\nERROR: required artifacts missing. "
            "Run scripts/build_reference_corpus.py and confirm "
            "release/working/data/record_schema.json exists.",
            file=sys.stderr,
        )
        return 1

    print()
    validation = validate_schemas()
    print("Schema validation:")
    for who, payload in validation.items():
        status = payload["status"]
        n_err = len(payload.get("errors") or [])
        mark = "✓" if status == "ok" else ("⊘" if status == "skipped" else "✗")
        print(f"  {mark} {who}: {status} ({n_err} errors)")

    print()
    gates = run_release_gates()
    _print_gates(gates)

    if args.gates_only:
        return 0 if _all_gates_pass(gates) else 2

    print()
    manifest = build_manifest(gates)

    manifest_path = RELEASE_DATA / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {_rel(manifest_path)}")

    stats_path = generate_statistics(manifest)
    if stats_path:
        print(f"wrote {_rel(stats_path)}")
    else:
        print("(STATISTICS.md skipped — dev/test records not on disk yet)")

    example_path = generate_example_submission()
    if example_path:
        print(f"wrote {_rel(example_path)}")
    else:
        print("(example submission skipped — test_frozen.json not on disk yet)")

    tree_hash = compute_tree_hash()
    print(f"release tree sha256: {tree_hash}")
    print(f"ready_for_distribution: {manifest['ready_for_distribution']}")

    if not manifest["ready_for_distribution"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
