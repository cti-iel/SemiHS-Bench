"""Rater-vs-authority calibration scorer for the SemiHS-Bench calibration stratum.

Reads two raters' completed calibration CSVs plus the held-out truth table
emitted by ``scripts/build_calibration_set.py`` and computes the metrics
specified in
the authority-calibration protocol (docs/IAA_PROTOCOL.md §7)
and [IAA_PROTOCOL.md §7.4](../../docs/IAA_PROTOCOL.md#74-metrics):

* Per-rater authority accuracy (rater_hs6 == authoritative_hs6).
* Joint authority accuracy (both raters correct on the same record).
* Per-HS4 stratified accuracy.
* Inter-rater Cohen's κ on proposed HS6.
* Confidence calibration (high-tier proposals should reach authority ≥ 85%).
* Citation usefulness (did the cited evidence include the actual authoritative
  ruling? did it include any other ruling at the same HS6?).
* Per-language and per-source breakdowns.

All metrics ship with non-parametric paired bootstrap 95% CIs (n=1000)
using ``src/annotation/iaa_report._paired_bootstrap_ci`` for continuity
with the IAA reporting module.

The §7.5 acceptance gate (joint authority accuracy ≥ 70%) is evaluated
and surfaced in the report's ``acceptance`` section.

Stdlib only — same constraint as ``iaa_report.py``.

CLI:

    python -m src.annotation.authority_calibration \\
        --rater-a data/intermediate/calibration_annotated_rater_a.csv \\
        --rater-b data/intermediate/calibration_annotated_rater_b.csv \\
        --truth data/intermediate/_calibration_truth.jsonl \\
        --out data/intermediate/calibration_report.json

Or with defaults (assumes standard project layout):

    python -m src.annotation.authority_calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# Reuse the bootstrap CI + Cohen's κ helpers from the IAA module.
from src.annotation.iaa_report import _paired_bootstrap_ci, cohens_kappa

ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_RELEASE = "working"
_INTERMEDIATE = ROOT / "data" / "intermediate"
_DEFAULT_TRUTH = _INTERMEDIATE / "_calibration_truth.jsonl"
_DEFAULT_REF_CORPUS = ROOT.parent / "data" / "reference_corpus.jsonl"
_DEFAULT_RATER_A = _INTERMEDIATE / "calibration_annotated_rater_a.csv"
_DEFAULT_RATER_B = _INTERMEDIATE / "calibration_annotated_rater_b.csv"
_DEFAULT_OUT = _INTERMEDIATE / "calibration_report.json"

# Joint authority accuracy ≥ 0.70 is the release gate.
_ACCEPTANCE_THRESHOLD_JOINT = 0.70
# Per IAA_PROTOCOL.md §7.4: among high-confidence proposals, accuracy ≥ 0.85.
_HIGH_CONF_ACCURACY_TARGET = 0.85

_HS6_RE = re.compile(r"^[0-9]{6}$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_rater_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """Return rows keyed by calibration_id."""
    out: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("calibration_id") or "").strip()
            if not cid:
                continue
            out[cid] = row
    return out


def _ref_corpus_hs6_lookup(path: Path) -> Dict[str, str]:
    """Map ``evidence_id → hs6_label`` for citation analysis."""
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for r in _read_jsonl(path):
        eid = r.get("evidence_id")
        hs6 = r.get("hs6_label")
        if eid and hs6:
            out[str(eid)] = str(hs6)
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_hs6(raw: str) -> Tuple[str, str]:
    """Return (canonical_hs6, status).

    status ∈ {"ok", "blank", "malformed_4", "malformed_other"}.

    Accepts:
        - "" / whitespace                  → ("", "blank")
        - "854231"                         → ("854231", "ok")
        - "8542.31" / "8542 31" / "8542-31"→ ("854231", "ok")
        - "8542"                           → ("8542", "malformed_4")
        - anything else                    → ("", "malformed_other")
    """
    s = (raw or "").strip()
    if not s:
        return ("", "blank")
    digits = re.sub(r"\D", "", s)
    if len(digits) == 6:
        return (digits, "ok")
    if len(digits) == 4:
        return (digits, "malformed_4")
    if len(digits) >= 6:
        # HTSUS may be 8-10 digits; take first 6.
        return (digits[:6], "ok")
    return ("", "malformed_other")


def _parse_cited(raw: str) -> List[str]:
    s = (raw or "").strip()
    if not s:
        return []
    return [tok.strip() for tok in re.split(r"[,;]", s) if tok.strip()]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _accuracy(pairs: Sequence[Tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def _joint_accuracy(triples: Sequence[Tuple[str, str, str]]) -> float:
    """triples = [(authority, rater_a, rater_b), ...]. Both must equal authority."""
    if not triples:
        return float("nan")
    return sum(
        1 for auth, a, b in triples if a == auth and b == auth
    ) / len(triples)


def _bool_rate(values: Sequence[bool]) -> float:
    if not values:
        return float("nan")
    return sum(1 for v in values if v) / len(values)


def _ci_for_accuracy(
    pairs: Sequence[Tuple[str, str]], n_boot: int = 1000, seed: int = 17
) -> Tuple[float, float]:
    return _paired_bootstrap_ci(pairs, _accuracy, n_boot=n_boot, seed=seed)


def _ci_for_joint(
    triples: Sequence[Tuple[str, str, str]], n_boot: int = 1000, seed: int = 17
) -> Tuple[float, float]:
    return _paired_bootstrap_ci(triples, _joint_accuracy, n_boot=n_boot, seed=seed)


def _ci_for_bool(
    flags: Sequence[bool], n_boot: int = 1000, seed: int = 17
) -> Tuple[float, float]:
    # Wrap as singleton tuples so _paired_bootstrap_ci can resample.
    wrapped = [(v,) for v in flags]
    return _paired_bootstrap_ci(
        wrapped,
        lambda resample: _bool_rate([t[0] for t in resample]),
        n_boot=n_boot,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def _stratified_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    stratify_key: str,
    rater_key: str,
    n_boot: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-stratum accuracy + CI for a given rater_key column."""
    by_stratum: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in rows:
        s = str(r.get(stratify_key) or "")
        by_stratum[s].append((r["authoritative_hs6"], r[rater_key]))
    out: Dict[str, Dict[str, Any]] = {}
    for s, pairs in sorted(by_stratum.items()):
        acc = _accuracy(pairs)
        lo, hi = _ci_for_accuracy(pairs, n_boot=n_boot)
        out[s] = {"n": len(pairs), "accuracy": acc, "ci95": [lo, hi]}
    return out


def score_calibration(
    truth_path: Path,
    rater_a_path: Path,
    rater_b_path: Path,
    *,
    ref_corpus_path: Optional[Path] = None,
    n_boot: int = 1000,
    seed: int = 17,
) -> Dict[str, Any]:
    truth_records = _read_jsonl(truth_path)
    rater_a = _read_rater_csv(rater_a_path)
    rater_b = _read_rater_csv(rater_b_path)
    corpus_hs6 = (
        _ref_corpus_hs6_lookup(ref_corpus_path) if ref_corpus_path else {}
    )

    # Join by calibration_id.
    joined: List[Dict[str, Any]] = []
    rater_a_missing: List[str] = []
    rater_b_missing: List[str] = []
    truth_orphans: List[str] = []
    for t in truth_records:
        cid = t["calibration_id"]
        a = rater_a.get(cid)
        b = rater_b.get(cid)
        if a is None:
            rater_a_missing.append(cid)
        if b is None:
            rater_b_missing.append(cid)
        if a is None or b is None:
            continue
        joined.append({
            "calibration_id": cid,
            "authoritative_hs6": t["authoritative_hs6"],
            "authoritative_hs4": t.get("authoritative_hs4") or t["authoritative_hs6"][:4],
            "source": t.get("source") or "",
            "jurisdiction": t.get("jurisdiction") or "",
            "language": t.get("language") or "",
            "authoritative_evidence_id": t.get("evidence_id") or "",
            "rater_a": a,
            "rater_b": b,
        })
    # Records present in rater files but not in truth (defensive).
    truth_ids = {t["calibration_id"] for t in truth_records}
    for cid in rater_a:
        if cid not in truth_ids:
            truth_orphans.append(cid)
    for cid in rater_b:
        if cid not in truth_ids:
            truth_orphans.append(cid)
    truth_orphans = sorted(set(truth_orphans))

    # Normalize rater HS6 + record format issues.
    n_norm: Dict[str, Counter] = {"rater_a": Counter(), "rater_b": Counter()}
    for row in joined:
        for who in ("rater_a", "rater_b"):
            raw = row[who].get("rater_hs6", "")
            canon, status = _normalize_hs6(raw)
            row[f"{who}_hs6"] = canon
            row[f"{who}_hs6_status"] = status
            n_norm[who][status] += 1

    # ---------- Per-rater authority accuracy ----------
    pairs_a = [(r["authoritative_hs6"], r["rater_a_hs6"]) for r in joined]
    pairs_b = [(r["authoritative_hs6"], r["rater_b_hs6"]) for r in joined]
    acc_a = _accuracy(pairs_a)
    acc_b = _accuracy(pairs_b)
    ci_a = _ci_for_accuracy(pairs_a, n_boot=n_boot, seed=seed)
    ci_b = _ci_for_accuracy(pairs_b, n_boot=n_boot, seed=seed)

    # ---------- Joint authority accuracy ----------
    triples = [
        (r["authoritative_hs6"], r["rater_a_hs6"], r["rater_b_hs6"])
        for r in joined
    ]
    joint = _joint_accuracy(triples)
    ci_joint = _ci_for_joint(triples, n_boot=n_boot, seed=seed)

    # ---------- Inter-rater Cohen's κ on proposed HS6 ----------
    rater_pairs = [(r["rater_a_hs6"], r["rater_b_hs6"]) for r in joined]
    kappa = cohens_kappa(rater_pairs)
    kappa_ci = _paired_bootstrap_ci(rater_pairs, cohens_kappa, n_boot=n_boot, seed=seed)

    # ---------- Stratified accuracy ----------
    per_hs4_a = _stratified_accuracy(joined, stratify_key="authoritative_hs4",
                                     rater_key="rater_a_hs6", n_boot=n_boot)
    per_hs4_b = _stratified_accuracy(joined, stratify_key="authoritative_hs4",
                                     rater_key="rater_b_hs6", n_boot=n_boot)
    per_source_a = _stratified_accuracy(joined, stratify_key="source",
                                        rater_key="rater_a_hs6", n_boot=n_boot)
    per_source_b = _stratified_accuracy(joined, stratify_key="source",
                                        rater_key="rater_b_hs6", n_boot=n_boot)
    per_language_a = _stratified_accuracy(joined, stratify_key="language",
                                          rater_key="rater_a_hs6", n_boot=n_boot)
    per_language_b = _stratified_accuracy(joined, stratify_key="language",
                                          rater_key="rater_b_hs6", n_boot=n_boot)

    # ---------- Confidence calibration ----------
    # Among rater_confidence_tier=high proposals, what fraction reach authority?
    def _high_pairs(who: str) -> List[Tuple[str, str]]:
        return [
            (r["authoritative_hs6"], r[f"{who}_hs6"])
            for r in joined
            if (r[who].get("rater_confidence_tier") or "").strip().lower() == "high"
        ]

    high_a = _high_pairs("rater_a")
    high_b = _high_pairs("rater_b")
    conf_calib_a = {
        "n_high": len(high_a),
        "accuracy_within_high": _accuracy(high_a) if high_a else None,
        "ci95": list(_ci_for_accuracy(high_a, n_boot=n_boot, seed=seed)) if high_a else None,
        "target": _HIGH_CONF_ACCURACY_TARGET,
        "meets_target": (
            (_accuracy(high_a) >= _HIGH_CONF_ACCURACY_TARGET) if high_a else None
        ),
    }
    conf_calib_b = {
        "n_high": len(high_b),
        "accuracy_within_high": _accuracy(high_b) if high_b else None,
        "ci95": list(_ci_for_accuracy(high_b, n_boot=n_boot, seed=seed)) if high_b else None,
        "target": _HIGH_CONF_ACCURACY_TARGET,
        "meets_target": (
            (_accuracy(high_b) >= _HIGH_CONF_ACCURACY_TARGET) if high_b else None
        ),
    }

    # ---------- Citation usefulness ----------
    # For each rater, among records where the rater reached authority, did
    # they cite (a) the actual authoritative evidence_id and (b) any other
    # ruling at the same authoritative HS6 (from the reference corpus)?
    def _citation_metrics(who: str) -> Dict[str, Any]:
        reached_records = [r for r in joined if r[f"{who}_hs6"] == r["authoritative_hs6"]]
        peek_flags: List[bool] = []
        same_hs6_flags: List[bool] = []
        for r in reached_records:
            cited = _parse_cited(r[who].get("rater_cited_evidence_ids", ""))
            auth_eid = r["authoritative_evidence_id"]
            peek_flags.append(auth_eid in cited)
            same_hs6 = False
            for c in cited:
                if c == auth_eid:
                    continue
                if corpus_hs6.get(c) == r["authoritative_hs6"]:
                    same_hs6 = True
                    break
            same_hs6_flags.append(same_hs6)
        return {
            "n_reached_authority": len(reached_records),
            "cited_authoritative_ruling_rate": _bool_rate(peek_flags) if peek_flags else None,
            "cited_authoritative_ci95": (
                list(_ci_for_bool(peek_flags, n_boot=n_boot, seed=seed))
                if peek_flags else None
            ),
            "cited_same_hs6_other_ruling_rate": _bool_rate(same_hs6_flags) if same_hs6_flags else None,
            "cited_same_hs6_other_ruling_ci95": (
                list(_ci_for_bool(same_hs6_flags, n_boot=n_boot, seed=seed))
                if same_hs6_flags else None
            ),
        }

    citation_a = _citation_metrics("rater_a")
    citation_b = _citation_metrics("rater_b")

    # ---------- Format issues summary ----------
    format_issues = {
        "rater_a": {
            "ok": n_norm["rater_a"].get("ok", 0),
            "blank": n_norm["rater_a"].get("blank", 0),
            "malformed_4_digit": n_norm["rater_a"].get("malformed_4", 0),
            "malformed_other": n_norm["rater_a"].get("malformed_other", 0),
        },
        "rater_b": {
            "ok": n_norm["rater_b"].get("ok", 0),
            "blank": n_norm["rater_b"].get("blank", 0),
            "malformed_4_digit": n_norm["rater_b"].get("malformed_4", 0),
            "malformed_other": n_norm["rater_b"].get("malformed_other", 0),
        },
    }

    # ---------- Acceptance gate ----------
    # The calibration must clear two checks:
    #   1. joint_authority_accuracy >= 0.70 (the headline ratio)
    #   2. COMPLETENESS: both raters submitted every calibration record,
    #      with a parseable rater_hs6 on every row.
    # Without the second check, partial submissions can spuriously pass
    # the gate — e.g., rater A submits only 30 of 60 rows with both
    # raters correct on those 30 ⇒ "joint=1.0" even though half the
    # set was never rated.
    n_truth = len(truth_records)
    n_joined = len(joined)
    n_rater_a_missing = len(rater_a_missing)
    n_rater_b_missing = len(rater_b_missing)
    n_blank_a = format_issues["rater_a"]["blank"]
    n_blank_b = format_issues["rater_b"]["blank"]
    completeness_complete = (
        n_truth > 0
        and n_joined == n_truth
        and n_rater_a_missing == 0
        and n_rater_b_missing == 0
        and n_blank_a == 0
        and n_blank_b == 0
    )
    accuracy_pass = (joint == joint) and (joint >= _ACCEPTANCE_THRESHOLD_JOINT)
    acceptance = {
        "release_gate": (
            "joint_authority_accuracy >= 0.70 AND complete coverage of all "
            "truth records by both raters (calibration protocol)"
        ),
        "joint_authority_accuracy": joint,
        "threshold": _ACCEPTANCE_THRESHOLD_JOINT,
        "accuracy_pass": accuracy_pass,
        "completeness": {
            "truth_records": n_truth,
            "joined_records": n_joined,
            "rater_a_missing_count": n_rater_a_missing,
            "rater_b_missing_count": n_rater_b_missing,
            "rater_a_blank_hs6_count": n_blank_a,
            "rater_b_blank_hs6_count": n_blank_b,
            "complete": completeness_complete,
        },
        "passes": accuracy_pass and completeness_complete,
    }

    return {
        "release": _DEFAULT_RELEASE,
        "scored_at": "build_time",
        "inputs": {
            "truth_path": str(truth_path),
            "rater_a_path": str(rater_a_path),
            "rater_b_path": str(rater_b_path),
            "reference_corpus_path": str(ref_corpus_path) if ref_corpus_path else None,
        },
        "n_truth_records": len(truth_records),
        "n_joined_records": len(joined),
        "missing": {
            "rater_a_missing_calibration_ids": rater_a_missing,
            "rater_b_missing_calibration_ids": rater_b_missing,
            "rater_csv_ids_not_in_truth": truth_orphans,
        },
        "format_issues": format_issues,
        "per_rater_authority_accuracy": {
            "rater_a": {"accuracy": acc_a, "ci95": [ci_a[0], ci_a[1]]},
            "rater_b": {"accuracy": acc_b, "ci95": [ci_b[0], ci_b[1]]},
        },
        "joint_authority_accuracy": {
            "accuracy": joint,
            "ci95": [ci_joint[0], ci_joint[1]],
        },
        "inter_rater_kappa": {
            "metric": "cohens_kappa_on_proposed_hs6",
            "kappa": kappa,
            "ci95": [kappa_ci[0], kappa_ci[1]],
        },
        "stratified_accuracy": {
            "by_hs4": {"rater_a": per_hs4_a, "rater_b": per_hs4_b},
            "by_source": {"rater_a": per_source_a, "rater_b": per_source_b},
            "by_language": {"rater_a": per_language_a, "rater_b": per_language_b},
        },
        "confidence_calibration": {
            "rater_a": conf_calib_a,
            "rater_b": conf_calib_b,
        },
        "citation_usefulness": {
            "rater_a": citation_a,
            "rater_b": citation_b,
        },
        "acceptance": acceptance,
        "bootstrap_iterations": n_boot,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score the rater-vs-authority calibration set. "
            "Reads two raters' completed CSVs + the held-out truth table "
            "and writes calibration_report.json."
        )
    )
    parser.add_argument("--release", default=_DEFAULT_RELEASE)
    parser.add_argument("--truth", type=Path, default=_DEFAULT_TRUTH)
    parser.add_argument("--rater-a", type=Path, default=_DEFAULT_RATER_A)
    parser.add_argument("--rater-b", type=Path, default=_DEFAULT_RATER_B)
    parser.add_argument("--reference-corpus", type=Path, default=_DEFAULT_REF_CORPUS,
                        help="Used for citation-usefulness analysis (evidence_id → HS6 lookup).")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    for p in (args.truth, args.rater_a, args.rater_b):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 1

    report = score_calibration(
        truth_path=args.truth,
        rater_a_path=args.rater_a,
        rater_b_path=args.rater_b,
        ref_corpus_path=args.reference_corpus if args.reference_corpus.exists() else None,
        n_boot=args.bootstrap_iterations,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Console summary.
    acc_a = report["per_rater_authority_accuracy"]["rater_a"]["accuracy"]
    acc_b = report["per_rater_authority_accuracy"]["rater_b"]["accuracy"]
    joint = report["joint_authority_accuracy"]["accuracy"]
    kappa = report["inter_rater_kappa"]["kappa"]
    passes = report["acceptance"]["passes"]

    print(f"Calibration scored: n={report['n_joined_records']}")
    print(f"  Rater A accuracy:  {acc_a:.3f}")
    print(f"  Rater B accuracy:  {acc_b:.3f}")
    print(f"  Joint accuracy:    {joint:.3f}  "
          f"(gate ≥ {_ACCEPTANCE_THRESHOLD_JOINT}: {'PASS' if passes else 'FAIL'})")
    print(f"  Inter-rater κ:     {kappa:.3f}")
    print(f"  -> {args.out}")
    return 0 if passes else 2  # non-zero exit when below acceptance gate


if __name__ == "__main__":
    sys.exit(main())
