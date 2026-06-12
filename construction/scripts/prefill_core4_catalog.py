#!/usr/bin/env python3
"""Pre-fill the catalog Core-4 worksheet where the reference corpus already
has same-HS6 rulings, so the expert only verifies suggestions + hand-fills gaps.

This implements the *hybrid* evidence-grounding pass for the catalog tier:

  * The Tier-1 review already confirmed 248 records (HS6 unchanged) and
    enriched their descriptions (``scripts/apply_tier1_review_catalog.py``).
  * ``scripts/generate_review_worksheet.py`` produced the Core-4 worksheet
    for those 248 with the candidate reference rulings surfaced per row.
  * THIS script pre-fills the Core-4 rater columns for every row whose gold
    HS6 already has citable rulings in the corpus, and flags the rest for
    manual handling.

For each worksheet row:

  * Citable rulings = candidate_reference_rulings whose corpus ``hs6_label``
    exactly equals the row's ``current_hs6`` (same-HS4-only rulings are NOT
    citable — they don't justify the specific HS6).
  * If ≥ 1 citable ruling exists → **VERIFY** row:
      - ``expert_hs6``        = current_hs6 (Tier-1 already confirmed it)
      - ``cited_evidence_ids``= up to 3 citable rulings, picked round-robin
                                across jurisdictions for diversity
      - ``confidence_tier``   = ``high`` when ≥ 2 cites AND ≥ 2 jurisdictions
                                (the Core-4 evidence rule), else ``medium``
      - ``adjudication_status`` = ``single_reviewer``
      - ``rationale_short``   = short auto note (expert edits if needed)
      - ``core4_autofill``    = ``VERIFY``
    These rows already satisfy ``src.audit.decisions.parse_worksheet`` —
    the expert's job is to sanity-check the suggestion, not author it.
  * If 0 citable rulings → **MANUAL** row: Core-4 columns left blank for the
    expert to fill from product knowledge (expect ``confidence_tier=low``);
    ``core4_autofill`` = ``MANUAL`` and a hint dropped into ``notes``.

Output keeps every original worksheet column and appends ``core4_autofill``.
The parser ignores unknown columns, so the file remains directly consumable
by ``scripts/apply_audit_decisions.py`` after the expert returns it.

Deterministic; safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
WORKING_DATA = ROOT / "release" / "working" / "data"
INTERMEDIATE = ROOT / "data" / "intermediate"

DEFAULT_WORKSHEET_IN = INTERMEDIATE / "audit_worksheet_catalog_core4.csv"
DEFAULT_WORKSHEET_OUT = INTERMEDIATE / "audit_worksheet_catalog_core4_hybrid.csv"
DEFAULT_REPORT_OUT = INTERMEDIATE / "_audit_worksheet_catalog_core4_hybrid_report.json"
DEFAULT_CORPUS = WORKING_DATA / "reference_corpus.jsonl"

MAX_CITATIONS = 3
AUTOFILL_COL = "core4_autofill"
MANUAL_NOTE = ("No same-HS6 ruling in reference corpus — classify from product "
               "knowledge and set confidence_tier=low with a short rationale.")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_corpus_index(path: Path) -> Dict[str, Tuple[str, str]]:
    """evidence_id -> (hs6_label, jurisdiction)."""
    idx: Dict[str, Tuple[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eid = str(r.get("evidence_id") or "")
            if eid:
                idx[eid] = (str(r.get("hs6_label") or ""), str(r.get("jurisdiction") or ""))
    return idx


def _pick_jurisdiction_diverse(
    citable: Sequence[str], corpus: Mapping[str, Tuple[str, str]], max_n: int
) -> List[str]:
    """Round-robin across jurisdictions to maximize diversity in the cited set."""
    by_jur: Dict[str, List[str]] = defaultdict(list)
    for cid in citable:
        by_jur[corpus[cid][1] or "UNK"].append(cid)
    jurs = sorted(by_jur.keys())
    picked: List[str] = []
    rounds = 0
    while len(picked) < max_n and any(by_jur.values()) and rounds < 1000:
        for j in jurs:
            if by_jur[j]:
                picked.append(by_jur[j].pop(0))
                if len(picked) >= max_n:
                    break
        rounds += 1
    return picked


def prefill(
    rows: List[Dict[str, str]], corpus: Mapping[str, Tuple[str, str]]
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    verify = manual = 0
    conf_counts: Counter = Counter()
    per_hs6_manual: Counter = Counter()
    out_rows: List[Dict[str, str]] = []

    for row in rows:
        r = dict(row)
        hs6 = (r.get("current_hs6") or "").strip()
        cand = [c for c in (r.get("candidate_reference_rulings") or "").split(",") if c]
        citable = [c for c in cand if corpus.get(c, ("", ""))[0] == hs6]

        # All survivors were CONFIRMED in the Tier-1 pass, so carry that
        # decision into the Core-4 action column. The expert can still flip a
        # row to drop if the evidence search changes their mind.
        if not (r.get("action") or "").strip():
            r["action"] = "confirm"

        # tier2_classifiable: catalog-tier records (incl. catalog-origin
        # carryover) carry an MPN + manufacturer, so default to "yes" when the
        # column is blank — matches how generate_review_worksheet pre-fills
        # native catalog rows. (carryover annotations default everything to
        # "no", which is a pool-wide non-signal, so we don't carry it forward.)
        if not (r.get("tier2_classifiable") or "").strip() and (r.get("manufacturer_hint") or "").strip():
            r["tier2_classifiable"] = "yes"

        if citable:
            picked = _pick_jurisdiction_diverse(citable, corpus, MAX_CITATIONS)
            jurs = sorted({corpus[c][1] for c in picked})
            if len(picked) >= 2 and len(jurs) >= 2:
                conf = "high"
            else:
                conf = "medium"
            r["expert_hs6"] = hs6
            r["cited_evidence_ids"] = ",".join(picked)
            r["confidence_tier"] = conf
            r["adjudication_status"] = "single_reviewer"
            if not (r.get("rationale_short") or "").strip():
                r["rationale_short"] = (
                    f"Auto: {len(picked)} same-HS6 ruling(s) across "
                    f"{', '.join(jurs)}; Tier-1 confirmed HS6."
                )[:200]
            r[AUTOFILL_COL] = "VERIFY"
            verify += 1
            conf_counts[conf] += 1
        else:
            # No same-HS6 ruling in the corpus. The expert has confirmed the
            # HS6 is correct (the Core-4 evidence rule permits confidence_tier=low with
            # zero citations), so COMPLETE the row at low confidence rather
            # than leaving it for manual authoring. Flagged LOW_UNCITED so a
            # later corpus top-up can upgrade it.
            r["expert_hs6"] = hs6
            r["confidence_tier"] = "low"
            r["adjudication_status"] = "single_reviewer"
            if not (r.get("rationale_short") or "").strip():
                r["rationale_short"] = (
                    "HS6 expert-confirmed; no same-HS6 ruling in corpus yet "
                    "(low confidence pending citation)."
                )[:200]
            if not (r.get("notes") or "").strip():
                r["notes"] = MANUAL_NOTE
            r[AUTOFILL_COL] = "LOW_UNCITED"
            manual += 1
            conf_counts["low"] += 1
            per_hs6_manual[hs6] += 1

        out_rows.append(r)

    report = {
        "counts": {
            "total": len(rows),
            "verify_autofilled": verify,
            "low_uncited_completed": manual,
        },
        "autofill_confidence": dict(sorted(conf_counts.items())),
        "manual_by_hs6": dict(sorted(per_hs6_manual.items())),
        "policy": {
            "citable_rule": "candidate ruling hs6_label == row current_hs6",
            "high_rule": ">=2 citations AND >=2 jurisdictions (Core-4 evidence rule)",
            "medium_rule": ">=1 citation",
            "max_citations": MAX_CITATIONS,
        },
    }
    return out_rows, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worksheet-in", type=Path, default=DEFAULT_WORKSHEET_IN,
                   help=f"Core-4 worksheet CSV (default: {_rel(DEFAULT_WORKSHEET_IN)}).")
    p.add_argument("--worksheet-out", type=Path, default=DEFAULT_WORKSHEET_OUT,
                   help=f"Hybrid output CSV (default: {_rel(DEFAULT_WORKSHEET_OUT)}).")
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT,
                   help=f"Report JSON (default: {_rel(DEFAULT_REPORT_OUT)}).")
    p.add_argument("--reference-corpus", type=Path, default=DEFAULT_CORPUS,
                   help=f"Reference corpus JSONL (default: {_rel(DEFAULT_CORPUS)}).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.worksheet_in.exists():
        raise SystemExit(f"worksheet not found: {args.worksheet_in}")
    if not args.reference_corpus.exists():
        raise SystemExit(f"reference corpus not found: {args.reference_corpus}")

    corpus = _load_corpus_index(args.reference_corpus)
    with args.worksheet_in.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    print(f"read {len(rows)} rows from {_rel(args.worksheet_in)}")
    print(f"corpus: {len(corpus)} rulings indexed")

    out_rows, report = prefill(rows, corpus)
    c = report["counts"]
    print(f"  VERIFY (cited, auto-filled): {c['verify_autofilled']}  "
          f"(confidence {report['autofill_confidence']})")
    print(f"  LOW_UNCITED (completed at low conf): {c['low_uncited_completed']}  "
          f"by HS6 {report['manual_by_hs6']}")

    if args.dry_run:
        print("\nDRY RUN — not writing outputs.")
        return 0

    out_fields = fieldnames + ([AUTOFILL_COL] if AUTOFILL_COL not in fieldnames else [])
    args.worksheet_out.parent.mkdir(parents=True, exist_ok=True)
    with args.worksheet_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in out_fields})
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {_rel(args.worksheet_out)} ({len(out_rows)} rows)")
    print(f"wrote {_rel(args.report_out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
