#!/usr/bin/env python3
"""Normalize Japan Customs browser-captured rulings into the reference-corpus input schema.

Reads ``data/raw/jp_customs/browser_captures.jsonl`` (Japan Customs advance
rulings; raw captures are not redistributed), filters to the in-scope
HS6 allowlist, and writes records in the same schema as
``data/raw/cross/new_pulls.jsonl`` so ``scripts/build_reference_corpus.py``
can consume them uniformly with EBTI and CROSS pulls.

Pure file → file. No network calls — the browser agent did the live fetch.

Source: Japan Customs Advance Classification Rulings (関税分類事前教示制度).
License: Public Data License v1.0 (CC-BY-compatible). Attribution required.
Reference: https://www.customs.go.jp/searchsv/jitsv002.jsp
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

ROOT = Path(__file__).resolve().parents[1]
SCOPE_PATH = ROOT / "configs" / "hs6_scope_tiers.yaml"
CAPTURE_PATH = ROOT / "data" / "raw" / "jp_customs" / "browser_captures.jsonl"
OUTPUT_PATH = ROOT / "data" / "raw" / "jp_customs" / "new_pulls.jsonl"
REPORT_PATH = ROOT / "data" / "raw" / "jp_customs" / "_new_pulls_report.json"


def _load_mapping(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return json.loads(content)


def _allow_hs6(path: Path = SCOPE_PATH) -> Set[str]:
    cfg = _load_mapping(path)
    return set(cfg.get("core") or []) | set(cfg.get("supply_chain") or [])


def _read_captures(path: Path) -> List[Dict[str, Any]]:
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


def _normalize_hs6(record: Mapping[str, Any]) -> str:
    """JP rulings carry a 9-digit statistical tariff number (e.g. 8543.40-000).
    Strip non-digits and return the first 6."""
    raw = (
        record.get("classification")
        or record.get("hs_code")
        or record.get("該当する関税分類番号")
        or record.get("税番")
        or ""
    )
    digits = "".join(c for c in str(raw) if c.isdigit())
    return digits[:6]


def _resolve_text(record: Mapping[str, Any], *keys: str) -> str:
    for k in keys:
        v = record.get(k)
        if v:
            return str(v)
    return ""


def _reshape(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Reshape a JP browser-capture record into the canonical
    _reference_corpus_inputs.jsonl schema used by CROSS / EBTI."""
    registration_no = _resolve_text(record, "registration_no", "登録番号", "evidence_id_hint")
    description = _resolve_text(
        record, "tier1_text_seed", "general_product_name", "一般的品名", "description",
    )
    cargo_overview = _resolve_text(record, "貨物の概要", "cargo_overview", "details")
    if cargo_overview and cargo_overview not in description:
        description = (description + "\n\n" + cargo_overview).strip()
    rationale = _resolve_text(
        record, "rationale_excerpt", "適用根拠", "classification_basis", "justification",
    )
    ruling_date = _resolve_text(record, "ruling_date", "処理年月日", "date")
    url = _resolve_text(record, "url", "source_url", "detail_url")
    issuing_office = _resolve_text(record, "担当部署", "issuing_office", "customs_office")
    language = (record.get("language") or "ja").lower()
    keywords = list(record.get("subject_terms") or record.get("keywords") or [])
    query_term = _resolve_text(record, "query_term", "search_term")

    hs6 = _normalize_hs6(record)

    return {
        "source": "JP_CUSTOMS",
        "language": language,
        "jurisdiction": "JP",
        "evidence_id_hint": registration_no,
        "hs6_label": hs6,
        "ruling_date": ruling_date or None,
        "url": url or None,
        "tier1_text_seed": description,
        "subject_terms": keywords,
        "rationale_excerpt": rationale,
        "raw_metadata": {
            "registration_no": registration_no,
            "issuing_office": issuing_office,
            "query_term": query_term,
            "captured_at": record.get("captured_at"),
            "captured_by": record.get("captured_by"),
            "source_release": "jp_customs_browser_capture",
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-path", type=Path, default=CAPTURE_PATH,
        help="Browser-captured JSONL input (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path", type=Path, default=OUTPUT_PATH,
        help="Normalized JSONL output (default: %(default)s).",
    )
    parser.add_argument(
        "--report-path", type=Path, default=REPORT_PATH,
        help="Run report JSON output (default: %(default)s).",
    )
    parser.add_argument(
        "--scope-path", type=Path, default=SCOPE_PATH,
        help="HS6 scope config (default: %(default)s).",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Do not filter to the in-scope HS6 allowlist. Useful for diagnostics.",
    )
    args = parser.parse_args(argv)

    allow_hs6 = _allow_hs6(args.scope_path)
    print(f"in-scope HS6 count: {len(allow_hs6)}")

    captures = _read_captures(args.capture_path)
    print(f"read {len(captures)} captures from {args.capture_path}")

    reshaped: List[Dict[str, Any]] = []
    dropped_out_of_scope = 0
    dropped_no_hs6 = 0
    for rec in captures:
        out = _reshape(rec)
        if not out["hs6_label"]:
            dropped_no_hs6 += 1
            continue
        if not args.no_filter and out["hs6_label"] not in allow_hs6:
            dropped_out_of_scope += 1
            continue
        reshaped.append(out)

    # Dedup by (source, evidence_id_hint).
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for r in reshaped:
        key = f"{r['source']}::{r['evidence_id_hint']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    deduped.sort(key=lambda x: (x["hs6_label"], x["evidence_id_hint"]))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {args.output_path} ({len(deduped)} records)")

    per_hs6: Counter = Counter()
    per_hs4: Counter = Counter()
    for r in deduped:
        per_hs6[r["hs6_label"]] += 1
        per_hs4[r["hs6_label"][:4]] += 1

    report = {
        "release": "working",
        "source": "JP_CUSTOMS",
        "capture_path": str(args.capture_path),
        "scope_config": str(args.scope_path),
        "captures_read": len(captures),
        "dropped_no_hs6": dropped_no_hs6,
        "dropped_out_of_scope": dropped_out_of_scope,
        "deduped_emitted": len(deduped),
        "per_hs4": dict(sorted(per_hs4.items())),
        "per_hs6": dict(sorted(per_hs6.items())),
        "output": str(args.output_path),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
