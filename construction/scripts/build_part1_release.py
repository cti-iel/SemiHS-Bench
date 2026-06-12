#!/usr/bin/env python3
"""Combine the catalog + BOL interim slices into one benchmark: **part1**.

Sources the two already-built interim packages (their final tier-shaped,
schema-valid records):
  * release/catalog-interim/  — 270 records (catalog tier)
  * release/bol-interim/      — 573 records (BOL tier)
→ release/part1/ (843 records).

Decisions:
  * **Splits preserved** — a record keeps the dev/test assignment it had in its
    interim package (catalog dev + BOL dev → dev; catalog test + BOL test →
    test). Nothing already shared moves split.
  * **IDs renumbered uniformly** — the two interim packages each used
    ``SH-NNNN`` independently and collide, so the union is renumbered to a
    single ``SH-NNNN`` / ``v2.0.{dev,test}.NNNN`` sequence. The interim-stage
    ``source_frozen_id`` is preserved; ``tier1_source`` (catalog|BOL)
    disambiguates origin.
  * No tier reconstruction — records are already in their agreed tier shapes.

Deterministic; re-running overwrites the package.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._release_utils import (  # noqa: E402
    _sha256, _git_commit, _dist, _write_corpus_subset,
)

WORKING = ROOT / "release" / "working"
WORKING_DATA = WORKING / "data"
CATALOG_PKG = ROOT / "release" / "catalog-interim"
BOL_PKG = ROOT / "release" / "bol-interim"
RECORD_SCHEMA = WORKING_DATA / "record_schema.json"
REF_CORPUS_SCHEMA = WORKING_DATA / "reference_corpus_schema.json"

OUT = ROOT / "release" / "part1"
OUT_DATA = OUT / "data"
RELEASE_LABEL = "part1"
SCHEMA_VERSION = "2.0.0"


def _load(pkg: Path):
    dev = json.loads((pkg / "data" / "dev.json").read_text(encoding="utf-8"))
    test = json.loads((pkg / "data" / "test_frozen.json").read_text(encoding="utf-8"))
    return dev, test


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cat_dev, cat_test = _load(CATALOG_PKG)
    bol_dev, bol_test = _load(BOL_PKG)
    print(f"catalog: dev={len(cat_dev)} test={len(cat_test)} | "
          f"bol: dev={len(bol_dev)} test={len(bol_test)}")

    dev = cat_dev + bol_dev
    test = cat_test + bol_test
    combined = dev + test
    print(f"combined: dev={len(dev)} test={len(test)} total={len(combined)}")

    # Renumber ids uniformly across the union (deterministic order). frozen_id
    # reassigned per preserved split below.
    combined.sort(key=lambda r: (str(r.get("hs4_label") or ""),
                                  str(r.get("hs6_label") or ""),
                                  str(r.get("tier1_source") or ""),
                                  str(r.get("source_frozen_id") or "")))
    for n, r in enumerate(combined, start=1):
        r["id"] = f"SH-{n:04d}"

    dev_set = {id(r) for r in dev}
    dev_out = [r for r in combined if id(r) in dev_set]
    test_out = [r for r in combined if id(r) not in dev_set]
    dev_out.sort(key=lambda r: r["id"]); test_out.sort(key=lambda r: r["id"])
    for i, r in enumerate(dev_out, start=1):
        r["split"] = "dev"; r["frozen_id"] = f"v2.0.dev.{i:04d}"
    for i, r in enumerate(test_out, start=1):
        r["split"] = "test"; r["frozen_id"] = f"v2.0.test.{i:04d}"

    # schema validation
    schema = json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
    import jsonschema
    v = jsonschema.Draft202012Validator(schema)
    errs = [f"{r['id']}: {e.message}" for r in combined for e in v.iter_errors(r)]
    if errs:
        print(f"✗ {len(errs)} schema errors (first 12):", file=sys.stderr)
        for e in errs[:12]:
            print("   -", e, file=sys.stderr)
        raise SystemExit(2)
    print(f"✓ all {len(combined)} records pass record_schema validation")

    if args.dry_run:
        print("DRY RUN — no writes.")
        return 0

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    dev_path = OUT_DATA / "dev.json"; test_path = OUT_DATA / "test_frozen.json"
    dev_path.write_text(json.dumps(dev_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    corpus_info = _write_corpus_subset(combined, OUT_DATA / "reference_corpus.jsonl")
    if corpus_info["unresolved_citations"]:
        raise SystemExit(f"unresolved citations: {corpus_info['unresolved_citations'][:5]}")
    print(f"corpus subset: {corpus_info['entries']} entries ({corpus_info['by_source']})")

    shutil.copy(RECORD_SCHEMA, OUT_DATA / "record_schema.json")
    shutil.copy(REF_CORPUS_SCHEMA, OUT_DATA / "reference_corpus_schema.json")
    shutil.copytree(WORKING / "eval", OUT / "eval")
    shutil.copytree(WORKING / "prompts", OUT / "prompts")

    hs_rows = sorted({(str(r["hs6_label"]), str(r["hs4_label"]), str(r["hs2_label"]),
                       str(r.get("scope_tier") or "")) for r in combined})
    with (OUT_DATA / "taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["hs6", "hs4", "hs2", "scope_tier"]); w.writerows(hs_rows)

    example = {
        "submission": {"name": "oracle-constrained-tier1.example", "model_id": "oracle",
                       "mode": "constrained", "tier": 1, "schema_version": SCHEMA_VERSION,
                       "notes": "Example — gold-first ranking on the first 5 test records (part1)."},
        "predictions": [],
    }
    for r in sorted(test_out, key=lambda x: x["frozen_id"])[:5]:
        codes = list((r.get("candidate_set") or {}).get("codes") or [])
        gold = str(r["hs6_label"])
        example["predictions"].append({"frozen_id": r["frozen_id"],
                                       "ranked_codes": [gold] + [c for c in codes if c != gold]})
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    (OUT / "examples" / "submission_constrained.example.json").write_text(
        json.dumps(example, indent=2) + "\n", encoding="utf-8")

    by_source = lambda rs: dict(sorted(Counter(str(r.get("tier1_source") or "") for r in rs).items()))
    manifest = {
        "release": RELEASE_LABEL, "interim": True, "ready_for_distribution": False,
        "schema_version": SCHEMA_VERSION, "hs_version": "HS2022",
        "built_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "coverage_note": ("part1 = the complete catalog + BOL expert-validated "
                          "set (843). Catalog: 248 net-new + 22 catalog "
                          "carryover. BOL: 573 expert-curated bill-of-lading records. "
                          "Carryover-BOL yielded no in-scope records and is closed; "
                          "the remaining BOL candidates were intentionally dropped "
                          "during expert curation."),
        "record_count": {"total": len(combined), "dev": len(dev_out), "test": len(test_out)},
        "by_source_tier1": {"all": by_source(combined), "dev": by_source(dev_out), "test": by_source(test_out)},
        "dev_distributions": _dist(dev_out), "test_distributions": _dist(test_out),
        "reference_corpus": {"entries": corpus_info["entries"],
                             "by_source": corpus_info["by_source"],
                             "by_jurisdiction": corpus_info["by_jurisdiction"]},
        "hashes": {"file_sha256": {
            "dev.json": _sha256(dev_path), "test_frozen.json": _sha256(test_path),
            "reference_corpus.jsonl": _sha256(OUT_DATA / "reference_corpus.jsonl")}},
    }
    (OUT_DATA / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_docs(combined, dev_out, test_out, manifest)
    print(f"\nwrote package: {OUT.relative_to(ROOT)}/")
    print(f"  by source: {by_source(combined)}")
    print(f"  per-HS4: {dict(sorted(Counter(str(r['hs4_label']) for r in combined).items()))}")
    return 0


def _write_docs(combined, dev, test, manifest) -> None:
    total = len(combined)
    src = dict(sorted(Counter(str(r["tier1_source"]) for r in combined).items()))
    hs4 = dict(sorted(Counter(str(r["hs4_label"]) for r in combined).items()))
    conf = dict(sorted(Counter(str(r["confidence_tier"]) for r in combined).items()))

    (OUT / "README.md").write_text(f"""# SemiHS-Bench — Part 1 (`part1`)

> **Combined benchmark — complete catalog + BOL expert-validated set.**
> {total} expert-validated records = the catalog tier + the BOL tier merged
> into one dataset. Carryover-BOL yielded no in-scope records and is closed;
> the remaining BOL candidates were intentionally dropped during expert
> curation. Full-benchmark gates (source mix, manufacturer floors) are not
> enforced at this size.

## Composition
- **By source** (`tier1_source`): {src}
- **dev / test**: {len(dev)} / {len(test)} (splits preserved from the catalog
  and BOL interim packages — no record changed split)

## Contents
- `data/dev.json` ({len(dev)}), `data/test_frozen.json` ({len(test)})
- `data/record_schema.json`, `data/reference_corpus.jsonl` (cited subset),
  `data/reference_corpus_schema.json`, `data/taxonomy.csv`, `data/MANIFEST.json`
- `eval/`, `prompts/`, `examples/submission_constrained.example.json`

## HS4 coverage
{chr(10).join(f"- `{k}`: {v}" for k, v in hs4.items())}

Confidence tiers: {conf}

## Three input tiers
- **Tier 1** — full expert-authored description.
- **Tier 2** — normalized short description + manufacturer (+ specs / origin).
- **Tier 3** — minimal: MPN/product code where available, else a 2–3-word identifier.

## How to evaluate
```bash
python eval/score_submission.py --submission your_submission.json \\
    --data data/test_frozen.json
```

## Provenance
Combined ids are a fresh uniform `SH-NNNN` / `v2.0.{{dev,test}}.NNNN`. Each
record keeps its interim-stage id in `source_frozen_id`; `tier1_source`
(`catalog` | `BOL`) marks origin. Built {manifest['built_at']} ·
commit {manifest['git_commit'][:12]}.
""", encoding="utf-8")

    (OUT / "DATASHEET.md").write_text(f"""# DATASHEET — SemiHS-Bench part1

**Status:** part1 — the complete catalog + BOL expert-validated set
({total} records). Carryover-BOL is closed (no in-scope records).

- **Task:** single-label HS6 tariff classification, semiconductor supply chain.
- **Sources:** manufacturer catalog + bill-of-lading line items, expert-validated.
- **By source:** {src}.
- **Labels:** Core-4 expert-validated; each cited to ≥1 authoritative ruling
  where available. Confidence: {conf}.
- **Splits:** {len(dev)} dev / {len(test)} test (preserved from the interim packages).
- **Limitations:** catalog tier-3 = MPN; BOL tier-3 = extracted identifier/keyword
  (no MPN in BOL); HS4 coverage {", ".join(hs4.keys())}; not the full benchmark.
- **License / citation:** inherit from the parent SemiHS-Bench release.
""", encoding="utf-8")

    def block(title, d):
        return f"### {title}\n\n| key | count |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in d.items())
    (OUT / "STATISTICS.md").write_text(
        f"# STATISTICS — part1 ({total} records)\n\n"
        + block("By source (tier1_source)", src) + "\n"
        + block("By HS4 (all)", hs4) + "\n"
        + block("By confidence tier (all)", conf) + "\n"
        + block("Dev — by HS6", manifest['dev_distributions']['by_hs6']) + "\n"
        + block("Test — by HS6", manifest['test_distributions']['by_hs6']),
        encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
