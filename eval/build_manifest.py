#!/usr/bin/env python3
"""Rebuild ``data/MANIFEST.json`` from the released data files.

Recomputes record counts (per split), the source/confidence/segment/HS
distributions (per split - ``eval`` and ``train`` are reported separately
because, while they share identical HS6 coverage, their per-code counts and
source mix differ), the reference-corpus composition, and SHA-256 hashes for
every data file present. Run this after updating anything under ``data/``
(e.g. after dropping in the ``train.json`` split or the full reference
corpus):

    python3 eval/build_manifest.py

Descriptive/meta fields already in MANIFEST.json (balance, hs_version,
ready_for_distribution, schema_version, split_scheme) are preserved. Standard-library only.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Hashed for integrity if present, in this order.
_HASH_FILES = [
    "eval.json",
    "train.json",
    "reference_corpus.jsonl",
    "taxonomy.csv",
    "hs6_descriptions.csv",
    "record_schema.json",
    "reference_corpus_schema.json",
]


def _difficulty_dist(records: list) -> dict:
    """Per-tag counts (a multi-tag record counts once per tag)."""
    counter: collections.Counter = collections.Counter()
    for r in records:
        for tag in r.get("difficulty_tags") or []:
            counter[tag] += 1
    return dict(sorted(counter.items()))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_records(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _dist(records: list, key: str) -> dict:
    return dict(sorted(collections.Counter(r.get(key) for r in records).items()))


def _split_distributions(records: list) -> dict:
    """All per-split distributions for one split's records."""
    tagged = sum(1 for r in records if r.get("difficulty_tags"))
    return {
        "by_confidence_tier": _dist(records, "confidence_tier"),
        "by_segment": _dist(records, "segment"),
        "by_hs2": _dist(records, "hs2_label"),
        "by_hs4": _dist(records, "hs4_label"),
        "by_hs6": _dist(records, "hs6_label"),
        "by_difficulty_tag": _difficulty_dist(records),
        "boundary_records": {"tagged": tagged, "untagged": len(records) - tagged},
    }


def main() -> int:
    eval_recs = _load_records(DATA / "eval.json")
    train_recs = _load_records(DATA / "train.json")
    corpus = _load_jsonl(DATA / "reference_corpus.jsonl")

    manifest_path = DATA / "MANIFEST.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    splits = {"eval": eval_recs, "train": train_recs}

    record_count = {name: len(recs) for name, recs in splits.items() if recs}
    record_count["total"] = sum(len(recs) for recs in splits.values())
    m["record_count"] = record_count

    # Distributions are reported per split: eval and train share identical HS6
    # coverage but differ in per-code counts and source mix.
    m["by_source_tier1"] = {name: _dist(recs, "tier1_source") for name, recs in splits.items() if recs}
    m["distributions"] = {name: _split_distributions(recs) for name, recs in splits.items() if recs}
    m["hs4_allocation"] = {name: dict(d["by_hs4"]) for name, d in m["distributions"].items()}

    m["reference_corpus"] = {
        "entries": len(corpus),
        "by_source": dict(sorted(collections.Counter(e.get("source") for e in corpus).items())),
        "by_jurisdiction": dict(sorted(collections.Counter(e.get("jurisdiction") for e in corpus).items())),
    }

    m["hashes"] = {
        "file_sha256": {name: _sha256(DATA / name) for name in _HASH_FILES if (DATA / name).exists()}
    }

    manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MANIFEST.json rebuilt: record_count={record_count}")
    print(f"  hashed files: {list(m['hashes']['file_sha256'])}")
    print(f"  reference corpus: {m['reference_corpus']['entries']} entries "
          f"({m['reference_corpus']['by_source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
