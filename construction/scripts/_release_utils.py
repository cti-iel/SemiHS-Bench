#!/usr/bin/env python3
"""Shared helpers for the release-packaging scripts
(``build_part1_release.py``, ``build_eval.py``): file hashing,
distribution summaries, tier-2 shortening, and the per-release reference
corpus subset."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.text_utils import normalize_text  # noqa: E402

# Source corpus the per-release subsets are cut from (non-public working pool).
REFERENCE_CORPUS = ROOT / "release" / "working" / "data" / "reference_corpus.jsonl"

# tier2 shortening
_TIER2_MAX_TOKENS = 6
_TIER2_MAX_CHARS = 48


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _shorten(text: str) -> str:
    """First ~6 tokens / 48 chars of a normalized string — the tier-2
    minimal-lookup form when no MPN is available."""
    norm = normalize_text(text)
    tokens = norm.split()
    short = " ".join(tokens[:_TIER2_MAX_TOKENS])
    if len(short) > _TIER2_MAX_CHARS:
        short = short[:_TIER2_MAX_CHARS].rstrip()
    return short


def _write_corpus_subset(records: Sequence[Mapping[str, Any]], dest: Path) -> Dict[str, Any]:
    hs6_present = {str(r.get("hs6_label") or "") for r in records}
    cited = set()
    for r in records:
        for e in r.get("cited_evidence_ids") or []:
            cited.add(e)

    kept: List[Dict[str, Any]] = []
    by_source: Counter = Counter()
    by_juris: Counter = Counter()
    with REFERENCE_CORPUS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            hs6 = str(e.get("hs6_label") or "")
            eid = str(e.get("evidence_id") or "")
            if hs6 in hs6_present or eid in cited:
                kept.append(e)
                by_source[str(e.get("source") or "")] += 1
                by_juris[str(e.get("jurisdiction") or "")] += 1

    with dest.open("w", encoding="utf-8") as f:
        for e in sorted(kept, key=lambda x: str(x.get("evidence_id") or "")):
            f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")

    # verify every citation resolves
    kept_ids = {str(e.get("evidence_id") or "") for e in kept}
    unresolved = sorted(cited - kept_ids)
    return {
        "entries": len(kept),
        "by_source": dict(sorted(by_source.items())),
        "by_jurisdiction": dict(sorted(by_juris.items())),
        "unresolved_citations": unresolved,
    }


def _dist(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def c(key):
        return dict(sorted(Counter(str(r.get(key) or "") for r in records).items()))
    return {
        "by_hs2": c("hs2_label"),
        "by_hs4": c("hs4_label"),
        "by_hs6": c("hs6_label"),
        "by_scope_tier": c("scope_tier"),
        "by_confidence_tier": c("confidence_tier"),
        "by_label_source": c("label_source"),
    }
