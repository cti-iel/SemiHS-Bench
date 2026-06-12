#!/usr/bin/env python3
"""Refresh ``difficulty_tags`` and ``boundary_note`` on the released splits.

For every record in ``../data/eval.json`` and ``../data/train.json``:

* migrate any legacy tag names (see ``OLD_TAG_MAP``),
* run the boundary detector over the candidate slate and the record text,
* union the migrated expert tags with the detected ones (expert tags first,
  preserving any an expert assigned that the heuristic no longer re-fires),
* write the union back to ``difficulty_tags`` and the composed comment to
  ``boundary_note``.

No other field is touched — ``tier2_classifiable`` and the rest are expert
data and are left exactly as-is. The rewrite is idempotent: running it twice
produces no change.

Usage::

    python scripts/annotate_difficulty.py            # rewrite both splits
    python scripts/annotate_difficulty.py --dry-run  # report only, write nothing
    python scripts/annotate_difficulty.py --check     # exit 2 if anything stale
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation.boundary_detector import (  # noqa: E402
    BOUNDARY_TAGS,
    OLD_TAG_MAP,
    BoundaryTag,
    compose_boundary_note,
    detect_boundaries,
    load_boundary_tags,
)

DEFAULT_DATA_FILES = (
    ROOT.parent / "data" / "eval.json",
    ROOT.parent / "data" / "train.json",
)


def resolve_tags(
    record: Mapping[str, object], specs: Sequence[BoundaryTag]
) -> List[str]:
    """Return the refreshed tag list for one record (expert tags first)."""
    migrated: List[str] = []
    for tag in record.get("difficulty_tags") or []:
        mapped = OLD_TAG_MAP.get(str(tag), str(tag))
        if mapped not in migrated:
            migrated.append(mapped)
    detected = detect_boundaries(record, tags=specs)
    union = list(migrated)
    for tag in detected:
        if tag not in union:
            union.append(tag)
    return union


def annotate_records(
    records: Sequence[Dict[str, object]], specs: Sequence[BoundaryTag]
) -> Tuple[int, Counter]:
    """Update records in place; return (changed_count, tag_counter)."""
    changed = 0
    counts: Counter = Counter()
    for record in records:
        tags = resolve_tags(record, specs)
        note = compose_boundary_note(tags, tags=specs)
        if record.get("difficulty_tags") != tags or record.get("boundary_note") != note:
            changed += 1
        record["difficulty_tags"] = tags
        record["boundary_note"] = note
        for tag in tags:
            counts[tag] += 1
    return changed, counts


def _serialize(records: Sequence[Mapping[str, object]]) -> str:
    return json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _print_summary(path: Path, records: Sequence[Mapping[str, object]], counts: Counter) -> None:
    total = len(records)
    tagged = sum(1 for r in records if r.get("difficulty_tags"))
    multi = sum(1 for r in records if len(r.get("difficulty_tags") or []) > 1)
    print(f"{path.name}: {tagged}/{total} tagged ({100 * tagged / total:.1f}%), {multi} multi-tag")
    specs = {spec.tag_id: spec.group for spec in load_boundary_tags()}
    for group in ("sibling_split", "cross_family"):
        print(f"  {group}:")
        for tag in BOUNDARY_TAGS:
            if specs.get(tag) == group and counts.get(tag):
                print(f"    {tag:30s} {counts[tag]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        action="append",
        type=Path,
        help="Data file to annotate (repeatable; defaults to eval + train).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 2 if any record's tags/note are stale; write nothing.",
    )
    args = parser.parse_args(argv)

    data_files = list(args.data) if args.data else list(DEFAULT_DATA_FILES)
    specs = load_boundary_tags()

    total_changed = 0
    for path in data_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit(f"{path}: expected a JSON array")
        before = _serialize(records)
        changed, counts = annotate_records(records, specs)
        after = _serialize(records)
        total_changed += changed
        _print_summary(path, records, counts)
        if args.check or args.dry_run:
            if changed:
                print(f"  -> {changed} record(s) would change")
            continue
        if before != after:
            path.write_text(after, encoding="utf-8")
            print(f"  -> wrote {path}")
        else:
            print("  -> no change")

    if args.check and total_changed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
