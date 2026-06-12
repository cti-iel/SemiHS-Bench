#!/usr/bin/env python3
"""Export a group-stratified IAA sample for second-annotator review.

Pools both released splits (``../data/eval.json`` + ``../data/train.json``)
and draws a 150-record sample stratified by boundary group:

  - cross_family       60 records (with a per-tag floor so thin frontiers
                          are represented)
  - sibling_split      60 records
  - control_no_boundary 30 records (no boundary tag)

Produces two CSVs:
  - iaa_input.csv               : blank annotation columns for rater A
  - iaa_annotated_rater_b.csv   : pipeline labels, treated as rater B

Rater B's labels are computed here from the released records: ``boundary_tags``
is the record's ``difficulty_tags``; ``ambiguity_score`` and
``classification_driver`` are inferred by the difficulty-tagger heuristics
(``ambiguity_score`` needs HS6 descriptions, read from
``../data/hs6_descriptions.csv``); ``tier2_classifiable`` is the record's
stored expert value and is passed through unchanged.

The sampling strata match docs/IAA_PROTOCOL.md §1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation.boundary_detector import load_boundary_tags  # noqa: E402
from src.annotation.difficulty_tagger import (  # noqa: E402
    _taxonomy_entries_by_hs4,
    infer_ambiguity,
    infer_classification_driver,
)
from src.collectors.hts_taxonomy import load_taxonomy_csv  # noqa: E402

REPO_DATA = ROOT.parent / "data"
DEFAULT_INPUTS = (REPO_DATA / "eval.json", REPO_DATA / "train.json")
DEFAULT_TAXONOMY = REPO_DATA / "hs6_descriptions.csv"

# (stratum name, group it draws from or None for control, target count)
STRATA: Tuple[Tuple[str, str | None, int], ...] = (
    ("cross_family", "cross_family", 60),
    ("sibling_split", "sibling_split", 60),
    ("control_no_boundary", None, 30),
)
# Try to seat at least this many records of each cross_family tag.
CROSS_FAMILY_FLOOR = 3

INPUT_COLUMNS: Tuple[str, ...] = (
    "frozen_id",
    "id",
    "hs4_label_hint",
    "tier1_description",
    "tier2_part_name",
    "tier2_manufacturer",
    "ambiguity_score",
    "boundary_tags",
    "classification_driver",
    "tier2_classifiable",
)


def _groups_by_tag() -> Dict[str, str]:
    return {spec.tag_id: spec.group for spec in load_boundary_tags()}


def _record_group(record: Mapping[str, object], tag_groups: Mapping[str, str]) -> str | None:
    """cross_family if any tag is cross-family, else sibling_split if any
    tag is a sibling split, else None (control)."""
    tags = record.get("difficulty_tags") or []
    groups = {tag_groups.get(str(t)) for t in tags}
    if "cross_family" in groups:
        return "cross_family"
    if "sibling_split" in groups:
        return "sibling_split"
    return None


def _record_row(
    record: Mapping[str, object],
    *,
    include_labels: bool,
    taxonomy=None,
    entries_by_hs4=None,
) -> Dict[str, str]:
    tier2 = record.get("tier2_minimal") or {}
    row: Dict[str, str] = {
        "frozen_id": str(record.get("frozen_id", "")),
        "id": str(record.get("id", "")),
        "hs4_label_hint": str(record.get("hs4_label", "")),
        "tier1_description": str(record.get("tier1_description", "")),
        "tier2_part_name": str(tier2.get("part_name", "")) if isinstance(tier2, Mapping) else "",
        "tier2_manufacturer": str(tier2.get("manufacturer", "")) if isinstance(tier2, Mapping) else "",
        "ambiguity_score": "",
        "boundary_tags": "",
        "classification_driver": "",
        "tier2_classifiable": "",
    }
    if include_labels:
        row["ambiguity_score"] = str(infer_ambiguity(record, taxonomy, entries_by_hs4=entries_by_hs4))
        row["boundary_tags"] = ";".join(record.get("difficulty_tags") or [])
        row["classification_driver"] = infer_classification_driver(record)
        row["tier2_classifiable"] = str(record.get("tier2_classifiable", ""))
    return row


def _partition(
    records: Sequence[Mapping[str, object]], tag_groups: Mapping[str, str]
) -> Dict[str, List[Mapping[str, object]]]:
    buckets: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        group = _record_group(record, tag_groups)
        buckets["control_no_boundary" if group is None else group].append(record)
    return buckets


def _cross_family_floor_draw(
    pool: Sequence[Mapping[str, object]],
    tag_groups: Mapping[str, str],
    rng: random.Random,
    target: int,
) -> List[Mapping[str, object]]:
    """Draw `target` records, seating CROSS_FAMILY_FLOOR of each cross-family
    tag first (rarest tag first), then filling the rest at random."""
    by_tag: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in pool:
        for tag in record.get("difficulty_tags") or []:
            if tag_groups.get(str(tag)) == "cross_family":
                by_tag[str(tag)].append(record)
    drawn_ids: set = set()
    drawn: List[Mapping[str, object]] = []
    for tag in sorted(by_tag, key=lambda t: len(by_tag[t])):
        candidates = [r for r in by_tag[tag] if r.get("frozen_id") not in drawn_ids]
        rng.shuffle(candidates)
        for record in candidates[:CROSS_FAMILY_FLOOR]:
            if len(drawn) >= target:
                break
            drawn.append(record)
            drawn_ids.add(record.get("frozen_id"))
    remaining = [r for r in pool if r.get("frozen_id") not in drawn_ids]
    rng.shuffle(remaining)
    for record in remaining:
        if len(drawn) >= target:
            break
        drawn.append(record)
        drawn_ids.add(record.get("frozen_id"))
    return drawn


def _sample(
    buckets: Mapping[str, Sequence[Mapping[str, object]]],
    tag_groups: Mapping[str, str],
    rng: random.Random,
) -> Tuple[List[Mapping[str, object]], List[Dict[str, object]]]:
    sample: List[Mapping[str, object]] = []
    manifest: List[Dict[str, object]] = []
    used_ids: set = set()
    deficits: List[Tuple[str, int]] = []
    for name, group, target in STRATA:
        pool = [r for r in buckets.get(name, []) if r.get("frozen_id") not in used_ids]
        if group == "cross_family":
            drawn = _cross_family_floor_draw(pool, tag_groups, rng, target)
        else:
            rng.shuffle(pool)
            drawn = pool[:target]
        used_ids.update(r.get("frozen_id") for r in drawn)
        sample.extend(drawn)
        manifest.append(
            {"stratum": name, "target": target, "drawn": len(drawn), "deficit": max(0, target - len(drawn))}
        )
        if len(drawn) < target:
            deficits.append((name, target - len(drawn)))

    if deficits:
        backup = [r for bucket in buckets.values() for r in bucket if r.get("frozen_id") not in used_ids]
        rng.shuffle(backup)
        for stratum_name, missing in deficits:
            fills = backup[:missing]
            backup = backup[missing:]
            used_ids.update(r.get("frozen_id") for r in fills)
            sample.extend(fills)
            for entry in manifest:
                if entry["stratum"] == stratum_name:
                    entry["filled_from_backup"] = len(fills)
                    entry["drawn"] += len(fills)
                    entry["deficit"] -= len(fills)
                    break
    return sample, manifest


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cross_family_coverage(
    sample: Sequence[Mapping[str, object]], tag_groups: Mapping[str, str]
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in sample:
        for tag in record.get("difficulty_tags") or []:
            if tag_groups.get(str(tag)) == "cross_family":
                counts[str(tag)] += 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Released split to pool (repeatable; defaults to eval + train).",
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "data" / "intermediate" / "review" / "iaa_input.csv",
        help="Blank-annotation template for rater A.",
    )
    parser.add_argument(
        "--rater-b-csv",
        type=Path,
        default=ROOT / "data" / "intermediate" / "review" / "iaa_annotated_rater_b.csv",
        help="Pipeline-label file used as rater B.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "intermediate" / "review" / "iaa_manifest.json",
        help="Stratum-by-stratum draw summary.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    inputs = list(args.input) if args.input else list(DEFAULT_INPUTS)
    records: List[Mapping[str, object]] = []
    for path in inputs:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise SystemExit(f"{path}: expected a JSON array")
        records.extend(loaded)

    taxonomy = load_taxonomy_csv(str(args.taxonomy))
    entries_by_hs4 = _taxonomy_entries_by_hs4(taxonomy)
    tag_groups = _groups_by_tag()

    rng = random.Random(args.seed)
    buckets = _partition(records, tag_groups)
    sample, manifest = _sample(buckets, tag_groups, rng)

    input_rows = [_record_row(r, include_labels=False) for r in sample]
    rater_b_rows = [
        _record_row(r, include_labels=True, taxonomy=taxonomy, entries_by_hs4=entries_by_hs4)
        for r in sample
    ]

    _write_csv(args.input_csv, input_rows)
    _write_csv(args.rater_b_csv, rater_b_rows)

    manifest_payload = {
        "sample_size": len(sample),
        "seed": args.seed,
        "strata": manifest,
        "cross_family_tag_coverage": _cross_family_coverage(sample, tag_groups),
        "source_datasets": [str(p) for p in inputs],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    print("iaa sample size:", len(sample))
    for entry in manifest:
        print(
            "  {0}: target={1} drawn={2} deficit={3}".format(
                entry["stratum"], entry["target"], entry["drawn"], entry["deficit"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
