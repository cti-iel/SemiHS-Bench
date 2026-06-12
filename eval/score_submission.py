#!/usr/bin/env python3
"""SemiHS-Bench - score a submission file and emit a metrics report.

Usage:

    python3 eval/score_submission.py \\
        --submission my_predictions.json \\
        [--data data/eval.json] \\
        [--output report.md]

Reads a JSON submission (see ``eval/submission_schema.json``), validates it
against the gold dataset, and writes a Markdown report plus a JSON sidecar
with all metrics. Stand-alone - no dependencies beyond the Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from lib import (  # noqa: E402  (vendored)
    hierarchical_distance,
    mrr,
    top_k_accuracy,
)


# Group of each difficulty tag, mirroring construction/configs/boundary_tags.yaml.
# Kept inline so the scorer stays stdlib-only and dependency-free.
_TAG_GROUP = {
    "8541_siblings": "sibling_split",
    "8542_ic_function": "sibling_split",
    "8486_process_stage": "sibling_split",
    "8504_power_splits": "sibling_split",
    "8536_connection_splits": "sibling_split",
    "9030_measurement_splits": "sibling_split",
    "9031_inspection_splits": "sibling_split",
    "2804_gas_purity": "sibling_split",
    "3707_photochemical": "sibling_split",
    "9027_analysis_splits": "sibling_split",
    "8471_adp_splits": "sibling_split",
    "8541_vs_8542": "cross_family",
    "populated_board_boundary": "cross_family",
    "storage_boundary": "cross_family",
    "doped_vs_undoped": "cross_family",
    "process_vs_metrology": "cross_family",
    "furnace_boundary": "cross_family",
    "machine_with_function": "cross_family",
    "parts_attribution": "cross_family",
    "led_device_vs_luminaire": "cross_family",
    "display_module_boundary": "cross_family",
    "amplifier_boundary": "cross_family",
    "cable_vs_connector": "cross_family",
    "sensor_boundary": "cross_family",
    "crystal_substrate_boundary": "cross_family",
}


# ----- I/O -------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _index_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(r["frozen_id"]): r for r in records}


# ----- validation ------------------------------------------------------------


class SubmissionError(ValueError):
    """Raised on any malformed-submission condition."""


def validate(
    submission: Mapping[str, Any],
    by_frozen: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Strict validation. Returns the parsed submission metadata + predictions."""
    meta = submission.get("submission") or {}
    if not isinstance(meta, Mapping):
        raise SubmissionError("'submission' block missing or not a mapping")
    for required in ("name", "model_id", "mode", "tier", "schema_version"):
        if not meta.get(required):
            raise SubmissionError(f"submission.{required} is required")
    mode = str(meta["mode"])
    if mode not in {"constrained", "open"}:
        raise SubmissionError(f"submission.mode must be 'constrained' or 'open', got {mode!r}")
    tier = int(meta["tier"])
    if tier not in (1, 2):
        raise SubmissionError(f"submission.tier must be 1 or 2, got {tier!r}")

    predictions = submission.get("predictions") or []
    if not isinstance(predictions, list) or not predictions:
        raise SubmissionError("'predictions' must be a non-empty list")

    seen_ids: set = set()
    parsed: List[Dict[str, Any]] = []
    issues: Dict[str, int] = defaultdict(int)
    for idx, pred in enumerate(predictions):
        if not isinstance(pred, Mapping):
            raise SubmissionError(f"predictions[{idx}] is not an object")
        frozen_id = str(pred.get("frozen_id") or "")
        if not frozen_id:
            raise SubmissionError(f"predictions[{idx}] missing frozen_id")
        if frozen_id in seen_ids:
            raise SubmissionError(f"predictions[{idx}] duplicate frozen_id {frozen_id!r}")
        seen_ids.add(frozen_id)
        record = by_frozen.get(frozen_id)
        if record is None:
            issues["unknown_frozen_id"] += 1
            continue
        ranked = pred.get("ranked_codes") or []
        if not isinstance(ranked, list) or not ranked:
            raise SubmissionError(f"predictions[{idx}] ranked_codes must be a non-empty list")
        ranked = [str(c) for c in ranked]
        if mode == "constrained":
            slate = list((record.get("candidate_set") or {}).get("codes") or [])
            if not slate:
                raise SubmissionError(f"record {frozen_id} has no candidate_set; "
                                      "did you mean mode=open?")
            if set(ranked) != set(slate):
                issues["codes_outside_slate"] += 1
                raise SubmissionError(
                    f"predictions[{idx}] ({frozen_id}, constrained mode): "
                    f"ranked_codes must be a permutation of the record's "
                    f"candidate_set.codes. Got {ranked!r}, expected permutation of {slate!r}"
                )
        parsed.append({"frozen_id": frozen_id, "ranked_codes": ranked, "record": record})

    return {
        "meta": dict(meta),
        "predictions": parsed,
        "issues": dict(issues),
    }


# ----- scoring ---------------------------------------------------------------


def _split_into_predictions_labels(parsed_predictions: Sequence[Mapping[str, Any]]):
    preds = [p["ranked_codes"] for p in parsed_predictions]
    labels = [str(p["record"]["hs6_label"]) for p in parsed_predictions]
    return preds, labels


def _slice(parsed_predictions, predicate):
    out = [p for p in parsed_predictions if predicate(p)]
    return out


def _metrics_block(parsed_predictions) -> Dict[str, Any]:
    if not parsed_predictions:
        return {"n": 0}
    preds, labels = _split_into_predictions_labels(parsed_predictions)
    return {
        "n": len(parsed_predictions),
        "hs6_top1": top_k_accuracy(preds, labels, k=1, level="hs6"),
        "hs6_top3": top_k_accuracy(preds, labels, k=3, level="hs6"),
        "hs6_top5": top_k_accuracy(preds, labels, k=5, level="hs6"),
        "hs4_top1": top_k_accuracy(preds, labels, k=1, level="hs4"),
        "hs2_top1": top_k_accuracy(preds, labels, k=1, level="hs2"),
        "mrr": mrr(preds, labels),
        "mean_hier_dist": statistics.fmean(
            hierarchical_distance(p[0] if p else "", l)
            for p, l in zip(preds, labels)
        ),
    }


def _hier_distribution(parsed_predictions) -> Dict[str, int]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in parsed_predictions:
        ranked = p["ranked_codes"]
        if not ranked:
            counts[3] += 1
            continue
        d = hierarchical_distance(ranked[0], str(p["record"]["hs6_label"]))
        counts[d] += 1
    return {
        "hs6_match": counts[0],
        "hs4_match_only": counts[1],
        "hs2_match_only": counts[2],
        "no_match": counts[3],
    }


def score(parsed_submission: Mapping[str, Any]) -> Dict[str, Any]:
    parsed_predictions = parsed_submission["predictions"]
    meta = parsed_submission["meta"]
    overall = _metrics_block(parsed_predictions)
    overall["hier_distribution"] = _hier_distribution(parsed_predictions)

    by_hs2: Dict[str, Any] = {}
    buckets_hs2: Dict[str, List] = defaultdict(list)
    for p in parsed_predictions:
        buckets_hs2[str(p["record"].get("hs2_label", ""))].append(p)
    for chapter, entries in sorted(buckets_hs2.items()):
        by_hs2[chapter] = _metrics_block(entries)

    return {
        "submission": meta,
        "issues": parsed_submission.get("issues") or {},
        "overall": overall,
        "by_hs2_chapter": by_hs2,
        "by_difficulty": _by_difficulty(parsed_predictions),
    }


def _by_difficulty(parsed_predictions) -> Dict[str, Any]:
    """Accuracy split by boundary difficulty: boundary vs non-boundary,
    by tag group, and per individual tag. A record contributes to every
    tag it carries, so multi-tag records appear in multiple per-tag buckets."""
    boundary: List = []
    non_boundary: List = []
    by_group: Dict[str, List] = defaultdict(list)
    by_tag: Dict[str, List] = defaultdict(list)
    for p in parsed_predictions:
        tags = p["record"].get("difficulty_tags") or []
        if tags:
            boundary.append(p)
        else:
            non_boundary.append(p)
        for tag in tags:
            by_tag[str(tag)].append(p)
            by_group[_TAG_GROUP.get(str(tag), "other")].append(p)
    return {
        "boundary": _metrics_block(boundary),
        "non_boundary": _metrics_block(non_boundary),
        "by_group": {g: _metrics_block(by_group[g]) for g in sorted(by_group)},
        "by_tag": {t: _metrics_block(by_tag[t]) for t in sorted(by_tag)},
    }


# ----- report formatting -----------------------------------------------------


def _format_metrics_row(name: str, m: Mapping[str, Any]) -> str:
    if not m or m.get("n") == 0:
        return f"| {name} | 0 | - | - | - | - | - |"
    return (
        f"| {name} | {m['n']} | "
        f"{m['hs6_top1']:.3f} | {m['hs6_top3']:.3f} | "
        f"{m['hs4_top1']:.3f} | {m['hs2_top1']:.3f} | {m['mrr']:.3f} |"
    )


def format_markdown(report: Mapping[str, Any]) -> str:
    meta = report["submission"]
    overall = report["overall"]
    issues = report.get("issues") or {}

    lines: List[str] = [
        f"# SemiHS-Bench - Submission Report: {meta['name']}",
        "",
        f"- **Model**: `{meta['model_id']}`",
        f"- **Mode**: `{meta['mode']}` · **Tier**: {meta['tier']}",
        f"- **Schema version**: {meta.get('schema_version', '2.0.0')}",
    ]
    if meta.get("notes"):
        lines.append(f"- **Notes**: {meta['notes']}")
    lines.extend(["", "## Overall metrics", ""])
    lines.append("| slice | n | top-1 (HS6) | top-3 (HS6) | top-1 (HS4) | top-1 (HS2) | MRR (HS6) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(_format_metrics_row("overall", overall))

    hd = overall.get("hier_distribution") or {}
    if hd:
        n = overall["n"]
        lines.extend([
            "",
            "**Hierarchical-distance distribution** (top-1 prediction vs. gold):",
            "",
            f"- HS6 exact match:        {hd['hs6_match']}/{n} ({hd['hs6_match']/n:.1%})",
            f"- HS4 match (HS6 wrong):  {hd['hs4_match_only']}/{n} ({hd['hs4_match_only']/n:.1%})",
            f"- HS2 match (HS4 wrong):  {hd['hs2_match_only']}/{n} ({hd['hs2_match_only']/n:.1%})",
            f"- No match:               {hd['no_match']}/{n} ({hd['no_match']/n:.1%})",
            f"- Mean hierarchical distance: {overall['mean_hier_dist']:.3f}",
        ])

    if report.get("by_hs2_chapter"):
        lines.extend(["", "## Per-HS2 chapter breakdown", ""])
        lines.append("| chapter | n | top-1 (HS6) | top-3 (HS6) | top-1 (HS4) | top-1 (HS2) | MRR (HS6) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for chapter, m in report["by_hs2_chapter"].items():
            lines.append(_format_metrics_row(chapter, m))

    diff = report.get("by_difficulty") or {}
    if diff:
        lines.extend(["", "## Difficulty breakdown", ""])
        lines.append("| slice | n | top-1 (HS6) | top-3 (HS6) | top-1 (HS4) | top-1 (HS2) | MRR (HS6) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        lines.append(_format_metrics_row("boundary", diff.get("boundary") or {}))
        lines.append(_format_metrics_row("non_boundary", diff.get("non_boundary") or {}))
        for group, m in (diff.get("by_group") or {}).items():
            lines.append(_format_metrics_row(f"group: {group}", m))
        for tag, m in (diff.get("by_tag") or {}).items():
            lines.append(_format_metrics_row(tag, m))

    if issues:
        lines.extend(["", "## Submission diagnostics", ""])
        for key, n in issues.items():
            lines.append(f"- {key}: {n}")

    lines.extend(["", "_See `INTERPRETING_RESULTS.md` for guidance on reading these numbers._"])
    return "\n".join(lines) + "\n"


# ----- CLI -------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "eval.json",
        help="Path to the gold dataset JSON (default: the 900-record eval split; use data/train.json for train).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Markdown report path (default: alongside submission, .report.md)",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="JSON metrics path (default: alongside submission, .report.json)",
    )
    args = parser.parse_args(argv)

    submission = _load_json(args.submission)
    records = _load_json(args.data)
    if not isinstance(records, list):
        print(f"ERROR: {args.data} expected JSON list", file=sys.stderr)
        return 1
    by_frozen = _index_records(records)

    try:
        parsed = validate(submission, by_frozen)
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = score(parsed)
    md = format_markdown(report)

    output = args.output or args.submission.with_suffix(".report.md")
    json_out = args.json_out or args.submission.with_suffix(".report.json")
    output.write_text(md, encoding="utf-8")
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(md)
    print(f"\n[wrote {output} and {json_out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
