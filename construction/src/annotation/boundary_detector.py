"""Boundary-case detection for benchmark annotation.

The controlled vocabulary lives in ``configs/boundary_tags.yaml``. Tags come
in two groups:

- ``sibling_split`` — subheading splits inside one HS4 family. A record
  carries the tag when its gold code and at least one candidate-slate
  distractor fall inside the same cluster, so the record poses that sibling
  decision problem by construction.
- ``cross_family`` — frontiers between HS families that customs practice
  finds genuinely confusable. A record carries the tag when its gold code
  sits on one side and either a slate distractor falls on another side, or
  the record text carries keyword evidence for the gold side and at least
  one opposing side.

``compose_boundary_note`` turns a record's tags into the released
``boundary_note`` comment (the deciding-criterion text from the config).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import yaml

from src.utils.text_utils import normalize_text, tokenize

_CONSTRUCTION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _CONSTRUCTION_ROOT / "configs" / "boundary_tags.yaml"
DEFAULT_TAXONOMY_PATH = _CONSTRUCTION_ROOT.parent / "data" / "taxonomy.csv"

# Stable tag order: sibling splits first, then cross-family frontiers.
# Must match the ids in configs/boundary_tags.yaml exactly.
BOUNDARY_TAGS = (
    "8541_siblings",
    "8542_ic_function",
    "8486_process_stage",
    "8504_power_splits",
    "8536_connection_splits",
    "9030_measurement_splits",
    "9031_inspection_splits",
    "2804_gas_purity",
    "3707_photochemical",
    "9027_analysis_splits",
    "8471_adp_splits",
    "8541_vs_8542",
    "populated_board_boundary",
    "storage_boundary",
    "doped_vs_undoped",
    "process_vs_metrology",
    "furnace_boundary",
    "machine_with_function",
    "parts_attribution",
    "led_device_vs_luminaire",
    "display_module_boundary",
    "amplifier_boundary",
    "cable_vs_connector",
    "sensor_boundary",
    "crystal_substrate_boundary",
)

# Renames from the earlier pair-style vocabulary still present in older
# annotations; ids not listed here are unchanged.
OLD_TAG_MAP = {
    "8542.31_vs_8542.32_vs_8542.33_vs_8542.39": "8542_ic_function",
    "3818_vs_2804": "doped_vs_undoped",
    "8534_vs_8542": "populated_board_boundary",
}


@dataclass(frozen=True)
class BoundaryTag:
    tag_id: str
    group: str
    sides: Tuple[FrozenSet[str], ...]
    keywords: Tuple[Tuple[str, ...], ...]
    note: str


def _load_taxonomy_codes(taxonomy_path: Path) -> FrozenSet[str]:
    with Path(taxonomy_path).open(encoding="utf-8-sig") as handle:
        return frozenset(
            str(row.get("hs6", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("hs6", "")).strip()
        )


def _expand_codes(entries: Sequence[str], taxonomy_codes: FrozenSet[str]) -> FrozenSet[str]:
    codes: set = set()
    for entry in entries:
        entry = str(entry).strip()
        if len(entry) == 6:
            codes.add(entry)
        else:
            codes.update(code for code in taxonomy_codes if code.startswith(entry))
    return frozenset(codes)


@lru_cache(maxsize=4)
def load_boundary_tags(
    config_path: Path = DEFAULT_CONFIG_PATH,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> Tuple[BoundaryTag, ...]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    taxonomy_codes = _load_taxonomy_codes(taxonomy_path)
    tags: List[BoundaryTag] = []
    for raw in config.get("tags", []):
        sides = tuple(
            _expand_codes(side.get("codes", []), taxonomy_codes)
            for side in raw.get("sides", [])
        )
        keywords = tuple(
            tuple(str(term).lower() for term in side.get("keywords", []))
            for side in raw.get("sides", [])
        )
        tags.append(
            BoundaryTag(
                tag_id=str(raw["id"]),
                group=str(raw["group"]),
                sides=sides,
                keywords=keywords,
                note=normalize_text(str(raw.get("note", ""))),
            )
        )
    return tuple(tags)


def _record_text(record: Mapping[str, object]) -> str:
    parts = [
        str(record.get("tier1_description", "")),
        str(record.get("justification_text", "")),
    ]
    return normalize_text(" ".join(parts)).lower()


def _matches_any(text: str, tokens: FrozenSet[str], terms: Sequence[str]) -> bool:
    for term in terms:
        if " " in term:
            if term in text:
                return True
        elif term in tokens:
            return True
    return False


def detect_boundaries(
    record: Mapping[str, object],
    tags: Optional[Sequence[BoundaryTag]] = None,
) -> List[str]:
    specs = tags if tags is not None else load_boundary_tags()
    gold = str(record.get("hs6_label", ""))
    candidate_codes = {
        str(code)
        for code in (record.get("candidate_set") or {}).get("codes", [])
        if str(code)
    }
    distractors = candidate_codes - {gold}

    text = _record_text(record)
    tokens = frozenset(token.strip(".+/-") for token in tokenize(text))

    detected: List[str] = []
    for spec in specs:
        gold_sides = [i for i, side in enumerate(spec.sides) if gold in side]
        if not gold_sides:
            continue
        if spec.group == "sibling_split":
            if distractors & spec.sides[gold_sides[0]]:
                detected.append(spec.tag_id)
            continue
        other_sides = [i for i in range(len(spec.sides)) if i not in gold_sides]
        slate_hit = any(distractors & spec.sides[i] for i in other_sides)
        keyword_hit = False
        if not slate_hit:
            gold_evidence = any(
                _matches_any(text, tokens, spec.keywords[i])
                for i in gold_sides
                if spec.keywords[i]
            )
            other_evidence = any(
                _matches_any(text, tokens, spec.keywords[i])
                for i in other_sides
                if spec.keywords[i]
            )
            keyword_hit = gold_evidence and other_evidence
        if slate_hit or keyword_hit:
            detected.append(spec.tag_id)
    return detected


def compose_boundary_note(
    boundary_tags: Sequence[str],
    tags: Optional[Sequence[BoundaryTag]] = None,
) -> str:
    if not boundary_tags:
        return ""
    specs = tags if tags is not None else load_boundary_tags()
    notes_by_id: Dict[str, str] = {spec.tag_id: spec.note for spec in specs}
    sentences: List[str] = []
    for tag_id in boundary_tags:
        note = notes_by_id.get(str(tag_id))
        if note is None:
            raise ValueError(f"unknown boundary tag: {tag_id!r}")
        sentences.append(note)
    return " ".join(sentences)
