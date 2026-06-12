"""Taxonomy loading helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.utils.io_utils import write_jsonl


@dataclass
class TaxonomyEntry:
    hs6: str
    description: str

    @property
    def hs4(self) -> str:
        return self.hs6[:4]

    @property
    def hs2(self) -> str:
        return self.hs6[:2]


class TaxonomyIndex:
    def __init__(self, entries: Iterable[TaxonomyEntry]) -> None:
        self._entries = {entry.hs6: entry for entry in entries}

    def is_valid_hs6(self, code: str) -> bool:
        return code in self._entries

    def get(self, code: str) -> Optional[TaxonomyEntry]:
        return self._entries.get(code)

    def all_entries(self) -> List[TaxonomyEntry]:
        return list(self._entries.values())


def resolve_hs6_alias(
    code: str,
    *,
    description: str = "",
    alias_config: Mapping[str, Any] | None = None,
) -> str:
    hs6 = "".join(character for character in str(code or "") if character.isdigit())[:6]
    if len(hs6) != 6 or not alias_config:
        return hs6

    exact_aliases = alias_config.get("exact_aliases", {})
    if isinstance(exact_aliases, dict):
        exact_match = str(exact_aliases.get(hs6, "")).strip()
        if exact_match:
            return exact_match

    lowered_description = str(description or "").lower()
    for rule in alias_config.get("rules", []):
        if not isinstance(rule, dict):
            continue
        if str(rule.get("from", "")).strip() != hs6:
            continue
        contains_any = [str(item).lower() for item in rule.get("contains_any", []) if str(item).strip()]
        contains_all = [str(item).lower() for item in rule.get("contains_all", []) if str(item).strip()]
        if contains_any and not any(item in lowered_description for item in contains_any):
            continue
        if contains_all and not all(item in lowered_description for item in contains_all):
            continue
        target = str(rule.get("to", "")).strip()
        if target:
            return target
    return hs6


def load_taxonomy_csv(path: str) -> TaxonomyIndex:
    entries_by_hs6: Dict[str, TaxonomyEntry] = {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_code = str(row.get("hs6") or row.get("hscode") or row.get("code") or row.get("HTS Number") or "").strip()
            hs_digits = "".join(character for character in raw_code if character.isdigit())
            hs6 = hs_digits[:6]
            description = str(row.get("description") or row.get("label") or row.get("Description") or "").strip()
            if len(hs6) != 6 or not description:
                continue
            existing = entries_by_hs6.get(hs6)
            if existing is None or len(description) > len(existing.description):
                entries_by_hs6[hs6] = TaxonomyEntry(hs6=hs6, description=description)
    return TaxonomyIndex(entries_by_hs6.values())


def export_taxonomy_jsonl(index: TaxonomyIndex, path: str) -> None:
    write_jsonl(path, [{"hs6": entry.hs6, "description": entry.description} for entry in index.all_entries()])
