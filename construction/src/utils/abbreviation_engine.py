"""Abbreviation lookup and replacement helpers."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .text_utils import normalize_text


def load_abbreviations(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            full_term = normalize_text(row["full_term"]).lower()
            abbreviation = normalize_text(row["abbreviation"])
            rows.append((full_term, abbreviation))
    rows.sort(key=lambda item: len(item[0]), reverse=True)
    return rows


def abbreviate_text(text: str, replacements: List[Tuple[str, str]]) -> str:
    result = normalize_text(text)
    for full_term, abbreviation in replacements:
        pattern = r"\b{}\b".format(re.escape(full_term))
        result = re.sub(pattern, abbreviation, result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    return result

