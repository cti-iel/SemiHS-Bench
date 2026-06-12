"""SemiHS-Bench evaluation library - standalone scoring utilities.

Self-contained scoring helpers: top-k accuracy, MRR, hierarchical
distance, ranked-prediction parsing, prompt rendering, and tier-input
selection.

This file has no dependencies beyond the Python standard library.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence


# ----- HS-code helpers -------------------------------------------------------


_LEVEL_WIDTHS = {"hs2": 2, "hs4": 4, "hs6": 6}


def _normalize_code(value: object, length: int) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())[:length]


def _level_width(level: str) -> int:
    if level not in _LEVEL_WIDTHS:
        raise ValueError("level must be one of hs2, hs4, hs6")
    return _LEVEL_WIDTHS[level]


# ----- metrics ---------------------------------------------------------------


def top_k_accuracy(
    predictions: Iterable[Sequence[str]],
    labels: Iterable[str],
    k: int = 1,
    level: str = "hs6",
) -> float:
    """Top-k accuracy at the given HS level (hs2/hs4/hs6)."""
    width = _level_width(level)
    total = 0
    correct = 0
    for prediction, label in zip(predictions, labels):
        total += 1
        label_code = _normalize_code(label, width)
        prediction_codes = [_normalize_code(c, width) for c in list(prediction)[:k]]
        if label_code and label_code in prediction_codes:
            correct += 1
    return float(correct) / float(total) if total else 0.0


def mrr(
    predictions: Iterable[Sequence[str]],
    labels: Iterable[str],
    level: str = "hs6",
) -> float:
    """Mean reciprocal rank of the first matching prediction."""
    width = _level_width(level)
    rranks: List[float] = []
    for prediction, label in zip(predictions, labels):
        label_code = _normalize_code(label, width)
        score = 0.0
        for rank, candidate in enumerate(prediction, start=1):
            if _normalize_code(candidate, width) == label_code:
                score = 1.0 / float(rank)
                break
        rranks.append(score)
    return sum(rranks) / len(rranks) if rranks else 0.0


def hierarchical_distance(pred: str, label: str) -> int:
    """0 if HS6 matches, 1 if HS4 matches, 2 if HS2 matches, 3 otherwise."""
    pred_hs6 = _normalize_code(pred, 6)
    label_hs6 = _normalize_code(label, 6)
    if not pred_hs6 or not label_hs6:
        return 3
    if pred_hs6 == label_hs6:
        return 0
    if pred_hs6[:4] == label_hs6[:4]:
        return 1
    if pred_hs6[:2] == label_hs6[:2]:
        return 2
    return 3


# ----- response parsing ------------------------------------------------------


_INDEX_RE = re.compile(r"-?\d+")


def parse_ranked_indices(text: str, *, expected: int) -> List[int]:
    """Extract a ranked list of distinct integer indices from a model response.

    Tries strict JSON first, then falls back to a regex scan. Truncated or
    duplicate outputs are repaired by appending the missing indices in
    natural order so the returned list always has length ``expected`` -
    callers can score a full ranking even on malformed model output.
    """
    indices: List[int] = []
    seen: set = set()
    text = (text or "").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            for item in payload:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < expected and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
    except ValueError:
        pass

    if len(indices) < expected:
        for match in _INDEX_RE.finditer(text):
            try:
                idx = int(match.group(0))
            except ValueError:
                continue
            if 0 <= idx < expected and idx not in seen:
                indices.append(idx)
                seen.add(idx)
                if len(indices) == expected:
                    break

    for idx in range(expected):
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices[:expected]


_HS6_RE = re.compile(r"\b\d{6}\b")


def extract_hs6_codes(text: str, allowed_codes: Optional[set] = None) -> List[str]:
    """Parse HS6 codes from free-text model output (open-mode submissions)."""
    codes: List[str] = []
    seen: set = set()
    for match in _HS6_RE.finditer(text or ""):
        code = match.group(0)
        if code in seen:
            continue
        if allowed_codes is not None and code not in allowed_codes:
            continue
        seen.add(code)
        codes.append(code)
    return codes


# ----- tier inputs -----------------------------------------------------------


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _join_nonempty(parts: Iterable[Any]) -> str:
    return " ".join(_normalize_text(p) for p in parts if _normalize_text(p))


def get_tier_input(record: Mapping[str, Any], tier: int) -> str:
    """Assemble the model-facing input text for a given tier per record.

    Tier 1: full ``tier1_description`` + manufacturer name.
    Tier 2: minimal ``part_name`` + ``manufacturer`` - the sparse ERP-style
        input (a manufacturer part number or short descriptor).
    """
    if tier == 1:
        desc = _normalize_text(record.get("tier1_description"))
        mfr = _normalize_text((record.get("tier2_minimal") or {}).get("manufacturer"))
        return _join_nonempty([desc, mfr])
    if tier == 2:
        minimal = record.get("tier2_minimal") or {}
        if not isinstance(minimal, Mapping):
            return _normalize_text(minimal)
        return _join_nonempty([minimal.get("part_name"), minimal.get("manufacturer")])
    raise ValueError("tier must be 1 or 2")


# ----- prompt rendering ------------------------------------------------------


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "bench" / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template by name from ``bench/prompts/``."""
    path = _PROMPT_DIR / f"{name}.txt"
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    return text


def render_prompt(name: str, **fields: Any) -> str:
    """Load a template and substitute via ``str.format``."""
    return load_prompt(name).format(**fields)


def render_candidates_block(codes: Sequence[str], taxonomy: Optional[Mapping[str, Any]] = None) -> str:
    """0-indexed candidate slate for inclusion in the constrained prompt."""
    hs6_desc = (taxonomy or {}).get("hs6_descriptions", {}) if taxonomy else {}
    hs4_desc = (taxonomy or {}).get("hs4_headings", {}) if taxonomy else {}
    lines: List[str] = []
    for idx, code in enumerate(codes):
        line = f"{idx}) {code}"
        desc = hs6_desc.get(code) or ""
        hs4 = code[:4]
        hs4_text = hs4_desc.get(hs4) or ""
        if desc or hs4_text:
            extras: List[str] = []
            if desc:
                extras.append(desc)
            if hs4_text:
                extras.append(f"parent {hs4}: {hs4_text}")
            line += f" - {'; '.join(extras)}"
        lines.append(line)
    return "\n".join(lines)


# ----- taxonomy CSV loader ---------------------------------------------------


def load_taxonomy_csv(path: Path) -> dict:
    """Load the bundled ``taxonomy.csv`` (the in-scope HS2/HS4/HS6 codes).

    Returns a dict with ``hs2_chapters``, ``hs4_headings``, ``hs6_descriptions``
    each mapping ``code → text``. The shipped taxonomy lists codes only (no
    description column); in that case the descriptions are empty strings and
    the codes still define the open-mode label space.
    """
    import csv

    hs2: dict = {}
    hs4: dict = {}
    hs6: dict = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        code_col = next((c for c in ("hscode", "hs6", "code") if c in (reader.fieldnames or [])), None)
        desc_col = next((c for c in ("description", "label") if c in (reader.fieldnames or [])), None)
        if code_col is None:
            raise ValueError(f"taxonomy CSV missing a code column; got {reader.fieldnames!r}")
        for row in reader:
            digits = "".join(ch for ch in str(row.get(code_col) or "") if ch.isdigit())
            if not digits:
                continue
            description = _normalize_text(row.get(desc_col)) if desc_col else ""
            if len(digits) == 2:
                hs2.setdefault(digits, description)
            elif len(digits) == 4:
                hs4.setdefault(digits, description)
            elif len(digits) >= 6:
                hs6.setdefault(digits[:6], description)
    return {
        "hs2_chapters": dict(sorted(hs2.items())),
        "hs4_headings": dict(sorted(hs4.items())),
        "hs6_descriptions": dict(sorted(hs6.items())),
    }


def in_scope_hs6_codes(taxonomy: Mapping[str, Any]) -> List[str]:
    """Sorted list of all in-scope HS6 codes (open-mode candidate space)."""
    return sorted(taxonomy.get("hs6_descriptions", {}).keys())
