"""Configuration loading with standard-library fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_mapping(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    if yaml is not None:
        loaded = yaml.safe_load(content)
        return loaded or {}

    return json.loads(content)

