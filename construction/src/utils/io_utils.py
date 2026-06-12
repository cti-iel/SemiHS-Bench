"""File IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Any, pretty: bool = True) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        if pretty:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        else:
            json.dump(payload, handle)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str, records: Iterable[Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def append_jsonl(path: str, record: Any) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def read_jsonl(path: str) -> List[Any]:
    items: List[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                items.append(json.loads(text))
    return items

