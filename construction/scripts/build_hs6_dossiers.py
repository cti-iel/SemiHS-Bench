#!/usr/bin/env python3
"""Generate per-HS6 evidence dossiers for the release.

Builds the per-HS6 evidence dossiers used during expert review:
for each in-scope HS6, emit a 1-page Markdown dossier with cited
EBTI/CROSS rulings, an expert-written synthesis paragraph, WCO HSN
section reference, and boundary-partner links.

Inputs:

* ``release/working/data/reference_corpus.jsonl`` — citable evidence pool.
* ``data/intermediate/dossier_rationales.csv`` — expert-written
  synthesis paragraphs (one row per HS6). The expert fills this CSV
  during the dossier-rationale pass of the build sequencing. Without it,
  this script emits stub dossiers flagged ``synthesis_pending=true``
  AND writes a blank CSV template at that path for the expert to fill.
* ``configs/hs6_scope_tiers.yaml`` — the canonical 34 in-scope HS6 set
  used as the dossier inventory in stub mode (before the benchmark
  records exist).
* ``release/working/data/_candidate_pool.jsonl`` (optional) — carryover
  records; unioned into the dossier inventory when present.
* ``release/working/data/dev.json`` + ``test_frozen.json`` (optional) —
  full benchmark records; when present, drives the inventory
  directly (production mode).
* ``configs/boundary_tags.yaml`` — boundary-partner links.
* ``configs/hs_chapters.yaml`` — HS heading display labels.

Outputs:

* ``release/working/dossiers/<hs6>.md`` — one file per in-scope HS6.
* ``release/working/dossiers/index.json`` — machine-readable index.
* ``data/intermediate/dossier_rationales.csv`` (TEMPLATE, only if
  missing) — pre-seeded with the HS6 inventory; expert fills the
  ``synthesis`` column.
* ``release/working/dossiers/_build_report.json`` — coverage + pending stats.

Inventory mode is auto-selected:

* **production**: dev.json + test_frozen.json present → use HS6s with
  ≥1 benchmark record.
* **stub_candidate**: candidate_pool.jsonl present → use HS6s from
  carryover ∪ in-scope set.
* **stub_inscope** (default): use the full 34-HS6 in-scope set.

Stub modes write dossiers for HS6s that may not yet have records —
useful for letting the expert pre-author synthesis paragraphs before
the bulk audit pass starts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation.boundary_detector import load_boundary_tags  # noqa: E402
WORKING_DATA = ROOT / "release" / "working" / "data"
DOSSIER_DIR = ROOT / "release" / "working" / "dossiers"
INTERMEDIATE = ROOT / "data" / "intermediate"

REF_CORPUS_PATH = WORKING_DATA / "reference_corpus.jsonl"
DEV_PATH = WORKING_DATA / "dev.json"
TEST_PATH = WORKING_DATA / "test_frozen.json"
CANDIDATE_POOL_PATH = WORKING_DATA / "_candidate_pool.jsonl"
SCOPE_PATH = ROOT / "configs" / "hs6_scope_tiers.yaml"
RATIONALES_PATH = INTERMEDIATE / "dossier_rationales.csv"
BOUNDARY_TAGS_PATH = ROOT / "configs" / "boundary_tags.yaml"
HS_CHAPTERS_PATH = ROOT / "configs" / "hs_chapters.yaml"

INDEX_PATH = DOSSIER_DIR / "index.json"
BUILD_REPORT_PATH = DOSSIER_DIR / "_build_report.json"

_MAX_CITATIONS_PER_DOSSIER = 5
_COVERAGE_FLOOR = 3  # every in-scope HS6 needs >=3 corpus rulings

_RATIONALES_TEMPLATE_HEADER = ["hs6", "synthesis", "wco_en_section", "author", "date"]


def _load_mapping(path: Path) -> Dict[str, Any]:
    """Stdlib-fallback YAML loader (consistent with sibling scripts)."""
    content = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return json.loads(content)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _read_rationales(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hs6 = (row.get("hs6") or "").strip()
            if not hs6:
                continue
            out[hs6] = {
                "synthesis": (row.get("synthesis") or "").strip(),
                "wco_en_section": (row.get("wco_en_section") or "").strip(),
                "author": (row.get("author") or "").strip(),
                "date": (row.get("date") or "").strip(),
            }
    return out


def _in_scope_hs6_set() -> Set[str]:
    cfg = _load_mapping(SCOPE_PATH)
    return set(cfg.get("core") or []) | set(cfg.get("supply_chain") or [])


def _resolve_inventory() -> Tuple[Set[str], str]:
    """Return (hs6_set, mode) where mode ∈ {production, stub_candidate, stub_inscope}."""
    if DEV_PATH.exists() and TEST_PATH.exists():
        hs6: Set[str] = set()
        for record in _read_json(DEV_PATH) + _read_json(TEST_PATH):
            h = str(record.get("hs6_label") or "")
            if h:
                hs6.add(h)
        if hs6:
            return hs6, "production"

    inscope = _in_scope_hs6_set()

    if CANDIDATE_POOL_PATH.exists():
        carry: Set[str] = set()
        for record in _read_jsonl(CANDIDATE_POOL_PATH):
            h = str(record.get("hs6_label") or "")
            if h:
                carry.add(h)
        # In stub mode the dossier inventory is the FULL in-scope set
        # (so the expert has the complete to-do list), unioned with any
        # carryover HS6s that happen to fall outside (shouldn't, but defensive).
        return inscope | carry, "stub_candidate"

    return inscope, "stub_inscope"


def _citations_for_hs6(
    hs6: str, corpus: List[Mapping[str, Any]], *, limit: int = _MAX_CITATIONS_PER_DOSSIER
) -> List[Dict[str, Any]]:
    """Return up to ``limit`` reference-corpus entries for this HS6.

    Selection prioritizes jurisdictional diversity (alternates EU/US) then
    ruling recency."""
    matches = [c for c in corpus if c.get("hs6_label") == hs6]
    eu = sorted(
        [c for c in matches if (c.get("jurisdiction") or "").startswith("EU-")],
        key=lambda c: (c.get("ruling_date") or "1900-01-01"),
        reverse=True,
    )
    us = sorted(
        [c for c in matches if c.get("jurisdiction") == "US"],
        key=lambda c: (c.get("ruling_date") or "1900-01-01"),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    while (eu or us) and len(out) < limit:
        if eu:
            out.append(eu.pop(0))
        if us and len(out) < limit:
            out.append(us.pop(0))
    return out


def _coverage_for_hs6(
    hs6: str, corpus: List[Mapping[str, Any]]
) -> Dict[str, int]:
    """Return total / EU / US citation counts for the evidence-coverage floor check."""
    total = 0
    eu = 0
    us = 0
    for c in corpus:
        if c.get("hs6_label") != hs6:
            continue
        total += 1
        j = str(c.get("jurisdiction") or "")
        if j.startswith("EU-"):
            eu += 1
        elif j == "US":
            us += 1
    return {"total": total, "EU": eu, "US": us}


def _boundary_partners_for_hs6(
    hs6: str, boundary_tags: Sequence[Any]
) -> List[Dict[str, str]]:
    """Return list of {partner_hs6, pair_id, rule_summary} for boundary tags
    whose code sets include this HS6.

    A sibling_split tag links the HS6 to the other codes in its own cluster;
    a cross_family tag links it to the codes on the opposing side(s)."""
    out: List[Dict[str, str]] = []
    for spec in boundary_tags:
        member_sides = [i for i, side in enumerate(spec.sides) if hs6 in side]
        if not member_sides:
            continue
        if spec.group == "sibling_split":
            partners = sorted(spec.sides[member_sides[0]] - {hs6})
        else:
            partners = sorted(
                code
                for i, side in enumerate(spec.sides)
                if i not in member_sides
                for code in side
            )
        for partner in partners:
            out.append({
                "partner_hs6": partner,
                "pair_id": spec.tag_id,
                "rule_summary": spec.note,
            })
    return out


def _hs4_label(hs6: str, hs_chapters_cfg: Mapping[str, Any]) -> str:
    hs4 = hs6[:4]
    for entry in hs_chapters_cfg.get("hs_headings") or []:
        if entry.get("hs4") == hs4:
            return entry.get("label", "")
    return ""


def _citation_short(c: Mapping[str, Any]) -> str:
    """Render a short product descriptor for the citation line.

    Falls back across tier2_minimal.part_name → subject_terms[0] →
    truncated tier1_text. (Earlier versions of this script tried
    tier2.description_short, but the actual degrader schema emits
    {part_name, manufacturer}, NOT description_short.
    See release/working/data/reference_corpus_schema.json.)
    """
    tier2 = c.get("tier2_minimal") or {}
    part_name = (tier2.get("part_name") or "").strip()
    if part_name:
        return part_name[:80]
    terms = c.get("subject_terms") or []
    if terms:
        return str(terms[0])[:80]
    tier1 = (c.get("tier1_text") or "").strip()
    if tier1:
        return tier1[:80]
    return ""


def _render_dossier(
    hs6: str,
    citations: List[Mapping[str, Any]],
    coverage: Mapping[str, int],
    boundary_partners: List[Mapping[str, str]],
    rationale: Dict[str, str],
    hs4_label: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# HS6 {hs6} — Evidence Dossier")
    lines.append("")
    lines.append(f"**HS4 family:** {hs6[:4]} — {hs4_label}")
    lines.append(f"**HS2 chapter:** {hs6[:2]}")
    lines.append("")
    # Coverage status row.
    cov_status = "✓" if coverage["total"] >= _COVERAGE_FLOOR else "⚠"
    lines.append(
        f"**Reference coverage:** {coverage['total']} total "
        f"({coverage['EU']} EU + {coverage['US']} US)  "
        f"{cov_status} evidence-coverage floor (≥{_COVERAGE_FLOOR})"
    )
    lines.append("")

    synthesis = rationale.get("synthesis") or ""
    synthesis_pending = not synthesis
    if synthesis:
        lines.append("## Synthesis")
        lines.append("")
        lines.append(synthesis)
        author = rationale.get("author")
        date = rationale.get("date")
        if author or date:
            byline = " · ".join(b for b in [author, date] if b)
            lines.append("")
            lines.append(f"*— {byline}*")
        lines.append("")
    else:
        lines.append("## Synthesis")
        lines.append("")
        lines.append(
            "> _Synthesis paragraph pending expert review. Fill the "
            "`synthesis` cell for this HS6 in "
            "[`data/intermediate/dossier_rationales.csv`]"
            "(../../../data/intermediate/dossier_rationales.csv) "
            "and re-run `scripts/build_hs6_dossiers.py`._"
        )
        lines.append("")

    lines.append("## Cited authoritative rulings")
    lines.append("")
    if not citations:
        lines.append(
            "_No reference-corpus entries available for this HS6._  "
            "This violates the evidence-coverage rule "
            f"(≥{_COVERAGE_FLOOR} reference entries per in-scope HS6). "
            "Investigate before release."
        )
    else:
        for c in citations:
            eid = c.get("evidence_id", "")
            jurisdiction = c.get("jurisdiction", "")
            date = c.get("ruling_date", "") or "no-date"
            short = _citation_short(c)
            url = c.get("url", "")
            line = f"- **{eid}** ({jurisdiction}, {date})"
            if short:
                line += f" — {short}"
            if url:
                line += f" · [source]({url})"
            lines.append(line)
    lines.append("")

    wco = rationale.get("wco_en_section") or ""
    if wco:
        lines.append("## WCO Explanatory Notes reference")
        lines.append("")
        lines.append(f"`{wco}` (citation only — see WCO HS Explanatory Notes.)")
        lines.append("")

    lines.append("## Boundary partners")
    lines.append("")
    if not boundary_partners:
        lines.append(
            "_No declared boundary partners in `configs/boundary_tags.yaml`._"
        )
    else:
        for bp in boundary_partners:
            partner = bp.get("partner_hs6", "")
            pair_id = bp.get("pair_id", "")
            summary = (bp.get("rule_summary") or "").strip().replace("\n", " ")
            lines.append(f"- ↔ **{partner}** (`{pair_id}`)")
            if summary:
                lines.append(f"  - {summary}")
    lines.append("")

    lines.append("---")
    lines.append("")
    if synthesis_pending:
        lines.append(
            "> ⚠️ This dossier is **incomplete** — synthesis pending expert "
            "review per the dossier design (docs/ADJUDICATION_PROTOCOL.md)."
        )
    lines.append("")
    return "\n".join(lines)


def _ensure_rationales_template(
    inventory: Set[str],
    hs_chapters_cfg: Mapping[str, Any],
    path: Path,
) -> bool:
    """Create a blank rationales CSV template if missing. Returns True if
    we wrote it; False if it already exists (we don't touch existing
    rater work)."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RATIONALES_TEMPLATE_HEADER)
        writer.writeheader()
        for hs6 in sorted(inventory):
            writer.writerow({
                "hs6": hs6,
                "synthesis": "",
                "wco_en_section": "",
                "author": "",
                "date": "",
            })
    return True


def main() -> int:
    corpus = _read_jsonl(REF_CORPUS_PATH)
    if not corpus:
        print(
            f"ERROR: {REF_CORPUS_PATH} missing or empty. "
            "Run scripts/build_reference_corpus.py first.",
            file=sys.stderr,
        )
        return 1

    inventory, mode = _resolve_inventory()
    if not inventory:
        print("ERROR: empty inventory; nothing to dossier.", file=sys.stderr)
        return 1

    rationales = _read_rationales(RATIONALES_PATH)
    boundary_tags = load_boundary_tags(BOUNDARY_TAGS_PATH)
    hs_chapters_cfg = _load_mapping(HS_CHAPTERS_PATH)

    print(f"inventory mode: {mode}")
    print(f"reference corpus: {len(corpus)} entries")
    print(f"dossier inventory: {len(inventory)} HS6 codes")

    # Emit rationales-CSV template if missing — gives the expert a place
    # to start. Won't overwrite an existing file (preserves rater edits).
    template_written = _ensure_rationales_template(
        inventory, hs_chapters_cfg, RATIONALES_PATH
    )
    if template_written:
        print(f"wrote rationales template at {RATIONALES_PATH.relative_to(ROOT)}")
    else:
        existing_with_synthesis = sum(
            1 for r in rationales.values() if r.get("synthesis")
        )
        print(
            f"rationales template exists at {RATIONALES_PATH.relative_to(ROOT)}: "
            f"{existing_with_synthesis}/{len(rationales)} synthesis paragraphs filled"
        )

    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, Any]] = []
    pending_count = 0
    below_floor_count = 0
    for hs6 in sorted(inventory):
        citations = _citations_for_hs6(hs6, corpus)
        coverage = _coverage_for_hs6(hs6, corpus)
        boundary_partners = _boundary_partners_for_hs6(hs6, boundary_tags)
        rationale = rationales.get(hs6, {})
        hs4_label = _hs4_label(hs6, hs_chapters_cfg)

        body = _render_dossier(
            hs6,
            citations,
            coverage,
            boundary_partners,
            rationale,
            hs4_label,
        )
        out_path = DOSSIER_DIR / f"{hs6}.md"
        out_path.write_text(body, encoding="utf-8")
        synthesis_pending = not rationale.get("synthesis")
        below_floor = coverage["total"] < _COVERAGE_FLOOR
        if synthesis_pending:
            pending_count += 1
        if below_floor:
            below_floor_count += 1
        index.append({
            "hs6": hs6,
            "file": f"{hs6}.md",
            "hs4_label": hs4_label,
            "citations": [c.get("evidence_id") for c in citations],
            "n_citations": len(citations),
            "coverage": coverage,
            "below_floor": below_floor,
            "boundary_partners": [bp.get("partner_hs6") for bp in boundary_partners],
            "synthesis_pending": synthesis_pending,
        })

    INDEX_PATH.write_text(
        json.dumps({
            "inventory_mode": mode,
            "dossier_count": len(index),
            "n_synthesis_pending": pending_count,
            "n_below_floor": below_floor_count,
            "coverage_floor": _COVERAGE_FLOOR,
            "dossiers": index,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(index)} dossiers to {DOSSIER_DIR.relative_to(ROOT)}")
    print(f"wrote {INDEX_PATH.relative_to(ROOT)}")
    print(f"  synthesis pending: {pending_count}/{len(index)}")
    print(f"  below evidence-coverage floor: {below_floor_count}")

    # Build-report sidecar (separate from index.json so the index stays
    # browser-friendly; sidecar carries diagnostic detail).
    build_report = {
        "release": "working",
        "inventory_mode": mode,
        "inventory": {
            "size": len(inventory),
            "from_in_scope_yaml": sorted(_in_scope_hs6_set()),
        },
        "outputs": {
            "dossier_dir": str(DOSSIER_DIR.relative_to(ROOT)),
            "index": str(INDEX_PATH.relative_to(ROOT)),
            "rationales_template": str(RATIONALES_PATH.relative_to(ROOT)),
        },
        "stats": {
            "n_dossiers": len(index),
            "n_synthesis_pending": pending_count,
            "n_below_floor": below_floor_count,
            "below_floor_hs6": [
                d["hs6"] for d in index if d["below_floor"]
            ],
        },
        "next_steps": (
            "1. Fill synthesis paragraphs in "
            f"{RATIONALES_PATH.relative_to(ROOT)} "
            "(one per HS6, ~150 words each).\n"
            "2. Re-run scripts/build_hs6_dossiers.py to refresh dossier "
            "files with synthesis content.\n"
            "3. Build the release (scripts/build_release.py — once "
            "the audit pass completes) so the dossiers ship alongside "
            "the data."
        ),
    }
    BUILD_REPORT_PATH.write_text(
        json.dumps(build_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {BUILD_REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
