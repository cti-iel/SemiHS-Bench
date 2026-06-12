#!/usr/bin/env python3
"""Build the eval split - a balanced 900-record evaluation set.

Sources
-------
* ``release/part1/data/{dev.json,test_frozen.json}`` — 843 fully-validated
  records, already in the record schema (preferred when a bucket has a
  choice; existing dev/test split is discarded).
* ``audited/part2_audited_worksheet_ALL_tier1_REVIEWED.xlsx`` (sheet
  ``Review``) — 785 catalog rows that have been through a two-stage audit
  (an automated HS-2022 suggestion + a human reviewer verdict). The
  reviewer's ``my_suggested_hs6`` is the gold label and relabels ~307 rows, so
  the combined pool now spans ~38 HS4 codes (vs the original 12).

Balance
-------
The 900 records are allocated by **water-filling across every HS4 code** in the
combined pool (equal per-HS4 target, capped at availability, surplus
redistributed to the largest codes), and within each HS4 the same water-fill
balances the HS6 children. Within a final (HS4, HS6) bucket, part1 records
are taken first, then part2 fills the remainder.

Provenance
----------
part2 rows are catalog (auxiliary) and not yet through the formal citation
re-audit, so they keep ``label_source="catalog_expert_validated_pending_reaudit"``
and empty ``cited_evidence_ids``; ``confidence_tier`` is taken from the
reviewer's ``my_confidence``. The reviewer verdict/confidence/reasoning, the
pre-audit HS6, candidate customs rulings, source URL and manufacturer-asserted
HS are all recorded under ``provenance_part2`` for a strict audit trail.

Anonymization
-------------
Per project rule, the brand/manufacturer lives ONLY in the structured
``manufacturer`` field of the tier — it must not appear in the *description*.
After selection every record is passed through ``scrub_brand``: the tier2
``manufacturer`` field is **kept**, but brand tokens are removed from
``part_name``. tier1 and ``provenance_*`` are left untouched.

Two further rules (``polish_descriptions``): **the catalog supplier must
never be named** — its name and SKUs (``…-ND``) are stripped everywhere, and a
supplier SKU is never used as an MPN (those catalog records fall back to a
tier2 descriptor);
and **common nouns are lowercased** in tier2 descriptions (the ALL-CAPS
BOL text — ``HYDROGEN``→``hydrogen``) while codes/specs, chemical symbols and
known acronyms are preserved.

Output: ``release/eval/`` (single ``split="eval"`` file + corpus subset +
schema + taxonomy + manifest + balance report). Deterministic; re-running
overwrites the package.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._release_utils import (  # noqa: E402
    _dist, _git_commit, _sha256, _shorten, _write_corpus_subset,
)
from src.assembly.build_candidates import _record_seed  # noqa: E402
from src.utils.text_utils import normalize_text  # noqa: E402

P1_DIR = ROOT / "release" / "part1" / "data"
RECORD_SCHEMA = ROOT / "release" / "working" / "data" / "record_schema.json"
REF_CORPUS_SCHEMA = ROOT / "release" / "working" / "data" / "reference_corpus_schema.json"
_CONF_MAP = {"high": "high", "medium": "medium", "low": "low"}

# --- brand / manufacturer scrubbing for tier2 (project rule: tier2
#     must not mention the product's own brand or manufacturer name) ---
_CORP_STOP = {
    # corporate / legal / regional suffixes
    "inc", "co", "ltd", "llc", "corp", "corporation", "incorporated", "company",
    "gmbh", "ag", "kk", "plc", "sa", "bv", "oy", "limited", "group", "holdings",
    "global", "usa", "us", "america", "americas", "intl", "international",
    # generic industry words
    "technology", "technologies", "tech", "supplies", "supply", "industrial",
    "industries", "electronics", "electronic", "semiconductor", "semiconductors",
    "materials", "material", "systems", "system", "instruments", "instrument",
    "products", "product", "studio", "scientific", "labs", "laboratories",
    "components", "component",
    # generic product/technical words that also occur in maker names — must NOT
    # be stripped from descriptions (e.g. Alliance MEMORY, Maxim INTEGRATED,
    # Analog DEVICES, Power INTEGRATIONS)
    "memory", "integrated", "integrations", "devices", "device", "micro",
    "circuit", "circuits", "power", "sensor", "sensors", "solutions", "digital",
    "photonics", "optoelectronics", "energy", "audio", "wireless", "networks",
    "network", "communications", "storage", "logic", "control", "controls",
    "electric",
}
_BRAND_TRAIL = r"(?:\s+(?:pro|premium|plus|series|brand|genuine|original))?"
_BRAND_DETAIL_KEYS = {"shipper", "consignee", "manufacturer", "supplier",
                      "vendor", "brand", "maker", "mfr", "mfg"}


def _brand_tokens(mfr: str) -> List[str]:
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", (mfr or "").lower()) if t]
    return [t for t in toks if len(t) >= 2 and t not in _CORP_STOP]


def _strip_brand(text: str, manufacturers: Sequence[str]) -> str:
    """Remove full manufacturer phrases and their significant tokens (plus an
    optional trailing marketing word like 'PRO') from ``text``, case-insensitive."""
    out = str(text or "")
    if not out:
        return out
    for mfr in manufacturers:
        mfr = (mfr or "").strip()
        if not mfr:
            continue
        out = re.sub(re.escape(mfr), " ", out, flags=re.IGNORECASE)
        for tok in _brand_tokens(mfr):
            out = re.sub(r"\b" + re.escape(tok) + r"\b" + _BRAND_TRAIL, " ", out,
                         flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip(" -—:,;|/.")
    return re.sub(r"\s+", " ", out).strip()


def scrub_brand(rec: Dict[str, Any]) -> None:
    """Keep the product's brand/manufacturer ONLY in the structured ``manufacturer``
    field of the tier; remove it from the *description* (in place). The
    ``manufacturer`` field is preserved as-is; brand tokens are stripped from
    ``part_name`` so the manufacturer name never appears in the descriptive text."""
    t2 = dict(rec.get("tier2_minimal") or {})
    mfr2 = str(t2.get("manufacturer") or "")
    mfrs = [m for m in dict.fromkeys([mfr2]) if m]
    fallback = _shorten(_strip_brand(str(rec.get("tier1_description") or ""), mfrs))

    p2 = _strip_brand(t2.get("part_name") or "", mfrs)
    t2["part_name"] = p2 if len(p2) >= 1 else (fallback or "component")
    t2["manufacturer"] = mfr2  # keep the structured manufacturer field
    rec["tier2_minimal"] = t2


_DESC_BOILERPLATE = re.compile(
    r"^(?:a |an )?(?:specialty |multi-?component |industrial |bulk )?"
    r"(?:mixture of gases|gas mixture|mixture of|blend of|gases?)\s*[,:-]?\s*",
    flags=re.IGNORECASE)


def _short_descriptor(text: str, max_chars: int = 40) -> str:
    """Compress a description to a readable ≤40-char tier2 identifier (used when
    there is no MPN): drop boilerplate, tidy grade codes like ``X50S_4 %`` →
    ``X50S 4%``, truncate at a word boundary."""
    s = _DESC_BOILERPLATE.sub("", str(text or "").strip())
    s = re.sub(r"_(?=\d)", " ", s)        # X50S_4 -> X50S 4
    s = re.sub(r"\s+%", "%", s)           # "4 %" -> "4%"
    s = re.sub(r"[,;]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -—:;|/.")
    if len(s) > max_chars:
        cut = s[:max_chars]
        s = (cut[:cut.rfind(" ")] if " " in cut else cut).strip()
    return s or "component"


def finalize_tiers(rec: Dict[str, Any]) -> None:
    """Apply the MPN rules to the tier (in place), after brand scrubbing:

    * catalog records carry a real MPN → ensure it appears in tier2.part_name
      and woven into the tier1 natural-language description.
    * no-MPN records (BOL) → tier2.part_name becomes a short readable descriptor
      (2–3 words / ≤40 chars), not a raw grade code.
    """
    t2 = rec["tier2_minimal"]
    cand = str(t2.get("part_name") or "").strip()
    # A usable MPN must be a manufacturer part number, NOT a supplier SKU
    # (supplier SKUs / the supplier name must never appear in the dataset).
    has_mpn = (rec.get("tier1_source") == "catalog"
               and rec.get("tier2_provenance") == "natural_mpn"
               and cand and not _is_supplier_sku(cand))
    mpn = cand if has_mpn else ""

    if mpn:
        t2["part_name"] = mpn
        t1 = str(rec.get("tier1_description") or "")
        if mpn.lower() not in t1.lower():
            rec["tier1_description"] = t1.rstrip().rstrip(".") + f". Part number {mpn}."
    else:
        t2["part_name"] = _short_descriptor(t2.get("part_name") or rec.get("tier1_description") or "")
        if rec.get("tier2_provenance") == "natural_mpn":
            rec["tier2_provenance"] = "degraded_manual"  # tier2 is now a descriptor, not an MPN


# --- the catalog supplier must never be named; lowercase common nouns ---
# Catalog-supplier SKUs end in the reseller suffix "-ND"; they are scrubbed
# from all released text and never used as a manufacturer part number.
_SUPPLIER_SKU_RE = re.compile(r"\b[0-9A-Za-z][0-9A-Za-z./]*-ND\b")


def _is_supplier_sku(s: str) -> bool:
    s = str(s or "").strip()
    return bool(re.search(r"-ND$", s, re.IGNORECASE))


def _scrub_supplier(text: str) -> str:
    s = _SUPPLIER_SKU_RE.sub(" ", str(text or ""))
    s = re.sub(r"(?i)\bpart\s+number\s*[.,;:]*\s*$", "", s)  # dangling "Part number ." if SKU removed
    return re.sub(r"\s+", " ", s).strip(" .,-—:;|/")


_LOWER_SHORT = {"of", "in", "and", "to", "for", "the", "with", "or", "by", "as",
                "on", "at", "from", "per", "gas", "oil", "kit", "bal", "ppm"}
_KEEP_ACRONYM = {
    "IC", "ADC", "DAC", "DAQ", "HAT", "PWM", "ESD", "EMI", "EMC", "LED", "RGB",
    "USB", "HDMI", "SATA", "PCIE", "GPIO", "SPI", "UART", "JTAG", "RTC", "PMIC",
    "LDO", "MEMS", "SMD", "SMT", "TVS", "SCR", "IGBT", "MOSFET", "CMOS", "TQFP",
    "SOIC", "QFN", "HVQFN", "BGA", "LFBGA", "FPGA", "EEPROM", "SRAM", "DRAM",
    "EMMC", "RFID", "EUV", "DUV", "BARC", "TMAH", "PTFE", "PEEK", "PES", "PVC",
    "ABS", "NTC", "PTC", "SAC", "EB", "ESR", "DIMM", "SODIMM", "RDIMM", "NAND",
    "NOR", "SSD", "HDD", "DDR", "LPDDR", "ASIC", "SOC", "PHY", "PLL", "FIFO",
    "CAN", "PCB", "SMPS", "NMOS", "PMOS", "JFET", "BJT", "TRIAC", "LCD", "OLED",
    "TFT", "CCD", "GNSS", "GPS", "IMU", "NFC", "BLE", "RISC", "WLAN", "MCU", "MPU",
}


def _is_shouting(s: str) -> bool:
    """True if the string's letters are ≥80% uppercase (BOL-native / all-caps
    catalog titles). Mixed-case text is left alone so intentional acronyms stay."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8


def _lower_common(text: str) -> str:
    """In a shouting (all-caps) string, lowercase common-noun runs
    (HYDROGEN→hydrogen) while preserving codes/specs (digits), chemical symbols
    & mixed-case (AlF3, eMMC), and known acronyms (IC, MOSFET). No-op on
    already mixed-case text."""
    s = str(text or "")
    if not _is_shouting(s):
        return s

    def repl(m: "re.Match") -> str:
        w = m.group(0)
        if w in _KEEP_ACRONYM:
            return w
        if len(w) >= 4 or w.lower() in _LOWER_SHORT:
            return w.lower()
        return w

    # uppercase letter-runs not glued to digits/lowercase (so X50S, AlF3 survive)
    return re.sub(r"(?<![A-Za-z0-9])[A-Z]{2,}(?![A-Za-z0-9])", repl, s)


def polish_descriptions(rec: Dict[str, Any]) -> None:
    """Final description hygiene (in place): strip any supplier name/SKU and
    lowercase common nouns in tier2 ``part_name``.
    tier1 prose is only supplier-scrubbed (case left natural)."""
    t2 = rec["tier2_minimal"]
    t2["part_name"] = _lower_common(_scrub_supplier(t2.get("part_name") or "")) or "component"
    rec["tier1_description"] = _scrub_supplier(rec.get("tier1_description") or "")

OUT = ROOT / "release" / "eval"
OUT_DATA = OUT / "data"
RELEASE_LABEL = "eval"
SCHEMA_VERSION = "2.0.0"
TARGET = 900
CANDIDATE_SIZE = 4


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_p1() -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for f in ("dev.json", "test_frozen.json"):
        for r in json.loads((P1_DIR / f).read_text(encoding="utf-8")):
            r["_origin"] = "p1"
            recs.append(r)
    return recs


def _parse_id_list(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [tok.strip() for tok in str(value).replace(";", ",").split(",") if tok.strip()]


def _cell(row: Any, key: str) -> Any:
    """Return a worksheet cell or None (handles missing column / NaN)."""
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v


def _norm_hs6(v: Any) -> Optional[str]:
    """Normalize an HS6 cell (often a float like 280530.0) to a digit string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        s = str(int(float(s)))
    except ValueError:
        pass
    return s


def load_p2(path: Path) -> List[Dict[str, Any]]:
    """Map the REVIEWED part2 audit worksheet rows into the full record schema.

    Gold HS6 = the human reviewer's ``my_suggested_hs6`` (falls back to
    ``current_hs6``). ``confidence_tier`` comes from ``my_confidence``. The full
    reviewer audit trail is preserved under ``provenance_part2``.
    """
    df = pd.read_excel(path, sheet_name="Review")
    out: List[Dict[str, Any]] = []
    skipped_short = 0
    skipped_hs = 0
    relabeled = 0
    for _, row in df.iterrows():
        desc = str(_cell(row, "tier1_description") or "").strip()
        if len(desc) < 10:  # schema minLength
            skipped_short += 1
            continue
        current = _norm_hs6(_cell(row, "current_hs6"))
        gold = _norm_hs6(_cell(row, "my_suggested_hs6")) or current
        if not gold or len(gold) != 6 or not gold.isdigit():
            skipped_hs += 1
            continue
        if current and gold != current:
            relabeled += 1

        mfr = str(_cell(row, "manufacturer_hint") or "").strip()
        mpn = str(_cell(row, "manufacturer_part_number") or "").strip()
        orig = str(_cell(row, "original_description") or desc)
        t2_name = normalize_text(orig) or normalize_text(desc) or desc[:80]
        t3_name = mpn or _shorten(orig) or t2_name
        rulings = _parse_id_list(_cell(row, "candidate_reference_rulings"))
        mfr_hs = _cell(row, "mfr_asserted_hs")
        mfr_hs = "" if mfr_hs is None else str(mfr_hs)
        conf = _CONF_MAP.get(str(_cell(row, "my_confidence") or "").strip().lower(), "low")
        verdict = str(_cell(row, "my_verdict_on_suggested") or "")
        reasoning = str(_cell(row, "my_reasoning") or "").strip()

        out.append({
            "tier1_description": desc,
            "tier1_source": "catalog",
            "tier2_minimal": {"part_name": t3_name, "manufacturer": mfr},
            "tier2_provenance": "natural_mpn" if mpn else "degraded_manual",
            "tier2_classifiable": str(_cell(row, "tier2_classifiable") or "yes"),
            "hs6_label": gold, "hs4_label": gold[:4], "hs2_label": gold[:2],
            "scope_tier": str(_cell(row, "scope_tier") or "core"),
            "label_source": "catalog_expert_validated_pending_reaudit",
            "confidence_tier": conf,
            "cited_evidence_ids": [],
            "adjudication_status": "single_reviewer",
            "adjudication_winning_evidence_id": None,
            "adjudication_rubric_score": None,
            "difficulty_tags": [],
            "rationale_short": (reasoning[:197] + "...") if len(reasoning) > 200 else (
                reasoning or f"Reviewer-assigned HS6 {gold} (pending citation re-audit)."),
            "justification_text": (
                f"Catalog product record from {mfr or 'supplier'}; gold HS6 {gold} "
                f"assigned by human reviewer (verdict={verdict or 'n/a'}, "
                f"confidence={conf}). {reasoning} "
                f"Manufacturer-asserted HS: {mfr_hs or 'n/a'}. "
                f"{len(rulings)} candidate customs rulings considered (see provenance_part2)."
            ).strip(),
            "bol_metadata": None,
            "source_frozen_id": str(_cell(row, "frozen_id") or ""),
            "provenance_part2": {
                "audit_status": "reviewer_audited_pending_citation_reaudit",
                "reviewer_verdict": verdict,
                "reviewer_confidence": str(_cell(row, "my_confidence") or ""),
                "reviewer_reasoning": reasoning,
                "pre_audit_hs6": current,
                "relabeled": bool(current and gold != current),
                "source_url": str(_cell(row, "tier1_source_url") or ""),
                "candidate_reference_rulings": rulings,
                "mfr_asserted_hs": mfr_hs,
                "worksheet_record_id": str(_cell(row, "record_id") or ""),
            },
            "_origin": "p2",
        })
    if skipped_short or skipped_hs:
        print(f"  (skipped {skipped_short} short-description + {skipped_hs} bad-HS6 part2 rows)")
    print(f"  part2 reviewed: {len(out)} usable, {relabeled} relabeled vs current_hs6")
    return out


# ---------------------------------------------------------------------------
# Balanced selection
# ---------------------------------------------------------------------------

def dedupe_true_repeats(combined: Sequence[Dict[str, Any]]):
    """Drop only TRUE repeats, returning (pool, dropped_indices).

    A record is a repeat when, within the same ``hs6_label``, it has an
    identical normalized ``tier1_description`` as one already kept. Records are
    processed part1-first then longest-description-first, so the preferred
    record claims the slot (this also removes any part2 row that exactly
    repeats a part1 record).

    Two deliberate non-choices:
    * The near-duplicate detector in ``deduplicator.py`` is NOT used — its
      raw-0.95 threshold collapses templated catalog text that differs only by a
      short but semantically critical token (CdF2 vs CeF3 sputtering targets,
      8 GB vs 4 GB compute modules). Those are distinct products with the same
      gold label and must be kept, or the HS4/HS6 balance breaks.
    * MPN equality is NOT used to merge — part1's degraded tier2 identifiers
      are often series/family codes ("3DXT", "T3XX/T7XX"), so MPN-merging would
      drop distinct products that part1's authors deliberately kept.
    """
    order = sorted(range(len(combined)),
                   key=lambda i: (0 if combined[i]["_origin"] == "p1" else 1,
                                  -len(str(combined[i]["tier1_description"])), i))
    seen_desc: set = set()
    kept = [False] * len(combined)
    dropped: List[int] = []
    for i in order:
        r = combined[i]
        dkey = (r["hs6_label"], normalize_text(str(r["tier1_description"])))
        if dkey in seen_desc:
            dropped.append(i)
            continue
        seen_desc.add(dkey)
        kept[i] = True
    pool = [combined[i] for i in range(len(combined)) if kept[i]]
    return pool, dropped


def waterfill(caps: Mapping[str, int], total: int) -> Dict[str, int]:
    """Capped equal allocation summing to min(total, sum(caps)).

    Equal shares are handed out round by round; whatever cannot be split
    equally (because some codes hit their cap) is given out one unit at a time
    to the codes with the most remaining capacity (deterministic, ties by key).
    """
    alloc = {k: 0 for k in caps}
    remaining = min(total, sum(caps.values()))
    active = [k for k in caps if caps[k] > 0]
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            break
        for k in list(active):
            add = min(share, caps[k] - alloc[k])
            alloc[k] += add
            remaining -= add
            if alloc[k] >= caps[k]:
                active.remove(k)
    while remaining > 0:
        order = sorted((k for k in caps if alloc[k] < caps[k]),
                       key=lambda k: (-(caps[k] - alloc[k]), k))
        if not order:
            break
        for k in order:
            if remaining == 0:
                break
            alloc[k] += 1
            remaining -= 1
    return alloc


def select_balanced(pool: Sequence[Dict[str, Any]], target: int):
    """Return (selected, hs4_alloc) — water-filled HS4 then HS6, p1 preferred."""
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        buckets[(r["hs4_label"], r["hs6_label"])].append(r)
    # p1 first within each (hs4,hs6) bucket; stable, then by source id for determinism
    for key in buckets:
        buckets[key].sort(key=lambda r: (0 if r["_origin"] == "p1" else 1,
                                         str(r.get("source_frozen_id") or "")))

    hs4_avail: Counter = Counter()
    for (h4, _h6), lst in buckets.items():
        hs4_avail[h4] += len(lst)

    hs4_alloc = waterfill(dict(hs4_avail), target)
    selected: List[Dict[str, Any]] = []
    for h4, n4 in hs4_alloc.items():
        h6_caps = {h6: len(buckets[(h4, h6)]) for (hh4, h6) in buckets if hh4 == h4}
        for h6, n6 in waterfill(h6_caps, n4).items():
            selected.extend(buckets[(h4, h6)][:n6])
    return selected, hs4_alloc


# ---------------------------------------------------------------------------
# Candidate sets (cross-chapter fallback for sparse chapters like 38/71)
# ---------------------------------------------------------------------------

def build_eval_candidate_set(rec: Mapping[str, Any], pool: Sequence[str],
                             size: int = CANDIDATE_SIZE) -> Dict[str, Any]:
    gold = str(rec["hs6_label"])
    seed_key = str(rec.get("id") or rec.get("source_frozen_id") or "") + gold
    rng = random.Random(_record_seed(seed_key))
    sib = sorted(c for c in pool if c[:4] == gold[:4] and c != gold)
    chap = sorted(c for c in pool if c[:2] == gold[:2] and c != gold and c not in sib)
    rest = sorted(c for c in pool if c != gold and c[:2] != gold[:2])
    rng.shuffle(sib); rng.shuffle(chap); rng.shuffle(rest)

    distractors: List[str] = []
    used_sibling = False
    for code in sib:
        if len(distractors) >= size - 1:
            break
        distractors.append(code); used_sibling = True
    for code in chap + rest:
        if len(distractors) >= size - 1:
            break
        distractors.append(code)
    if len(distractors) < size - 1:
        raise ValueError(f"only {len(distractors)} distractors for {gold}; pool too small")

    construction = "sibling_heading" if used_sibling else "random_chapter"
    slate = [gold] + distractors
    rng.shuffle(slate)
    return {"size": len(slate), "codes": slate, "construction": construction,
            "gold_rank_in_candidates": slate.index(gold)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part2", type=Path, required=True,
                   help="path to part2 audited worksheet xlsx")
    p.add_argument("--target", type=int, default=TARGET)
    p.add_argument("--zip-dir", type=Path, default=Path.home() / "Downloads",
                   help="directory to write the release zip into")
    p.add_argument("--no-zip", action="store_true", help="skip writing the release zip")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    # ---- load ----
    p1 = load_p1()
    p2 = load_p2(args.part2)
    combined = p1 + p2
    print(f"loaded: part1={len(p1)}  part2={len(p2)}  combined={len(combined)}")

    # ---- dedup: true repeats only (see dedupe_true_repeats docstring) ----
    pool, drop = dedupe_true_repeats(combined)
    if drop:
        dp = sum(1 for i in drop if combined[i]["_origin"] == "p1")
        print(f"dedup: removed {len(drop)} exact repeats (p1={dp}, p2={len(drop) - dp}); pool={len(pool)}")
    else:
        print(f"dedup: no exact repeats found; pool={len(pool)}")

    if len(pool) < args.target:
        raise SystemExit(f"pool {len(pool)} < target {args.target}")

    # ---- balanced selection ----
    selected, hs4_alloc = select_balanced(pool, args.target)
    assert len(selected) == args.target, f"selected {len(selected)} != {args.target}"

    # ---- scrub brand, apply MPN rules, then polish (no supplier name, lowercase) ----
    for r in selected:
        scrub_brand(r)
        finalize_tiers(r)
        polish_descriptions(r)
    print("scrubbed brand + MPN/tier2 rules + supplier-name removal + lowercasing (900 records)")

    # ---- candidate sets: keep p1's; build for p2 (and any record missing one) ----
    code_pool = sorted({r["hs6_label"] for r in selected}
                       | {c for r in selected if r["_origin"] == "p1"
                          for c in (r.get("candidate_set") or {}).get("codes") or []})
    for r in selected:
        if r["_origin"] == "p2" or not r.get("candidate_set"):
            r["candidate_set"] = build_eval_candidate_set(r, code_pool)

    # ---- assign uniform ids / single eval split ----
    selected.sort(key=lambda r: (r["hs4_label"], r["hs6_label"], r["_origin"],
                                 str(r.get("source_frozen_id") or "")))
    for n, r in enumerate(selected, start=1):
        r["source_frozen_id"] = str(r.get("source_frozen_id") or r.get("frozen_id") or "")
        r["id"] = f"SH-{n:04d}"
        r["split"] = "eval"
        r["frozen_id"] = f"v2.0.eval.{n:04d}"
    origins = Counter(r.pop("_origin") for r in selected)
    print(f"selected {len(selected)}: by origin {dict(origins)}")

    # ---- schema (variant allowing split='eval'/'train') ----
    schema = json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
    schema["properties"]["split"]["enum"] = ["dev", "test", "eval", "train"]
    schema["properties"]["frozen_id"]["pattern"] = \
        r"^v(1\.5|2\.0)\.(dev|test|eval|train)\.\d{4,}(\.[a-z_]+)?$"

    import jsonschema  # type: ignore
    validator = jsonschema.Draft202012Validator(schema)
    errs = [f"{r['id']}: {e.message}" for r in selected for e in validator.iter_errors(r)]
    if errs:
        print(f"✗ {len(errs)} schema errors (first 12):", file=sys.stderr)
        for e in errs[:12]:
            print("   -", e, file=sys.stderr)
        raise SystemExit(2)
    print(f"✓ all {len(selected)} records pass record_schema validation")

    if args.dry_run:
        _print_balance(selected, hs4_alloc, origins)
        print("DRY RUN — no writes.")
        return 0

    # ---- write package ----
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    eval_path = OUT_DATA / "eval.json"
    eval_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    corpus_info = _write_corpus_subset(selected, OUT_DATA / "reference_corpus.jsonl")
    if corpus_info["unresolved_citations"]:
        print(f"WARN: {len(corpus_info['unresolved_citations'])} unresolved citations "
              f"(first 5): {corpus_info['unresolved_citations'][:5]}", file=sys.stderr)
    print(f"corpus subset: {corpus_info['entries']} entries ({corpus_info['by_source']})")

    (OUT_DATA / "record_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    shutil.copy(REF_CORPUS_SCHEMA, OUT_DATA / "reference_corpus_schema.json")

    hs_rows = sorted({(str(r["hs6_label"]), str(r["hs4_label"]), str(r["hs2_label"]),
                       str(r.get("scope_tier") or "")) for r in selected})
    with (OUT_DATA / "taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["hs6", "hs4", "hs2", "scope_tier"]); w.writerows(hs_rows)

    by_source = dict(sorted(Counter(str(r["tier1_source"]) for r in selected).items()))
    manifest = {
        "release": RELEASE_LABEL, "interim": True, "ready_for_distribution": False,
        "schema_version": SCHEMA_VERSION, "hs_version": "HS2022",
        "built_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "split_scheme": "single 'eval' split (prior dev/test/train splits discarded)",
        "balance": "water-fill across HS4 (priority) then HS6 within HS4; part1 preferred per bucket",
        "coverage_note": (
            "eval split = balanced 900-record evaluation set drawn from part1 "
            f"({origins['p1']}) + the REVIEWED part2 catalog audit ({origins['p2']}). "
            "part2 gold = reviewer my_suggested_hs6 (relabels many rows, widening "
            "HS4 coverage); confidence_tier from reviewer my_confidence; "
            "label_source=catalog_expert_validated_pending_reaudit with empty "
            "cited_evidence_ids (full audit trail in provenance_part2). Balanced "
            "across all HS4 codes present in the combined pool."
        ),
        "record_count": {"total": len(selected)},
        "by_origin": dict(origins),
        "by_source_tier1": by_source,
        "hs4_allocation": dict(sorted(hs4_alloc.items())),
        "distributions": _dist(selected),
        "reference_corpus": {"entries": corpus_info["entries"],
                             "by_source": corpus_info["by_source"],
                             "by_jurisdiction": corpus_info["by_jurisdiction"]},
        "hashes": {"file_sha256": {
            "eval.json": _sha256(eval_path),
            "reference_corpus.jsonl": _sha256(OUT_DATA / "reference_corpus.jsonl")}},
    }
    (OUT_DATA / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_balance_report(selected, hs4_alloc, origins, manifest)

    # ---- complete the package: eval harness + prompts + example + docs ----
    shutil.copytree(ROOT / "release" / "working" / "eval", OUT / "eval")
    shutil.copytree(ROOT / "release" / "working" / "prompts", OUT / "prompts")
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    example = {
        "submission": {"name": "oracle-constrained-tier1.example", "model_id": "oracle",
                       "mode": "constrained", "tier": 1, "schema_version": SCHEMA_VERSION,
                       "notes": "Example - gold-first ranking on the first 5 eval records (eval split)."},
        "predictions": [],
    }
    for r in sorted(selected, key=lambda x: x["frozen_id"])[:5]:
        codes = list((r.get("candidate_set") or {}).get("codes") or [])
        gold = str(r["hs6_label"])
        example["predictions"].append({"frozen_id": r["frozen_id"],
                                       "ranked_codes": [gold] + [c for c in codes if c != gold]})
    (OUT / "examples" / "submission_constrained.example.json").write_text(
        json.dumps(example, indent=2) + "\n", encoding="utf-8")
    _write_eval_docs(selected, manifest, origins)

    _print_balance(selected, hs4_alloc, origins)
    print(f"\nwrote package: {OUT.relative_to(ROOT)}/  (eval.json = {len(selected)} records)")

    # ---- zip the package into the Downloads folder (final deliverable) ----
    if not args.no_zip:
        import zipfile
        args.zip_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d")
        zip_path = args.zip_dir / f"semihs_eval_{stamp}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(OUT.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(Path("eval") / path.relative_to(OUT)))
        print(f"zipped → {zip_path}  ({zip_path.stat().st_size // 1024} KB)")
    return 0


def _print_balance(selected, hs4_alloc, origins) -> None:
    hs4 = Counter(r["hs4_label"] for r in selected)
    print("\nHS4 distribution (alloc == realized):")
    for h4 in sorted(hs4_alloc, key=lambda k: -hs4_alloc[k]):
        print(f"  {h4}: {hs4[h4]}")
    print(f"by origin: {dict(origins)}  | total {len(selected)}")


def _write_balance_report(selected, hs4_alloc, origins, manifest) -> None:
    hs4 = Counter(r["hs4_label"] for r in selected)
    by_h6 = defaultdict(Counter)
    for r in selected:
        by_h6[r["hs4_label"]][r["hs6_label"]] += 1
    conf = manifest["distributions"]["by_confidence_tier"]
    scope = manifest["distributions"]["by_scope_tier"]
    src = manifest["by_source_tier1"]

    lines = [
        f"# Eval split - balance report ({len(selected)} records)", "",
        f"- **By source** (`tier1_source`): {src}",
        f"- **By origin**: {dict(origins)} (part1 reused / part2 added)",
        f"- **By confidence tier**: {conf}",
        f"- **By scope tier**: {scope}",
        f"- **Distinct HS4**: {len(hs4)}  |  **Distinct HS6**: "
        f"{len({r['hs6_label'] for r in selected})}", "",
        "## HS4 allocation (target == realized)", "",
        "| HS4 | records |", "|---|---|",
    ]
    for h4 in sorted(hs4_alloc, key=lambda k: -hs4_alloc[k]):
        lines.append(f"| {h4} | {hs4[h4]} |")
    lines += ["", "## HS6 spread within each HS4", ""]
    for h4 in sorted(by_h6, key=lambda k: -hs4[k]):
        spread = dict(sorted(by_h6[h4].items()))
        lo, hi = min(spread.values()), max(spread.values())
        lines.append(f"- **{h4}** ({hs4[h4]} across {len(spread)} HS6, min={lo} max={hi}): {spread}")
    (OUT / "BALANCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_eval_docs(selected, manifest, origins) -> None:
    total = len(selected)
    d = manifest["distributions"]
    hs4 = d["by_hs4"]
    conf = d["by_confidence_tier"]
    scope = d["by_scope_tier"]
    src = manifest["by_source_tier1"]
    hs6_n = len(d["by_hs6"])

    (OUT / "README.md").write_text(f"""# SemiHS-Bench - Evaluation Set (eval split)

> **Balanced {total}-record evaluation set** for single-label HS6 tariff
> classification of semiconductor-supply-chain goods. One held-out `eval` split
> (prior dev/test/train splits are discarded).

## Composition
- **By source** (`tier1_source`): {src}
- **By origin**: part1 {origins['p1']} / part2 (re-audited catalog) {origins['p2']}
- **Coverage**: {len(hs4)} HS4 families, {hs6_n} HS6 codes — balanced by water-fill
  across HS4 (priority) then HS6 within each HS4.
- **Confidence**: {conf}  ·  **Scope**: {scope}

## Contents
- `data/eval.json` - the {total} records (full record schema, `split="eval"`)
- `data/record_schema.json` — per-record JSON Schema
- `data/reference_corpus.jsonl` — EBTI/CROSS/JP_CUSTOMS rulings cited by these
  records (every `cited_evidence_ids` entry resolves here)
- `data/reference_corpus_schema.json`, `data/taxonomy.csv`, `data/MANIFEST.json`
- `eval/` — scoring harness; `prompts/` — constrained + open top-k prompts
- `examples/submission_constrained.example.json` — oracle example
- `BALANCE_REPORT.md`, `STATISTICS.md`, `DATASHEET.md`

## Two input tiers
- **Tier 1** (`tier1_description`) — full natural-language description (MPN woven
  in for catalog parts).
- **Tier 2** (`tier2_minimal`) — MPN, else a ≤40-char descriptor.

## Anonymization rules
- The product's **brand/manufacturer name never appears in the description text**
  (`part_name` / specs); it lives only in the structured `manufacturer` field.
- **The catalog supplier is never named**, and supplier SKUs/internal part numbers are excluded.
- Common nouns in all-caps source text are lowercased; codes, specs, chemical
  symbols and acronyms are preserved.

## How to evaluate
```bash
python3 eval/score_submission.py --submission your_submission.json \\
    --data data/eval.json
```
See `examples/submission_constrained.example.json` for the format.

Built {manifest['built_at']} · commit {manifest['git_commit'][:12]}.
""", encoding="utf-8")

    (OUT / "DATASHEET.md").write_text(f"""# DATASHEET - SemiHS-Bench eval split

**Status:** balanced evaluation set ({total} records), single `eval` split.

- **Task:** single-label HS6 tariff classification, semiconductor supply chain.
- **Sources:** part1 (BOL + catalog, expert-validated) + part2 (re-audited
  catalog). By source: {src}.
- **Labels:** gold HS6 per record; part2 golds are the human reviewer's
  `my_suggested_hs6`. Confidence: {conf}.
- **Balance:** water-fill across {len(hs4)} HS4 families, then HS6 within each.
- **Anonymization:** no brand/manufacturer in descriptions (structured field only);
  no supplier name or SKUs; common nouns lowercased.
- **Limitations:** part2 rows are `catalog_expert_validated_pending_reaudit`
  (citation re-audit pending; `cited_evidence_ids` empty, candidate rulings in
  `provenance_part2`). Long tail of small HS4 families retained intentionally.
- **License / citation:** inherit from the parent SemiHS-Bench release.
""", encoding="utf-8")

    def block(title, dd):
        return f"### {title}\n\n| key | count |\n|---|---|\n" + "".join(
            f"| {k} | {v} |\n" for k, v in dd.items())
    (OUT / "STATISTICS.md").write_text(
        f"# STATISTICS - eval split ({total} records)\n\n"
        + block("By source (tier1_source)", src) + "\n"
        + block("By origin", dict(origins)) + "\n"
        + block("By HS4", dict(sorted(hs4.items()))) + "\n"
        + block("By confidence tier", conf) + "\n"
        + block("By scope tier", scope),
        encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
