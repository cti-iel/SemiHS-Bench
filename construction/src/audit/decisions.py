"""Parse expert audit decisions and turn them into a corrections payload.

The reviewer fills out a CSV worksheet (one row per record). This module
parses that CSV, validates it strictly, and produces a structured
``Corrections`` object that can be applied to records during freeze.

Validation is conservative: any malformed row stops the parse with a clear
error message. The reviewer is told *exactly* which row to fix.

The worksheet follows the Core-4 evidence protocol (see
docs/ADJUDICATION_PROTOCOL.md). Required per row: ``frozen_id``, ``action``
(``confirm`` / ``change`` / ``drop``), ``new_hs6`` (when changing),
``expert_hs6``, ``confidence_tier``, ``cited_evidence_ids``,
``rationale_short``, ``tier2_classifiable``, ``adjudication_status``,
``adjudication_winning_evidence_id``, ``adjudication_rubric_score``.

Validation enforces the evidence-count → confidence-tier binding
(``high`` requires ≥ 2 cross-jurisdiction citations, ``medium`` requires
≥ 1, ``low`` allowed with 0) and the adjudication-status → rubric-score
binding.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.models import TIER2_CLASSIFIABILITY


VALID_ACTIONS = ("confirm", "change", "drop")
# Reviewer-friendly aliases. Maps any synonym to the canonical action.
ACTION_ALIASES = {
    # confirm
    "confirm": "confirm",
    "correct": "confirm",
    "ok": "confirm",
    "keep": "confirm",
    "verified": "confirm",
    # change
    "change": "change",
    "wrong": "change",
    "relabel": "change",
    "update": "change",
    "fix": "change",
    # drop
    "drop": "drop",
    "remove": "drop",
    "delete": "drop",
    "exclude": "drop",
}
HS6_RE = re.compile(r"^\d{6}$")

# Core-4 protocol fields.
CONFIDENCE_TIERS = ("high", "medium", "low")
TIER2_CLASSIFIABLE_VALUES = tuple(sorted(TIER2_CLASSIFIABILITY))
ADJUDICATION_STATUSES = (
    "single_reviewer",
    "adjudicated_consensus",
    "adjudicated_evidence_resolved",   # rubric-resolved
    "adjudicated_direct",              # direct tie-break
    "unresolved_dropped",              # drop path
    "pending",                         # carryover not yet audited
)
# Statuses that REQUIRE adjudication_rubric_score + winning_evidence_id.
_ADJUDICATION_REQUIRES_RUBRIC = {"adjudicated_evidence_resolved"}


@dataclass
class Decision:
    row_kind: str
    frozen_id: str
    record_id: str
    current_hs6: str
    action: str
    new_hs6: Optional[str]
    notes: str
    # Evidence/adjudication fields (Optional / default-empty).
    expert_hs6: Optional[str] = None
    confidence_tier: Optional[str] = None
    cited_evidence_ids: List[str] = field(default_factory=list)
    rationale_short: Optional[str] = None
    tier2_classifiable: Optional[str] = None
    adjudication_status: Optional[str] = None
    adjudication_winning_evidence_id: Optional[str] = None
    adjudication_rubric_score: Optional[int] = None


@dataclass
class Corrections:
    """Structured payload applied to records during freeze.

    * ``confirmed``: ``frozen_id``s whose ``label_source`` should flip to
      the validated form (``BOL_expert_validated`` /
      ``catalog_expert_validated``).
    * ``relabeled``: ``frozen_id`` → new HS6.
    * ``dropped``: ``frozen_id``s removed from the benchmark.
    * ``decisions``: full list, including notes + evidence fields, for archival.
    """

    confirmed: List[str] = field(default_factory=list)
    relabeled: Dict[str, str] = field(default_factory=dict)
    dropped: List[str] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)


class DecisionParseError(ValueError):
    """Raised when the worksheet has malformed rows."""


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _parse_cited_evidence_ids(raw: str) -> List[str]:
    """Comma/semicolon separated evidence_ids → list of trimmed strings."""
    s = _strip(raw)
    if not s:
        return []
    return [tok.strip() for tok in re.split(r"[,;]", s) if tok.strip()]


def _validate_row(
    row: Mapping[str, str],
    line_no: str,
    frozen_id: str,
    action: str,
    new_hs6: str,
    errors: List[str],
) -> Dict[str, Any]:
    """Validate the Core-4 evidence columns. Returns the parsed fields as a
    dict (with keys matching the Decision dataclass fields). Errors are
    appended to ``errors``; the caller decides whether to abort.

    Rules (see docs/ADJUDICATION_PROTOCOL.md):

    * ``expert_hs6`` required for action=confirm or change. For confirm
      it must equal current_hs6; for change it must equal new_hs6.
      For drop it is optional.
    * ``confidence_tier`` required and ∈ {high, medium, low}, EXCEPT when
      adjudication_status == "pending" (carryover not yet audited) or
      action == "drop" with status == "unresolved_dropped".
    * ``cited_evidence_ids`` count vs confidence_tier:
        high   → ≥ 2 citations
        medium → ≥ 1
        low    → any (including 0)
    * ``tier2_classifiable`` ∈ {yes, partial, no} for non-drop rows.
    * ``adjudication_status`` required and ∈ ADJUDICATION_STATUSES.
    * If status == ``adjudicated_evidence_resolved``: both
      ``adjudication_rubric_score`` (0–6) and
      ``adjudication_winning_evidence_id`` are required.
    * If status == ``unresolved_dropped``: action MUST be ``drop``.
    """
    out: Dict[str, Any] = {
        "expert_hs6": None,
        "confidence_tier": None,
        "cited_evidence_ids": [],
        "rationale_short": None,
        "tier2_classifiable": None,
        "adjudication_status": None,
        "adjudication_winning_evidence_id": None,
        "adjudication_rubric_score": None,
    }
    initial_error_count = len(errors)

    # adjudication_status
    raw_adj = _strip(row.get("adjudication_status")).lower()
    if not raw_adj:
        errors.append(
            f"line {line_no} ({frozen_id}): the protocol requires "
            f"adjudication_status; expected one of {ADJUDICATION_STATUSES}"
        )
    elif raw_adj not in ADJUDICATION_STATUSES:
        errors.append(
            f"line {line_no} ({frozen_id}): unknown adjudication_status "
            f"{raw_adj!r}; expected one of {ADJUDICATION_STATUSES}"
        )
    else:
        out["adjudication_status"] = raw_adj

    # confidence_tier
    raw_conf = _strip(row.get("confidence_tier")).lower()
    needs_confidence = raw_adj != "pending" and not (
        action == "drop" and raw_adj == "unresolved_dropped"
    )
    if needs_confidence:
        if not raw_conf:
            errors.append(
                f"line {line_no} ({frozen_id}): the protocol requires "
                f"confidence_tier; expected one of {CONFIDENCE_TIERS}"
            )
        elif raw_conf not in CONFIDENCE_TIERS:
            errors.append(
                f"line {line_no} ({frozen_id}): unknown confidence_tier "
                f"{raw_conf!r}; expected one of {CONFIDENCE_TIERS}"
            )
        else:
            out["confidence_tier"] = raw_conf
    elif raw_conf:
        # Allowed to be present, but if so it must still be valid.
        if raw_conf in CONFIDENCE_TIERS:
            out["confidence_tier"] = raw_conf
        else:
            errors.append(
                f"line {line_no} ({frozen_id}): unknown confidence_tier "
                f"{raw_conf!r}; expected one of {CONFIDENCE_TIERS}"
            )

    # cited_evidence_ids
    raw_cited = row.get("cited_evidence_ids", "")
    cited = _parse_cited_evidence_ids(raw_cited)
    out["cited_evidence_ids"] = cited

    # Evidence-count → confidence-tier binding.
    if out["confidence_tier"] == "high" and len(cited) < 2:
        errors.append(
            f"line {line_no} ({frozen_id}): confidence_tier=high requires "
            f"≥ 2 cited_evidence_ids (got {len(cited)}). "
            f"Down-rank to medium/low if only one citation is available."
        )
    elif out["confidence_tier"] == "medium" and len(cited) < 1:
        errors.append(
            f"line {line_no} ({frozen_id}): confidence_tier=medium requires "
            f"≥ 1 cited_evidence_ids (got 0). Down-rank to low if no "
            f"citation is available."
        )

    # expert_hs6
    raw_expert = _strip(row.get("expert_hs6"))
    if action in ("confirm", "change"):
        if not HS6_RE.match(raw_expert):
            errors.append(
                f"line {line_no} ({frozen_id}): the protocol requires "
                f"expert_hs6 (6 digits) for action={action} (got {raw_expert!r})"
            )
        else:
            out["expert_hs6"] = raw_expert
            current_hs6 = _strip(row.get("current_hs6"))
            if action == "confirm" and current_hs6 and raw_expert != current_hs6:
                errors.append(
                    f"line {line_no} ({frozen_id}): action=confirm with "
                    f"expert_hs6={raw_expert!r} disagrees with current_hs6="
                    f"{current_hs6!r}; use action=change instead"
                )
            if action == "change" and new_hs6 and raw_expert != new_hs6:
                errors.append(
                    f"line {line_no} ({frozen_id}): action=change with "
                    f"expert_hs6={raw_expert!r} disagrees with new_hs6="
                    f"{new_hs6!r}"
                )
    elif raw_expert:
        # action=drop may carry an expert_hs6 (the expert's preferred label
        # before adjudication failed); preserve it if valid.
        if HS6_RE.match(raw_expert):
            out["expert_hs6"] = raw_expert
        else:
            errors.append(
                f"line {line_no} ({frozen_id}): expert_hs6 must be 6 digits "
                f"when populated (got {raw_expert!r})"
            )

    # tier2_classifiable
    raw_t2 = _strip(row.get("tier2_classifiable")).lower()
    if raw_t2:
        if raw_t2 not in TIER2_CLASSIFIABLE_VALUES:
            errors.append(
                f"line {line_no} ({frozen_id}): unknown tier2_classifiable "
                f"{raw_t2!r}; expected one of {TIER2_CLASSIFIABLE_VALUES}"
            )
        else:
            out["tier2_classifiable"] = raw_t2
    elif action != "drop":
        # The protocol lists this as a required expert assignment.
        errors.append(
            f"line {line_no} ({frozen_id}): the protocol requires "
            f"tier2_classifiable for non-drop rows; expected one of "
            f"{TIER2_CLASSIFIABLE_VALUES}"
        )

    # rationale_short
    raw_rat = _strip(row.get("rationale_short"))
    if len(raw_rat) > 200:
        errors.append(
            f"line {line_no} ({frozen_id}): rationale_short exceeds 200 "
            f"chars ({len(raw_rat)})"
        )
    if raw_rat:
        out["rationale_short"] = raw_rat

    # adjudication_winning_evidence_id + rubric_score
    raw_winner = _strip(row.get("adjudication_winning_evidence_id"))
    if raw_winner:
        out["adjudication_winning_evidence_id"] = raw_winner

    raw_rubric = _strip(row.get("adjudication_rubric_score"))
    if raw_rubric:
        try:
            score = int(raw_rubric)
        except ValueError:
            errors.append(
                f"line {line_no} ({frozen_id}): adjudication_rubric_score "
                f"must be an integer (got {raw_rubric!r})"
            )
        else:
            if not (0 <= score <= 6):
                errors.append(
                    f"line {line_no} ({frozen_id}): adjudication_rubric_score "
                    f"out of range (got {score}; expected 0–6 per the adjudication rubric)"
                )
            else:
                out["adjudication_rubric_score"] = score

    # Status-specific consistency.
    if out["adjudication_status"] in _ADJUDICATION_REQUIRES_RUBRIC:
        if out["adjudication_rubric_score"] is None:
            errors.append(
                f"line {line_no} ({frozen_id}): adjudication_status="
                f"{out['adjudication_status']} requires adjudication_rubric_score"
            )
        if not out["adjudication_winning_evidence_id"]:
            errors.append(
                f"line {line_no} ({frozen_id}): adjudication_status="
                f"{out['adjudication_status']} requires "
                f"adjudication_winning_evidence_id"
            )

    if out["adjudication_status"] == "unresolved_dropped" and action != "drop":
        errors.append(
            f"line {line_no} ({frozen_id}): adjudication_status="
            f"unresolved_dropped requires action=drop (got action={action!r})"
        )

    return out if len(errors) == initial_error_count else out


def parse_worksheet(path: Path) -> Corrections:
    """Read the completed CSV and return validated corrections.

    Every row is validated against the Core-4 evidence protocol
    (docs/ADJUDICATION_PROTOCOL.md).
    """
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):  # start=2 to match line numbers
            row["_line_no"] = str(idx)
            rows.append(row)

    if not rows:
        raise DecisionParseError(f"{path}: worksheet is empty")

    corrections = Corrections()
    errors: List[str] = []
    seen_ids: set = set()

    for row in rows:
        line_no = row["_line_no"]
        frozen_id = _strip(row.get("frozen_id"))
        if not frozen_id:
            errors.append(f"line {line_no}: blank frozen_id")
            continue
        if frozen_id in seen_ids:
            errors.append(f"line {line_no}: duplicate frozen_id {frozen_id!r}")
            continue
        seen_ids.add(frozen_id)

        raw_action = _strip(row.get("action")).lower()
        if not raw_action:
            errors.append(
                f"line {line_no} ({frozen_id}): action is blank — "
                f"expected one of {VALID_ACTIONS}"
            )
            continue
        action = ACTION_ALIASES.get(raw_action)
        if action is None:
            errors.append(
                f"line {line_no} ({frozen_id}): unknown action {raw_action!r} — "
                f"expected one of {sorted(ACTION_ALIASES.keys())}"
            )
            continue

        new_hs6 = _strip(row.get("new_hs6"))
        if action == "change":
            if not HS6_RE.match(new_hs6):
                errors.append(
                    f"line {line_no} ({frozen_id}): action=change requires a "
                    f"6-digit new_hs6, got {new_hs6!r}"
                )
                continue
        else:
            if new_hs6:
                errors.append(
                    f"line {line_no} ({frozen_id}): new_hs6 must be blank "
                    f"when action={action} (got {new_hs6!r})"
                )
                continue

        fields = _validate_row(
            row, line_no, frozen_id, action, new_hs6, errors,
        )

        decision = Decision(
            row_kind=_strip(row.get("row_kind")),
            frozen_id=frozen_id,
            record_id=_strip(row.get("record_id")),
            current_hs6=_strip(row.get("current_hs6")),
            action=action,
            new_hs6=new_hs6 or None,
            notes=_strip(row.get("notes")),
            expert_hs6=fields.get("expert_hs6"),
            confidence_tier=fields.get("confidence_tier"),
            cited_evidence_ids=fields.get("cited_evidence_ids") or [],
            rationale_short=fields.get("rationale_short"),
            tier2_classifiable=fields.get("tier2_classifiable"),
            adjudication_status=fields.get("adjudication_status"),
            adjudication_winning_evidence_id=fields.get(
                "adjudication_winning_evidence_id"
            ),
            adjudication_rubric_score=fields.get("adjudication_rubric_score"),
        )
        corrections.decisions.append(decision)

        if action == "confirm":
            corrections.confirmed.append(frozen_id)
        elif action == "change":
            assert new_hs6 is not None
            corrections.relabeled[frozen_id] = new_hs6
        elif action == "drop":
            corrections.dropped.append(frozen_id)

    if errors:
        msg = f"{path}: {len(errors)} validation error(s):\n  - " + "\n  - ".join(errors)
        raise DecisionParseError(msg)

    return corrections


def apply_corrections(
    records: List[Mapping[str, Any]],
    corrections: Corrections,
) -> List[Dict[str, Any]]:
    """Apply parsed corrections to a record list. Returns a new list.

    * confirm: BOL_expert_validated_pending_reaudit → BOL_expert_validated;
               catalog_expert_validated_pending_reaudit → catalog_expert_validated.
               Plus: populate the Core-4 fields (confidence_tier,
               cited_evidence_ids, rationale_short, tier2_classifiable,
               adjudication_*) on the record from the Decision.
    * change:  set hs6/hs4/hs2; label_source = expert_relabeled;
               populate the Core-4 fields as above.
    * drop:    omit the record entirely.
    """
    confirmed = set(corrections.confirmed)
    relabeled = dict(corrections.relabeled)
    dropped = set(corrections.dropped)
    by_id: Dict[str, Decision] = {d.frozen_id: d for d in corrections.decisions}

    out: List[Dict[str, Any]] = []
    for record in records:
        frozen_id = str(record.get("frozen_id") or "")
        if frozen_id in dropped:
            continue

        new_record = dict(record)
        if frozen_id in confirmed:
            current_source = new_record.get("label_source")
            if current_source == "BOL_expert_validated_pending_reaudit":
                new_record["label_source"] = "BOL_expert_validated"
            elif current_source == "catalog_expert_validated_pending_reaudit":
                new_record["label_source"] = "catalog_expert_validated"

        if frozen_id in relabeled:
            new_hs6 = relabeled[frozen_id]
            new_record["hs6_label"] = new_hs6
            new_record["hs4_label"] = new_hs6[:4]
            new_record["hs2_label"] = new_hs6[:2]
            new_record["label_source"] = "expert_relabeled"

        # Populate the Core-4 fields on the record when present.
        if frozen_id in by_id:
            decision = by_id[frozen_id]
            if decision.confidence_tier is not None:
                new_record["confidence_tier"] = decision.confidence_tier
            if decision.cited_evidence_ids:
                new_record["cited_evidence_ids"] = list(decision.cited_evidence_ids)
            if decision.rationale_short is not None:
                new_record["rationale_short"] = decision.rationale_short
            if decision.tier2_classifiable is not None:
                new_record["tier2_classifiable"] = decision.tier2_classifiable
            if decision.adjudication_status is not None:
                new_record["adjudication_status"] = decision.adjudication_status
            if decision.adjudication_winning_evidence_id is not None:
                new_record["adjudication_winning_evidence_id"] = (
                    decision.adjudication_winning_evidence_id
                )
            if decision.adjudication_rubric_score is not None:
                new_record["adjudication_rubric_score"] = (
                    decision.adjudication_rubric_score
                )

        out.append(new_record)

    return out


def corrections_to_dict(corrections: Corrections) -> Dict[str, Any]:
    return {
        "confirmed": sorted(corrections.confirmed),
        "relabeled": dict(sorted(corrections.relabeled.items())),
        "dropped": sorted(corrections.dropped),
        "decisions": [
            {
                "row_kind": d.row_kind,
                "frozen_id": d.frozen_id,
                "record_id": d.record_id,
                "current_hs6": d.current_hs6,
                "action": d.action,
                "new_hs6": d.new_hs6,
                "notes": d.notes,
                "expert_hs6": d.expert_hs6,
                "confidence_tier": d.confidence_tier,
                "cited_evidence_ids": list(d.cited_evidence_ids),
                "rationale_short": d.rationale_short,
                "tier2_classifiable": d.tier2_classifiable,
                "adjudication_status": d.adjudication_status,
                "adjudication_winning_evidence_id": d.adjudication_winning_evidence_id,
                "adjudication_rubric_score": d.adjudication_rubric_score,
            }
            for d in corrections.decisions
        ],
    }


# ---------------------------------------------------------------------------
# Worksheet column template (used by scripts/generate_review_worksheet.py)
# ---------------------------------------------------------------------------

WORKSHEET_COLUMNS = (
    # Stable identifiers.
    "row_kind",
    "frozen_id",
    "record_id",
    "split",
    "current_hs6",
    "current_hs4",
    "label_source",
    "tier1_description",
    # Audit-context fields surfaced to the reviewer.
    "candidate_reference_rulings",  # up to 5 EBTI/CROSS evidence_ids in same HS4
    "manufacturer_hint",
    "manufacturer_part_number",  # informational MPN from the catalog
                                 # source; blank for BOL rows.
    "scope_tier",  # core | supply_chain
    "mfr_asserted_hs",  # informational; populated for catalog rows from
                        # the catalog's manufacturer-self-asserted HTSUS code
                        # (e.g. "8541.10.0080"). NOT authoritative — surfaced
                        # to the reviewer as a hint. Blank for BOL rows.
    # The reviewer's decision.
    "action",                                # confirm | change | drop
    "new_hs6",                                # required when action=change
    # Core-4 evidence fields.
    "expert_hs6",                             # the gold HS6 (6 digits)
    "confidence_tier",                        # high | medium | low
    "cited_evidence_ids",                     # comma-separated evidence_ids
    "rationale_short",                        # ≤ 200 chars
    "tier2_classifiable",                     # yes | partial | no
    "adjudication_status",                    # see ADJUDICATION_STATUSES
    "adjudication_winning_evidence_id",       # populated when rubric-resolved
    "adjudication_rubric_score",              # int 0–6
    # Free-form.
    "notes",
)
