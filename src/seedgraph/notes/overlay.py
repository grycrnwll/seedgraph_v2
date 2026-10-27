"""Researcher edits as an overlay on an extraction (stage 3 design §4.1-4.2, ADR-0011).

An overlay belongs to one ``(paper_id, schema_id)`` and is pinned to a **base** note
(the ``note_current`` when it was created). ``overlay_versions`` is the source of
truth: every save writes version n+1 holding the **complete** op set and the
researcher notes. Revert writes a new version copying an old op set, so nothing is
ever deleted and a revert can itself be reverted.

Ops (``ops_json``):

- ``{op: "edit", target: ClaimRef, set: {claim_text?, normalized_label?, claim_type?}}``
- ``{op: "hide", target: ClaimRef}``
- ``{op: "add", add_id: "rc_…", kind: "quote" | "assertion", claim_type, normalized_label,
  claim_text, quote?: {markdown_id, markdown_hash, start_char, end_char, quote_hash, span_id}}``

``ClaimRef = {claim_id, fingerprint}``: ``claim_id`` names the row in the base
extraction and ``fingerprint`` (:func:`fingerprint`) is computed at edit time from
the agent's original claim, so a later replay never needs the old rows.

Incoming requests to :func:`save_edit` are single ops: ``edit``, ``hide``, ``unhide``,
``add``, ``remove_added`` and ``notes``. They are folded into the op set (one merged
``edit`` per claim; an edit of an added claim changes its ``add`` op; ``unhide`` and
``remove_added`` delete ops).

One save, in **one** library transaction (:func:`write_version`): check
``expected_version``, validate the op set, create the overlay on the first save,
write the version, rebuild ``interpretation_claims`` / ``interpretation_notes``,
replace the paper's ``claim_fts`` / ``note_fts`` rows (:func:`rebuild`) and upsert
``concept_refresh`` (:func:`mark_concepts_out_of_date`). U2's replay accept calls
:func:`write_version` inside its own transaction with a new base.
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .. import anchor
from ..errors import SeedgraphError
from ..ids import new_id
from ..semantic.merge import normalize as label_fold
from .format import _ASSUMPTION_CLAIM_TYPE, ARRAY_FIELDS, SCALAR_FIELDS, SCHEMA_ID
from .spans import ensure_span, read_markdown

if TYPE_CHECKING:
    from ..library.store import Library

#: The fixed claim types of the note schema (stage 2 §4.1): what an edit or add may use.
NOTE_CLAIM_TYPES: frozenset[str] = frozenset(
    {t for _, t in SCALAR_FIELDS} | {t for _, t in ARRAY_FIELDS}
    | set(_ASSUMPTION_CLAIM_TYPE.values()) | {"model"}
)
EDITABLE_FIELDS: tuple[str, ...] = ("claim_text", "normalized_label", "claim_type", "plain_words")
ACTIONS: tuple[str, ...] = ("edit", "hide", "unhide", "add", "remove_added", "notes", "revert",
                            "accept_extraction")
ADD_KINDS: tuple[str, ...] = ("quote", "assertion")
QUOTE_KEYS: tuple[str, ...] = ("markdown_id", "markdown_hash", "start_char", "end_char", "quote_hash",
                               "span_id")
STALE_MESSAGE = "this note changed in another tab; reload"
#: ``concept_refresh.reason`` written by a researcher save.
REFRESH_REASON = "researcher_edit"

_FIELD_INDEX_RE = re.compile(r"\[\d+\]$")


class OverlayError(SeedgraphError):
    """A researcher edit was refused (invalid op, unknown claim, no note)."""


class OverlayConflict(OverlayError):
    """The form's ``expected_version`` is not the overlay's head version."""


@dataclass(frozen=True)
class OverlayVersion:
    """One immutable overlay version (``ops`` is the complete op set at that version)."""

    overlay_id: str
    paper_id: str
    schema_id: str | None
    version_no: int
    base_note_id: str
    base_extraction_run_id: str
    ops: list[dict[str, Any]]
    researcher_notes: str | None
    action: str
    summary: str | None
    reverted_to: int | None
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- reading ---------------------------------------------------------------------------------

_VERSION_SQL = (
    "SELECT o.overlay_id, o.paper_id, o.schema_id, v.version_no, v.base_note_id, "
    "v.base_extraction_run_id, v.ops_json, v.researcher_notes, v.action, v.summary, v.reverted_to, "
    "v.created_at FROM note_overlays o JOIN overlay_versions v ON v.overlay_id = o.overlay_id "
    "WHERE o.paper_id = ? AND o.schema_id IS ? "
)


def _version(row: sqlite3.Row) -> OverlayVersion:
    return OverlayVersion(
        overlay_id=row[0], paper_id=row[1], schema_id=row[2], version_no=row[3], base_note_id=row[4],
        base_extraction_run_id=row[5], ops=json.loads(row[6]), researcher_notes=row[7], action=row[8],
        summary=row[9], reverted_to=row[10], created_at=row[11])


def head(conn: sqlite3.Connection, paper_id: str, schema_id: str | None = SCHEMA_ID) -> OverlayVersion | None:
    """The overlay's head version, or ``None`` when the paper has no overlay."""
    row = conn.execute(_VERSION_SQL + "AND v.version_no = o.head_version_no", (paper_id, schema_id)).fetchone()
    return _version(row) if row else None


def head_version_no(conn: sqlite3.Connection, paper_id: str, schema_id: str | None = SCHEMA_ID) -> int:
    """The ``expected_version`` a form should carry: the head version, ``0`` with no overlay."""
    h = head(conn, paper_id, schema_id)
    return h.version_no if h else 0


def version(conn: sqlite3.Connection, paper_id: str, version_no: int,
            schema_id: str | None = SCHEMA_ID) -> OverlayVersion | None:
    row = conn.execute(_VERSION_SQL + "AND v.version_no = ?", (paper_id, schema_id, version_no)).fetchone()
    return _version(row) if row else None


def history(conn: sqlite3.Connection, paper_id: str, schema_id: str | None = SCHEMA_ID) -> list[OverlayVersion]:
    """Every version, newest first."""
    return [_version(r) for r in conn.execute(_VERSION_SQL + "ORDER BY v.version_no DESC",
                                              (paper_id, schema_id))]


def has_edits(conn: sqlite3.Connection, paper_id: str) -> bool:
    """Whether the paper has an overlay (any schema)."""
    return conn.execute("SELECT 1 FROM note_overlays WHERE paper_id = ? LIMIT 1", (paper_id,)).fetchone() is not None


# --- fingerprints ----------------------------------------------------------------------------

def _top_span(conn: sqlite3.Connection, claim_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT e.span_id, e.quote_hash, e.markdown_hash, e.start_char, e.end_char "
        "FROM claim_spans cs JOIN evidence_spans e ON e.span_id = cs.span_id "
        "WHERE cs.claim_id = ? ORDER BY cs.rank, cs.span_id LIMIT 1", (claim_id,)).fetchone()


def fingerprint(conn: sqlite3.Connection, claim_id: str) -> dict[str, Any]:
    """The stable identity of an agent claim (design §4.4), computed from its stored row:
    ``claim_type``, ``field_root``, ``label_fold``, ``text_tokens`` (sorted) and, for an
    anchored claim, ``quote_hash``, ``markdown_hash``, ``start_char``, ``end_char``
    (``None`` otherwise). U2's ``notes/identity.py`` matches on these fields."""
    row = conn.execute("SELECT claim_type, field_key, normalized_label, claim_text "
                       "FROM extracted_claims WHERE claim_id = ?", (claim_id,)).fetchone()
    if row is None:
        raise OverlayError(f"claim {claim_id!r} does not exist")
    span = _top_span(conn, claim_id)
    return {
        "claim_type": row[0],
        "field_root": _FIELD_INDEX_RE.sub("", row[1]),
        "label_fold": label_fold(row[2] or ""),
        "text_tokens": sorted(set(label_fold(row[3] or "").split())),
        "quote_hash": span[1] if span else None,
        "markdown_hash": span[2] if span else None,
        "start_char": span[3] if span else None,
        "end_char": span[4] if span else None,
    }


# --- validation ------------------------------------------------------------------------------

def _base_claim_ids(conn: sqlite3.Connection, base_note_id: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT claim_id FROM extracted_claims WHERE structured_note_id = ?",
                                       (base_note_id,))}


def _check_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OverlayError(f"{what} must be non-empty")
    return value


def _check_set(set_: Any) -> dict[str, str]:
    if not isinstance(set_, Mapping) or not set_:
        raise OverlayError("an edit with nothing to change")
    bad = sorted(set(set_) - set(EDITABLE_FIELDS))
    if bad:
        raise OverlayError(f"an edit cannot set {', '.join(bad)} (only {', '.join(EDITABLE_FIELDS)})")
    out: dict[str, str] = {}
    for key in EDITABLE_FIELDS:
        if key not in set_:
            continue
        if key == "claim_type":
            if set_[key] not in NOTE_CLAIM_TYPES:
                raise OverlayError(f"unknown claim type {set_[key]!r}; allowed: {', '.join(sorted(NOTE_CLAIM_TYPES))}")
            out[key] = set_[key]
        elif key == "normalized_label":
            out[key] = _check_text(set_[key], "a label")
        elif key == "plain_words":
            if not isinstance(set_[key], str):
                raise OverlayError("plain words must be text")
            out[key] = set_[key]  # may be empty: the researcher cleared them
        else:
            out[key] = _check_text(set_[key], "claim text")
    return out


def _check_add_fields(op: Mapping[str, Any]) -> None:
    if op.get("kind") not in ADD_KINDS:
        raise OverlayError(f"an added claim's kind must be one of {', '.join(ADD_KINDS)}")
    if op.get("claim_type") not in NOTE_CLAIM_TYPES:
        raise OverlayError(f"unknown claim type {op.get('claim_type')!r}; "
                           f"allowed: {', '.join(sorted(NOTE_CLAIM_TYPES))}")
    _check_text(op.get("normalized_label"), "a label")
    _check_text(op.get("claim_text"), "claim text")
    if op["kind"] == "assertion" and op.get("quote") is not None:
        raise OverlayError("a researcher assertion has no quote")


def validate_ops(conn: sqlite3.Connection, ops: Sequence[Mapping[str, Any]], base_note_id: str) -> None:
    """Check a complete op set against its base note (raises :class:`OverlayError`)."""
    base_ids = _base_claim_ids(conn, base_note_id)
    edited: set[str] = set()
    hidden: set[str] = set()
    add_ids: set[str] = set()
    for op in ops:
        kind = op.get("op") if isinstance(op, Mapping) else None
        if kind in ("edit", "hide"):
            target = op.get("target") or {}
            cid = target.get("claim_id")
            if cid not in base_ids:
                raise OverlayError(f"{cid!r} is not a claim of the base note {base_note_id}")
            if not isinstance(target.get("fingerprint"), Mapping):
                raise OverlayError(f"the {kind} of {cid} has no fingerprint")
            seen = edited if kind == "edit" else hidden
            if cid in seen:
                raise OverlayError(f"two {kind} ops for {cid}")
            seen.add(cid)
            if kind == "edit":
                _check_set(op.get("set"))
        elif kind == "add":
            add_id = op.get("add_id")
            if not isinstance(add_id, str) or not add_id.startswith("rc_") or add_id in add_ids:
                raise OverlayError(f"bad or duplicate add_id {add_id!r}")
            add_ids.add(add_id)
            _check_add_fields(op)
            if op["kind"] == "quote":
                quote = op.get("quote")
                if not isinstance(quote, Mapping) or any(quote.get(k) is None for k in QUOTE_KEYS):
                    raise OverlayError(f"the quote of {add_id} needs {', '.join(QUOTE_KEYS)}")
        else:
            raise OverlayError(f"unknown op {kind!r} in the op set")


# --- folding one request into the op set -----------------------------------------------------

def _resolve_quote(conn: sqlite3.Connection, paper_id: str, base_note_id: str,
                   quote: Any) -> dict[str, Any]:
    """Anchor a quote-backed add exactly and uniquely; create or reuse its span."""
    if not isinstance(quote, Mapping):
        raise OverlayError("a quote-backed claim needs a quote {markdown_id, start_char, end_char}")
    md_id = quote.get("markdown_id")
    allowed = {r[0] for r in conn.execute(
        "SELECT markdown_id FROM structured_notes WHERE note_id = ? "
        "UNION SELECT markdown_id FROM paper_files WHERE paper_id = ? AND role = 'primary' "
        "AND markdown_id IS NOT NULL", (base_note_id, paper_id))}
    if md_id not in allowed:
        raise OverlayError(f"markdown {md_id!r} is not this paper's note or primary markdown")
    text = read_markdown(conn, md_id)
    if text is None:
        raise OverlayError(f"markdown {md_id} is missing from the library")
    start, end = quote.get("start_char"), quote.get("end_char")
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
        raise OverlayError("the quote does not anchor: its offsets are outside the markdown")
    exact = text[start:end]
    md_hash = conn.execute("SELECT markdown_hash FROM markdown_documents WHERE markdown_id = ?",
                           (md_id,)).fetchone()[0]
    if quote.get("markdown_hash") not in (None, md_hash):
        raise OverlayError("the quote was taken from a different version of the markdown")
    if anchor.locate_quote(text, exact) != (start, end):
        raise OverlayError("the quote does not anchor exactly and uniquely in the markdown; "
                           "select a longer passage")
    span_id = ensure_span(conn, markdown_id=md_id, exact_quote=exact, markdown_text=text)
    if span_id is None:
        raise OverlayError("the quote does not anchor exactly and uniquely in the markdown")
    return {"markdown_id": md_id, "markdown_hash": md_hash, "start_char": start, "end_char": end,
            "quote_hash": anchor.quote_hash(exact), "span_id": span_id}


def _find(ops: list[dict], kind: str, claim_id: str) -> dict | None:
    return next((o for o in ops if o["op"] == kind and o["target"]["claim_id"] == claim_id), None)


def fold_op(conn: sqlite3.Connection, paper_id: str, base_note_id: str, ops: list[dict[str, Any]],
            notes: str | None, request: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    """Apply one request to a copy of the op set: ``(action, new_ops, new_notes)``."""
    ops = copy.deepcopy(ops)
    kind = request.get("op") if isinstance(request, Mapping) else None
    if kind == "notes":
        text = request.get("text")
        if text is not None and not isinstance(text, str):
            raise OverlayError("researcher notes must be text")
        return "notes", ops, (text if text and text.strip() else None)
    if kind == "add":
        _check_add_fields(request)
        op: dict[str, Any] = {"op": "add", "add_id": new_id("rc"), "kind": request["kind"],
                              "claim_type": request["claim_type"],
                              "normalized_label": request["normalized_label"],
                              "claim_text": request["claim_text"]}
        if request["kind"] == "quote":
            op["quote"] = _resolve_quote(conn, paper_id, base_note_id, request.get("quote"))
        ops.append(op)
        return "add", ops, notes
    if kind == "remove_added":
        add_id = request.get("add_id")
        kept = [o for o in ops if not (o["op"] == "add" and o["add_id"] == add_id)]
        if len(kept) == len(ops):
            raise OverlayError(f"{add_id!r} is not an added claim of this note")
        return "remove_added", kept, notes
    if kind not in ("edit", "hide", "unhide"):
        raise OverlayError(f"unknown op {kind!r}; expected edit, hide, unhide, add, remove_added or notes")

    cid = (request.get("target") or {}).get("claim_id")
    if kind == "edit" and isinstance(cid, str) and cid.startswith("rc_"):
        set_ = _check_set(request.get("set"))
        added = next((o for o in ops if o["op"] == "add" and o["add_id"] == cid), None)
        if added is None:
            raise OverlayError(f"{cid!r} is not an added claim of this note")
        added.update(set_)
        return "edit", ops, notes
    if cid not in _base_claim_ids(conn, base_note_id):
        raise OverlayError(f"{cid!r} is not a claim of the base note {base_note_id}")
    if kind == "edit":
        set_ = _check_set(request.get("set"))
        existing = _find(ops, "edit", cid)
        if existing is None:
            ops.append({"op": "edit", "target": {"claim_id": cid, "fingerprint": fingerprint(conn, cid)},
                        "set": set_})
        else:
            existing["set"] = {**existing["set"], **set_}
        return "edit", ops, notes
    if kind == "hide":
        if _find(ops, "hide", cid) is not None:
            raise OverlayError(f"{cid} is already hidden")
        ops.append({"op": "hide", "target": {"claim_id": cid, "fingerprint": fingerprint(conn, cid)}})
        return "hide", ops, notes
    existing = _find(ops, "hide", cid)
    if existing is None:
        raise OverlayError(f"{cid} is not hidden")
    ops.remove(existing)
    return "unhide", ops, notes


# --- writing ---------------------------------------------------------------------------------

def _check_expected(current: OverlayVersion | None, expected_version: int) -> None:
    actual = current.version_no if current else 0
    if expected_version != actual:
        raise OverlayConflict(f"{STALE_MESSAGE} (expected version {expected_version}, found {actual})")


def _current_base(conn: sqlite3.Connection, paper_id: str, schema_id: str | None) -> tuple[str, str]:
    row = conn.execute("SELECT note_id, extraction_run_id FROM note_current WHERE paper_id = ? AND schema_id IS ?",
                       (paper_id, schema_id)).fetchone()
    if row is None:
        raise OverlayError(f"paper {paper_id} has no current note to edit")
    return row[0], row[1]


def mark_concepts_out_of_date(conn: sqlite3.Connection, paper_id: str, reason: str = REFRESH_REASON) -> None:
    """Upsert ``concept_refresh`` (``since`` keeps the first time; stage 4 clears the row)."""
    conn.execute("INSERT INTO concept_refresh (paper_id, reason, since) VALUES (?, ?, ?) "
                 "ON CONFLICT(paper_id) DO UPDATE SET reason = excluded.reason", (paper_id, reason, _now()))


def write_version(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    ops: Sequence[Mapping[str, Any]],
    researcher_notes: str | None,
    action: str,
    expected_version: int,
    base_note_id: str | None = None,
    base_extraction_run_id: str | None = None,
    summary: str | None = None,
    reverted_to: int | None = None,
    schema_id: str | None = SCHEMA_ID,
) -> OverlayVersion:
    """Write version n+1 with the complete ``ops`` and rebuild everything derived from it.

    Runs **inside the caller's write transaction** (never begins or commits). The base
    defaults to the head's base (or ``note_current`` for the first version); pass a new
    base (U2's ``accept_extraction``) to move the pin.
    """
    if not conn.in_transaction:
        raise OverlayError("write_version must run inside the caller's write transaction")
    if action not in ACTIONS:
        raise OverlayError(f"unknown action {action!r}")
    current = head(conn, paper_id, schema_id)
    _check_expected(current, expected_version)
    if base_note_id is None:
        if current is not None:
            base_note_id, base_extraction_run_id = current.base_note_id, current.base_extraction_run_id
        else:
            base_note_id, base_extraction_run_id = _current_base(conn, paper_id, schema_id)
    if base_extraction_run_id is None:
        row = conn.execute("SELECT extraction_run_id FROM structured_notes WHERE note_id = ? AND paper_id = ?",
                           (base_note_id, paper_id)).fetchone()
        if row is None:
            raise OverlayError(f"{base_note_id!r} is not a note of {paper_id}")
        base_extraction_run_id = row[0]
    ops = [dict(o) for o in ops]
    validate_ops(conn, ops, base_note_id)

    now = _now()
    if current is None:
        overlay_id = new_id("ov")
        version_no = 1
        conn.execute(
            "INSERT INTO note_overlays (overlay_id, paper_id, schema_id, base_note_id, base_extraction_run_id, "
            "head_version_no, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (overlay_id, paper_id, schema_id, base_note_id, base_extraction_run_id, now, now))
    else:
        overlay_id = current.overlay_id
        version_no = current.version_no + 1
    conn.execute(
        "INSERT INTO overlay_versions (overlay_id, version_no, base_note_id, base_extraction_run_id, ops_json, "
        "researcher_notes, action, summary, reverted_to, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (overlay_id, version_no, base_note_id, base_extraction_run_id,
         json.dumps(ops, sort_keys=True, ensure_ascii=False), researcher_notes, action, summary, reverted_to, now))
    conn.execute("UPDATE note_overlays SET head_version_no = ?, base_note_id = ?, base_extraction_run_id = ?, "
                 "updated_at = ? WHERE overlay_id = ?",
                 (version_no, base_note_id, base_extraction_run_id, now, overlay_id))
    rebuild(conn, paper_id, schema_id)
    mark_concepts_out_of_date(conn, paper_id)
    result = head(conn, paper_id, schema_id)
    assert result is not None
    return result


def apply_op(conn: sqlite3.Connection, paper_id: str, request: Mapping[str, Any], *, expected_version: int,
             summary: str | None = None, schema_id: str | None = SCHEMA_ID) -> OverlayVersion:
    """Fold one request into the op set and write it (caller's transaction)."""
    current = head(conn, paper_id, schema_id)
    _check_expected(current, expected_version)
    if current is not None:
        base_note_id, ops, notes = current.base_note_id, current.ops, current.researcher_notes
    else:
        base_note_id, _ = _current_base(conn, paper_id, schema_id)
        ops, notes = [], None
    action, new_ops, new_notes = fold_op(conn, paper_id, base_note_id, ops, notes, request)
    return write_version(conn, paper_id, ops=new_ops, researcher_notes=new_notes, action=action,
                         expected_version=expected_version, summary=summary, schema_id=schema_id)


def save_edit(lib: Library, paper_id: str, op: Mapping[str, Any], *, expected_version: int,
              summary: str | None = None, schema_id: str | None = SCHEMA_ID) -> OverlayVersion:
    """Save one edit request as a new version, in one library transaction."""
    with lib.write() as conn:
        return apply_op(conn, paper_id, op, expected_version=expected_version, summary=summary,
                        schema_id=schema_id)


def save_notes(lib: Library, paper_id: str, text: str | None, *, expected_version: int,
               schema_id: str | None = SCHEMA_ID) -> OverlayVersion:
    """Save the researcher notes box (``action = 'notes'``)."""
    return save_edit(lib, paper_id, {"op": "notes", "text": text}, expected_version=expected_version,
                     schema_id=schema_id)


def revert(lib: Library, paper_id: str, to_version: int, *, expected_version: int,
           schema_id: str | None = SCHEMA_ID) -> OverlayVersion:
    """Write a new version whose op set, notes and base are version ``to_version``'s."""
    with lib.write() as conn:
        old = version(conn, paper_id, to_version, schema_id)
        current = head(conn, paper_id, schema_id)
        _check_expected(current, expected_version)
        if old is None:
            raise OverlayError(f"paper {paper_id} has no overlay version {to_version}")
        return write_version(conn, paper_id, ops=old.ops, researcher_notes=old.researcher_notes,
                             action="revert", expected_version=expected_version,
                             base_note_id=old.base_note_id, base_extraction_run_id=old.base_extraction_run_id,
                             summary=f"revert to version {to_version}", reverted_to=to_version,
                             schema_id=schema_id)


# --- derived rows and FTS --------------------------------------------------------------------

def _interpretation_rows(conn: sqlite3.Connection, h: OverlayVersion) -> list[dict[str, Any]]:
    edits = {o["target"]["claim_id"]: o["set"] for o in h.ops if o["op"] == "edit"}
    hides = {o["target"]["claim_id"] for o in h.ops if o["op"] == "hide"}
    out: list[dict[str, Any]] = []
    for c in conn.execute(
            "SELECT claim_id, claim_type, claim_subtype, field_key, normalized_label, claim_text, plain_words, "
            "status FROM extracted_claims WHERE structured_note_id = ? ORDER BY claim_id", (h.base_note_id,)):
        span = _top_span(conn, c["claim_id"])
        set_ = edits.get(c["claim_id"])
        if set_ is not None:
            trust = "researcher"
        elif c["status"] == "found" and span is not None:
            trust = "paper_states"
        else:
            trust = "agent_inferred"
        out.append(dict(
            claim_id=c["claim_id"], claim_type=(set_ or {}).get("claim_type", c["claim_type"]),
            claim_subtype=c["claim_subtype"], field_key=c["field_key"],
            normalized_label=(set_ or {}).get("normalized_label", c["normalized_label"]),
            claim_text=(set_ or {}).get("claim_text", c["claim_text"]),
            plain_words=(set_ or {}).get("plain_words", c["plain_words"]) or None,
            status=c["status"], trust=trust, edited=int(set_ is not None),
            hidden=int(c["claim_id"] in hides), kind="agent", span_id=span[0] if span else None))
    adds = [o for o in h.ops if o["op"] == "add"]
    for i, o in enumerate(adds):
        out.append(dict(
            claim_id=o["add_id"], claim_type=o["claim_type"], claim_subtype=None, field_key=f"added[{i}]",
            normalized_label=o["normalized_label"], claim_text=o["claim_text"], plain_words=None,
            status="found", trust="researcher", edited=0, hidden=0, kind=o["kind"],
            span_id=(o.get("quote") or {}).get("span_id")))
    return out


def note_text_for(claims: Sequence[Mapping[str, Any]], researcher_notes: str | None) -> str:
    """The interpretation's note text: v2's line per visible found claim, then the notes."""
    lines = []
    for c in claims:
        if c["status"] == "found" and not c.get("hidden"):
            line = f"{c['normalized_label'] or c['claim_type']}: {c['claim_text'] or ''}".strip()
            if line:
                lines.append(line)
    text = "\n".join(lines)
    if researcher_notes:
        text = f"{text}\n\nResearcher notes: {researcher_notes}" if text else f"Researcher notes: {researcher_notes}"
    return text


def rebuild(conn: sqlite3.Connection, paper_id: str, schema_id: str | None = SCHEMA_ID) -> None:
    """Recompute ``interpretation_claims`` / ``interpretation_notes`` from the head version and
    replace the paper's FTS rows (caller's transaction).

    FTS scope: every note that has ever been a base of this overlay. The current base's
    claims and note are indexed as the interpretation (hidden claims left out, added
    claims indexed under their ``rc_`` id); earlier bases get their agent rows back.
    Other notes of the paper (a newer re-note, ``note_latest``) are never touched.
    """
    h = head(conn, paper_id, schema_id)
    ov = conn.execute("SELECT overlay_id FROM note_overlays WHERE paper_id = ? AND schema_id IS ?",
                      (paper_id, schema_id)).fetchone()
    old_added = [r[0] for r in conn.execute(
        "SELECT claim_id FROM interpretation_claims WHERE overlay_id = ? AND kind <> 'agent'",
        (ov[0] if ov else None,))]
    if ov is not None:
        conn.execute("DELETE FROM interpretation_claims WHERE overlay_id = ?", (ov[0],))
        bases = [r[0] for r in conn.execute(
            "SELECT DISTINCT base_note_id FROM overlay_versions WHERE overlay_id = ? ORDER BY base_note_id",
            (ov[0],))]
    else:
        bases = []
    conn.execute("DELETE FROM interpretation_notes WHERE paper_id = ? AND schema_id IS ?", (paper_id, schema_id))

    for cid in old_added:
        conn.execute("DELETE FROM claim_fts WHERE claim_id = ?", (cid,))
    for note_id in bases:
        conn.execute("DELETE FROM claim_fts WHERE claim_id IN "
                     "(SELECT claim_id FROM extracted_claims WHERE structured_note_id = ?)", (note_id,))
        conn.execute("DELETE FROM note_fts WHERE note_id = ?", (note_id,))
        if h is not None and note_id == h.base_note_id:
            continue
        conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text, plain_words) "
                     "SELECT claim_id, normalized_label, claim_text, plain_words FROM extracted_claims "
                     "WHERE structured_note_id = ? ORDER BY claim_id", (note_id,))
        conn.execute("INSERT INTO note_fts (note_id, note_text) SELECT note_id, note_text FROM structured_notes "
                     "WHERE note_id = ?", (note_id,))
    if h is None:
        return

    claims = _interpretation_rows(conn, h)
    for c in claims:
        conn.execute(
            "INSERT INTO interpretation_claims (overlay_id, claim_id, paper_id, version_no, claim_type, "
            "claim_subtype, field_key, normalized_label, claim_text, plain_words, status, trust, edited, hidden, "
            "kind, span_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (h.overlay_id, c["claim_id"], paper_id, h.version_no, c["claim_type"], c["claim_subtype"],
             c["field_key"], c["normalized_label"], c["claim_text"], c["plain_words"], c["status"], c["trust"],
             c["edited"], c["hidden"], c["kind"], c["span_id"]))
        if not c["hidden"]:
            conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text, plain_words) "
                         "VALUES (?, ?, ?, ?)",
                         (c["claim_id"], c["normalized_label"], c["claim_text"], c["plain_words"]))
    text = note_text_for(claims, h.researcher_notes)
    conn.execute("INSERT INTO interpretation_notes (paper_id, schema_id, note_text) VALUES (?, ?, ?)",
                 (paper_id, schema_id, text))
    conn.execute("INSERT INTO note_fts (note_id, note_text) VALUES (?, ?)", (h.base_note_id, text))


def delete_overlays(conn: sqlite3.Connection, paper_id: str) -> None:
    """Remove a paper's overlays, derived rows, added-claim FTS rows and ``concept_refresh``
    (only ``library.lifecycle.delete_paper`` calls this; the caller deletes the notes)."""
    conn.execute("DELETE FROM claim_fts WHERE claim_id IN (SELECT claim_id FROM interpretation_claims "
                 "WHERE paper_id = ? AND kind <> 'agent')", (paper_id,))
    conn.execute("DELETE FROM interpretation_claims WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM interpretation_notes WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM overlay_versions WHERE overlay_id IN "
                 "(SELECT overlay_id FROM note_overlays WHERE paper_id = ?)", (paper_id,))
    conn.execute("DELETE FROM note_overlays WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM concept_refresh WHERE paper_id = ?", (paper_id,))
