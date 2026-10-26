"""Replay researcher edits on a re-extraction (stage 3 design §4.5, ADR-0011).

An overlaid paper whose ``note_latest`` is newer than its overlay base offers a replay.

:func:`plan_replay` writes nothing. It matches the base's claims onto the new
extraction (:mod:`notes.identity`; an op's claim uses the fingerprint stored at edit
time, so a re-typed claim matches on the agent's original type) and lists, per op:

- ``kept``: a matched edit (the researcher's version wins over the new agent wording);
- ``still_hidden``: a matched hide;
- ``added_kept`` / ``added_unanchored``: a researcher-added claim; a quote-backed one is
  re-checked for an exact unique anchor in the new markdown (relocated there when the
  markdown changed) and, when it no longer anchors, is kept as an assertion;
- ``unmatched``: an edit whose claim did not reappear. Default on accept: **keep** it as
  a researcher claim (quote-backed when its old quote anchors exactly and uniquely in
  the new markdown, else an assertion), unless discarded or attached to a new claim;
- ``ambiguous``: a tie the researcher should resolve with ``attach``; left undecided it
  is handled as ``unmatched``;
- ``hide_gone``: a hide whose claim did not reappear (reported, no decision). An edit of
  a hidden claim that is gone is ``hide_gone`` too: it is not re-added as a visible
  researcher claim.

plus the new agent claims, the matched claims without ops whose wording changes (they
take the new wording on accept) and the gone ones. Old quotes on a different markdown
are relocated through the text first (``library.rematch.match_quote``: exact, then
mojibake-folded, then whitespace-collapsed, never a guess among duplicates).

:func:`accept` refuses a stale plan (the overlay head or ``note_latest`` moved), then
writes one overlay version (``action = 'accept_extraction'``) on the new base with ops
re-keyed to the new claim ids and fingerprints of the new agent claims, dismissed new
claims as ``hide`` ops, all in one library transaction (``overlay.write_version``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .. import anchor
from ..ids import new_id
from . import identity, overlay
from .format import SCHEMA_ID
from .identity import Ambiguity, ClaimFingerprint, Matching
from .spans import ensure_span, read_markdown

if TYPE_CHECKING:
    from ..library.store import Library

OpStatus = Literal["kept", "still_hidden", "added_kept", "added_unanchored", "unmatched", "ambiguous",
                   "hide_gone"]
ACTION = "accept_extraction"


class ReplayError(overlay.OverlayError):
    """A replay could not be planned or accepted."""


class ReplayStale(overlay.OverlayConflict, ReplayError):
    """The overlay head or the latest extraction changed after the plan was made."""


@dataclass(frozen=True)
class ClaimText:
    claim_type: str
    normalized_label: str | None
    claim_text: str | None


@dataclass(frozen=True)
class OpPlan:
    """What accept will do with one op of the head."""

    key: str                         # "edit:<claim_id>", "hide:<claim_id>" or the add_id
    op: dict[str, Any]
    status: OpStatus
    old_claim_id: str | None = None  # edit / hide: the base claim
    new_claim_id: str | None = None  # kept / still_hidden
    candidates: tuple[str, ...] = ()  # ambiguous: the tied new claims
    keep_as: Literal["quote", "assertion"] | None = None  # unmatched / ambiguous edit, unanchored add
    quote: dict[str, Any] | None = None  # the quote on the new markdown (an add, or a kept-as-quote edit)
    new_agent: ClaimText | None = None   # the new agent claim beside a kept edit or hide


@dataclass(frozen=True)
class NewClaim:
    claim_id: str
    claim_type: str
    field_key: str
    normalized_label: str | None
    claim_text: str | None
    status: str


@dataclass(frozen=True)
class WordingChange:
    """A matched claim with no op whose agent wording changes on accept."""

    old_claim_id: str
    claim_id: str
    field_key: str
    old_label: str | None
    old_text: str | None
    new_label: str | None
    new_text: str | None


@dataclass(frozen=True)
class ReplayPlan:
    paper_id: str
    schema_id: str | None
    overlay_id: str
    head_version_no: int
    base_note_id: str
    new_note_id: str
    new_markdown_id: str
    ops: tuple[OpPlan, ...]
    new_claims: tuple[NewClaim, ...]
    changed: tuple[WordingChange, ...]
    gone: tuple[str, ...]            # base claim ids with no op and no match
    ambiguous: tuple[Ambiguity, ...]  # every tie, with or without ops
    matching: Matching = field(repr=False)

    def _count(self, status: str, kind: str | None = None) -> int:
        return sum(1 for o in self.ops if o.status == status and (kind is None or o.op["op"] == kind))

    @property
    def kept(self) -> int:
        return self._count("kept")

    @property
    def still_hidden(self) -> int:
        return self._count("still_hidden")

    @property
    def unmatched(self) -> int:
        return self._count("unmatched")

    @property
    def hides_gone(self) -> int:
        """ "n hidden claims are not in the new extraction"."""
        return self._count("hide_gone", "hide")

    @property
    def needs_decision(self) -> tuple[OpPlan, ...]:
        return tuple(o for o in self.ops if o.status in ("unmatched", "ambiguous", "added_unanchored"))


# --- reading ---------------------------------------------------------------------------------

def _claims(conn: sqlite3.Connection, note_id: str) -> dict[str, sqlite3.Row]:
    return {r["claim_id"]: r for r in conn.execute(
        "SELECT claim_id, claim_type, field_key, normalized_label, claim_text, status "
        "FROM extracted_claims WHERE structured_note_id = ? ORDER BY claim_id", (note_id,))}


class _Texts:
    """Markdown texts by id and by hash (read lazily, never written)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._by_id: dict[str, str | None] = {}

    def by_id(self, markdown_id: str) -> str | None:
        if markdown_id not in self._by_id:
            self._by_id[markdown_id] = read_markdown(self.conn, markdown_id)
        return self._by_id[markdown_id]

    def by_hash(self, markdown_hash: str) -> str | None:
        for (md_id,) in self.conn.execute("SELECT markdown_id FROM markdown_documents WHERE markdown_hash = ? "
                                          "ORDER BY markdown_id", (markdown_hash,)):
            text = self.by_id(md_id)
            if text is not None:
                return text
        return None


def _old_quote(conn: sqlite3.Connection, texts: _Texts, fp: ClaimFingerprint) -> str | None:
    """The exact text of an anchored fingerprint's quote: sliced from its markdown, else
    looked up by ``quote_hash``."""
    if not fp.anchored:
        return None
    text = texts.by_hash(fp.markdown_hash)  # type: ignore[arg-type]
    if text is not None:
        piece = text[fp.start_char:fp.end_char]
        if fp.quote_hash is None or anchor.quote_hash(piece) == fp.quote_hash:
            return piece
    row = conn.execute("SELECT exact_quote FROM evidence_spans WHERE quote_hash = ? LIMIT 1",
                       (fp.quote_hash,)).fetchone()
    return row[0] if row else None


def _relocate(old_quote: str | None, new_text: str) -> tuple[int, int] | None:
    from ..library.rematch import match_quote  # library imports notes: import late

    if not old_quote:
        return None
    m = match_quote(old_quote, new_text)
    return (m.start, m.end) if m else None


def _state(conn: sqlite3.Connection, paper_id: str, schema_id: str | None):
    h = overlay.head(conn, paper_id, schema_id)
    if h is None:
        raise ReplayError(f"paper {paper_id} has no researcher edits to replay")
    latest = conn.execute("SELECT note_id, markdown_id, markdown_hash FROM note_latest "
                          "WHERE paper_id = ? AND schema_id IS ?", (paper_id, schema_id)).fetchone()
    return h, latest


# --- planning --------------------------------------------------------------------------------

def plan_replay(lib: Library | sqlite3.Connection, paper_id: str, *,
                schema_id: str | None = SCHEMA_ID) -> ReplayPlan:
    """Compare the overlay head with the newest extraction; writes nothing."""
    conn = lib if isinstance(lib, sqlite3.Connection) else lib.conn
    h, latest = _state(conn, paper_id, schema_id)
    if latest is None or latest["note_id"] == h.base_note_id:
        raise ReplayError(f"paper {paper_id} has no newer extraction than its edited note")
    base_v, new_v = (conn.execute("SELECT version_no FROM structured_notes WHERE note_id = ?", (n,)).fetchone()[0]
                     for n in (h.base_note_id, latest["note_id"]))
    if new_v <= base_v:
        raise ReplayError(f"paper {paper_id} has no newer extraction than its edited note")
    new_md, new_hash = latest["markdown_id"], latest["markdown_hash"]
    texts = _Texts(conn)
    new_text = texts.by_id(new_md)
    if new_text is None:
        raise ReplayError(f"markdown {new_md} of the new extraction is missing from the library")

    old_rows = _claims(conn, h.base_note_id)
    new_rows = _claims(conn, latest["note_id"])
    op_fps: dict[str, dict[str, Any]] = {}
    for op in h.ops:
        if op["op"] in ("edit", "hide"):
            op_fps.setdefault(op["target"]["claim_id"], op["target"]["fingerprint"])

    old_fps = []
    for cid, row in old_rows.items():
        fp = ClaimFingerprint.from_dict(cid, op_fps.get(cid) or overlay.fingerprint(conn, cid),
                                        field_key=row["field_key"])
        if fp.anchored and fp.markdown_hash != new_hash:
            fp = fp.relocated(new_hash, _relocate(_old_quote(conn, texts, fp), new_text))
        old_fps.append(fp)
    new_fps = [ClaimFingerprint.from_dict(cid, overlay.fingerprint(conn, cid), field_key=row["field_key"])
               for cid, row in new_rows.items()]
    matching = identity.match(old_fps, new_fps)
    amb_by_old = {o: a for a in matching.ambiguous for o in a.old}
    hidden = {o["target"]["claim_id"] for o in h.ops if o["op"] == "hide"}

    def agent(cid: str) -> ClaimText:
        r = new_rows[cid]
        return ClaimText(r["claim_type"], r["normalized_label"], r["claim_text"])

    plans: list[OpPlan] = []
    for op in h.ops:
        kind = op["op"]
        if kind == "add":
            plans.append(_plan_add(op, texts, new_md, new_hash, new_text))
            continue
        cid = op["target"]["claim_id"]
        key = f"{kind}:{cid}"
        new_cid = matching.new_for(cid)
        if new_cid is not None:
            plans.append(OpPlan(key, op, "kept" if kind == "edit" else "still_hidden", cid, new_cid,
                                new_agent=agent(new_cid)))
        elif cid in hidden and cid not in amb_by_old:
            plans.append(OpPlan(key, op, "hide_gone", cid))
        elif kind == "hide":
            plans.append(OpPlan(key, op, "ambiguous", cid, candidates=amb_by_old[cid].new))
        else:
            old_quote = _old_quote(conn, texts, ClaimFingerprint.from_dict(cid, op["target"]["fingerprint"]))
            span = anchor.locate_quote(new_text, old_quote) if old_quote else None
            quote = ({"markdown_id": new_md, "markdown_hash": new_hash, "start_char": span[0],
                      "end_char": span[1]} if span else None)
            amb = amb_by_old.get(cid)
            plans.append(OpPlan(key, op, "ambiguous" if amb else "unmatched", cid,
                                candidates=amb.new if amb else (),
                                keep_as="quote" if quote else "assertion", quote=quote))

    op_claims = set(op_fps)
    changed = []
    for p in matching.pairs:
        if p.old in op_claims:
            continue
        o, n = old_rows[p.old], new_rows[p.new]
        if (o["normalized_label"], o["claim_text"]) != (n["normalized_label"], n["claim_text"]):
            changed.append(WordingChange(p.old, p.new, n["field_key"], o["normalized_label"], o["claim_text"],
                                         n["normalized_label"], n["claim_text"]))
    new_claims = tuple(NewClaim(c, new_rows[c]["claim_type"], new_rows[c]["field_key"],
                                new_rows[c]["normalized_label"], new_rows[c]["claim_text"], new_rows[c]["status"])
                       for c in matching.unmatched_new)
    return ReplayPlan(
        paper_id=paper_id, schema_id=schema_id, overlay_id=h.overlay_id, head_version_no=h.version_no,
        base_note_id=h.base_note_id, new_note_id=latest["note_id"], new_markdown_id=new_md, ops=tuple(plans),
        new_claims=new_claims, changed=tuple(changed),
        gone=tuple(c for c in matching.unmatched_old if c not in op_claims),
        ambiguous=matching.ambiguous, matching=matching)


def _plan_add(op: dict[str, Any], texts: _Texts, new_md: str, new_hash: str, new_text: str) -> OpPlan:
    key = op["add_id"]
    q = op.get("quote")
    if op["kind"] != "quote" or not q:
        return OpPlan(key, op, "added_kept")
    if q["markdown_hash"] == new_hash:
        return OpPlan(key, op, "added_kept", quote=dict(q))
    old_text = texts.by_id(q["markdown_id"]) or texts.by_hash(q["markdown_hash"])
    exact = old_text[q["start_char"]:q["end_char"]] if old_text else None
    span = anchor.locate_quote(new_text, exact) if exact else None
    if span is None:
        return OpPlan(key, op, "added_unanchored", keep_as="assertion")
    return OpPlan(key, op, "added_kept", quote={"markdown_id": new_md, "markdown_hash": new_hash,
                                               "start_char": span[0], "end_char": span[1]})


# --- accepting -------------------------------------------------------------------------------

def _quote_op(conn: sqlite3.Connection, quote: Mapping[str, Any]) -> dict[str, Any] | None:
    """Anchor ``quote`` (offsets on the new markdown) as a span; ``None`` if it no longer anchors."""
    text = read_markdown(conn, quote["markdown_id"])
    if text is None:
        return None
    exact = text[quote["start_char"]:quote["end_char"]]
    if not exact or anchor.locate_quote(text, exact) != (quote["start_char"], quote["end_char"]):
        return None
    span_id = ensure_span(conn, markdown_id=quote["markdown_id"], exact_quote=exact, markdown_text=text)
    if span_id is None:
        return None
    return {"markdown_id": quote["markdown_id"], "markdown_hash": quote["markdown_hash"],
            "start_char": quote["start_char"], "end_char": quote["end_char"],
            "quote_hash": anchor.quote_hash(exact), "span_id": span_id}


def _kept_edit_as_add(conn: sqlite3.Connection, p: OpPlan) -> dict[str, Any]:
    """An unmatched edit kept as a researcher claim (the researcher's fields over the agent's)."""
    row = conn.execute("SELECT claim_type, normalized_label, claim_text FROM extracted_claims WHERE claim_id = ?",
                       (p.old_claim_id,)).fetchone()
    set_ = p.op["set"]
    claim_type = set_.get("claim_type") or (row[0] if row else p.op["target"]["fingerprint"]["claim_type"])
    label = set_.get("normalized_label") or (row[1] if row else None) or claim_type
    text = set_.get("claim_text") or (row[2] if row else None) or label
    add: dict[str, Any] = {"op": "add", "add_id": new_id("rc"), "kind": "assertion", "claim_type": claim_type,
                           "normalized_label": label, "claim_text": text}
    quote = _quote_op(conn, p.quote) if p.keep_as == "quote" and p.quote else None
    if quote is not None:
        add.update(kind="quote", quote=quote)
    return add


def accept(
    lib: Library,
    paper_id: str,
    plan: ReplayPlan,
    *,
    accept_new: Literal["all"] | Iterable[str] = "all",
    dismiss: Iterable[str] = (),
    discard: Iterable[str] = (),
    attach: Mapping[str, str] | None = None,
    summary: str | None = None,
) -> overlay.OverlayVersion:
    """Accept the new extraction under the researcher's edits (see the module doc).

    - ``accept_new``: ``"all"`` or the ids of the plan's new agent claims to show; the
      others are hidden. ``dismiss``: new-extraction claim ids to hide (``hide`` ops).
    - ``discard``: op keys (:attr:`OpPlan.key`) to drop.
    - ``attach``: ``{old_claim_id: new_claim_id}``: re-key that claim's edit/hide onto a
      chosen new claim (resolves an ambiguous or unmatched op).
    """
    if paper_id != plan.paper_id:
        raise ReplayError(f"this plan is for {plan.paper_id}, not {paper_id}")
    dismiss, discard, attach = set(dismiss), set(discard), dict(attach or {})
    with lib.write() as conn:
        h, latest = _state(conn, paper_id, plan.schema_id)
        if (h.version_no != plan.head_version_no or h.base_note_id != plan.base_note_id
                or latest is None or latest["note_id"] != plan.new_note_id):
            raise ReplayStale(f"{overlay.STALE_MESSAGE} (the edits or the newest extraction changed "
                              "after this comparison was made)")
        new_ids = set(_claims(conn, plan.new_note_id))
        bad = sorted((dismiss | set(attach.values())) - new_ids)
        if bad:
            raise ReplayError(f"{', '.join(bad)} are not claims of the new extraction")
        if accept_new != "all":
            chosen = set(accept_new)
            listed = {c.claim_id for c in plan.new_claims}
            if chosen - listed:
                raise ReplayError(f"{', '.join(sorted(chosen - listed))} are not new agent claims of this plan")
            dismiss |= listed - chosen

        def target(cid: str) -> dict[str, Any]:
            return {"claim_id": cid, "fingerprint": overlay.fingerprint(conn, cid)}

        was_hidden = {p.old_claim_id for p in plan.ops if p.op["op"] == "hide" and p.key not in discard}
        ops: list[dict[str, Any]] = []
        adds_from_edits: list[dict[str, Any]] = []
        for p in plan.ops:
            if p.key in discard:
                continue
            kind = p.op["op"]
            if kind == "add":
                op = json.loads(json.dumps(p.op))
                if p.status == "added_unanchored":
                    op["kind"] = "assertion"
                    op.pop("quote", None)
                elif p.quote is not None and p.quote["markdown_hash"] != (p.op.get("quote") or {}).get(
                        "markdown_hash"):
                    quote = _quote_op(conn, p.quote)
                    if quote is None:
                        op["kind"] = "assertion"
                        op.pop("quote", None)
                    else:
                        op["quote"] = quote
                ops.append(op)
                continue
            new_cid = attach.get(p.old_claim_id) or p.new_claim_id  # type: ignore[arg-type]
            if new_cid is not None:
                op = {"op": kind, "target": target(new_cid)}
                if kind == "edit":
                    op["set"] = dict(p.op["set"])
                ops.append(op)
            elif kind == "edit" and p.status in ("unmatched", "ambiguous") and p.old_claim_id not in was_hidden:
                adds_from_edits.append(_kept_edit_as_add(conn, p))
            # a hide that is gone or undecided, and an edit of a gone hidden claim, are dropped
        hidden = {o["target"]["claim_id"] for o in ops if o["op"] == "hide"}
        for cid in sorted(dismiss - hidden):
            ops.append({"op": "hide", "target": target(cid)})
        ops.extend(adds_from_edits)
        return overlay.write_version(
            conn, paper_id, ops=ops, researcher_notes=h.researcher_notes, action=ACTION,
            expected_version=plan.head_version_no, base_note_id=plan.new_note_id,
            summary=summary or _summary(plan, ops), schema_id=plan.schema_id)


def _summary(plan: ReplayPlan, ops: list[dict[str, Any]]) -> str:
    return (f"accepted the newer extraction: {plan.kept} edits kept, {plan.still_hidden} still hidden, "
            f"{sum(1 for o in ops if o['op'] == 'add')} researcher claims, {len(plan.new_claims)} new agent claims")


#: The design's name (§1.2).
plan = plan_replay

__all__ = ["ACTION", "ClaimText", "NewClaim", "OpPlan", "ReplayError", "ReplayPlan", "ReplayStale",
           "WordingChange", "accept", "plan", "plan_replay"]

