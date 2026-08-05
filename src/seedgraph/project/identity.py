"""Identifier normalization + conservative strong-id merge (decision 7/8/13/64).

This module carries the v1 ``identity.py`` forward, including the first-class
**cross-work identifier-collision** path. The cardinal rule: strong ids merge
**only** when they resolve to exactly one existing work; title hashes NEVER
auto-merge; and incoming ids that resolve to two or more distinct works are routed
to ``review_queue`` as a ``duplicate_candidate`` rather than ever crashing on the
``UNIQUE(id_type, id_value)`` constraint.

Strong-id precedence (highest canonical authority first):
``doi > openalex > arxiv > s2 > ssrn``.

The small title-normalization helpers (``sha1_hex`` / ``normalize_title`` /
``title_hash``) live here rather than in the phase_0 ``ids.py`` so this phase adds
no edit to a foundation module; they are pure and deterministic.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, TypeAlias

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from ..db.project_models import Identifier, Work
from ..ids import new_id
from . import review

# Closed set of strong external id kinds (validated at the app boundary; the
# DB intentionally carries no CHECK on identifiers.id_type so new kinds can land
# in a later phase without a schema break).
IdType = Literal["doi", "openalex", "arxiv", "s2", "ssrn"]

# Canonical-key precedence: earlier wins.
ID_TYPE_PRECEDENCE: tuple[IdType, ...] = ("doi", "openalex", "arxiv", "s2", "ssrn")

# An incoming, user-supplied work record: a mapping of normalized fields
# (title/authors/year/venue + any subset of the strong id kinds above).
WorkLike: TypeAlias = Mapping[str, Any]


class CrossPaperIdentifierCollision(Exception):
    """Incoming strong ids resolve to >1 existing work.

    Raised/handled by routing the case to ``review_queue`` as a
    ``duplicate_candidate`` (decision 7/8) — the merge flow NEVER auto-merges the
    conflicting works and NEVER blind-inserts a colliding identifier.
    """

    def __init__(self, conflicting_work_ids: list[str]) -> None:
        self.conflicting_work_ids = list(conflicting_work_ids)
        super().__init__(
            "incoming strong ids resolve to multiple existing works: "
            f"{self.conflicting_work_ids}"
        )


# --- Pure title-normalization helpers (trivial; implemented) ----------------

def sha1_hex(text: str) -> str:
    """SHA-1 of ``text`` (UTF-8) as a lowercase hex string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching: lower, NFKD-fold (drop diacritics),
    strip punctuation, collapse whitespace to single spaces.

    Deterministic and idempotent: ``normalize_title(normalize_title(t)) ==
    normalize_title(t)``. Recomputable on demand — no ``normalized_title`` column
    is stored (only the derived ``title_hash``).
    """
    decomposed = unicodedata.normalize("NFKD", title)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9]+", " ", no_marks.lower())
    return " ".join(cleaned.split())


def title_hash(title: str) -> str:
    """``sha1_hex(normalize_title(title))`` — the indexed fuzzy matcher.

    NEVER an auto-merge key (decision 13/64): a title-hash collision routes to
    review, it does not merge works.
    """
    return sha1_hex(normalize_title(title))


# --- ID normalization -------------------------------------------------------

# The ``works`` scalar id column that mirrors each strong id_type (decision 25 —
# denormalized onto the work for the per-column indexes).
ID_TYPE_TO_WORK_COLUMN: dict[IdType, str] = {
    "doi": "doi",
    "openalex": "openalex_id",
    "arxiv": "arxiv_id",
    "s2": "semantic_scholar_id",
    "ssrn": "ssrn_id",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# v1-ported surface-form normalizers (prototype identity.py:163-240, pure regex).
_DOI_PREFIX_RE = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"^(?:https?://)?arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)
_ARXIV_LABEL_RE = re.compile(r"^arxiv:", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_OPENALEX_URL_RE = re.compile(r"^(?:https?://)?openalex\.org/", re.IGNORECASE)

#: The arXiv-DOI alias family (``10.48550/arxiv.{id}``) — the identity/cache/fetch
#: layers must agree on this fold (gap scan §4.7, live-caught 2026-05-31). Single
#: definition: ``acquisition/fetch.py`` imports it from here.
ARXIV_DOI_RE = re.compile(r"(?i)^10\.48550/arxiv\.(.+)$")


def normalize_id(id_type: str, value: Any) -> str | None:
    """Normalize a raw external id value deterministically, or ``None`` if empty.

    Trims surrounding whitespace, then per type (v1 parity):

    * ``doi`` — strip ``doi:`` and any ``doi.org`` / ``dx.doi.org`` URL prefix
      (scheme optional); lowercase.
    * ``arxiv`` — strip ``arxiv.org/abs|pdf/`` URL prefixes and the ``arXiv:``
      label; DROP the ``v\\d+`` version suffix (v1/v2 of a preprint are the same
      work — the strip is load-bearing for the merge contract).
    * ``openalex`` — strip the ``openalex.org/`` URL prefix; uppercase a leading
      ``w``.
    * ``s2`` / ``ssrn`` — opaque, trimmed only.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if id_type == "doi":
        text = _DOI_PREFIX_RE.sub("", text)
        if text.lower().startswith("doi:"):
            text = text[len("doi:"):]
        text = text.strip().lower()
    elif id_type == "arxiv":
        text = _ARXIV_URL_RE.sub("", text)
        text = _ARXIV_LABEL_RE.sub("", text).strip()
        text = _ARXIV_VERSION_RE.sub("", text)
    elif id_type == "openalex":
        text = _OPENALEX_URL_RE.sub("", text).strip()
        if text and text[0] in ("w", "W"):
            text = "W" + text[1:]
    return text or None


# --- Identity resolution + merge decision -----------------------------------

def canonical_key(rec: WorkLike) -> tuple[IdType, str] | None:
    """Return the single highest-precedence ``(id_type, normalized_value)`` present
    on ``rec``, or ``None`` if it carries no strong id.

    Precedence: ``doi > openalex > arxiv > s2 > ssrn`` (:data:`ID_TYPE_PRECEDENCE`).
    Defined as ``strong_ids(rec)[0]`` so key and lookup can never disagree (the
    arXiv-DOI fold applies to both by construction).
    """
    pairs = strong_ids(rec)
    return pairs[0] if pairs else None


def strong_ids(rec: WorkLike) -> list[tuple[IdType, str]]:
    """Return ALL strong ids present on ``rec`` as normalized ``(id_type, value)``
    pairs, in precedence order (used to look the record up by every id it carries).

    A DOI in the arXiv-DOI alias family (``10.48550/arxiv.{id}``) is FOLDED to an
    ``("arxiv", id)`` pair and the doi pair is NOT emitted — otherwise the doi-first
    precedence would keep ``{doi: 10.48550/arxiv.X}`` and ``{arxiv: X}`` as two
    disjoint identities (gap scan §4.7).
    """
    values: dict[IdType, str] = {}
    for id_type in ID_TYPE_PRECEDENCE:
        value = normalize_id(id_type, rec.get(id_type))
        if value:
            values[id_type] = value
    doi = values.get("doi")
    if doi is not None:
        match = ARXIV_DOI_RE.match(doi)
        if match:
            del values["doi"]
            folded = normalize_id("arxiv", match.group(1))
            if folded:
                values.setdefault("arxiv", folded)
    return [(id_type, values[id_type]) for id_type in ID_TYPE_PRECEDENCE if id_type in values]


def match_works(session: Session, incoming: WorkLike) -> list[Work]:
    """Look up EVERY incoming strong id in ``identifiers`` and return the DISTINCT
    set of works any of them resolve to (0, 1, or >=2), order-stable.

    The >=2 case is the cross-work identifier collision (decision 7/8)."""
    work_ids: list[str] = []
    seen: set[str] = set()
    for id_type, id_value in strong_ids(incoming):
        rows = session.exec(
            select(Identifier).where(
                Identifier.id_type == id_type, Identifier.id_value == id_value
            )
        ).all()
        for row in rows:
            if row.work_id not in seen:
                seen.add(row.work_id)
                work_ids.append(row.work_id)
    works: list[Work] = []
    for work_id in work_ids:
        work = session.get(Work, work_id)
        if work is not None:
            works.append(work)
    return works


def merge_decision(
    matched: list[Work], incoming: WorkLike
) -> Literal["create", "merge", "duplicate_candidate"]:
    """Map the matched-work set to an outcome: 0 -> ``create``; exactly 1 ->
    ``merge``; >=2 distinct -> ``duplicate_candidate`` (cross-id collision)."""
    distinct = {work.work_id for work in matched}
    if not distinct:
        return "create"
    if len(distinct) == 1:
        return "merge"
    return "duplicate_candidate"


def _existing_id_values(session: Session, sids: list[tuple[IdType, str]]) -> set[tuple[str, str]]:
    """The subset of ``sids`` already present in ``identifiers`` (any owner)."""
    claimed: set[tuple[str, str]] = set()
    for id_type, id_value in sids:
        row = session.exec(
            select(Identifier).where(
                Identifier.id_type == id_type, Identifier.id_value == id_value
            )
        ).first()
        if row is not None:
            claimed.add((id_type, id_value))
    return claimed


def _attach_unclaimed_ids(
    session: Session, work_id: str, sids: list[tuple[IdType, str]]
) -> None:
    """Insert each strong id NOT already present in ``identifiers`` (never
    blind-insert a colliding id, so ``UNIQUE(id_type,id_value)`` is never violated)."""
    claimed = _existing_id_values(session, sids)
    for id_type, id_value in sids:
        if (id_type, id_value) in claimed:
            continue
        session.add(
            Identifier(
                work_id=work_id,
                id_type=id_type,
                id_value=id_value,
                resolution_source="user",
                confidence=None,
            )
        )
    session.flush()


def _create_work(session: Session, incoming: WorkLike, th: str | None) -> Work:
    """Mint and insert a new ``works`` row from the incoming metadata (ids mirrored
    onto the indexed scalar columns)."""
    work = Work(
        work_id=new_id("work"),
        canonical_title=incoming.get("title"),
        title_hash=th,
        authors=list(incoming["authors"]) if incoming.get("authors") else None,
        venue=incoming.get("venue"),
        year=incoming.get("year"),
        abstract=incoming.get("abstract"),
        oa_status=incoming.get("oa_status"),
        created_at=_now(),
    )
    for id_type, id_value in strong_ids(incoming):
        setattr(work, ID_TYPE_TO_WORK_COLUMN[id_type], id_value)
    session.add(work)
    session.flush()
    return work


def backfill(work: Work, incoming: WorkLike, th: str | None) -> None:
    """Fill only the still-empty scalar fields of ``work`` from ``incoming`` (a merge
    never overwrites a value already present)."""
    # Empty-only means BLANK-only for the title (Build D ch9 title-backfill fold):
    # a once-NULL canonical_title can be persisted as "" by an intermediate upsert
    # (the v1 lesson), and a blank title carries no information — filling it is
    # the read-repair acquisition/backfill.py relies on. A present (non-blank)
    # title is still NEVER overwritten.
    if not (work.canonical_title or "").strip() and incoming.get("title") is not None:
        work.canonical_title = incoming["title"]
    if work.title_hash is None and th is not None:
        work.title_hash = th
    if work.authors is None and incoming.get("authors"):
        work.authors = list(incoming["authors"])
    if work.venue is None and incoming.get("venue") is not None:
        work.venue = incoming["venue"]
    if work.year is None and incoming.get("year") is not None:
        work.year = incoming["year"]
    if work.abstract is None and incoming.get("abstract") is not None:
        work.abstract = incoming["abstract"]
    if work.oa_status is None and incoming.get("oa_status") is not None:
        work.oa_status = incoming["oa_status"]
    for id_type, id_value in strong_ids(incoming):
        column = ID_TYPE_TO_WORK_COLUMN[id_type]
        if getattr(work, column) is None:
            setattr(work, column, id_value)


def _works_by_title_hash(session: Session, th: str) -> list[Work]:
    return list(session.exec(select(Work).where(Work.title_hash == th)).all())


def _snapshot(incoming: WorkLike) -> dict:
    """A JSON-able snapshot of the incoming WorkLike for the review payload."""
    snap: dict[str, Any] = {}
    for key, value in incoming.items():
        if key == "authors" and value is not None:
            snap[key] = list(value)
        else:
            snap[key] = value
    return snap


# Weak id kinds (v1 parity): strong ONLY when corroborated — a single-work match
# shared solely on these types with conflicting title evidence demotes to review.
_ALWAYS_STRONG: frozenset[str] = frozenset({"doi", "openalex"})


def _conflicting_id(
    session: Session, work: Work, sids: list[tuple[IdType, str]]
) -> tuple[str, str, str] | None:
    """First incoming ``(id_type, value)`` whose value the matched ``work`` already
    owns DIFFERENTLY (walked in precedence order), as ``(id_type, incoming, owned)``,
    or ``None`` (v1 Fix 1 — a present-and-conflicting strong id blocks a merge)."""
    for id_type, id_value in sids:
        column = ID_TYPE_TO_WORK_COLUMN[id_type]
        owned = _owned_id_value(session, work.work_id, id_type, work, column)
        if owned is not None and owned != id_value:
            return (id_type, id_value, owned)
    return None


def _shared_id_types(
    session: Session, work: Work, sids: list[tuple[IdType, str]]
) -> set[str]:
    """Incoming id types whose value the matched ``work`` owns identically."""
    shared: set[str] = set()
    for id_type, id_value in sids:
        column = ID_TYPE_TO_WORK_COLUMN[id_type]
        if _owned_id_value(session, work.work_id, id_type, work, column) == id_value:
            shared.add(id_type)
    return shared


def _demote_to_duplicate(
    session: Session,
    incoming: WorkLike,
    sids: list[tuple[IdType, str]],
    th: str | None,
    *,
    reason: str,
    conflicting_work_ids: list[str],
) -> tuple[Work, str]:
    """Blocked-merge path: create the incoming as a DISTINCT work, attach only
    unclaimed ids, and enqueue a ``duplicate_candidate`` (never silently merge)."""
    work = _create_work(session, incoming, th)
    _attach_unclaimed_ids(session, work.work_id, sids)
    review.enqueue_in_session(
        session,
        "duplicate_candidate",
        target_type="work",
        target_id=work.work_id,
        payload={
            "kind": "duplicate_candidate",
            "reason": reason,
            "conflicting_work_ids": conflicting_work_ids,
            "incoming": _snapshot(incoming),
        },
    )
    return work, "duplicate_candidate"


def upsert_work(session: Session, incoming: WorkLike) -> tuple[Work, str]:
    """Resolve-or-create a work from ``incoming`` and return ``(work, outcome)``
    with ``outcome in {"created", "merged", "duplicate_candidate"}``.

    Implements the 5-step flow (plan §6) that resolves the must-fix — no UNIQUE
    crash, no silent over-merge:

    1. ``matched = match_works(session, incoming)``.
    2. 0 matches -> ``create``: mint ``work_id``, insert work + each strong id.
    3. exactly 1 -> ``merge``: backfill missing scalars; insert only the incoming
       strong ids not already present (skip ids already owned by this work) —
       UNLESS (a) a strong id the matched work owns DIFFERS from the incoming value
       (v1 Fix 1: preprint/published pairs share an arXiv id with different DOIs)
       or (b) the ONLY shared id types are weak (arxiv/s2/ssrn) and the two title
       hashes conflict (v1 Fix 5): either demotes to ``duplicate_candidate``
       (reasons ``id_disagreement`` / ``weak_id_title_conflict``).
    4. >=2 distinct -> ``duplicate_candidate``: do NOT merge, do NOT blind-insert
       any colliding id; mint a fresh work with the incoming metadata, attach only
       the as-yet-unclaimed strong ids, and enqueue a ``duplicate_candidate`` review
       item (``reason="cross_id_collision"``, ``conflicting_work_ids=[...]``).
    5. Title-only path: when steps 2-4 yield ``create`` but ``title_hash`` matches
       an existing work, still create the distinct work (titles NEVER auto-merge)
       and additionally enqueue a ``duplicate_candidate`` item with
       ``reason="title_collision"``.
    """
    matched = match_works(session, incoming)
    decision = merge_decision(matched, incoming)
    sids = strong_ids(incoming)
    title = incoming.get("title")
    th = title_hash(title) if title else None

    if decision == "merge":
        work = matched[0]
        if _conflicting_id(session, work, sids) is not None:
            return _demote_to_duplicate(
                session, incoming, sids, th,
                reason="id_disagreement", conflicting_work_ids=[work.work_id],
            )
        shared = _shared_id_types(session, work, sids)
        if (
            not (shared & _ALWAYS_STRONG)
            and th is not None
            and work.title_hash is not None
            and th != work.title_hash
        ):
            return _demote_to_duplicate(
                session, incoming, sids, th,
                reason="weak_id_title_conflict", conflicting_work_ids=[work.work_id],
            )
        backfill(work, incoming, th)
        _attach_unclaimed_ids(session, work.work_id, sids)
        session.add(work)
        session.flush()
        return work, "merged"

    if decision == "duplicate_candidate":
        conflicting = [work.work_id for work in matched]
        work = _create_work(session, incoming, th)
        _attach_unclaimed_ids(session, work.work_id, sids)
        review.enqueue_in_session(
            session,
            "duplicate_candidate",
            target_type="work",
            target_id=work.work_id,
            payload={
                "kind": "duplicate_candidate",
                "reason": "cross_id_collision",
                "conflicting_work_ids": conflicting,
                "incoming": _snapshot(incoming),
            },
        )
        return work, "duplicate_candidate"

    # decision == "create"
    work = _create_work(session, incoming, th)
    _attach_unclaimed_ids(session, work.work_id, sids)
    if th is not None:
        title_dups = [w.work_id for w in _works_by_title_hash(session, th) if w.work_id != work.work_id]
        if title_dups:
            review.enqueue_in_session(
                session,
                "duplicate_candidate",
                target_type="work",
                target_id=work.work_id,
                payload={
                    "kind": "duplicate_candidate",
                    "reason": "title_collision",
                    "conflicting_work_ids": title_dups,
                    "incoming": _snapshot(incoming),
                },
            )
    return work, "created"


# --- post-conversion bibliographic backfill (metadata_extraction) -----------

# (id_type, ``works`` scalar column, BiblioMeta attribute) for the two strong ids
# the LLM/regex backfill can attach. Never a NEW work — attach-or-review only.
_BIBLIO_ID_FIELDS: tuple[tuple[IdType, str], ...] = (("doi", "doi"), ("arxiv", "arxiv_id"))


@dataclass
class BiblioUpdate:
    """Outcome of one :func:`update_work_bibliography` call (the CLI per-work line).

    ``ids_attached`` are ``(id_type, id_value)`` pairs newly written for THIS work;
    ``ids_collided`` are pairs already owned by a DIFFERENT work (routed to review,
    never merged/inserted); ``fields_filled`` names the scalar columns filled.
    """

    work_id: str
    old_title: str | None = None
    new_title: str | None = None
    title_changed: bool = False
    fields_filled: list[str] = field(default_factory=list)
    ids_attached: list[tuple[str, str]] = field(default_factory=list)
    ids_collided: list[tuple[str, str]] = field(default_factory=list)


def work_has_strong_id(work: Work) -> bool:
    """True if any of the work's scalar strong-id columns is populated."""
    return any(getattr(work, column, None) for column in ID_TYPE_TO_WORK_COLUMN.values())


def has_resolution_provenance(session: Session, work_id: str) -> bool:
    """True if any identifier of ``work_id`` carries real resolver provenance.

    ``resolution_source`` NULL or ``'user'`` is NOT provenance (a user-typed id on a
    filename-derived stub); any other value (an id_type name, ``'title'``, a provider
    name) means the work was authoritatively resolved.
    """
    rows = session.exec(select(Identifier).where(Identifier.work_id == work_id)).all()
    return any((row.resolution_source or None) not in (None, "user") for row in rows)


def is_filename_derived_title(session: Session, work: Work) -> bool:
    """Heuristic: is the work's ``canonical_title`` a replaceable filename-derived stub?

    True (safe to replace) iff the work carries NO strong id AND has no resolution
    provenance — i.e. nothing authoritative has ever named it, so the title is just
    the user's PDF-filename acquisition convention. An authoritative resolve title
    (strong id or ``resolution_source`` set) returns False and is NEVER clobbered.
    """
    return not work_has_strong_id(work) and not has_resolution_provenance(session, work.work_id)


def _owned_id_value(
    session: Session, work_id: str, id_type: str, work: Work, column: str
) -> str | None:
    """The value THIS work already owns for ``id_type`` — the scalar mirror column
    if set, else the first matching ``identifiers`` row — or ``None`` if unowned.

    Used to detect a same-work id disagreement (a --force backfill that extracted a
    strong id different from the one the work already carries)."""
    scalar = getattr(work, column, None)
    if scalar:
        return scalar
    row = session.exec(
        select(Identifier).where(
            Identifier.work_id == work_id, Identifier.id_type == id_type
        )
    ).first()
    return row.id_value if row is not None else None


def _enqueue_id_collision(
    session: Session,
    *,
    work_id: str,
    conflicting_work_id: str,
    id_type: str,
    id_value: str,
    title: str | None,
    reason: str,
) -> None:
    """Route a strong-id conflict to ``review_queue`` (never merge/insert the id).

    ``reason='cross_id_collision'`` when a DIFFERENT work owns the id;
    ``reason='id_disagreement'`` when THIS work already owns a different value."""
    review.enqueue_in_session(
        session,
        "duplicate_candidate",
        target_type="work",
        target_id=work_id,
        payload={
            "kind": "duplicate_candidate",
            "reason": reason,
            "conflicting_work_ids": [conflicting_work_id],
            "incoming": {
                "id_type": id_type,
                "id_value": id_value,
                "title": title,
                "backfill_work_id": work_id,
            },
        },
    )


def update_work_bibliography(
    session: Session,
    work_id: str,
    meta: Any,
    *,
    allow_title_overwrite: bool = False,
) -> BiblioUpdate:
    """Apply extracted bibliographic ``meta`` to an EXISTING work (never creates one).

    ``meta`` is any object exposing ``title`` / ``authors`` / ``year`` / ``venue`` /
    ``doi`` / ``arxiv_id`` (duck-typed to avoid an extraction↔identity import cycle).

    * ``authors`` / ``venue`` / ``year`` are filled only when currently empty.
    * ``canonical_title`` is replaced from ``meta.title`` ONLY when the current title
      is filename-derived (:func:`is_filename_derived_title`) or ``allow_title_overwrite``
      is set — an authoritative resolve title is preserved.
    * for each of ``meta.doi`` / ``meta.arxiv_id``: normalize; if that
      ``(id_type, id_value)`` already exists on a DIFFERENT work, enqueue a
      ``duplicate_candidate`` (``cross_id_collision``) review item and skip (NO merge,
      NO insert); otherwise insert the identifier row for THIS work AND set the work's
      scalar mirror column.

    Flushes (the caller commits). Returns a :class:`BiblioUpdate` summary.
    """
    work = session.get(Work, work_id)
    result = BiblioUpdate(work_id=work_id)
    if work is None:
        return result
    result.old_title = work.canonical_title

    authors = getattr(meta, "authors", None)
    if authors and (work.authors is None or len(work.authors) == 0):
        work.authors = list(authors)
        result.fields_filled.append("authors")
    venue = getattr(meta, "venue", None)
    if venue and work.venue is None:
        work.venue = venue
        result.fields_filled.append("venue")
    year = getattr(meta, "year", None)
    if year is not None and work.year is None:
        work.year = year
        result.fields_filled.append("year")

    new_title = getattr(meta, "title", None)
    if new_title:
        replace = allow_title_overwrite or is_filename_derived_title(session, work)
        if replace and new_title != work.canonical_title:
            work.canonical_title = new_title
            work.title_hash = title_hash(new_title)
            result.title_changed = True
    result.new_title = work.canonical_title

    # Persist the scalar/title fills BEFORE the per-id savepoint inserts so a
    # savepoint rollback (concurrent-insert race, below) can never revert them.
    session.add(work)
    session.flush()

    for id_type, attr in _BIBLIO_ID_FIELDS:
        value = normalize_id(id_type, getattr(meta, attr, None))
        if not value:
            continue
        column = ID_TYPE_TO_WORK_COLUMN[id_type]

        # This work already OWNS a value for this id_type (scalar mirror or an
        # identifier row). If it DISAGREES with the extracted one (only reachable
        # under --force), route to review — never append a second, conflicting id
        # row. If it matches, there is nothing to do.
        owned = _owned_id_value(session, work_id, id_type, work, column)
        if owned is not None:
            if owned != value:
                _enqueue_id_collision(
                    session, work_id=work_id, conflicting_work_id=work_id,
                    id_type=id_type, id_value=value, title=new_title,
                    reason="id_disagreement",
                )
                result.ids_collided.append((id_type, value))
            continue

        existing = session.exec(
            select(Identifier).where(
                Identifier.id_type == id_type, Identifier.id_value == value
            )
        ).first()
        if existing is not None:
            # A DIFFERENT work owns this exact id (this work owns none — ruled out
            # above) — route to review, NEVER merge/insert.
            _enqueue_id_collision(
                session, work_id=work_id, conflicting_work_id=existing.work_id,
                id_type=id_type, id_value=value, title=new_title,
                reason="cross_id_collision",
            )
            result.ids_collided.append((id_type, value))
            continue

        # Insert under a SAVEPOINT so a concurrent backfill (one _convert_worker per
        # GPU) that wins the UNIQUE(id_type,id_value) race between our SELECT and this
        # INSERT is caught + re-resolved here, not swallowed by best_effort_backfill.
        try:
            with session.begin_nested():
                session.add(
                    Identifier(
                        work_id=work_id,
                        id_type=id_type,
                        id_value=value,
                        resolution_source="metadata_backfill",
                        confidence=None,
                    )
                )
                session.flush()
        except IntegrityError:
            owner = session.exec(
                select(Identifier).where(
                    Identifier.id_type == id_type, Identifier.id_value == value
                )
            ).first()
            if owner is not None and owner.work_id != work_id:
                # The race winner is a DIFFERENT work — a real cross-id collision.
                _enqueue_id_collision(
                    session, work_id=work_id, conflicting_work_id=owner.work_id,
                    id_type=id_type, id_value=value, title=new_title,
                    reason="cross_id_collision",
                )
                result.ids_collided.append((id_type, value))
            elif getattr(work, column, None) is None:
                # Now owned by THIS work (our own parallel insert) — mirror the scalar.
                setattr(work, column, value)
                session.flush()
            continue

        if getattr(work, column, None) is None:
            setattr(work, column, value)
        result.ids_attached.append((id_type, value))
        session.flush()

    return result
