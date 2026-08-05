"""Polymorphic ``review_queue`` surface + per-item_type validated payloads (decision 80).

``review_queue`` is a single polymorphic table (owned by phase_0's
``project/0001_foundation.sql``; this phase only reads/writes it, never re-authors
it). Each row's JSON ``payload`` is validated against a discriminated union keyed on
``kind`` *at the app boundary* before it is persisted — so unvalidated JSON never
reaches the table. The union grows one variant per ``item_type`` as later phases add
reviewers (citation resolution, concept merge, claim audit); this phase ships the
single ``duplicate_candidate`` variant exercised by the identity-merge flow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter
from sqlmodel import Session, select

from ..display import derive_label
from ..errors import ValidationError
from ..ids import new_id
from ..db.project_models import ReviewQueueItem
from ..vocab import ReviewAction

if TYPE_CHECKING:
    from .service import ProjectHandle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DuplicateCandidatePayload(BaseModel):
    """Payload for a ``duplicate_candidate`` review item (decision 7/8/80).

    ``reason`` distinguishes the cross-work strong-id collision from a title-only
    collision; ``conflicting_work_ids`` lists the existing works involved; ``incoming``
    is a snapshot of the incoming WorkLike that triggered the item.
    """

    kind: Literal["duplicate_candidate"]
    # ``id_disagreement``: a work already owns a DIFFERENT value for a strong
    # id_type the incoming asserts (the upsert merge blocker, v1 Fix 1; also the
    # --force backfill disagreement path). ``weak_id_title_conflict``: a single-work
    # match shared ONLY on weak ids (arxiv/s2/ssrn) with conflicting title hashes
    # (v1 Fix 5) — demoted to review instead of silently merging.
    reason: Literal[
        "cross_id_collision",
        "title_collision",
        "id_disagreement",
        "weak_id_title_conflict",
    ]
    conflicting_work_ids: list[str]
    incoming: dict


class CitationResolutionPayload(BaseModel):
    """Payload for a ``citation_resolution`` review item (phase_3b; decision 80).

    Enqueued for an ``ambiguous`` (title-only / unpinnable) or ``suspect``
    (precision-guard-rejected) parsed-bibliography reference. The entry stays raw in
    ``reference_entries``; this item carries the parsed fields, the provider
    candidate(s) (``ambiguous``), and the rejected record + reason (``suspect``) so a
    reviewer can approve a ``manual_override`` edge or reject. ``run_id`` scopes it to
    the provider walk that produced the projection.
    """

    kind: Literal["citation_resolution"]
    status: Literal["ambiguous", "suspect"]
    citing_work_id: str
    reference_id: str
    raw: str
    title: str | None = None
    year: int | None = None
    first_author: str | None = None
    doi: str | None = None
    arxiv: str | None = None
    candidates: list[dict] = []
    rejected_record: dict | None = None
    reject_reason: str | None = None
    run_id: str


class ConceptMergeCandidatePayload(BaseModel):
    """Payload for a ``concept_merge_candidate`` review item (phase_7; decision 80).

    Enqueued for a BORDERLINE LLM-proposed synonym fold that passed the
    deterministic guardrails but is not independently justified by an exact-key /
    acronym fold. The two concepts STAY SPLIT until resolved (plan §4.3): ``approve``
    applies the fold (writes a ``concept_aliases`` row + a sticky ``must_link``
    constraint + marks the surviving concept ``user_confirmed``); ``reject``/``split``
    writes a sticky ``cannot_link`` constraint + marks ``user_split``. Both
    constraints are consulted before every subsequent build (no cross-run flip).
    Labels are NORMALIZED merge-keys; surfaces are the human-facing forms.
    """

    kind: Literal["concept_merge_candidate"]
    canonical_label: str
    member_label: str
    canonical_concept_id: str
    member_concept_id: str
    canonical_surface: str | None = None
    member_surface: str | None = None
    confidence: float | None = None
    run_id: str | None = None


class ConceptEdgeCandidatePayload(BaseModel):
    """Payload for a ``concept_edge_candidate`` review item (phase_7; decision 80).

    The ``UserValidation --validates/rejects--> Edge`` workflow (doc 07 §10/§11) for
    a borderline interpretive ``Concept--Concept`` relation the LLM inferred
    (``related_to``/``broader_than``/``narrower_than``/``contrasts_with``). ``approve``
    flips the edge's ``epistemic_type`` to ``user_validated``; ``reject`` deletes the
    edge row. ``edge_id`` is the ``project_graph_edges`` row under review (also the
    item's ``target_id``).
    """

    kind: Literal["concept_edge_candidate"]
    edge_id: str
    edge_type: str
    source_concept_id: str
    target_concept_id: str
    confidence: float | None = None
    run_id: str | None = None


class LensRecordPayload(BaseModel):
    """Payload for a ``lens_record`` review item (phase_6; decision 80).

    Enqueued for an ``ambiguous`` lens record (a ``found`` record whose verbatim
    quote could not anchor to >=1 evidence_span under
    ``evidence_policy.require_evidence_span``) or an ``extraction_failed`` record
    (invalid JSON overall, or an ``inferred`` record with empty notes under
    ``inferred_requires_explanation``). ``lens_output_id`` back-references the
    lens_outputs row that holds the full ``fields_json``; ``raw`` carries the
    verbatim quote / offending payload and ``reason`` the gate that fired.
    """

    kind: Literal["lens_record"]
    lens_id: str
    work_id: str
    status: Literal["ambiguous", "extraction_failed"]
    reason: str
    lens_output_id: str | None = None
    raw: str | None = None


class UnmatchedUploadPayload(BaseModel):
    """Payload for an ``unmatched_upload`` review item (corpus ingest-folder).

    A PDF dropped into ``corpus ingest-folder`` (or the web bulk drop-zone) that
    matched NO pending work (``reason='no_match'``) or matched AMBIGUOUSLY
    (``reason='ambiguous_match'`` — a cross-id / title collision, never auto-merged).
    The bytes are already ingested (fail-closed ``user_supplied_private``) and
    converted, so this payload carries the cache anchors (``source_file_id`` /
    ``file_hash`` / ``markdown_id``) + the deterministically-extracted identity
    (``doi`` / ``arxiv`` / ``extracted_title``) a reviewer needs to adopt it into a
    new work. ``candidates`` lists the colliding work_ids for an ambiguous match.
    Carries ids/hashes/filename/extracted-identity ONLY — never full text.
    """

    kind: Literal["unmatched_upload"]
    reason: Literal["no_match", "ambiguous_match"]
    source_file_id: str
    file_hash: str
    markdown_id: str | None = None
    filename: str
    doi: str | None = None
    arxiv: str | None = None
    extracted_title: str | None = None
    candidates: list[str] = []


# Discriminated union over ``kind`` — grows one variant per item_type. Validation
# by discriminator happens at the enqueue boundary (decision 80).
ReviewPayload = Annotated[
    Union[
        DuplicateCandidatePayload,
        CitationResolutionPayload,
        ConceptMergeCandidatePayload,
        ConceptEdgeCandidatePayload,
        LensRecordPayload,
        UnmatchedUploadPayload,
    ],
    Field(discriminator="kind"),
]

# Module-level adapter so payload validation is the same machinery the union grows.
_PAYLOAD_ADAPTER: TypeAdapter = TypeAdapter(ReviewPayload)


def _validate_payload(payload: dict | None) -> dict | None:
    """Validate ``payload`` against :data:`ReviewPayload` (decision 80) and return a
    normalized JSON-able dict, or ``None`` when there is no payload."""
    if payload is None:
        return None
    model = _PAYLOAD_ADAPTER.validate_python(payload)
    return model.model_dump()


def enqueue_in_session(
    session: Session,
    item_type: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    payload: dict | None = None,
) -> str:
    """Validate ``payload`` then add a new ``review_queue`` row to ``session``
    (flushed, not committed); return the minted ``item_id``.

    The transaction-participating core used by both the standalone :func:`enqueue`
    and the identity-merge flow (``project.identity.upsert_work``), so a
    ``duplicate_candidate`` is enqueued atomically with the new work it concerns.
    Validation happens BEFORE any write (decision 80) — an invalid payload raises
    and nothing is added.
    """
    validated = _validate_payload(payload)
    item_id = new_id("rq")
    row = ReviewQueueItem(
        item_id=item_id,
        item_type=item_type,
        target_type=target_type,
        target_id=target_id,
        payload=json.dumps(validated) if validated is not None else None,
        status="open",
        action=None,
        created_at=_now(),
        resolved_at=None,
    )
    session.add(row)
    session.flush()
    return item_id


def enqueue(
    h: "ProjectHandle",
    item_type: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    payload: dict | None = None,
) -> str:
    """Validate ``payload`` via :data:`ReviewPayload` then persist a new
    ``review_queue`` row; return the minted ``item_id`` (``"rq_" + uuid4().hex``).

    Validation happens BEFORE any write (decision 80) — a payload that fails the
    discriminated-union validation raises and nothing is persisted.
    """
    with Session(h.engine, expire_on_commit=False) as session:
        item_id = enqueue_in_session(
            session,
            item_type,
            target_type=target_type,
            target_id=target_id,
            payload=payload,
        )
        session.commit()
    return item_id


def enqueue_raw(
    conn,
    item_type: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    payload: dict | None = None,
) -> str:
    """Validate ``payload`` then INSERT a ``review_queue`` row on a raw sqlite3
    connection; return the minted ``item_id``.

    The transaction-participating path used by the phase_7 concept build (which
    holds a raw ``sqlite3.Connection``, not an ORM session). Validation happens
    BEFORE the write (decision 80) — an invalid payload raises and nothing is
    inserted. The caller owns the surrounding transaction/commit.
    """
    validated = _validate_payload(payload)
    item_id = new_id("rq")
    conn.execute(
        "INSERT INTO review_queue "
        "(item_id, item_type, target_type, target_id, payload, status, action, "
        "created_at, resolved_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            item_id,
            item_type,
            target_type,
            target_id,
            json.dumps(validated) if validated is not None else None,
            "open",
            None,
            _now(),
            None,
        ),
    )
    return item_id


def _apply_concept_merge(conn, payload: dict, action: str) -> None:
    """Idempotent side effects for a ``concept_merge_candidate`` resolution.

    ``approve``: write a sticky ``must_link`` constraint over the two normalized
    labels, fold the member surface into the surviving concept as a
    ``llm_proposed_reviewed`` alias, and mark the surviving concept
    ``user_confirmed`` (the actual re-key fold happens on the next build, driven by
    the constraint — sticky, no flip). ``reject``/``split``: write a sticky
    ``cannot_link`` constraint and mark both concepts ``user_split``.
    """
    canonical_label = payload["canonical_label"]
    member_label = payload["member_label"]
    canonical_id = payload["canonical_concept_id"]
    member_id = payload["member_concept_id"]
    a, b = sorted((canonical_label, member_label))
    now = _now()
    if action == "approve":
        conn.execute(
            "INSERT OR IGNORE INTO concept_constraints "
            "(kind, label_a, label_b, source, created_at) VALUES "
            "('must_link', ?, ?, 'user', ?)",
            (a, b, now),
        )
        member_surface = payload.get("member_surface") or member_label
        conn.execute(
            "INSERT OR IGNORE INTO concept_aliases "
            "(concept_id, alias_label, fold_reason, epistemic_type, run_id, created_at) "
            "VALUES (?, ?, 'llm_proposed_reviewed', 'user_validated', NULL, ?)",
            (canonical_id, member_surface, now),
        )
        conn.execute(
            "UPDATE concepts SET status='user_confirmed', epistemic_type='user_validated', "
            "updated_at=? WHERE concept_id=?",
            (now, canonical_id),
        )
    else:  # reject | split
        conn.execute(
            "INSERT OR IGNORE INTO concept_constraints "
            "(kind, label_a, label_b, source, created_at) VALUES "
            "('cannot_link', ?, ?, 'user', ?)",
            (a, b, now),
        )
        conn.execute(
            "UPDATE concepts SET status='user_split', updated_at=? WHERE concept_id IN (?, ?)",
            (now, canonical_id, member_id),
        )


def _apply_concept_edge(conn, payload: dict, target_id: str | None, action: str) -> None:
    """Idempotent side effects for a ``concept_edge_candidate`` resolution.

    ``approve``: flip the edge's ``epistemic_type`` to ``user_validated``.
    ``reject``: delete the edge row (the ``UserValidation→Edge`` workflow)."""
    edge_id = (payload or {}).get("edge_id") or target_id
    if edge_id is None:
        return
    if action == "approve":
        conn.execute(
            "UPDATE project_graph_edges SET epistemic_type='user_validated' WHERE edge_id=?",
            (edge_id,),
        )
    else:  # reject
        conn.execute("DELETE FROM project_graph_edges WHERE edge_id=?", (edge_id,))


def _apply_unmatched_upload(session: Session, payload: dict, action: str) -> None:
    """Side effect for an ``unmatched_upload`` resolution (corpus ingest-folder).

    ``approve``: adopt the already-ingested + converted PDF into a work — mint (or
    merge) a work from the extracted identity (``doi``/``arxiv``/``extracted_title``,
    or the filename stem), set it ``included`` / ``user_supplied_private`` with
    ``inclusion_reason='provided_pdf'``, and write the single bridge row reusing the
    payload's cache anchors (``source_file_id`` / ``file_hash`` / ``markdown_id``).
    ``reject``: pure status flip (the ingested artifacts stay in the cache,
    unreferenced). Runs in the SAME transaction as the status flip; the bridge upsert
    (ON ``work_id``) makes a re-approve idempotent."""
    if action != "approve":
        return
    from pathlib import PurePosixPath

    from ..acquisition import bridge as bridge_mod
    from ..db.project_models import ProjectDocument
    from ..vocab import AccessClass
    from . import identity

    source_file_id = payload.get("source_file_id")
    file_hash = payload.get("file_hash")
    if not source_file_id or not file_hash:
        return
    markdown_id = payload.get("markdown_id")
    # markdown_id IS "md_" + markdown_hash (content-addressed; convert.py), so the
    # bridge's markdown_hash is derivable without carrying it on the payload.
    markdown_hash = (
        markdown_id[len("md_"):]
        if markdown_id and markdown_id.startswith("md_")
        else None
    )
    title = payload.get("extracted_title") or (
        PurePosixPath(payload.get("filename") or "").stem or None
    )
    incoming: dict = {}
    if title:
        incoming["title"] = title
    if payload.get("doi"):
        incoming["doi"] = payload["doi"]
    if payload.get("arxiv"):
        incoming["arxiv"] = payload["arxiv"]

    work, _outcome = identity.upsert_work(session, incoming)
    now = _now()
    doc = session.get(ProjectDocument, work.work_id)
    if doc is None:
        session.add(
            ProjectDocument(
                work_id=work.work_id,
                inclusion_status="included",
                inclusion_reason="provided_pdf",
                is_seed=0,
                access_status=AccessClass.user_supplied_private.value,
                created_at=now,
                updated_at=now,
            )
        )
    else:
        doc.inclusion_status = "included"
        doc.access_status = AccessClass.user_supplied_private.value
        doc.updated_at = now
        session.add(doc)
    bridge_mod.write_bridge(
        session,
        work_id=work.work_id,
        source_file_id=source_file_id,
        file_hash=file_hash,
        markdown_id=markdown_id,
        markdown_hash=markdown_hash,
        acquisition_method="manual_upload",
    )


def list_open(h: "ProjectHandle") -> list[ReviewQueueItem]:
    """Return all open (``status='open'``) review items for the project, newest last."""
    with Session(h.engine, expire_on_commit=False) as session:
        items = list(
            session.exec(
                select(ReviewQueueItem)
                .where(ReviewQueueItem.status == "open")
                .order_by(ReviewQueueItem.created_at)
            ).all()
        )
        session.expunge_all()
    return items


def _apply_duplicate_candidate(
    session: Session,
    row: ReviewQueueItem,
    payload: dict,
    action: str,
    *,
    survivor_id: str | None,
) -> None:
    """In-transaction side effects for a ``duplicate_candidate`` resolution.

    ``merge``: collapse the pair via :mod:`seedgraph.project.merge`. Default pair
    (Design 7 / v1 rule): survivor = ``conflicting_work_ids[0]`` (the pre-existing
    canonical work), victim = the item's ``target_id`` (the newcomer);
    ``survivor_id=`` flips it. A STALE pair (either work already collapsed by a
    sibling resolution) is REFUSED with a message and the item stays open — never
    record ``action='merge'`` on a resolve that collapsed nothing.

    ``exclude``: create-or-update the target work's membership row to
    ``excluded`` / ``user_excluded`` in-session (NOT via
    ``service.set_inclusion_status``, which opens its own transaction;
    ``access_status`` untouched per decision 30).

    ``reject``: pure status flip (genuinely distinct works) — handled by the caller.
    """
    from ..db.project_models import ProjectDocument, Work
    from . import merge as merge_mod

    if action == "merge":
        conflicting = list(payload.get("conflicting_work_ids") or [])
        target_id = row.target_id
        if not conflicting or not target_id:
            raise ValidationError(
                f"review item {row.item_id!r} carries no (target, conflicting) work "
                f"pair to merge"
            )
        survivor = survivor_id or conflicting[0]
        if survivor == target_id:
            victim = conflicting[0]
        elif survivor in conflicting:
            victim = target_id
        else:
            raise ValidationError(
                f"--survivor {survivor!r} is neither the item's target work "
                f"{target_id!r} nor one of its conflicting works {conflicting!r}"
            )
        # Stale-pair guard BEFORE any mutation: a sibling resolution may have
        # already collapsed one side. Refuse (item stays open; resolve with
        # `reject` after checking) rather than recording a no-op merge.
        for label, work_id in (("survivor", survivor), ("victim", victim)):
            if session.get(Work, work_id) is None:
                raise ValidationError(
                    f"stale duplicate pair: {label} work {work_id!r} no longer "
                    f"exists (already merged/deleted?) — resolve with 'reject' "
                    f"if the pair is already collapsed"
                )
        merge_mod.merge_works(session, survivor, victim)
        # merge_works retargeted open review rows (incl. this one) via raw SQL;
        # mirror it on the ORM object so the later status flip flushes consistently.
        if row.target_id == victim:
            row.target_id = survivor
    elif action == "exclude":
        target_id = row.target_id
        if not target_id or session.get(Work, target_id) is None:
            raise ValidationError(
                f"review item {row.item_id!r} targets no existing work to exclude"
            )
        now = _now()
        doc = session.get(ProjectDocument, target_id)
        if doc is None:
            # A duplicate_candidate work may have no membership row yet.
            session.add(
                ProjectDocument(
                    work_id=target_id,
                    inclusion_status="excluded",
                    inclusion_reason="user_excluded",
                    is_seed=0,
                    access_status=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            doc.inclusion_status = "excluded"
            doc.inclusion_reason = "user_excluded"
            doc.updated_at = now
            session.add(doc)
        session.flush()


def _apply_citation_resolution(
    session: Session,
    row: ReviewQueueItem,
    payload: dict,
    action: str,
    *,
    candidate_index: int | None,
    target_work_id: str | None,
) -> None:
    """In-transaction side effects for a ``citation_resolution`` resolution.

    ``approve`` / ``re_resolve``: make the human verdict durable TWICE (Design 8) —
    write a ``manual_override`` edge (confidence 1.0) under the payload's
    ``run_id`` (immediate authority within that run — the edges.py ladder needs no
    change) AND update the ``reference_entries`` row
    (``resolution_source='manual_override'``, admitted by migration 0014) so every
    FUTURE Stage-B projection targets the chosen work.

    Target choice: ``target_work_id=`` names an already-known work;
    otherwise ``candidate_index=`` picks from ``payload.candidates`` (defaulting
    to 0 only when exactly one candidate exists — ambiguity is refused, never
    guessed). A not-yet-a-work candidate is upserted ``metadata_only``.

    ``reject``: pure status flip (entry stays ambiguous/suspect, raw text kept —
    recall stays measurable) — handled by the caller.
    """
    if action not in ("approve", "re_resolve"):
        return

    from ..citation.edges import write_edge
    from ..citation.parsed_bib import ensure_metadata_only
    from ..citation.resolve import record_to_identity
    from ..db.adapter import raw_conn
    from ..db.project_models import Work
    from . import identity

    run_id = (payload.get("run_id") or "").strip()
    if not run_id:
        # v1 _require_run_id: an edge-writing resolution needs a run scope.
        raise ValidationError(
            f"review item {row.item_id!r} carries no run_id — refusing to write "
            f"an unscoped manual_override edge"
        )
    citing = payload.get("citing_work_id")
    reference_id = payload.get("reference_id")
    if not citing or not reference_id:
        raise ValidationError(
            f"review item {row.item_id!r} payload lacks citing_work_id/reference_id"
        )

    if target_work_id is not None:
        if session.get(Work, target_work_id) is None:
            raise ValidationError(f"--target-work {target_work_id!r} is not a work")
        chosen = target_work_id
    else:
        candidates = list(payload.get("candidates") or [])
        if not candidates:
            raise ValidationError(
                f"review item {row.item_id!r} has no candidates; pass --target-work"
            )
        idx = candidate_index
        if idx is None:
            if len(candidates) != 1:
                raise ValidationError(
                    f"review item {row.item_id!r} has {len(candidates)} candidates; "
                    f"pass --candidate N (0-based) or --target-work"
                )
            idx = 0
        if not (0 <= idx < len(candidates)):
            raise ValidationError(
                f"--candidate {idx} out of range (item has {len(candidates)} candidates)"
            )
        incoming = record_to_identity(dict(candidates[idx]))
        if not incoming:
            raise ValidationError(
                f"candidate {idx} of review item {row.item_id!r} carries no usable "
                f"identity fields; pass --target-work"
            )
        work, _outcome = identity.upsert_work(session, incoming)
        ensure_metadata_only(session, work.work_id)
        chosen = work.work_id

    conn = raw_conn(session)
    write_edge(
        conn,
        source=citing,
        target=chosen,
        provenance="manual_override",
        confidence=1.0,
        run_id=run_id,
        reference_id=reference_id,
    )
    conn.execute(
        "UPDATE reference_entries SET resolved_work_id = ?, "
        "resolution_status = 'resolved', resolution_source = 'manual_override', "
        "confidence = 1.0 WHERE reference_id = ?",
        (chosen, reference_id),
    )


def resolve(
    h: "ProjectHandle",
    item_id: str,
    action: str,
    *,
    survivor_id: str | None = None,
    candidate_index: int | None = None,
    target_work_id: str | None = None,
) -> None:
    """Resolve a review item: apply the item-type's side effects IN THE SAME
    TRANSACTION as the ``action`` + ``resolved_at`` + ``status`` flip (decision
    80). Idempotent — resolving an already-resolved item is a no-op.

    ``action`` must be a member of :class:`~seedgraph.vocab.ReviewAction`
    (``approve|reject|merge|re_resolve|exclude|split``). The keyword-only knobs
    default to today's behavior for every existing caller:

    * ``survivor_id`` — ``duplicate_candidate`` merge override (default survivor =
      ``payload.conflicting_work_ids[0]``, the pre-existing canonical work).
    * ``candidate_index`` / ``target_work_id`` — ``citation_resolution``
      approve/re_resolve target choice (default: the single candidate; ambiguity
      is refused, never guessed).
    """
    valid = {a.value for a in ReviewAction}
    if action not in valid:
        raise ValidationError(
            f"invalid review action {action!r}: must be one of {sorted(valid)}"
        )
    from ..db.adapter import raw_conn

    with Session(h.engine, expire_on_commit=False) as session:
        row = session.get(ReviewQueueItem, item_id)
        if row is None:
            raise ValidationError(f"no review item {item_id!r}")
        if row.status == "resolved":
            return  # idempotent no-op
        # Side effects run in the SAME transaction as the status flip (decision
        # 80) — the graph mutation and the resolution commit atomically; a raised
        # refusal (stale pair, missing run_id, candidate ambiguity) leaves the
        # item open and nothing written.
        if row.item_type in ("concept_merge_candidate", "concept_edge_candidate"):
            conn = raw_conn(session)
            payload = json.loads(row.payload) if row.payload else {}
            if row.item_type == "concept_merge_candidate":
                _apply_concept_merge(conn, payload, action)
            else:
                _apply_concept_edge(conn, payload, row.target_id, action)
        elif row.item_type == "unmatched_upload":
            # approve -> new work + bridge (ORM side effect, same transaction).
            payload = json.loads(row.payload) if row.payload else {}
            _apply_unmatched_upload(session, payload, action)
        elif row.item_type == "duplicate_candidate":
            payload = json.loads(row.payload) if row.payload else {}
            _apply_duplicate_candidate(
                session, row, payload, action, survivor_id=survivor_id
            )
        elif row.item_type == "citation_resolution":
            payload = json.loads(row.payload) if row.payload else {}
            _apply_citation_resolution(
                session, row, payload, action,
                candidate_index=candidate_index, target_work_id=target_work_id,
            )
        row.status = "resolved"
        row.action = action
        row.resolved_at = _now()
        session.add(row)
        session.commit()


# --- review-list rendering -------------------------------------------------
# Shared row-shaping seam consumed by the `review list` CLI table, the web
# review screen, and the MCP `review_list` tool (a neutral, adapter-free home).

def _truncate(text: str, limit: int = 60) -> str:
    """One-line, sanely-truncated cell text (whitespace collapsed)."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def describe_review_item(
    item_type: str,
    payload: dict,
    target_id: Optional[str],
    *,
    work_labels: Optional[dict] = None,
) -> str:
    """One-line human *decision* for a review row — WHAT is being decided, not
    which opaque ids are involved (the `review list` UX gap from the first live
    semantic run).

    ``concept_merge_candidate``: the merge pair ``member -> canonical``
    (surfaces preferred over normalized merge-key labels).
    ``citation_resolution``: ``first_author (year) — title`` (title truncated;
    identifier-placeholder / raw-text fallbacks follow
    :func:`seedgraph.display.derive_label` conventions), the citing work's
    label when ``work_labels`` carries it, then ``status, N candidate(s)``.
    Every other item type keeps the old ``target_id`` rendering. Shared
    verbatim by the CLI table and the web review screen via :func:`review_rows`.
    """
    if item_type == "concept_merge_candidate":
        member = payload.get("member_surface") or payload.get("member_label") or "?"
        canonical = (
            payload.get("canonical_surface") or payload.get("canonical_label") or "?"
        )
        # ASCII arrow — matches `project set-status` output and stays
        # cp1252-encodable when CLI output is piped on Windows (U+2192 is not).
        return f"{_truncate(member)} -> {_truncate(canonical)}"
    if item_type == "citation_resolution":
        head = str(payload.get("first_author") or "").strip()
        year = payload.get("year")
        if year:
            head = f"{head} ({year})" if head else f"({year})"
        title = str(payload.get("title") or "").strip()
        if not title:
            # derive_label conventions: a visibly-placeholder identifier, never
            # title-shaped; the raw reference text beats a bare "untitled".
            title = derive_label(doi=payload.get("doi"), arxiv_id=payload.get("arxiv"))
            if title == "untitled" and str(payload.get("raw") or "").strip():
                title = str(payload["raw"])
        decision = f"{head} — {_truncate(title)}" if head else _truncate(title)
        citing = (work_labels or {}).get(payload.get("citing_work_id"))
        if citing:
            decision += f" · in: {_truncate(citing, 40)}"
        n = len(payload.get("candidates") or [])
        status = payload.get("status") or "?"
        decision += f" · {status}, {n} candidate{'' if n == 1 else 's'}"
        return decision
    return target_id or ""


# Optional applier knobs `resolve()` reads for some item types (Build A ch7/8):
# rendered only when the item type's menu below actually lists them.
_INPUT_SURVIVOR = {
    "name": "survivor_id", "label": "Survivor work id", "placeholder": "survivor_id (merge)",
}
_INPUT_CANDIDATE = {
    "name": "candidate_index", "label": "Candidate #", "placeholder": "cand #",
}
_INPUT_TARGET = {
    "name": "target_work_id", "label": "Target work id", "placeholder": "target_work_id",
}


def review_action_menu(item_type: str) -> dict:
    """The review UI's per-item-type governance: the human *question* a row is
    asking, which :class:`~seedgraph.vocab.ReviewAction` values actually apply
    (each with a plain-language ``label`` and a one-line ``effect``), and which
    of the three optional applier inputs (``survivor_id``/``candidate_index``/
    ``target_work_id``) are relevant. Consulted by :func:`review_rows` so a row
    offers only the actions/inputs its item type's ``resolve()`` branch reads —
    see the ``_apply_*`` functions in :mod:`seedgraph.project.review`.

    Unknown item types fall back to the full six-action / three-input surface
    (today's behavior) so nothing regresses.
    """
    if item_type == "concept_merge_candidate":
        return {
            "question": "Merge these two concepts into one, or keep them separate?",
            "actions": [
                {
                    "value": "approve",
                    "label": "Approve - merge into one concept",
                    "effect": "folds the member into the canonical concept, sticky",
                },
                {
                    "value": "reject",
                    "label": "Reject - keep separate",
                    "effect": "records the two concepts as permanently distinct",
                },
            ],
            "inputs": [],
        }
    if item_type == "concept_edge_candidate":
        return {
            "question": "Is this relationship between concepts correct?",
            "actions": [
                {
                    "value": "approve",
                    "label": "Approve - keep this edge",
                    "effect": "marks the edge user-validated",
                },
                {
                    "value": "reject",
                    "label": "Reject - remove this edge",
                    "effect": "deletes the edge",
                },
            ],
            "inputs": [],
        }
    if item_type == "citation_resolution":
        return {
            "question": "Resolve this citation to a candidate work?",
            "actions": [
                {
                    "value": "approve",
                    "label": "Approve - resolve to the chosen candidate",
                    "effect": "records the choice as a manual override for this citation",
                },
                {
                    "value": "re_resolve",
                    "label": "Re-resolve - pick a different candidate",
                    "effect": "same as approve; redoes an already-resolved citation",
                },
            ],
            "inputs": [_INPUT_CANDIDATE, _INPUT_TARGET],
        }
    if item_type == "duplicate_candidate":
        return {
            "question": (
                "Is this the same work as an existing one (merge), or a "
                "different work (keep separate)?"
            ),
            "actions": [
                {
                    "value": "merge",
                    "label": "Merge - same work",
                    "effect": "collapses the two works into one (default survivor: the pre-existing work)",
                },
                {
                    "value": "exclude",
                    "label": "Exclude - different work, drop this one",
                    "effect": "marks this work excluded from the project",
                },
            ],
            "inputs": [_INPUT_SURVIVOR],
        }
    if item_type == "unmatched_upload":
        return {
            "question": "Adopt this dropped PDF as a new work in the project?",
            "actions": [
                {
                    "value": "approve",
                    "label": "Approve - adopt as a new work",
                    "effect": "creates (or matches) a work from the PDF and links it in",
                },
                {
                    "value": "reject",
                    "label": "Reject - ignore this upload",
                    "effect": "leaves the file unlinked; no work is created",
                },
            ],
            "inputs": [],
        }
    if item_type == "lens_record":
        return {
            "question": "Accept this lens record?",
            "actions": [
                {"value": "approve", "label": "Approve", "effect": "marks the record accepted"},
                {"value": "reject", "label": "Reject", "effect": "marks the record rejected"},
            ],
            "inputs": [],
        }
    return {
        "question": "",
        "actions": [
            {"value": a.value, "label": a.value, "effect": ""} for a in ReviewAction
        ],
        "inputs": [_INPUT_SURVIVOR, _INPUT_CANDIDATE, _INPUT_TARGET],
    }


def _citing_work_labels(handle, payloads: list[dict]) -> dict:
    """Batch :func:`derive_label` labels for the citing works named by
    ``citation_resolution`` payloads — one IN query; ``{}`` when none apply."""
    ids = sorted(
        {
            p.get("citing_work_id")
            for p in payloads
            if p.get("kind") == "citation_resolution" and p.get("citing_work_id")
        }
    )
    if not ids:
        return {}
    from sqlmodel import Session, select

    from ..db.project_models import Work

    with Session(handle.engine) as session:
        works = session.exec(select(Work).where(Work.work_id.in_(ids))).all()
    return {
        w.work_id: derive_label(
            title=w.canonical_title,
            authors=w.authors,
            year=w.year,
            doi=w.doi,
            openalex_id=w.openalex_id,
            arxiv_id=w.arxiv_id,
            work_id=w.work_id,
        )
        for w in works
    }


def review_rows(handle, items) -> list[dict]:
    """Display dicts for review items — the CLI table rows AND the web review
    screen's template context (one shared rendering seam). ``decision`` is the
    human line from :func:`describe_review_item`; ``menu`` is the item type's
    action/input governance from :func:`review_action_menu` (the web screen's
    scoped resolve form; the CLI table ignores it); ``item_id`` stays available
    because ``review resolve`` takes it."""
    payloads: list[dict] = []
    for item in items:
        try:
            payloads.append(json.loads(item.payload) if item.payload else {})
        except ValueError:  # defensive: one bad payload must not kill the list
            payloads.append({})
    labels = _citing_work_labels(handle, payloads)
    return [
        {
            "item_id": item.item_id,
            "item_type": item.item_type,
            "target_id": item.target_id,
            "status": item.status,
            "decision": describe_review_item(
                item.item_type, payload, item.target_id, work_labels=labels
            ),
            "menu": review_action_menu(item.item_type),
        }
        for item, payload in zip(items, payloads)
    ]
