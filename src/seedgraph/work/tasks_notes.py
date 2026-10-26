"""The ``notes`` task (design §1.3, §2.2, §3, §4): whole paper minus references, parts
above the threshold, ``default_research_note_v1`` at schema 2.0.0.

Part ownership: part 1 owns the four scalars; every part contributes array elements.
A later part that sends a scalar gets the warning "scalars belong to part 1" and the
scalar is removed before checking, so it can neither block acceptance nor need a
quote. The note is assembled only when every part is saved (``normalize_parts`` ->
one ``save_note``), so it becomes current only when the whole paper is complete.

**Freshness** (one definition, used by eligibility and the selection language): a
current note is *fresh* when its ``markdown_hash`` equals the paper's current
markdown hash, whatever its schema version (a re-note of a fresh note needs
``replace``; design §10.1 step 2 re-notes v2 notes with ``replace``). "Current" here
is the latest extraction (``note_latest``): freshness and the save conflict check are
about extractions, so a researcher-edited paper (whose ``note_current`` is pinned to
its overlay base, stage 3 §4.1) is judged on its extractions alone.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .. import notes
from ..notes.format import ARRAY_FIELDS, SCALAR_FIELDS, NoteV2
from ..vocab import StatusValue
from . import packages
from .tasks import (
    AcceptedPart,
    CheckResult,
    Eligibility,
    Instructions,
    PackagePlan,
    QuoteRef,
    SaveContext,
    SaveResult,
    Selection,
    resolve_selection,
)

if TYPE_CHECKING:
    from ..library.store import Library
    from .store import BatchItem, WorkPackage

SCALARS_BELONG_TO_PART_1 = "scalars belong to part 1; this part's {field} was dropped"
_FOUND = StatusValue.found.value


def current_note(conn: sqlite3.Connection, paper_id: str) -> tuple[str, str] | None:
    """``(note_id, markdown_hash)`` of the paper's latest extraction (``note_latest``), or ``None``."""
    row = conn.execute("SELECT note_id, markdown_hash FROM note_latest WHERE paper_id = ? AND schema_id IS ?",
                       (paper_id, notes.SCHEMA_ID)).fetchone()
    return (row[0], row[1]) if row else None


def note_freshness(conn: sqlite3.Connection, paper_id: str) -> str:
    """``'none'`` / ``'fresh'`` / ``'stale'`` (see the module doc)."""
    cur = current_note(conn, paper_id)
    if cur is None:
        return "none"
    md = packages.primary_markdown(conn, paper_id)
    return "fresh" if md is not None and md[1] == cur[1] else "stale"


class WholePaperTask:
    """Shared by notes and lens: whole-paper packages, the gate and title hold-back."""

    subject_kind = "paper"
    allowed_workers = frozenset({"interactive", "background"})

    def __init__(self, lib: Library) -> None:
        self.lib = lib

    def resolve(self, lib: Library, query: Selection) -> list[str]:
        return resolve_selection(lib, query)

    def is_preserved(self, conn: sqlite3.Connection, paper_id: str) -> bool:
        raise NotImplementedError

    def expected_current(self, conn: sqlite3.Connection, paper_id: str) -> str | None:
        return None

    def eligibility(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> Eligibility:
        if not lib.conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (subject_id,)).fetchone():
            return Eligibility("failed", reason="not a library paper")
        paper = packages.load_paper_text(lib.conn, subject_id)
        if paper is None:
            return Eligibility("failed", reason="no converted markdown (or its file is missing)")
        binding = dict(markdown_id=paper.markdown_id, markdown_hash=paper.markdown_hash,
                       expected_current_note_id=self.expected_current(lib.conn, subject_id))
        if not options.get("replace") and self.is_preserved(lib.conn, subject_id):
            return Eligibility("preserved", reason="a fresh current result exists (use replace to redo it)",
                               **binding)
        held = packages.hold_back_reason(lib, paper)
        if held is not None:
            return Eligibility("held_back", reason=held.reason, attention_item_id=held.attention_item_id,
                               **binding)
        return Eligibility("queued", **binding)

    def plan(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> list[PackagePlan]:
        paper = packages.load_paper_text(lib.conn, subject_id)
        assert paper is not None
        parts = packages.plan_parts(
            paper.text, paper.reference_ranges, paper.section_starts,
            split_threshold_tokens=int(options.get("split_threshold_tokens") or packages.DEFAULT_SPLIT_THRESHOLD_TOKENS),
            part_target_tokens=int(options.get("part_target_tokens") or packages.DEFAULT_PART_TARGET_TOKENS))
        return [PackagePlan(p.part, p.part_count, p.start_char, p.end_char, p.text, p.est_tokens) for p in parts]


class NotesTask(WholePaperTask):
    name = "notes"
    version = notes.SCHEMA_VERSION

    def is_preserved(self, conn: sqlite3.Connection, paper_id: str) -> bool:
        return note_freshness(conn, paper_id) == "fresh"

    def expected_current(self, conn: sqlite3.Connection, paper_id: str) -> str | None:
        cur = current_note(conn, paper_id)
        return cur[0] if cur else None

    def instructions(self, lib: Library, *, has_parts: bool) -> Instructions:
        return Instructions(notes.render_instructions(has_parts=has_parts), notes.labels_json(lib.conn))

    def check(self, lib: Library, package: WorkPackage, payload: dict[str, Any]) -> CheckResult:
        result = CheckResult()
        note, errors = notes.validate_note(payload)
        if note is None:
            result.errors = errors
            return result
        if package.part > 1:
            note = _drop_scalars(note, result)
        checked = notes.check_note(note)
        result.errors = [f"{i.path}: {i.message}" for i in checked.errors]
        result.warnings += [{"field_key": i.path, "message": i.message} for i in checked.warnings]
        if result.errors:
            return result
        # claims_kept = the claim objects the worker sent and Seedgraph keeps: non-null
        # scalars (part 1 only; later parts' scalars were dropped above) plus array
        # elements. The not_found rows stored for null scalars / empty arrays don't count.
        scalar_names = [name for name, _ in SCALAR_FIELDS] if package.part == 1 else []
        result.claims_kept = (sum(1 for name in scalar_names if getattr(note, name) is not None)
                              + sum(len(getattr(note, name)) for name, _ in ARRAY_FIELDS))
        for name in scalar_names:
            env = getattr(note, name)
            if env is not None and _status(env.status) == _FOUND:
                result.quotes.append(QuoteRef(name, env.exact_quote))
        for name, _ in ARRAY_FIELDS:
            for i, env in enumerate(getattr(note, name)):
                if _status(env.status) == _FOUND:
                    result.quotes.append(QuoteRef(f"{name}[{i}]", env.exact_quote))
        return result

    def save(self, conn: sqlite3.Connection, item: BatchItem, parts: Sequence[AcceptedPart],
             ctx: SaveContext) -> SaveResult:
        models: list[NoteV2] = []
        for p in sorted(parts, key=lambda x: x.part):
            note, errors = notes.validate_note(p.payload)
            if note is None:  # cannot happen: it was validated at submit
                raise notes.NoteSaveError(f"part {p.part} no longer validates: {errors}")
            models.append(note)
        normalized = notes.normalize_parts(models)
        last = max(parts, key=lambda x: x.part)
        provenance = notes.NoteProvenance(
            client=last.client or last.worker_kind, model=last.model or ctx.options.get("model"),
            input_tokens=_usage_sum(parts, "input_tokens"), output_tokens=_usage_sum(parts, "output_tokens"))
        raw = json.dumps([p.payload for p in sorted(parts, key=lambda x: x.part)] if len(parts) > 1
                         else parts[0].payload, ensure_ascii=False)
        saved = notes.save_note(
            conn, paper_id=item.subject_id, markdown_id=item.markdown_id or "", markdown_hash=item.markdown_hash or "",
            drafts=normalized.drafts, note_text=normalized.note_text, archetype=normalized.archetype,
            raw_note_json=raw, provenance=provenance, expected_current=item.expected_current_note_id,
            extraction_mode="agent_parts" if len(parts) > 1 else "agent_whole", part_count=len(parts),
            batch_id=ctx.batch_id)
        return SaveResult(saved.note_id, {"version_no": saved.version_no, "claim_count": saved.claim_count,
                                          "quotes_found": saved.quotes_found,
                                          "quotes_anchored": saved.quotes_anchored,
                                          "downgraded": saved.downgraded})


def _status(value: str | None) -> str:
    try:
        return StatusValue(value).value
    except (ValueError, TypeError):
        return _FOUND


def _drop_scalars(note: NoteV2, result: CheckResult) -> NoteV2:
    updates = {}
    for name, _ in SCALAR_FIELDS:
        if getattr(note, name) is not None:
            result.warnings.append({"field_key": name, "message": SCALARS_BELONG_TO_PART_1.format(field=name)})
            updates[name] = None
    return note.model_copy(update=updates) if updates else note


def _usage_sum(parts: Sequence[AcceptedPart], key: str) -> int | None:
    values = [int(p.usage[key]) for p in parts if p.usage and isinstance(p.usage.get(key), (int, float))]
    return sum(values) if values else None
