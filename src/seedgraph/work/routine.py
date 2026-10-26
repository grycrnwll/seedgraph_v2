"""The routine (design §1.1): the protocol's deep-module interface. The MCP tools (P7),
the CLI mirror (P8) and the Ollama worker loop (P6) are thin adapters over these calls:

    create_batch(lib, task, *, paper_ids | query | graph, options, client) -> BatchView
    claim_next(lib, batch_id, worker, *, paper_id)                         -> ClaimReply
    submit(lib, packet_id, claim_id, *, note | note_path, accept_unanchored,
           model, usage)                                                   -> SubmitResult
    release(lib, packet_id, claim_id, reason, *, message)                  -> BatchView
    stop(lib, batch_id, *, mode)                                           -> BatchView
    report_limit(lib, batch_id, worker, *, resets_at, message)             -> BatchView
    continue_batch(lib, batch_id)                                          -> BatchView
    batch_view(lib, batch_id, *, offset, limit)                            -> BatchView
    list_batches(lib, *, active_only)                                      -> list[BatchView]

Errors: a refused ``claim_id`` raises :class:`seedgraph.work.claims.ClaimRefused`
(a ``WorkError`` with ``.code``); any other refusal raises ``WorkError``. Pauses,
``invalid`` and ``needs_revision`` are normal replies, never exceptions.

Files (design §1.2, §1.3) live in ``<home>/work/<batch_id>/``: ``instructions.md`` and
``labels.json`` (once per batch, frozen), ``<packet_id>.md`` (the package text) and
``<packet_id>.result.json`` (where the worker writes its result). Every file is
written as UTF-8 bytes with ``\\n`` line ends, so package text equals the stored
markdown slice byte for byte on Windows too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..library.store import transaction
from ..llm.parse import extract_json_object
from ..notes import NoteConflict, NoteStale
from ..notes.spans import read_markdown
from . import claims, packages, state, store, tasks
from .commands import WorkCommands, work_commands
from .feedback import IN_REFERENCES, diagnose_quote
from .state import BatchState
from .store import WorkError

if TYPE_CHECKING:
    from ..library.store import Library
    from .claims import WorkerRef

MAX_PAPERS = 1000
STALE_REASON = "the paper was re-converted; create a new batch"
CONFLICT_REASON = "the paper's current note changed since the batch was created; create a new batch (replace)"

NEXT_INVALID = "Fix the errors and submit the same packet_id again."
NEXT_REVISION = ("Copy each failed quote's markdown_slice exactly (or set that claim's status to ambiguous) and "
                 "submit the same packet_id again; accept_unanchored=true saves unanchored quotes as ambiguous.")
NEXT_CLAIM = "Claim the next package."
NEXT_CLAIMED = ("Read instructions.md once per batch, then the package text (in ranges if it is long); write "
                "the result to result_path and submit that file (MCP work_submit note_path=result_path, "
                "CLI work submit --note-file result_path).")
NEXT_NO_PACKAGE = ("No package is open right now: other workers hold the rest. Claim again later, "
                   "or stop and report your counts.")


# --- options and views --------------------------------------------------------------------

@dataclass(frozen=True)
class BatchOptions:
    claim_ttl_s: int = store.DEFAULT_CLAIM_TTL_S
    split_threshold_tokens: int = packages.DEFAULT_SPLIT_THRESHOLD_TOKENS
    part_target_tokens: int = packages.DEFAULT_PART_TARGET_TOKENS
    replace: bool = False
    model: str | None = None
    auto_continue_after_limit: bool = False  # stage 6
    stop_after_papers: int | None = None  # stage 6

    @classmethod
    def coerce(cls, options: BatchOptions | Mapping[str, Any] | None) -> BatchOptions:
        if options is None:
            return cls()
        if isinstance(options, BatchOptions):
            return options
        known = {f.name for f in fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise WorkError(f"unknown batch option(s) {sorted(unknown)}; known: {sorted(known)}")
        opts = cls(**dict(options))
        if opts.claim_ttl_s < 1 or opts.split_threshold_tokens < 1 or opts.part_target_tokens < 1:
            raise WorkError("claim_ttl_s, split_threshold_tokens and part_target_tokens must be positive")
        return opts

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ItemView:
    ordinal: int
    subject_kind: str
    subject_id: str
    state: str
    reason: str | None
    part_count: int | None
    parts_done: int
    result_ref: str | None
    attention_item_id: str | None
    #: Claim history, oldest first: ``{worker, client, claimed_at, ended}`` (+ ``part`` for a
    #: split paper). ``ended``: ``live`` while held, else ``submitted`` / ``expired`` /
    #: ``released`` / ``stopped`` / ``limit``.
    claims: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchView:
    batch_id: str
    task_type: str
    task_version: str
    selection_query: str | None
    graph: str | None
    options: dict[str, Any]
    state: BatchState
    items: list[ItemView]
    item_total: int
    commands: WorkCommands
    accepted_unanchored: int
    work_dir: str

    def as_dict(self) -> dict[str, Any]:
        st = self.state
        return {
            "batch_id": self.batch_id, "task_type": self.task_type, "task_version": self.task_version,
            "selection_query": self.selection_query, "graph": self.graph, "options": self.options,
            "state": st.state, "label": st.label, "next_action": st.next_action, "stop_mode": st.stop_mode,
            "resets_at": st.resets_at,
            "pause_detail": st.pause_detail, "live_claims": st.live_claims, "workers": list(st.workers),
            "packages": st.packages, "item_counts": st.items, "item_total": self.item_total,
            "items": [i.as_dict() for i in self.items], "commands": self.commands.as_dict(),
            "accepted_unanchored": self.accepted_unanchored, "work_dir": self.work_dir,
        }


@dataclass(frozen=True)
class ClaimReply:
    """``status``: ``claimed`` (``message`` is the package message), ``paused``,
    ``no_package``, ``finished`` or ``not_allowed`` (the worker kind may not take this
    task type; no claim was written)."""

    status: str
    state: BatchState
    message: dict[str, Any] | None
    next_action: str

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "state": self.state.state, "label": self.state.label,
                               "next_action": self.next_action}
        if self.status == "paused":
            out["resets_at"] = self.state.resets_at
        if self.message is not None:
            out["package"] = self.message
        return out


@dataclass
class SubmitResult:
    status: str  # 'accepted' | 'needs_revision' | 'invalid'
    packet_id: str
    claim_id: str
    part: int
    part_count: int
    paper_status: str  # 'in_progress' | 'done' | 'stale' | 'conflict'
    parts_done: int
    claim_expires_at: str | None
    next_action: str
    errors: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    failed_quotes: list[dict[str, Any]] = field(default_factory=list)
    claims_kept: int = 0
    quotes_found: int = 0
    quotes_anchored: int = 0
    accepted_unanchored: int = 0
    note_id: str | None = None  # the item's result_ref once the paper is done
    submission_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["parts_done_label"] = f"{self.parts_done}/{self.part_count}"
        return out


# --- paths --------------------------------------------------------------------------------

def work_root(lib: Library) -> Path:
    """``<home>/work`` (a sibling of the library folder, like ``graphs``)."""
    return Path(lib.root).parent / "work"


def _home(lib: Library) -> Path:
    return Path(lib.root).parent


def _write_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# --- create -------------------------------------------------------------------------------

def _select(lib: Library, task: tasks.TaskType, paper_ids: Sequence[str] | None, query: str | None,
            graph: str | None) -> tuple[list[str], str | None, str | None]:
    if paper_ids is not None and query is not None:
        raise WorkError("give paper_ids or a query, not both")
    if paper_ids is not None:
        ids = [str(p) for p in paper_ids]
        if len(set(ids)) != len(ids):
            raise WorkError("paper_ids must be distinct")
        return ids, None, graph
    if query is not None:
        sel = tasks.parse_query(query)
        if graph and sel.graph and graph != sel.graph:
            raise WorkError(f"the query names graph {sel.graph!r} but graph={graph!r} was given")
        sel = replace(sel, graph=sel.graph or graph)
        return task.resolve(lib, sel), sel.text, sel.graph
    if graph is not None:
        sel = tasks.Selection(text=f"held in graph {graph}", graph=graph)
        return task.resolve(lib, sel), sel.text, graph
    raise WorkError("a batch needs paper_ids, a query or a graph")


def create_batch(lib: Library, task: str, *, paper_ids: Sequence[str] | None = None, query: str | None = None,
                 graph: str | None = None, options: BatchOptions | Mapping[str, Any] | None = None,
                 client: str | None = None, now: datetime | None = None) -> BatchView:
    """Create a batch: resolve and freeze the selection, decide each subject's
    eligibility (gate and title hold-back), write ``instructions.md``, ``labels.json``
    and one text file per package, and insert the batch, items and packages."""
    tt = tasks.get_task(lib, task)
    opts = BatchOptions.coerce(options)
    ids, selection_query, graph_slug = _select(lib, tt, paper_ids, query, graph)
    if not ids:
        raise WorkError("the selection names no papers")
    if len(ids) > MAX_PAPERS:
        raise WorkError(f"a batch takes at most {MAX_PAPERS} papers, not {len(ids)}")
    option_map = opts.as_dict()
    decided = [(sid, tt.eligibility(lib, sid, option_map)) for sid in ids]
    plans = {sid: tt.plan(lib, sid, option_map) for sid, el in decided if el.state == "queued"}
    instr = tt.instructions(lib, has_parts=any(len(p) > 1 for p in plans.values()))

    batch_id = store.new_batch_id(now)
    folder = work_root(lib) / batch_id
    folder.mkdir(parents=True, exist_ok=False)
    instructions_sha = _write_text(folder / "instructions.md", instr.instructions_md)
    labels_sha = _write_text(folder / "labels.json", instr.labels_json)
    with lib.write() as conn:
        store.insert_batch(conn, task_type=tt.name, task_version=tt.version, selection=ids,
                           selection_query=selection_query, graph_slug=graph_slug, options=option_map,
                           instructions_sha256=instructions_sha, labels_sha256=labels_sha,
                           created_by_client=client, now=now, batch_id=batch_id)
        for ordinal, (sid, el) in enumerate(decided, start=1):
            parts = plans.get(sid, [])
            store.insert_item(conn, batch_id, ordinal, sid, subject_kind=tt.subject_kind, state=el.state,
                              markdown_id=el.markdown_id, markdown_hash=el.markdown_hash,
                              expected_current_note_id=el.expected_current_note_id,
                              part_count=len(parts) or None, reason=el.reason,
                              attention_item_id=el.attention_item_id, now=now)
            for plan in parts:
                packet_id = store.packet_id_for(
                    batch_id=batch_id, subject_kind=tt.subject_kind, subject_id=sid,
                    markdown_hash=el.markdown_hash, part=plan.part, part_count=plan.part_count,
                    start_char=plan.start_char, end_char=plan.end_char, task_version=tt.version, attempt=1)
                _write_text(folder / f"{packet_id}.md", plan.text)
                store.insert_package(conn, batch_id=batch_id, ordinal=ordinal, part=plan.part,
                                     part_count=plan.part_count, task_version=tt.version,
                                     markdown_hash=el.markdown_hash, start_char=plan.start_char,
                                     end_char=plan.end_char, est_tokens=plan.est_tokens,
                                     text_path=f"work/{batch_id}/{packet_id}.md", now=now)
        store.append_event(conn, batch_id, "items_decided", detail={
            s: sum(1 for _, el in decided if el.state == s)
            for s in ("queued", "preserved", "held_back", "failed")}, now=now)
    return batch_view(lib, batch_id, now=now)


# --- claim --------------------------------------------------------------------------------

def _package_message(lib: Library, batch: store.Batch, package: store.WorkPackage,
                     claim: store.PackageClaim) -> dict[str, Any]:
    item = store.get_item(lib.conn, batch.batch_id, package.ordinal)
    assert item is not None
    paper = lib.conn.execute("SELECT title, year FROM papers WHERE paper_id = ?", (item.subject_id,)).fetchone()
    folder = work_root(lib) / batch.batch_id
    text_file = _home(lib) / package.text_path if package.text_path else None
    try:
        lines = len(text_file.read_bytes().decode("utf-8").splitlines()) if text_file else None
    except (OSError, UnicodeDecodeError):
        lines = None
    return {
        "packet_id": package.packet_id, "claim_id": claim.claim_id, "batch_id": batch.batch_id,
        "task_type": batch.task_type, "paper_id": item.subject_id,
        "title": paper[0] if paper else None, "year": paper[1] if paper else None,
        "part": package.part, "part_count": package.part_count,
        "text_path": str(text_file) if text_file else None, "lines": lines,
        "instructions_path": str(folder / "instructions.md"), "labels_path": str(folder / "labels.json"),
        "result_path": str(folder / f"{package.packet_id}.result.json"),
        "est_tokens": package.est_tokens, "claim_expires_at": claim.expires_at,
    }


def claim_next(lib: Library, batch_id: str, worker: WorkerRef, *, paper_id: str | None = None,
               now: datetime | None = None) -> ClaimReply:
    """Claim the next package for ``worker`` (design §1.4). A worker kind the task type
    does not allow (e.g. Ollama on a notes batch) gets ``not_allowed`` and no claim."""
    batch = store.require_batch(lib.conn, batch_id)
    tt = tasks.task_for_batch(lib, batch.task_type, batch.task_version)
    if worker.worker_kind not in tt.allowed_workers:
        st = state.batch_state(lib, batch_id, now=now)
        return ClaimReply("not_allowed", st, None,
                          f"{batch.task_type} packages are not offered to {worker.worker_kind} workers; "
                          f"allowed: {', '.join(sorted(tt.allowed_workers))}.")
    res = claims.claim_package(lib, batch_id, worker, paper_id=paper_id, now=now)
    if res.status != "claimed":
        nxt = NEXT_NO_PACKAGE if res.status == "no_package" else res.state.next_action
        return ClaimReply(res.status, res.state, None, nxt)
    assert res.package is not None and res.claim is not None
    msg = _package_message(lib, batch, res.package, res.claim)
    return ClaimReply("claimed", res.state, msg, NEXT_CLAIMED)


# --- submit -------------------------------------------------------------------------------

def _load_payload(note: dict | str | None, note_path: Path | str | None) -> tuple[dict | None, list[str]]:
    if (note is None) == (note_path is None):
        return None, ["give the note inline (note) or as a file (note_path), not both or neither"]
    raw: Any = note
    if note_path is not None:
        path = Path(note_path)
        if path.suffix.lower() != ".json":
            return None, [f"note_path must be a .json file, not {path.name!r}"]
        try:
            raw = path.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            return None, [f"cannot read {path}: {exc}"]
    if isinstance(raw, dict):
        return raw, []
    if isinstance(raw, str):
        candidate = extract_json_object(raw)
        if candidate is None:
            return None, ["no JSON object found"]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, [f"JSON decode error: {exc}"]
        if isinstance(data, dict):
            return data, []
    return None, [f"the note must be a JSON object, not {type(raw).__name__}"]


def _parts_done(conn, batch_id: str, ordinal: int) -> tuple[int, int]:
    pkgs = [p for p in store.list_packages(conn, batch_id, ordinal=ordinal) if p.state != "void"]
    return sum(1 for p in pkgs if p.state == "saved"), len(pkgs)


def submit(lib: Library, packet_id: str, claim_id: str, *, note: dict | str | None = None,
           note_path: Path | str | None = None, accept_unanchored: bool = False, model: str | None = None,
           usage: Mapping[str, Any] | None = None, now: datetime | None = None) -> SubmitResult:
    """Validate -> task checks -> anchor preview -> ``invalid`` / ``needs_revision``
    (claim kept and renewed) or saved; the paper's note is assembled and saved when its
    last package is saved (design §1.5)."""
    package = store.get_package(lib.conn, packet_id)
    if package is None:
        raise WorkError(f"no package {packet_id!r}")
    claim = claims.renew_claim(lib, packet_id, claim_id, now=now)  # raises ClaimRefused
    batch = store.require_batch(lib.conn, package.batch_id)
    item = store.get_item(lib.conn, batch.batch_id, package.ordinal)
    assert item is not None
    tt = tasks.task_for_batch(lib, batch.task_type, batch.task_version)
    done, total = _parts_done(lib.conn, batch.batch_id, package.ordinal)
    result = SubmitResult(status="invalid", packet_id=packet_id, claim_id=claim_id, part=package.part,
                          part_count=package.part_count, paper_status="in_progress", parts_done=done,
                          claim_expires_at=claim.expires_at, next_action=NEXT_INVALID)
    usage_dict = dict(usage) if usage else None
    submit_model = model or claim.model

    payload, load_errors = _load_payload(note, note_path)
    if payload is None:
        result.errors = load_errors
        raw_json = json.dumps(note if isinstance(note, str) else None)
        return _record(lib, result, claim.worker_id, batch.batch_id, raw_json, usage_dict, submit_model, now)
    payload_json = json.dumps(payload, ensure_ascii=False)
    check = tt.check(lib, package, payload)
    result.warnings = list(check.warnings)
    result.claims_kept = check.claims_kept
    if check.errors:
        result.errors = list(check.errors)
        return _record(lib, result, claim.worker_id, batch.batch_id, payload_json, usage_dict, submit_model, now)

    if check.quotes:
        text = read_markdown(lib.conn, item.markdown_id or "")
        if text is None:
            result.errors = ["the paper's stored markdown is missing; release the package with paper_problem"]
            return _record(lib, result, claim.worker_id, batch.batch_id, payload_json, usage_dict, submit_model, now)
        ranges = packages.reference_ranges(lib.conn, item.markdown_id or "")
        for q in check.quotes:
            d = diagnose_quote(q.quote, text, reference_ranges=ranges)
            if d.anchored:
                result.quotes_anchored += 1
                if d.reason == IN_REFERENCES:
                    result.warnings.append({"field_key": q.field_key, "reason": IN_REFERENCES, "message": d.hint})
            else:
                result.failed_quotes.append({"field_key": q.field_key, "reason": d.reason, "hint": d.hint,
                                             "markdown_slice": d.markdown_slice})
    result.quotes_found = len(check.quotes)
    if result.failed_quotes and not accept_unanchored:
        result.status = "needs_revision"
        result.next_action = NEXT_REVISION
        return _record(lib, result, claim.worker_id, batch.batch_id, payload_json, usage_dict, submit_model, now)

    result.status = "accepted"
    result.accepted_unanchored = len(result.failed_quotes)
    result.next_action = NEXT_CLAIM
    with lib.write() as conn:
        result.submission_id = store.record_submission(
            conn, packet_id=packet_id, claim_id=claim_id, payload_json=payload_json, outcome="accepted",
            feedback=_feedback(result), usage=usage_dict, model=submit_model, now=now)
        claims.complete_claim(lib, packet_id, claim_id, now=now)
        store.append_event(conn, batch.batch_id, "submitted", packet_id=packet_id, worker_id=claim.worker_id,
                           detail={"outcome": "accepted", "accepted_unanchored": result.accepted_unanchored},
                           now=now)
        result.parts_done, total = _parts_done(conn, batch.batch_id, package.ordinal)
        result.next_action = _after_accept(state.derive_state(conn, batch.batch_id))
        if result.parts_done == total:
            _complete_item(lib, conn, batch, store.get_item(conn, batch.batch_id, package.ordinal), tt, result, now)
    return result


def _after_accept(st: BatchState) -> str:
    """What to do after a saved package: the batch state's next_action (claim again,
    wait while others hold every package, or stop and report when paused or finished)."""
    return st.next_action


def _feedback(result: SubmitResult) -> dict[str, Any]:
    return {"errors": result.errors, "warnings": result.warnings, "failed_quotes": result.failed_quotes,
            "claims_kept": result.claims_kept, "quotes_found": result.quotes_found,
            "quotes_anchored": result.quotes_anchored, "accepted_unanchored": result.accepted_unanchored}


def _record(lib: Library, result: SubmitResult, worker_id: str, batch_id: str, payload_json: str,
            usage: dict | None, model: str | None, now: datetime | None) -> SubmitResult:
    with lib.write() as conn:
        result.submission_id = store.record_submission(
            conn, packet_id=result.packet_id, claim_id=result.claim_id, payload_json=payload_json,
            outcome=result.status, feedback=_feedback(result), usage=usage, model=model, now=now)
        store.append_event(conn, batch_id, "submitted", packet_id=result.packet_id, worker_id=worker_id,
                           detail={"outcome": result.status, "failed_quotes": len(result.failed_quotes),
                                   "errors": len(result.errors)}, now=now)
    return result


def _accepted_parts(conn, batch_id: str, ordinal: int) -> list[tasks.AcceptedPart]:
    parts: list[tasks.AcceptedPart] = []
    for pkg in store.list_packages(conn, batch_id, ordinal=ordinal):
        if pkg.state != "saved":
            continue
        sub = store.latest_accepted_submission(conn, pkg.packet_id)
        assert sub is not None
        claim = store.get_claim(conn, sub.claim_id)
        parts.append(tasks.AcceptedPart(
            part=pkg.part, part_count=pkg.part_count, payload=json.loads(sub.payload_json),
            worker_kind=claim.worker_kind if claim else "interactive", client=claim.client if claim else None,
            model=sub.model, usage=json.loads(sub.usage_json) if sub.usage_json else None))
    return parts


def _complete_item(lib: Library, conn, batch: store.Batch, item: store.BatchItem | None, tt: tasks.TaskType,
                   result: SubmitResult, now: datetime | None) -> None:
    """Every package of the item is saved: check staleness, then assemble and save
    through the task in a savepoint, so a refused save leaves the submissions saved
    and the item ``stale`` / ``conflict``."""
    assert item is not None
    current = packages.primary_markdown(conn, item.subject_id) if item.subject_kind == "paper" else None
    outcome, reason, ref = "done", None, None
    if item.subject_kind == "paper" and current != (item.markdown_id, item.markdown_hash):
        outcome, reason = "stale", STALE_REASON
    else:
        text = read_markdown(conn, item.markdown_id or "") or ""
        ctx = tasks.SaveContext(lib=lib, batch_id=batch.batch_id, markdown_text=text, options=batch.options)
        try:
            with transaction(conn):
                saved = tt.save(conn, item, _accepted_parts(conn, batch.batch_id, item.ordinal), ctx)
        except NoteStale:
            outcome, reason = "stale", STALE_REASON
        except NoteConflict:
            outcome, reason = "conflict", CONFLICT_REASON
        else:
            ref = saved.result_ref
    store.update_item(conn, batch.batch_id, item.ordinal, state=outcome, reason=reason, result_ref=ref, now=now)
    store.append_event(conn, batch.batch_id, f"paper_{outcome}", detail={"subject_id": item.subject_id,
                                                                         "result_ref": ref}, now=now)
    result.paper_status = outcome
    result.note_id = ref
    if outcome != "done":
        result.next_action = f"{reason}. {result.next_action}"


# --- release, pauses, views ---------------------------------------------------------------

RELEASE_REASONS = ("give_back", "paper_problem")


def release(lib: Library, packet_id: str, claim_id: str, reason: str, *, message: str | None = None,
            now: datetime | None = None) -> BatchView:
    """Hand a package back early. ``give_back`` returns it to the pool; ``paper_problem``
    fails the paper with the worker's message and retires all its packages."""
    if reason not in RELEASE_REASONS:
        raise WorkError(f"release reason must be one of {RELEASE_REASONS}, not {reason!r}")
    package = store.get_package(lib.conn, packet_id)
    if package is None:
        raise WorkError(f"no package {packet_id!r}")
    if reason == "give_back":
        claims.release_claim(lib, packet_id, claim_id, package_state="open", note=message, now=now)
        return batch_view(lib, package.batch_id, now=now)
    with lib.write() as conn:
        claims.release_claim(lib, packet_id, claim_id, package_state="void", note=message, now=now)
        ts = store.fmt_ts(now or store.now_utc())
        for pkg in store.list_packages(conn, package.batch_id, ordinal=package.ordinal):
            if pkg.state in ("open", "claimed"):
                conn.execute("UPDATE package_claims SET released_at = ?, release_reason = 'released' "
                             "WHERE packet_id = ? AND released_at IS NULL", (ts, pkg.packet_id))
                store.set_package_state(conn, pkg.packet_id, "void")
        store.update_item(conn, package.batch_id, package.ordinal, state="failed",
                          reason=f"paper_problem: {message or 'reported by the worker'}", now=now)
        store.append_event(conn, package.batch_id, "paper_failed", packet_id=packet_id,
                           detail={"message": message}, now=now)
    return batch_view(lib, package.batch_id, now=now)


def stop(lib: Library, batch_id: str, *, mode: str, now: datetime | None = None) -> BatchView:
    state.stop(lib, batch_id, mode=mode, now=now)  # type: ignore[arg-type]
    return batch_view(lib, batch_id, now=now)


def report_limit(lib: Library, batch_id: str, worker: WorkerRef, *, resets_at: str | None = None,
                 message: str | None = None, now: datetime | None = None) -> BatchView:
    state.report_limit(lib, batch_id, worker, resets_at=resets_at, message=message, now=now)
    return batch_view(lib, batch_id, now=now)


def continue_batch(lib: Library, batch_id: str, *, now: datetime | None = None) -> BatchView:
    state.continue_batch(lib, batch_id, now=now)
    return batch_view(lib, batch_id, now=now)


def _accepted_unanchored(conn, batch_id: str) -> int:
    total = 0
    for (fb,) in conn.execute(
            "SELECT s.feedback_json FROM package_submissions s JOIN work_packages p ON p.packet_id = s.packet_id "
            "WHERE p.batch_id = ? AND s.outcome = 'accepted' ORDER BY s.packet_id, s.submission_no", (batch_id,)):
        if fb:
            total += int(json.loads(fb).get("accepted_unanchored") or 0)
    return total


def _claim_history(conn, batch_id: str) -> dict[int, list[dict[str, Any]]]:
    """``{ordinal: [claim row, ...]}`` in claim order (see :attr:`ItemView.claims`)."""
    rows = conn.execute(
        "SELECT p.ordinal, p.part, p.part_count, c.worker_id, c.client, c.claimed_at, c.released_at, "
        "c.release_reason FROM package_claims c JOIN work_packages p ON p.packet_id = c.packet_id "
        "WHERE p.batch_id = ? ORDER BY p.ordinal, c.claimed_at, p.part, c.claim_no", (batch_id,))
    out: dict[int, list[dict[str, Any]]] = {}
    for ordinal, part, part_count, worker, client, claimed_at, released_at, reason in rows:
        row: dict[str, Any] = {"worker": worker, "client": client, "claimed_at": claimed_at,
                               "ended": "live" if released_at is None else reason}
        if part_count > 1:
            row["part"] = part
        out.setdefault(ordinal, []).append(row)
    return out


def batch_view(lib: Library, batch_id: str, *, offset: int = 0, limit: int | None = None,
               now: datetime | None = None) -> BatchView:
    """Read-only status (after lazy expiry); ``offset``/``limit`` page the items."""
    st = state.batch_state(lib, batch_id, now=now)
    conn = lib.conn
    batch = store.require_batch(conn, batch_id)
    all_items = store.list_items(conn, batch_id)
    saved = {r[0]: r[1] for r in conn.execute(
        "SELECT ordinal, COUNT(*) FROM work_packages WHERE batch_id = ? AND state = 'saved' GROUP BY ordinal",
        (batch_id,))}
    page = all_items[offset:] if limit is None else all_items[offset:offset + limit]
    history = _claim_history(conn, batch_id) if page else {}
    items = [ItemView(i.ordinal, i.subject_kind, i.subject_id, i.state, i.reason, i.part_count,
                      saved.get(i.ordinal, 0), i.result_ref, i.attention_item_id,
                      history.get(i.ordinal, [])) for i in page]
    return BatchView(batch_id=batch_id, task_type=batch.task_type, task_version=batch.task_version,
                     selection_query=batch.selection_query, graph=batch.graph_slug, options=batch.options,
                     state=st, items=items, item_total=len(all_items), commands=work_commands(batch_id),
                     accepted_unanchored=_accepted_unanchored(conn, batch_id),
                     work_dir=str(work_root(lib) / batch_id))


def list_batches(lib: Library, *, active_only: bool = False, now: datetime | None = None) -> list[BatchView]:
    """Every batch (oldest first) without its items; ``active_only`` drops finished ones."""
    views = [batch_view(lib, b.batch_id, limit=0, now=now) for b in store.list_batches(lib.conn)]
    return [v for v in views if not active_only or v.state.state != "finished"]
