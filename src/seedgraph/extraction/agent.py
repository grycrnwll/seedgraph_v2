"""Host-driven extraction: frozen packets, local validation, resumable activation.

No model dispatch or provider credentials live here. The signed-in host supplies
JSON; only the existing note writer can create searchable interpretations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace as replace_dataclass
from pathlib import Path
from contextlib import closing

from sqlmodel import Session

from .. import cache_access, run
from ..acquisition.bridge import resolve_work_source
from ..db.adapter import raw_conn
from ..config.loader import load_project_config
from ..errors import ConfigError, SeedgraphError
from ..llm.routing import assert_external_content_allowed
from ..llm.tokens import estimate_tokens
from ..vocab import AccessClass, is_shareable
from ..sections.store import load_sections
from .chunker import plan_chunks
from .normalize import normalize_note
from .prompt import build_prompt
from .reduce import merge_chunk_drafts
from .runner import NoteConflict, latest_note_id, save_normalized_note
from .schema import DefaultNoteV1, PROMPT_VERSION, SCHEMA_ID, SCHEMA_VERSION
from .validator import validate_note

_SESSION_FAILURES = {"authentication", "quota", "batch_access"}
_PAPER_FAILURES = {"extraction_failed", "invalid_response", "source_unavailable"}
_MAX_RESULT_BYTES = 4_000_000


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def _project_key(handle) -> str:
    return _hash(str(Path(handle.db_path).resolve()).casefold())


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(handle, batch_id, directory, state, event="awaiting_host"):
    run._atomic_write_json(directory / "state.json", state)
    run.append_event(handle.slug, batch_id, phase="agent_extraction", event=event,
                     data={"paused": state["paused"],
                           "papers": [p["status"] for p in state["papers"]]}, root=handle.root)


def _load(handle, batch_id):
    manifest = run.read_manifest(handle.slug, batch_id, root=handle.root)
    plan = manifest.sections.get("agent_extraction")
    if not plan or plan.get("project_key") != _project_key(handle):
        raise SeedgraphError("batch does not belong to this project")
    directory = run._run_dir(handle.slug, batch_id, handle.root)
    return directory, plan, _read(directory / "state.json")


def _gate(handle, access_class, plan):
    cfg = load_project_config(handle.slug, handle.root)
    if plan.get("confirm_external"):
        cfg = cfg.model_copy(deep=True)
        cfg.content_policy.external_llm_for_private_full_text = True
    assert_external_content_allowed(access_class, cfg)


def _source(session, cache, handle, work_id):
    bridge = resolve_work_source(session, work_id=work_id)
    if bridge is None or bridge.markdown_id is None or bridge.markdown_hash is None:
        raise ValueError("missing converted full text; acquire and convert this work first")
    md_id, md_hash = bridge.markdown_id, bridge.markdown_hash
    md = cache_access.read_markdown(cache, handle.root, md_id)
    if md is None:
        raise ValueError("missing cached text; restore or reconvert this work")
    if (md.markdown_hash != md_hash or not cache_access.proven_markdown_source(
            cache, bridge.source_file_id, bridge.file_hash, md_hash)):
        raise ValueError("source and converted text do not match; convert the current source and reconcile its work link before preparing again")
    latest = cache_access.current_markdown_for_source(cache, bridge.source_file_id,
                                                     bridge.file_hash)
    if latest and latest != (md_id, md_hash):
        md_id, md_hash = latest
        md = cache_access.read_markdown(cache, handle.root, md_id)
        if md is None:
            raise ValueError("newest converted text is unavailable; restore or reconvert")
        if (md.markdown_hash != md_hash or not cache_access.proven_markdown_source(
                cache, bridge.source_file_id, bridge.file_hash, md_hash)):
            raise ValueError("newest text has inconsistent source identity; reconcile conversion before preparing again")
    source_class = cache.execute("SELECT access_class FROM source_files WHERE source_file_id=?",
                                 (bridge.source_file_id,)).fetchone()[0]
    md = replace_dataclass(md, access_class=str(AccessClass.most_restrictive(md.access_class, source_class)))
    return md_id, md_hash, md


def prepare_batch(handle, work_ids: list[str], *, input_budget_tokens: int = 12000,
                  replace: bool = False, confirm_external: bool = False,
                  client: str | None = None) -> dict:
    """Freeze an ordered, explicitly selected batch without returning source text."""
    if not work_ids or len(work_ids) > 1000 or len(set(work_ids)) != len(work_ids):
        raise ValueError("select 1-1000 distinct work IDs in the desired order")
    if isinstance(input_budget_tokens, bool) or not 1024 <= input_budget_tokens <= 128000:
        raise ValueError("input_budget_tokens must be between 1024 and 128000")
    if client is not None and (not isinstance(client, str) or len(client) > 100):
        raise ValueError("client must be a short reported client name")
    schema = DefaultNoteV1.model_json_schema()
    system, empty_user = build_prompt("")
    overhead = (estimate_tokens(system) + estimate_tokens(empty_user)
                + estimate_tokens(json.dumps(schema)) + 256)
    if input_budget_tokens <= overhead:
        raise ValueError(f"input budget must exceed prompt/schema overhead ({overhead} tokens)")
    plan = {"owner": "external_host", "project_key": _project_key(handle),
            "schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION, "input_budget_tokens": input_budget_tokens,
            "prompt_overhead_tokens": overhead, "schema": schema, "client": client,
            "replace": replace, "confirm_external": confirm_external, "works": []}
    state = {"paused": None, "papers": []}
    with Session(handle.engine) as session, closing(cache_access.open_cache_ro(handle.root)) as cache:
        conn = raw_conn(session)
        for work_id in work_ids:
            if not conn.execute("SELECT 1 FROM works WHERE work_id=?", (work_id,)).fetchone():
                raise ValueError(f"unknown work: {work_id}")
            paper = {"work_id": work_id, "chunks": [],
                     "expected_current": latest_note_id(conn, work_id)}
            ps = {"status": "pending", "attempt": 0, "chunks": {}, "note_id": None}
            try:
                md_id, md_hash, md = _source(session, cache, handle, work_id)
                _gate(handle, md.access_class, plan)
                bridge = resolve_work_source(session, work_id=work_id)
                paper.update(bridge_source_file_id=bridge.source_file_id,
                             bridge_file_hash=bridge.file_hash,
                             markdown_id=md_id, markdown_hash=md_hash,
                             source_file_id=bridge.source_file_id, source_file_hash=bridge.file_hash,
                             access_class=md.access_class)
                latest = conn.execute("SELECT markdown_hash,schema_version,prompt_version FROM "
                                      "structured_notes WHERE note_id=?",
                                      (paper["expected_current"],)).fetchone()
                if latest and tuple(latest) == (md_hash, SCHEMA_VERSION, PROMPT_VERSION) and not replace:
                    ps.update(status="preserved", note_id=paper["expected_current"])
                else:
                    sections = load_sections(conn, md_id) or [{"start_char": 0,
                        "end_char": len(md.text), "ordinal": 0, "section_id": None}]
                    chunks = plan_chunks(sections, md.text, input_budget_tokens=input_budget_tokens,
                                         prompt_overhead_tokens=overhead,
                                         reserved_output_tokens=0, overlap_tokens=0)
                    if not chunks:
                        raise ValueError("empty converted text; restore or reconvert")
                    # Include inter-paragraph whitespace so the plan covers every codepoint.
                    for i, chunk in enumerate(chunks):
                        start = 0 if i == 0 else chunks[i - 1].end_char
                        end = len(md.text) if i == len(chunks) - 1 else chunk.end_char
                        sys_prompt, user_prompt = build_prompt(md.text[start:end])
                        total = (estimate_tokens(sys_prompt) + estimate_tokens(user_prompt)
                                 + estimate_tokens(json.dumps(schema)) + 256)
                        if total > input_budget_tokens:
                            raise ValueError("indivisible text exceeds input budget; prepare again with a larger budget")
                        paper["chunks"].append({"index": i, "start_char": start, "end_char": end,
                            "section_ids": chunk.section_ids, "system_prompt": sys_prompt,
                            "user_prompt": user_prompt, "estimated_input_tokens": total})
            except (ValueError, ConfigError) as exc:
                paper["error"] = str(exc)
                paper["chunks"] = []  # incomplete planning never declares partial coverage complete
                ps.update(status="failed", reason=str(exc))
            plan["works"].append(paper)
            state["papers"].append(ps)
    batch_id = run.ensure_run(handle.slug, root=handle.root)
    run.update_manifest(handle.slug, batch_id, {"agent_extraction": plan}, root=handle.root)
    directory = run._run_dir(handle.slug, batch_id, handle.root)
    _checkpoint(handle, batch_id, directory, state)
    return batch_status(handle, batch_id)


def _packet(plan, batch_id, paper_index, ps, chunk):
    paper = plan["works"][paper_index]
    binding = {key: plan[key] for key in ("project_key", "schema_id", "schema_version", "prompt_version")}
    binding.update({key: paper[key] for key in ("work_id", "markdown_id", "markdown_hash",
                                              "source_file_id", "source_file_hash",
                                              "bridge_source_file_id", "bridge_file_hash")})
    binding.update(batch_id=batch_id, chunk_index=chunk["index"], chunk_count=len(paper["chunks"]),
                   start_char=chunk["start_char"], end_char=chunk["end_char"], attempt=ps["attempt"],
                   prompt_hash=_hash([chunk["system_prompt"], chunk["user_prompt"], plan["schema"]]))
    return {"packet_id": "packet_" + _hash(binding), "binding": binding,
            "system_prompt": chunk["system_prompt"], "user_prompt": chunk["user_prompt"],
            "schema": plan["schema"], "access_class": paper["access_class"],
            "estimated_input_tokens": chunk["estimated_input_tokens"],
            "input_budget_tokens": plan["input_budget_tokens"]}


def _receipt_path(directory, packet_id):
    # Packet IDs are compared to generated IDs before constructing this path.
    return directory / "submissions" / (packet_id + ".json")


def _activate(handle, batch_id, directory, plan, state, pi):
    paper, ps = plan["works"][pi], state["papers"][pi]
    if ps["status"] != "pending" or len(ps["chunks"]) != len(paper["chunks"]):
        return
    with Session(handle.engine) as session, closing(cache_access.open_cache_ro(handle.root)) as cache:
        conn = raw_conn(session)
        # DB is authoritative after commit-before-checkpoint crashes.
        committed = conn.execute("SELECT n.note_id FROM structured_notes n JOIN extraction_runs r "
                                 "ON r.extraction_run_id=n.extraction_run_id "
                                 "WHERE r.run_id=? AND n.work_id=? ORDER BY n.rowid DESC LIMIT 1",
                                 (batch_id, paper["work_id"])).fetchone()
        if committed:
            ps.update(status="accepted", note_id=committed[0])
            return
        try:
            md_id, md_hash, md = _source(session, cache, handle, paper["work_id"])
            def source_matches(current_id, current_hash, current_md):
                bridge = resolve_work_source(session, work_id=paper["work_id"])
                return (current_id == paper["markdown_id"] and current_hash == paper["markdown_hash"]
                        and cache_access.proven_markdown_source(cache, paper["source_file_id"],
                                                                paper["source_file_hash"], current_hash)
                        and bridge is not None
                        and bridge.source_file_id == paper["bridge_source_file_id"]
                        and bridge.file_hash == paper["bridge_file_hash"])

            if (not source_matches(md_id, md_hash, md) or plan["schema_id"] != SCHEMA_ID
                    or plan["schema_version"] != SCHEMA_VERSION or plan["prompt_version"] != PROMPT_VERSION):
                ps.update(status="stale", reason="source/schema/prompt changed; retained result requires fresh preparation")
                return
            notes = []
            models = []
            for index in range(len(paper["chunks"])):
                receipt = _read(_receipt_path(directory, ps["chunks"][str(index)]))
                note, errors = validate_note(json.dumps(receipt["note"]))
                if errors or note is None:
                    raise ValueError("saved chunk failed validation; retry this paper")
                notes.append(note)
                models.append(receipt.get("model"))
            normalized = [normalize_note(note) for note in notes]
            if len(notes) == 1:
                drafts, text, archetype = normalized[0]
                raw_json = notes[0].model_dump_json()
            else:
                drafts, text, archetype = merge_chunk_drafts([x[0] for x in normalized])
                raw_json = json.dumps({"schema_id": SCHEMA_ID, "chunks": [n.model_dump() for n in notes],
                                       "merged_claims": [asdict(d) for d in drafts]})
            def check_source():
                current_id, current_hash, current_md = _source(session, cache, handle, paper["work_id"])
                if not source_matches(current_id, current_hash, current_md):
                    raise ValueError("source changed; retained result requires fresh preparation")

            saved = save_normalized_note(session, cache, work_id=paper["work_id"],
                markdown_id=md_id, markdown_hash=md_hash, drafts=drafts, note_text=text,
                archetype=archetype, raw_note_json=raw_json, resolved_class=md.access_class,
                model_name=models[0] if all(m == models[0] for m in models) else None,
                provider=plan["client"], access_mode="subscription_host_reported",
                external_full_text=1, build_run_id=batch_id, cache_root=handle.root,
                extraction_mode="agent_whole" if len(notes) == 1 else "agent_reduce",
                chunk_count=len(notes), expected_current=paper["expected_current"],
                before_activation=check_source)
            ps.update(status="accepted", note_id=saved.note_id)
        except NoteConflict as exc:
            ps.update(status="conflict", reason=str(exc))
        except ValueError as exc:
            ps.update(status="stale", reason=str(exc))


def _reconcile(handle, batch_id, directory, plan, state):
    for pi, (paper, ps) in enumerate(zip(plan["works"], state["papers"])):
        if ps["status"] != "pending":
            continue
        for chunk in paper["chunks"]:
            key = str(chunk["index"])
            if key in ps["chunks"]:
                continue
            packet = _packet(plan, batch_id, pi, ps, chunk)
            path = _receipt_path(directory, packet["packet_id"])
            if path.exists():
                receipt = _read(path)
                if receipt.get("status") == "accepted":
                    ps["chunks"][key] = packet["packet_id"]
                elif receipt.get("status") == "failed":
                    ps.update(status="failed", reason=receipt["failure"])
                    break
                elif receipt.get("status") == "paused" and not receipt.get("resumed"):
                    state["paused"] = receipt["failure"]
        _activate(handle, batch_id, directory, plan, state, pi)


def next_packet(handle, batch_id: str, *, retry_failed: bool = False, resume: bool = False,
                redact_private: bool = False) -> dict:
    with run.batch_writer(handle.slug, batch_id, root=handle.root) as directory:
        _, plan, state = _load(handle, batch_id)
        if resume:
            # Persist acknowledgment on the receipt before clearing the checkpoint.
            for paper_index, (paper, ps) in enumerate(zip(plan["works"], state["papers"])):
                for chunk in paper["chunks"]:
                    packet = _packet(plan, batch_id, paper_index, ps, chunk)
                    path = _receipt_path(directory, packet["packet_id"])
                    if path.exists():
                        receipt = _read(path)
                        if receipt.get("status") == "paused":
                            receipt["resumed"] = True
                            run._atomic_write_json(path, receipt)
            state["paused"] = None
        if retry_failed:
            for paper, ps in zip(plan["works"], state["papers"]):
                if ps["status"] == "failed" and paper["chunks"]:
                    ps.update(status="pending", attempt=ps["attempt"] + 1)
        _reconcile(handle, batch_id, directory, plan, state)
        if state["paused"]:
            _checkpoint(handle, batch_id, directory, state, "paused")
            return {"status": "paused", "batch_id": batch_id, "reason": state["paused"],
                    "next_action": "resolve host problem then call next with resume=true"}
        for pi, (paper, ps) in enumerate(zip(plan["works"], state["papers"])):
            if ps["status"] != "pending":
                continue
            try:
                with closing(cache_access.open_cache_ro(handle.root)) as cache:
                    try:
                        pinned = cache_access.read_markdown(cache, handle.root, paper["markdown_id"])
                    except (ValueError, UnicodeError):
                        # Only the cache read's content/integrity boundary is recoverable.
                        ps.update(status="failed", reason="pinned text failed integrity or UTF-8 validation; restore or reconvert, then retry")
                        continue
                    if pinned is None:
                        ps.update(status="failed", reason="pinned text unavailable; prepare again")
                        continue
                    if not cache_access.proven_markdown_source(
                            cache, paper["source_file_id"], paper["source_file_hash"], paper["markdown_hash"]):
                        ps.update(status="failed", reason="pinned source provenance unavailable; restore or reconcile before retry")
                        continue
                    selected_class = cache.execute(
                        "SELECT access_class FROM source_files WHERE source_file_id=?",
                        (paper["source_file_id"],),
                    ).fetchone()[0]
                    pinned = replace_dataclass(pinned, access_class=str(
                        AccessClass.most_restrictive(pinned.access_class, selected_class)))
                    if redact_private and not is_shareable(pinned.access_class):
                        _checkpoint(handle, batch_id, directory, state)
                        return {"status": "policy_withheld", "batch_id": batch_id,
                                "reason": "server redaction withholds private source packets",
                                "next_action": "use an authorized session with private redaction disabled"}
                    _gate(handle, pinned.access_class, plan)
            except ConfigError:
                state["paused"] = "batch_access"
                _checkpoint(handle, batch_id, directory, state, "paused")
                return {"status": "paused", "batch_id": batch_id, "reason": "batch_access",
                        "next_action": "authorize external full text or prepare permitted works"}
            for chunk in paper["chunks"]:
                if str(chunk["index"]) not in ps["chunks"]:
                    _checkpoint(handle, batch_id, directory, state)
                    packet = _packet(plan, batch_id, pi, ps, chunk)
                    packet["access_class"] = pinned.access_class
                    return {"status": "ready", "batch_id": batch_id, **packet}
        _checkpoint(handle, batch_id, directory, state, "finished")
        return {"status": "finished", "batch_id": batch_id,
                "summary": _summary(plan, state, 0, 50)}


def submit_packet(handle, batch_id: str, packet_id: str, *, result: dict | str | None = None,
                  failure: str | None = None, model: str | None = None) -> dict:
    if (result is None) == (failure is None):
        raise ValueError("provide exactly one result or failure")
    if failure is not None and failure not in _SESSION_FAILURES | _PAPER_FAILURES:
        raise ValueError("unknown failure code")
    if model is not None and (not isinstance(model, str) or len(model) > 200):
        raise ValueError("model must be a short reported model name")
    if result is not None and len(json.dumps(result).encode()) > _MAX_RESULT_BYTES:
        raise ValueError("result too large; maximum is 4 MB")
    canonical_result = result
    if isinstance(result, str):
        try:
            canonical_result = json.loads(result)
        except ValueError:
            pass  # preserve malformed text for inspection and ordinary validation
    payload_hash = _hash({"result": canonical_result, "failure": failure})
    with run.batch_writer(handle.slug, batch_id, root=handle.root) as directory:
        _, plan, state = _load(handle, batch_id)
        # Enumerate generated IDs, so caller-controlled IDs never become paths.
        selected = None
        for pi, (paper, ps) in enumerate(zip(plan["works"], state["papers"])):
            for attempt in range(ps["attempt"] + 1):
                for chunk in paper["chunks"]:
                    packet = _packet(plan, batch_id, pi, {**ps, "attempt": attempt}, chunk)
                    if packet["packet_id"] == packet_id:
                        selected = pi, paper, ps, chunk, packet
                        break
                if selected:
                    break
            if selected:
                break
        if selected is None:
            return {"status": "conflict", "reason": "packet does not belong to this batch or attempt"}
        pi, paper, ps, chunk, packet = selected
        path = _receipt_path(directory, packet_id)
        if path.exists():
            prior = _read(path)
            if prior["status"] in {"accepted", "failed"}:
                if prior["payload_hash"] != payload_hash:
                    return {"status": "conflict", "reason": "packet already consumed with a different payload"}
                _reconcile(handle, batch_id, directory, plan, state)
                _checkpoint(handle, batch_id, directory, state)
                return {"status": "duplicate", "packet_id": packet_id, "paper_status": ps["status"],
                        "note_id": ps["note_id"], "original": prior.get("outcome")}
        if (ps["status"] != "pending" or state["paused"]
                or packet["binding"]["attempt"] != ps["attempt"]):
            return {"status": "conflict", "reason": "paper is not pending or batch is paused; resume/retry explicitly"}
        active = next((c for c in paper["chunks"] if str(c["index"]) not in ps["chunks"]), None)
        earlier_pending = any(p["status"] == "pending" for p in state["papers"][:pi])
        if active is None or active["index"] != chunk["index"] or earlier_pending:
            return {"status": "conflict", "reason": "submit the unfinished packet returned by next"}
        receipt = {"payload_hash": payload_hash, "packet_id": packet_id,
                   "binding": packet["binding"], "payload": result, "failure": failure,
                   "model": model, "status": "invalid"}
        if failure:
            if failure in _SESSION_FAILURES:
                state["paused"] = failure
                receipt["status"] = "paused"
            else:
                ps.update(status="failed", reason=failure)
                receipt["status"] = "failed"
            outcome = {"status": "accepted", "packet_id": packet_id,
                       "paper_status": ps["status"], "pause_reason": state["paused"]}
        else:
            try:
                value = json.loads(result) if isinstance(result, str) else result
                if isinstance(value, dict) and "binding" in value:
                    if value["binding"] != packet["binding"]:
                        raise ValueError("result binding does not match prepared packet")
                    value = value.get("note")
                note, errors = validate_note(json.dumps(value))
                if note is None or errors:
                    raise ValueError("; ".join(errors))
                normalize_note(note)
                receipt.update(status="accepted", note=note.model_dump())
                # Save the validated chunk before any activation. Crash recovery reads it.
                run._atomic_write_json(path, receipt)
                ps["chunks"][str(chunk["index"])] = packet_id
                _activate(handle, batch_id, directory, plan, state, pi)
                outcome = {"status": ps["status"] if ps["status"] in {"stale", "conflict"} else "accepted",
                           "packet_id": packet_id, "paper_status": ps["status"], "note_id": ps["note_id"],
                           "chunks_completed": len(ps["chunks"]), "chunks_planned": len(paper["chunks"])}
                if ps.get("reason"):
                    outcome["reason"] = ps["reason"]
            except (ValueError, TypeError) as exc:
                outcome = {"status": "invalid", "packet_id": packet_id, "reason": str(exc)[:2000]}
        outcome["artifact_path"] = str(path)
        receipt["outcome"] = outcome
        run._atomic_write_json(path, receipt)
        _checkpoint(handle, batch_id, directory, state, "paused" if state["paused"] else "awaiting_host")
        return outcome


def _summary(plan, state, offset, limit):
    counts = {}
    for ps in state["papers"]:
        counts[ps["status"]] = counts.get(ps["status"], 0) + 1
    papers = []
    for paper, ps in list(zip(plan["works"], state["papers"]))[offset:offset + limit]:
        papers.append({"work_id": paper["work_id"], "status": ps["status"], "note_id": ps["note_id"],
                       "chunks_completed": len(ps["chunks"]), "chunks_planned": len(paper["chunks"]),
                       "reason": ps.get("reason"), "attempt": ps["attempt"]})
    return {"counts": counts, "papers": papers, "total": len(plan["works"]),
            "next_offset": offset + limit if offset + limit < len(plan["works"]) else None}


def batch_status(handle, batch_id: str, *, offset: int = 0, limit: int = 50) -> dict:
    if isinstance(offset, bool) or isinstance(limit, bool) or offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative and limit between 1 and 100")
    _, plan, state = _load(handle, batch_id)
    return {"batch_id": batch_id, "status": "paused" if state["paused"] else (
                "awaiting_host" if any(p["status"] == "pending" for p in state["papers"]) else "finished"),
            "pause_reason": state["paused"], **_summary(plan, state, offset, limit)}
