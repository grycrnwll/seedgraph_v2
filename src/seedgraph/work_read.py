"""Bounded, nonmutating reads of a work's pinned Markdown source.

Offsets are Unicode codepoints, half-open [start, end). Cursors contain IDs and
selection state, never paths; they are fingerprints, not authorization tokens.
Access policy is checked afresh on every request, before any source text leaves.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from typing import Any

from . import anchor, cache_access
from .config.loader import load_project_config
from .db.connection import cache_db_path, connect_readonly
from .db.migrations import current_version, latest_version
from .errors import ConfigError, ValidationError
from .ids import sha256_hex
from .llm.routing import assert_external_content_allowed
from .project.service import ProjectHandle
from .sections.parser import Section, parse_sections
from .sections.store import load_sections
from .semantic.access import is_shareable
from .vocab import AccessClass


def _fingerprint(data: dict) -> str:
    return sha256_hex(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def _cursor(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(
        {"state": data, "fingerprint": _fingerprint(data)}, separators=(",", ":")
    ).encode()).decode()


def _decode(cursor: str) -> dict:
    try:
        if len(cursor) > 8192:
            raise ValueError()
        decoded = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        state = decoded["state"]
        if not isinstance(state, dict) or decoded["fingerprint"] != _fingerprint(state):
            raise ValueError()
        if set(state) != {"version", "project", "work", "markdown", "hash", "source",
                          "source_hash", "section", "subtree", "span", "start", "end", "next"}:
            raise ValueError()
        if state["version"] != 1 or any(type(state[k]) is not int for k in ("start", "end", "next")):
            raise ValueError()
        if any(not isinstance(state[k], str) for k in
               ("project", "work", "markdown", "hash", "source", "source_hash")):
            raise ValueError()
        if type(state["subtree"]) is not bool or any(
            state[k] is not None and not isinstance(state[k], str) for k in ("section", "span")
        ):
            raise ValueError()
        return state
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise ValidationError("invalid work_read cursor; restart the selected read") from exc


def _bounded_section(section: Section) -> dict:
    result = asdict(section)
    # Headings are source text, and are bounded independently of the body budget.
    result["heading_truncated"] = any(len(result[k] or "") > limit for k, limit in
                                      (("heading_text", 256), ("heading_path", 512)))
    result["heading_text"] = (section.heading_text or "")[:256] or None
    result["heading_path"] = (section.heading_path or "")[:512] or None
    return result


def read_work(
    handle: ProjectHandle, *, work_id: str, section_id: str | None = None,
    subtree: bool = False, span_id: str | None = None, cursor: str | None = None,
    max_chars: int = 12000, section_offset: int = 0, section_limit: int = 50,
    redact_private: bool = False, external: bool = False, confirm_external: bool = False,
) -> dict[str, Any]:
    """Read a work, one section (own/subtree), or an anchored span's source.

    A span selects its full source document, starting at zero; ``anchor`` in the
    result identifies its exact location and section. Missing source and policy
    outcomes carry metadata and next actions without acquisition or conversion.
    """
    for name, value, lower, upper in (("max_chars", max_chars, 1, 48000),
                                      ("section_limit", section_limit, 1, 100)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValidationError(f"{name} must be between {lower} and {upper}")
    if type(section_offset) is not int or section_offset < 0:
        raise ValidationError("section_offset must be a nonnegative integer")
    if section_id and span_id or subtree and not section_id:
        raise ValidationError("select section_id or span_id; subtree requires section_id")
    state = _decode(cursor) if cursor else None
    project = _fingerprint({"slug": handle.slug, "db": str(handle.db_path.resolve()).casefold()})
    requested = {"project": project, "work": work_id, "section": section_id,
                 "subtree": subtree, "span": span_id}
    if state and any(state[k] != value for k, value in requested.items()):
        raise ValidationError("cursor belongs to another project, work or selection; restart")

    with closing(connect_readonly(handle.db_path)) as conn:
        work = conn.execute("SELECT * FROM works WHERE work_id=?", (work_id,)).fetchone()
        if work is None:
            raise ValidationError(f"unknown work_id {work_id!r}")
        membership = conn.execute(
            "SELECT inclusion_status, inclusion_reason, is_seed FROM project_documents WHERE work_id=?",
            (work_id,),
        ).fetchone()
        bridge = conn.execute("SELECT * FROM work_source_files WHERE work_id=?", (work_id,)).fetchone()
        span = None
        if span_id:
            span = conn.execute("SELECT * FROM evidence_spans WHERE span_id=? AND work_id=?",
                                (span_id, work_id)).fetchone()
            if span is None:
                raise ValidationError("span_id does not belong to the selected work")
        response = {
            "status": "ok", "reason": None, "next_action": None,
            "work": {"work_id": work_id, "title": work["canonical_title"],
                     "authors": json.loads(work["authors"] or "[]"), "year": work["year"],
                     "venue": work["venue"], "doi": work["doi"], "oa_status": work["oa_status"]},
            "membership": dict(membership) if membership else None,
            "access_class": None, "source": None, "anchor": None, "selection": None,
            "text": None, "start_char": None, "end_char": None,
            "selection_complete": False, "continuation": None,
            "sections": {"items": [], "total": 0, "next_offset": None},
            "newer_version_available": False,
        }

        def unavailable(reason: str, action: str, *, status="unavailable") -> dict:
            response.update(status=status, reason=reason, next_action=action)
            return response

        source = span if span is not None else bridge
        if source is None and not state:
            return unavailable("source_missing", "Acquire or upload this work's source, then convert it.")
        source_hash = (source["source_file_hash"] if span is not None else source["file_hash"]) if source else None
        source_id = source["source_file_id"] if source else None
        md_id = state["markdown"] if state else source["markdown_id"]
        md_hash = state["hash"] if state else source["markdown_hash"]
        lineage_changed = bool(state and (state["source"] != source_id or state["source_hash"] != source_hash))
        if lineage_changed:
            # The mutable bridge alone cannot prove old source ownership after a
            # re-upload. Existing project provenance may; a cursor is not proof.
            known = conn.execute(
                "SELECT 1 FROM document_sections WHERE work_id=? AND markdown_id=? AND markdown_hash=? "
                "UNION ALL SELECT 1 FROM evidence_spans WHERE work_id=? AND markdown_id=? AND markdown_hash=? "
                "UNION ALL SELECT 1 FROM structured_notes WHERE work_id=? AND markdown_id=? AND markdown_hash=? LIMIT 1",
                (work_id, md_id, md_hash) * 3,
            ).fetchone()
            if known is None:
                return unavailable("pinned_lineage_unavailable",
                                   "The source file was replaced and its old work link is unavailable; restart this read.")
            source_id, source_hash = state["source"], state["source_hash"]
        if md_id is None:
            return unavailable("text_not_converted", "Convert the existing source, then retry work_read.")
        if not cache_db_path(handle.root).is_file():
            return unavailable("cache_missing", "Restore or initialize the source cache, then restart this read.")
        with closing(connect_readonly(cache_db_path(handle.root))) as cache:
            try:
                version = current_version(cache)
                expected = latest_version("cache")
            except sqlite3.Error:
                return unavailable("cache_unavailable", "Restore or repair the source cache, then retry this read.")
            if version != expected:
                return unavailable("cache_setup_required" if version == 0 else "cache_schema_mismatch",
                                   f"Cache schema is {version}; expected {expected}. Initialize or upgrade the cache separately, then retry.")
            # Verify cached bytes without the mutating spans.verify or cache initializer.
            try:
                md = cache_access.read_markdown(cache, handle.root, md_id)
            except sqlite3.Error:
                return unavailable("cache_schema_unavailable", "Repair or upgrade the source cache separately, then retry this read.")
            except (ValueError, UnicodeError):
                return unavailable("source_integrity_error", "Restore or reconvert the source, then restart this read.")
            if md is None:
                return unavailable("pinned_version_missing" if state or span is not None else "text_missing",
                                   "Restore the original cached version or restart from available source text.")
            try:
                proven = cache_access.proven_markdown_source(cache, source_id, source_hash, md.markdown_hash)
                source_row = cache.execute(
                    "SELECT access_class FROM source_files WHERE source_file_id=? AND file_hash=?",
                    (source_id, source_hash),
                ).fetchone()
            except sqlite3.Error:
                return unavailable("cache_schema_unavailable", "Repair or upgrade the source cache separately, then retry this read.")
            if md.markdown_hash != md_hash or not proven or source_row is None:
                return unavailable("source_identity_mismatch", "Repair source lineage, then restart this read.")
            selected_access = source_row[0]
            access_class = str(AccessClass.most_restrictive(
                md.access_class, selected_access,
                span["access_class"] if span is not None else selected_access,
            ))
            response["access_class"] = access_class
            response["source"] = {"markdown_id": md_id, "markdown_hash": md.markdown_hash,
                                  "source_file_id": source_id,
                                  "source_file_hash": source_hash,
                                  "total_chars": len(md.text)}
            if redact_private and not is_shareable(access_class):
                return unavailable("private_content_redacted", "Use an authorized local reading session.",
                                   status="policy_withheld")
            if external:
                cfg = load_project_config(handle.slug, handle.root)
                if confirm_external:
                    cfg = cfg.model_copy(deep=True)
                    cfg.content_policy.external_llm_for_private_full_text = True
                try:
                    assert_external_content_allowed(access_class, cfg, task_type="work_read")
                except ConfigError as exc:
                    response["policy_reason"] = str(exc)[:600]
                    route = cfg.llm.routes.get("work_read")
                    if not cfg.content_policy.allow_external_llm:
                        action = "Read locally or enable external LLM handoff in the applicable content policy."
                    elif (route is not None and route.allowed_access_classes is not None
                          and access_class not in route.allowed_access_classes):
                        action = "Read locally or change the route's allowed access classes."
                    else:
                        action = "Read locally or explicitly confirm external sharing for this request."
                    return unavailable("external_content_forbidden",
                                       action, status="policy_withheld")
            try:
                newest = cache_access.current_markdown_for_source(cache, source_id, source_hash)
            except sqlite3.Error:
                return unavailable("cache_schema_unavailable", "Repair or upgrade the source cache separately, then retry this read.")
            response["newer_version_available"] = lineage_changed or bool(newest and newest[0] != md_id)
            if response["newer_version_available"]:
                response["newer_markdown_id"] = bridge["markdown_id"] if lineage_changed and bridge else newest[0] if newest else None

        if span is not None:
            if (not 0 <= span["start_char"] <= span["end_char"] <= len(md.text)
                    or md.text[span["start_char"]:span["end_char"]] != span["exact_quote"]
                    or anchor.quote_hash(span["exact_quote"]) != span["quote_hash"]):
                return unavailable("span_anchor_invalid", "Verify or reanchor this span before reading its source.")
            response["anchor"] = {key: span[key] for key in
                                  ("span_id", "section_id", "start_char", "end_char", "page_start", "page_end")}

        sections = load_sections(conn, md_id)
        if not sections or any(
            s.markdown_hash != md.markdown_hash or s.source_file_id != source_id
            or not 0 <= s.start_char <= s.end_char <= len(md.text) for s in sections
        ):
            sections = parse_sections(md.text, markdown_id=md_id, markdown_hash=md.markdown_hash,
                                      source_file_id=source_id, source_file_hash=source_hash,
                                      work_id=work_id)
        start, end = 0, len(md.text)
        if section_id:
            selected = next((s for s in sections if s.section_id == section_id), None)
            if selected is None:
                raise ValidationError("section_id is absent from the pinned source; restart the read")
            start, end = selected.start_char, selected.end_char
            if subtree and selected.level > 0:
                for section in sections:
                    if section.ordinal <= selected.ordinal:
                        continue
                    if section.level <= selected.level:
                        break
                    end = section.end_char
        if state:
            if (state["start"] != start or state["end"] != end or not start <= state["next"] <= end):
                raise ValidationError("cursor offsets do not match the selected source; restart")
            position = state["next"]
        else:
            position = start
        stop = min(position + max_chars, end)
        complete = stop == end
        response.update(text=md.text[position:stop], start_char=position, end_char=stop,
                        selection_complete=complete,
                        selection={"kind": "section" if section_id else "span_source" if span_id else "work",
                                   "section_id": section_id, "subtree": subtree,
                                   "start_char": start, "end_char": end})
        response["sections"] = {
            "items": [_bounded_section(s) for s in sections[section_offset:section_offset + section_limit]],
            "total": len(sections),
            "next_offset": section_offset + section_limit if section_offset + section_limit < len(sections) else None,
        }
        if not complete:
            response["continuation"] = _cursor({
                "version": 1, **requested, "markdown": md_id, "hash": md.markdown_hash,
                "source": source_id, "source_hash": source_hash,
                "start": start, "end": end, "next": stop,
            })
        return response
