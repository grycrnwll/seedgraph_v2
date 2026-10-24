"""The ``lens:<lens_id>`` task (design §2.2): v2 lens behaviour, dispatched as packages.

Kept as-is from v2 (``lenses/runner.py``, which P8 deletes, so its record helpers are
copied here): the same definitions (``lenses.schema.LensDefinition``), the same record
validation and evidence policy (``validate_record``), the same prompt
(``lenses.prompt.build_lens_prompt``, in ``instructions.md``), and the same projection
of a record: anchor its verbatim quote first, then validate with the span filled in;
``found`` -> an ``extracted_claims`` row (``lens_claim_type``, ``<object_type>[i]``),
``claim_spans``, ``claim_fts`` and a ``found`` output; ``ambiguous`` /
``extraction_failed`` -> an output with no claim; no records -> one ``not_found``
output. Current = the latest run per (paper, lens) matching the definition hash.

Changes from v2, all from the dispatch: the version is the definition hash and
definitions come from ``lens_definitions.definition_yaml`` (the canonical snapshot) in
the library; ``paper_id`` replaces ``work_id``; a payload that is not an object with a
``records`` list is an ``invalid`` submission (v2 stored an ``extraction_failed`` run),
so the worker fixes it; a found record's quote goes through the protocol's anchor
preview (``needs_revision`` unless ``accept_unanchored``); v2's ``lens_record`` review
queue item has no library home in stage 2 and is dropped (the output row keeps the
status, and the submission feedback carries the issues).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import yaml

from ..ids import new_id
from ..lenses.prompt import LENS_PROMPT_VERSION, build_lens_prompt
from ..lenses.schema import LensDefinition
from ..notes.spans import ensure_span
from ..vocab import lens_claim_type
from . import packages
from .store import WorkError
from .tasks import (
    AcceptedPart,
    CheckResult,
    Instructions,
    QuoteRef,
    SaveContext,
    SaveResult,
)
from .tasks_notes import WholePaperTask

if TYPE_CHECKING:
    from ..library.store import Library
    from .store import BatchItem, WorkPackage

LENS_SCHEMA_VERSION = "lens_schema_v1"  # as v2 lenses/runner.py
_PREVIEW_SPAN = "span_preview"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- definitions ---------------------------------------------------------------------------

def register_lens(lib: Library, lens: LensDefinition, *, status: str = "active") -> LensDefinition:
    """Store a lens version in ``lens_definitions`` (idempotent per definition hash)."""
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO lens_definitions (lens_id, name, object_type, status, definition_hash, definition_yaml, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(lens_id, definition_hash) DO UPDATE SET "
            "status = excluded.status",
            (lens.lens_id, lens.name, lens.object_type, status, lens.definition_hash(), lens.definition_snapshot(),
             _now()))
    return lens


def load_lens(conn: sqlite3.Connection, lens_id: str, definition_hash: str | None = None) -> LensDefinition:
    """The lens at ``definition_hash`` (any status), or else its current version: the
    newest ``active`` row, else the newest non-archived row."""
    if definition_hash is not None:
        row = conn.execute("SELECT definition_hash, definition_yaml FROM lens_definitions "
                           "WHERE lens_id = ? AND definition_hash = ?", (lens_id, definition_hash)).fetchone()
    else:
        row = conn.execute(
            "SELECT definition_hash, definition_yaml FROM lens_definitions WHERE lens_id = ? "
            "AND status <> 'archived' ORDER BY (status = 'active') DESC, created_at DESC, definition_hash DESC "
            "LIMIT 1", (lens_id,)).fetchone()
    if row is None:
        version = f" at version {definition_hash}" if definition_hash else ""
        raise WorkError(f"no lens {lens_id!r}{version} in the library")
    if not row[1]:
        raise WorkError(f"lens {lens_id!r} has no stored definition (definition_yaml is empty)")
    lens = LensDefinition.from_dict(yaml.safe_load(row[1]))
    if lens.definition_hash() != row[0]:
        raise WorkError(f"lens {lens_id!r}: the stored definition does not hash to its definition_hash")
    return lens


def _latest_run(conn: sqlite3.Connection, lens_id: str, paper_id: str) -> tuple[str, str, str] | None:
    row = conn.execute(
        "SELECT extraction_run_id, lens_definition_hash, markdown_hash FROM extraction_runs "
        "WHERE lens_id = ? AND paper_id = ? ORDER BY created_at DESC, extraction_run_id DESC LIMIT 1",
        (lens_id, paper_id)).fetchone()
    return (row[0], row[1], row[2]) if row else None


def lens_is_current(conn: sqlite3.Connection, lens: LensDefinition, paper_id: str) -> bool:
    """The paper's latest run of this lens matches the definition hash and its current markdown."""
    run = _latest_run(conn, lens.lens_id, paper_id)
    md = packages.primary_markdown(conn, paper_id)
    return run is not None and md is not None and run[1] == lens.definition_hash() and run[2] == md[1]


def lens_results(lib: Library, lens_id: str, *, status: str = "found",
                 paper_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Library-backed lens results: the outputs of each paper's current run (latest run
    matching the lens's definition hash and the paper's current markdown)."""
    conn = lib.conn
    lens = load_lens(conn, lens_id)
    papers = paper_ids if paper_ids is not None else [r[0] for r in conn.execute(
        "SELECT DISTINCT paper_id FROM extraction_runs WHERE lens_id = ? ORDER BY paper_id", (lens_id,))]
    out: list[dict[str, Any]] = []
    for pid in papers:
        if not lens_is_current(conn, lens, pid):
            continue
        run = _latest_run(conn, lens_id, pid)
        assert run is not None
        for lo_id, st, claim_id, confidence, fields_json in conn.execute(
                "SELECT lens_output_id, status, claim_id, confidence, fields_json FROM lens_outputs "
                "WHERE extraction_run_id = ? ORDER BY created_at, lens_output_id", (run[0],)):
            if status != "all" and st != status:
                continue
            fields = json.loads(fields_json) if fields_json else {}
            claim = conn.execute("SELECT claim_text, normalized_label FROM extracted_claims WHERE claim_id = ?",
                                 (claim_id,)).fetchone() if claim_id else None
            spans = [r[0] for r in conn.execute("SELECT span_id FROM claim_spans WHERE claim_id = ? ORDER BY rank",
                                                (claim_id,))] if claim_id else []
            out.append({"lens_output_id": lo_id, "paper_id": pid, "status": st, "claim_id": claim_id,
                        "claim_text": claim[0] if claim else None, "normalized_label": claim[1] if claim else None,
                        "section": fields.get("section") if isinstance(fields, dict) else None,
                        "confidence": confidence, "span_ids": spans, "fields": fields,
                        "extraction_run_id": run[0]})
    return out


# --- record helpers (copied from v2 lenses/runner.py) -----------------------------------------

def _record_quote(record: dict) -> str | None:
    for key in ("condition_text_verbatim", "exact_quote", "quote", "text_verbatim", "verbatim"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key, value in record.items():
        if isinstance(value, str) and value.strip() and key.endswith("verbatim"):
            return value
    return None


def _record_label(record: dict) -> str | None:
    for key in ("normalized_condition_label", "normalized_label", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _records(payload: dict[str, Any]) -> list[Any]:
    records = payload.get("records")
    return records if isinstance(records, list) else []


# --- the task --------------------------------------------------------------------------------

class LensTask(WholePaperTask):
    def __init__(self, lib: Library, lens_id: str, version: str | None = None) -> None:
        super().__init__(lib)
        self.lens = load_lens(lib.conn, lens_id, version)
        self.name = f"lens:{lens_id}"
        self.version = self.lens.definition_hash()

    def is_preserved(self, conn: sqlite3.Connection, paper_id: str) -> bool:
        return lens_is_current(conn, self.lens, paper_id)

    def instructions(self, lib: Library, *, has_parts: bool) -> Instructions:
        system, user = build_lens_prompt(self.lens, markdown_text="(the paper text is in your package file)")
        lines = [
            f"# Lens task: `{self.lens.lens_id}` ({self.lens.name})",
            "",
            f"Definition hash `{self.version}`, prompt `{LENS_PROMPT_VERSION}`. Write ONE JSON object of the shape "
            "below to the package's `result_path` and submit it. Copy every quote character for character from "
            "the package text; a quote that does not anchor comes back with the exact text to copy.",
            "",
        ]
        if has_parts:
            lines += ["Long papers come in parts (`part` k of n): report the instances in your part's text only.", ""]
        lines += ["## Role", "", system, "", "## Lens prompt", "", user.rstrip(), ""]
        return Instructions("\n".join(lines), "{}\n")

    def check(self, lib: Library, package: WorkPackage, payload: dict[str, Any]) -> CheckResult:
        result = CheckResult()
        if not isinstance(payload.get("records", []), list):
            result.errors = ["`records` must be an array of record objects"]
            return result
        ot = self.lens.object_type
        for i, rec in enumerate(_records(payload)):
            if not isinstance(rec, dict):
                result.warnings.append({"field_key": f"{ot}[{i}]", "message": "not an object; skipped"})
                continue
            preview = self.lens.validate_record({**rec, "evidence_span_ids": [_PREVIEW_SPAN]})
            if preview.status == "found":
                result.claims_kept += 1
                result.quotes.append(QuoteRef(f"{ot}[{i}]", _record_quote(rec)))
            elif preview.status == "extraction_failed":
                result.warnings.append({"field_key": f"{ot}[{i}]",
                                        "message": "saved as extraction_failed: " + "; ".join(preview.issues)})
        return result

    def save(self, conn: sqlite3.Connection, item: BatchItem, parts: Sequence[AcceptedPart],
             ctx: SaveContext) -> SaveResult:
        lens = self.lens
        ordered = sorted(parts, key=lambda p: p.part)
        multi = len(ordered) > 1
        last = ordered[-1]
        markdown_id, paper_id = item.markdown_id or "", item.subject_id
        access_class = conn.execute(
            "SELECT s.access_class FROM markdown_documents m JOIN source_files s "
            "ON s.source_file_id = m.source_file_id WHERE m.markdown_id = ?", (markdown_id,)).fetchone()[0]
        run_id = new_id("extr")
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, schema_id, "
            "lens_id, schema_version, prompt_version, model_name, provider, access_mode, access_class, run_status, "
            "input_tokens, output_tokens, run_id, created_at, extraction_mode, chunk_count, lens_definition_hash) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 'agent_session', ?, 'success', ?, ?, ?, ?, ?, ?, ?)",
            (run_id, paper_id, markdown_id, item.markdown_hash, lens.lens_id, LENS_SCHEMA_VERSION,
             LENS_PROMPT_VERSION, last.model or ctx.options.get("model"), last.client or last.worker_kind,
             access_class, _usage(ordered, "input_tokens"), _usage(ordered, "output_tokens"), ctx.batch_id, _now(),
             "agent_parts" if multi else "agent_whole", len(ordered), self.version))
        counts = {"found": 0, "not_found": 0, "ambiguous": 0, "not_applicable": 0, "extraction_failed": 0}
        produced = 0
        for p in ordered:
            for rec in _records(p.payload):
                if not isinstance(rec, dict):
                    continue
                status = self._project(conn, rec, run_id=run_id, paper_id=paper_id, markdown_id=markdown_id,
                                       text=ctx.markdown_text, field_index=produced, access_class=access_class,
                                       chunk=p.part - 1 if multi else None)
                counts[status] = counts.get(status, 0) + 1
                produced += 1
        if produced == 0:
            nfs = [p.payload.get("not_found") for p in ordered]
            nfs = [nf if isinstance(nf, dict) else {} for nf in nfs]
            sections = [s for nf in nfs for s in (nf.get("searched_sections") or [])]
            notes = [nf.get("notes") for nf in nfs if nf.get("notes")]
            fields = {"status": "not_found", "searched_sections": sections,
                      "notes": notes[0] if len(notes) == 1 else ("; ".join(notes) if notes else None),
                      "confidence": nfs[0].get("confidence")}
            _insert_output(conn, run_id=run_id, lens_id=lens.lens_id, paper_id=paper_id, status="not_found",
                           fields=fields, access_class=access_class, confidence=fields["confidence"])
            counts["not_found"] += 1
        return SaveResult(run_id, counts)

    def _project(self, conn: sqlite3.Connection, record: dict, *, run_id: str, paper_id: str, markdown_id: str,
                 text: str, field_index: int, access_class: str, chunk: int | None) -> str:
        """v2 ``_project_found_record`` on the library: anchor, validate, project."""
        lens = self.lens
        quote = _record_quote(record)
        span_id = ensure_span(conn, markdown_id=markdown_id, exact_quote=quote, markdown_text=text) if quote else None
        record = dict(record)
        record["evidence_span_ids"] = [span_id] if span_id else []
        vr = lens.validate_record(record)
        if vr.status != "found":
            _insert_output(conn, run_id=run_id, lens_id=lens.lens_id, paper_id=paper_id, status=vr.status,
                           fields=record, access_class=access_class, confidence=vr.confidence)
            return vr.status
        claim_id = new_id("claim")
        label = _record_label(record)
        explanation = (record.get("notes") or None) if vr.assertion_status == "inferred" else None
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, paper_id, claim_type, "
            "claim_subtype, field_key, normalized_label, claim_text, status, epistemic_type, assertion_status, "
            "inferred_explanation, confidence, access_class, created_at, source_chunk_index) "
            "VALUES (?, NULL, ?, ?, ?, NULL, ?, ?, ?, 'found', 'llm_extracted', ?, ?, ?, ?, ?, ?)",
            (claim_id, run_id, paper_id, lens_claim_type(lens.object_type, lens.lens_id),
             f"{lens.object_type}[{field_index}]", label, quote, vr.assertion_status, explanation, vr.confidence,
             access_class, _now(), chunk))
        assert span_id is not None
        conn.execute("INSERT OR IGNORE INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, 0, ?)",
                     (claim_id, span_id, _now()))
        conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text, plain_words) "
                     "VALUES (?, ?, ?, NULL)", (claim_id, label, quote))
        _insert_output(conn, run_id=run_id, lens_id=lens.lens_id, paper_id=paper_id, status="found", fields=record,
                       access_class=access_class, claim_id=claim_id, confidence=vr.confidence)
        return "found"


def _insert_output(conn: sqlite3.Connection, *, run_id: str, lens_id: str, paper_id: str, status: str,
                   fields: Mapping[str, Any], access_class: str, claim_id: str | None = None,
                   confidence: Any = None) -> str:
    lens_output_id = new_id("lensout")
    conn.execute(
        "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, paper_id, status, claim_id, "
        "confidence, fields_json, access_class, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (lens_output_id, run_id, lens_id, paper_id, status, claim_id,
         confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
         json.dumps(dict(fields), ensure_ascii=False), access_class, _now()))
    return lens_output_id


def _usage(parts: Sequence[AcceptedPart], key: str) -> int | None:
    values = [int(p.usage[key]) for p in parts if p.usage and isinstance(p.usage.get(key), (int, float))]
    return sum(values) if values else None
