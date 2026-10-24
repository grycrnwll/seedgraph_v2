"""Build a small, deterministic v2-shaped home for the migration tests.

``build_v2_home(root)`` writes ``cache/cache.db``, ``cache/pdfs/<sha>.pdf``,
``cache/markdown/<sha>.md``, ``projects/<slug>/{project.db,project.yaml}``, a lens
YAML folder and saved answers, all against the frozen v2 schema
(:mod:`seedgraph.migrate.v2_schema`). Rows are inserted directly; no v2 code runs.

It returns a :class:`V2Fixture` whose manifest (``items``) names every planted
defect or edge case, the ids involved and the outcome the design expects, so the
migration tests can assert that the report accounts for each one.

Determinism: ids are hashes of fixed keys, timestamps are literals, files are
written as LF-only bytes, PDFs carry no dates. Two builds into different folders
give identical database dumps and identical files.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seedgraph.migrate.v2_schema import create_v2_db

from .corpus import (
    ANSWERS_TEXT, ASSETS, REF_AMBIG, REF_CITED, REF_OVERLAP, REF_PORT, REF_SUSPECT, REF_TOLLS,
    REF_TP_ONLY, Asset,
)
from .pdf import minimal_pdf

APPLIED_AT = "2026-06-30T00:00:00+00:00"
MARKER_CONFIG = ('{"force_ocr":false,"llm_model":null,"llm_prompt_version":null,"llm_service":null,'
                 '"output_format":"markdown","paginate_output":true,"redo_inline_math":false,"use_llm":false}')
SCHEMA_ID = "default_research_note_v1"
NO_SECTIONS = {"oct2026test"}        # as in the real data: oct2026test was never section-parsed
PARAGRAPH_SPANS = {"land_value"}     # only land_value has auto 'paragraph' spans


# --- manifest ----------------------------------------------------------------

@dataclass(frozen=True)
class WorkRef:
    """One planted v2 ``works`` row (key ``"<name>@<slug>"``)."""

    key: str
    slug: str
    work_id: str
    title: str | None
    year: int | None
    doi: str | None = None
    openalex: str | None = None
    arxiv: str | None = None
    source_file_id: str | None = None
    markdown_id: str | None = None


@dataclass(frozen=True)
class Planted:
    """One planted case: ``kind`` (the class), the ``ids`` involved, and the outcome
    the design (docs/V3_LIBRARY_DESIGN.md) expects in ``expect``."""

    kind: str
    ids: Mapping[str, Any]
    expect: Mapping[str, Any]
    note: str = ""


@dataclass
class V2Fixture:
    home: Path
    works: dict[str, WorkRef] = field(default_factory=dict)
    items: list[Planted] = field(default_factory=list)

    @property
    def kinds(self) -> list[str]:
        return sorted({i.kind for i in self.items})

    def by_kind(self, kind: str) -> list[Planted]:
        return [i for i in self.items if i.kind == kind]

    def one(self, kind: str) -> Planted:
        found = self.by_kind(kind)
        if len(found) != 1:
            raise KeyError(f"{kind!r}: expected one planted item, found {len(found)}")
        return found[0]

    def to_json(self) -> str:
        return json.dumps(
            {"works": {k: vars(v) for k, v in sorted(self.works.items())},
             "items": [{"kind": i.kind, "ids": i.ids, "expect": i.expect, "note": i.note} for i in self.items]},
            sort_keys=True, indent=2, ensure_ascii=False,
        )


# --- small helpers -----------------------------------------------------------

def _h(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{_h(prefix, *parts)[:32]}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_title(title: str) -> str:
    import re
    decomposed = unicodedata.normalize("NFKD", title)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", no_marks.lower()).split())


def _title_hash(title: str | None) -> str | None:
    return hashlib.sha1(_normalize_title(title).encode("utf-8")).hexdigest() if title else None


def _quote_hash(q: str) -> str:
    return _sha(unicodedata.normalize("NFC", q).encode("utf-8"))


def _section_id(md_hash: str, ordinal: int) -> str:
    return "sec_" + _sha(f"{md_hash}|{ordinal}".encode())[:16]


def _auto_span_id(md_hash: str, start: int, end: int, kind: str) -> str:
    return "span_" + _sha(f"{md_hash}|{start}|{end}|{kind}".encode())[:24]


def _page_at(text: str, offset: int) -> int | None:
    import re
    page = None
    for m in re.finditer(r"(?m)^\{(\d+)\}-{10,}$", text):
        if m.end() <= offset:
            page = int(m.group(1)) + 1
    return page


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@dataclass(frozen=True)
class Md:
    markdown_id: str
    md_hash: str
    source_file_id: str
    sf_hash: str
    text: str


# --- the builder -------------------------------------------------------------

class _Builder:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.fx = V2Fixture(home=home)
        create_v2_db(home / "cache" / "cache.db", "cache", applied_at=APPLIED_AT)
        self.cache = sqlite3.connect(home / "cache" / "cache.db")
        self.cache.execute("PRAGMA foreign_keys=ON")
        self.db: dict[str, sqlite3.Connection] = {}
        self.pdfs: dict[str, str] = {}             # asset key -> sha
        self.mds: dict[str, Md] = {}               # asset key -> markdown
        self._sections: set[tuple[str, str]] = set()
        self._section_rows: dict[tuple[str, str], list[tuple[str, int, int]]] = {}

    # cache -------------------------------------------------------------------
    def event(self, object_type: str, object_id: str, event_type: str, detail: dict | None, at: str) -> None:
        self.cache.execute(
            "INSERT INTO cache_events (object_type, object_id, event_type, detail_json, created_at) "
            "VALUES (?,?,?,?,?)",
            (object_type, object_id, event_type,
             json.dumps(detail, sort_keys=True, separators=(",", ":")) if detail else None, at))

    def pdf(self, asset: Asset, at: str, *, acquisition: str = "upload",
            access: str = "user_supplied_private") -> str:
        data = minimal_pdf(asset.pdf_pages())
        sha = _sha(data)
        _write(self.home / "cache" / "pdfs" / f"{sha}.pdf", data)
        self.cache.execute(
            "INSERT INTO source_files (source_file_id, file_hash, file_type, access_class, storage_uri, "
            "acquisition_method, created_at, byte_size, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"sf_{sha}", sha, "pdf", access, f"pdfs/{sha}.pdf", acquisition, at, len(data), at))
        self.event("source_file", f"sf_{sha}", "source_file_ingested",
                   {"access_class": access, "acquisition_method": acquisition, "file_hash": sha,
                    "file_type": "pdf"}, at)
        self.pdfs[asset.key] = sha
        return sha

    def convert(self, asset: Asset, at: str, done: str | None, *, status: str = "success",
                variant: str = "marker", tag: str = "") -> tuple[str, Md | None]:
        sha = self.pdfs[asset.key]
        sf = f"sf_{sha}"
        name, package, version, config = {
            "marker": ("marker", "marker-pdf", "1.10.2", MARKER_CONFIG),
            "fake": ("marker", "marker-pdf", "fake-0.0.0", MARKER_CONFIG),
            "external": ("external_import", "external_import", "n/a", "{}"),
        }[variant]
        run_id = _id("conv", asset.key, status, variant, tag)
        fingerprint = _h("fp", name, version, config, sha)
        md: Md | None = None
        md_hash = None
        if status == "success":
            data = asset.markdown().encode("utf-8")
            md_hash = _sha(data)
            md = Md(f"md_{md_hash}", md_hash, sf, sha, data.decode("utf-8"))
        self.event("conversion_run", run_id, "markdown_imported" if variant == "external" else "conversion_started",
                   {"conversion_fingerprint": fingerprint, "source_file_id": sf}, at)
        self.cache.execute(
            "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, converter_name, "
            "converter_package, converter_version, python_version, config_json, conversion_fingerprint, "
            "llm_enabled, conversion_epistemic_type, run_status, warnings_json, error, markdown_hash, "
            "created_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, sf, sha, name, package, version, "3.13.5", config, fingerprint, 0,
             "local_deterministic_conversion", status,
             "[]" if status == "success" else None,
             "backend error: synthetic converter failure" if status == "failed" else None,
             md_hash, at, done))
        if md is not None:
            assert done is not None
            _write(self.home / "cache" / "markdown" / f"{md.md_hash}.md", md.text.encode("utf-8"))
            self.cache.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, markdown_hash, "
                "storage_uri, conversion_status, byte_size, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (md.markdown_id, run_id, sf, md.md_hash, f"markdown/{md.md_hash}.md", "success",
                 len(md.text.encode("utf-8")), done))
            if variant != "external":
                self.event("conversion_run", run_id, "conversion_succeeded", None, done)
            self.event("markdown_document", md.markdown_id, "markdown_written",
                       {"markdown_id": md.markdown_id, "warnings": []}, done)
            self.mds[asset.key] = md
        elif status == "failed":
            self.event("conversion_run", run_id, "conversion_failed", None, done or at)
        return run_id, md

    # projects ----------------------------------------------------------------
    def project(self, slug: str, created_at: str) -> sqlite3.Connection:
        pdir = self.home / "projects" / slug
        create_v2_db(pdir / "project.db", "project", applied_at=APPLIED_AT)
        _write(pdir / "project.yaml", (
            f"project_id: {slug}\nproject_name: {slug}\ndescription: null\nschema_version: 1\n"
            "storage_mode: local_first\nanswer_policy:\n  require_project_sources: true\n"
            "  allow_external_search_by_default: false\n  distinguish_source_claims_from_synthesis: true\n"
            f"created_at: '{created_at}'\n").encode("utf-8"))
        conn = sqlite3.connect(pdir / "project.db")
        conn.execute("PRAGMA foreign_keys=ON")
        self.db[slug] = conn
        return conn

    def work(self, slug: str, name: str, *, title: str | None, year: int | None, at: str,
             authors: Sequence[str] = (), venue: str | None = None, doi: str | None = None,
             openalex: str | None = None, arxiv: str | None = None, column_only: Sequence[str] = (),
             inclusion: str = "included", reason: str | None = "provided_pdf", seed: bool = False,
             abstract: str | None = None, work_id: str | None = None) -> WorkRef:
        key = f"{name}@{slug}"
        wid = work_id or _id("work", key)
        conn = self.db[slug]
        conn.execute(
            "INSERT INTO works (work_id, canonical_title, title_hash, authors, venue, year, doi, arxiv_id, "
            "openalex_id, created_at, abstract) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (wid, title, _title_hash(title), json.dumps(list(authors)) if authors else None, venue, year,
             doi, arxiv, openalex, at, abstract))
        for id_type, value in (("doi", doi), ("openalex", openalex), ("arxiv", arxiv)):
            if value and id_type not in column_only:
                conn.execute("INSERT INTO identifiers (work_id, id_type, id_value, resolution_source) "
                             "VALUES (?,?,?,?)", (wid, id_type, value, "user"))
        if inclusion:
            conn.execute(
                "INSERT INTO project_documents (work_id, inclusion_status, inclusion_reason, is_seed, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?)", (wid, inclusion, reason, int(seed), at, at))
        ref = WorkRef(key, slug, wid, title, year, doi, openalex, arxiv)
        self.fx.works[key] = ref
        return ref

    def attach(self, ref: WorkRef, asset_key: str, md: Md | None, at: str,
               acquisition: str = "manual_upload") -> WorkRef:
        sha = self.pdfs[asset_key]
        self.db[ref.slug].execute(
            "INSERT INTO work_source_files (work_id, source_file_id, file_hash, markdown_id, markdown_hash, "
            "acquisition_method, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (ref.work_id, f"sf_{sha}", sha, md.markdown_id if md else None, md.md_hash if md else None,
             acquisition, at, at))
        new = WorkRef(**{**vars(ref), "source_file_id": f"sf_{sha}",
                         "markdown_id": md.markdown_id if md else None})
        self.fx.works[ref.key] = new
        return new

    def held(self, slug: str, name: str, asset_key: str, *, at: str, md_key: str | None = None,
             **kw: Any) -> WorkRef:
        """A held work: a works row, its project document and its PDF bridge."""
        asset = ASSETS[asset_key]
        kw.setdefault("title", asset.title)
        kw.setdefault("authors", asset.authors)
        ref = self.work(slug, name, at=at, **kw)
        md = self.mds.get(md_key or asset_key) if md_key != "" else None
        return self.attach(ref, asset_key, md, at)

    # text ---------------------------------------------------------------------
    def sections(self, slug: str, ref: WorkRef, md: Md, at: str) -> None:
        if slug in NO_SECTIONS or (slug, md.markdown_id) in self._sections:
            return
        self._sections.add((slug, md.markdown_id))
        text = md.text
        heads = [("## Abstract", "abstract", "Abstract"), ("## 1 Introduction", "body", "1 Introduction"),
                 ("## References", "references", "References")]
        starts = [text.index(h) for h, _, _ in heads]
        rows = []
        for i, ((head, kind, label), start) in enumerate(zip(heads, starts)):
            end = starts[i + 1] if i + 1 < len(heads) else len(text)
            sid = _section_id(md.md_hash, i)
            rows.append((sid, start, end))
            self.db[slug].execute(
                "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, source_file_id, "
                "source_file_hash, work_id, parent_section_id, level, ordinal, heading_text, heading_path, "
                "section_kind, start_char, end_char, page_start, page_end, section_parser_version, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, md.markdown_id, md.md_hash, md.source_file_id, md.sf_hash, ref.work_id, None, 2, i,
                 label, label, kind, start, end, _page_at(text, start), _page_at(text, max(start, end - 1)),
                 "secparse-2", at))
        self._section_rows[(slug, md.markdown_id)] = rows
        if slug in PARAGRAPH_SPANS:
            pos = 0
            for block in text.split("\n\n"):
                start = text.index(block, pos) if block else pos
                pos = start + len(block)
                stripped = block.strip()
                if not stripped or stripped.startswith(("{", "#")):
                    continue
                s = text.index(stripped, start)
                self.span(slug, ref, md, s, s + len(stripped), at, span_id=None, kind="paragraph")

    def section_of(self, slug: str, md: Md, offset: int) -> str | None:
        for sid, start, end in self._section_rows.get((slug, md.markdown_id), []):
            if start <= offset < end:
                return sid
        return None

    def span(self, slug: str, ref: WorkRef, md: Md, start: int, end: int, at: str, *,
             span_id: str | None, kind: str = "manual") -> str:
        quote = md.text[start:end]
        sid = span_id or _auto_span_id(md.md_hash, start, end, kind)
        section = self.section_of(slug, md, start)
        self.db[slug].execute(
            "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, source_file_hash, "
            "work_id, section_id, start_char, end_char, exact_quote, quote_hash, span_kind, page_start, page_end, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, md.markdown_id, md.md_hash, md.source_file_id, md.sf_hash, ref.work_id, section, start, end,
             quote, _quote_hash(quote), kind, _page_at(md.text, start), _page_at(md.text, max(start, end - 1)),
             at))
        self.db[slug].execute(
            "INSERT INTO span_fts (quote_text, span_id, markdown_id, work_id, section_id) VALUES (?,?,?,?,?)",
            (quote, sid, md.markdown_id, ref.work_id, section))
        return sid

    def note(self, ref: WorkRef, md: Md, at: str, *, tag: str = "", flavor: str = "",
             schema_id: str = SCHEMA_ID) -> dict[str, Any]:
        """One extraction run + structured note + two claims, each anchored by a manual span."""
        slug = ref.slug
        conn = self.db[slug]
        self.sections(slug, ref, md, at)
        asset = next(a for a in ASSETS.values() if a.markdown() == md.text)
        run_id = _id("extr", ref.key, tag)
        note_id = _id("note", ref.key, tag)
        quotes = [("research_question", "research_question", asset.rq),
                  ("result", "results[0]", asset.result)]
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, schema_id, "
            "schema_version, prompt_version, model_name, provider, access_mode, temperature, run_status, "
            "input_tokens, output_tokens, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, ref.work_id, md.markdown_id, md.md_hash, schema_id, "1.1.0", "1.1.0", "fixture-model",
             "fixture", "local", 0.0, "success", 1200, 300, at))
        raw = {fkey: {"claim_text": q, "normalized_label": asset.topic, "status": "found",
                       "epistemic_type": None, "assertion_status": "stated", "inferred_explanation": None,
                       "confidence": 0.9, "exact_quote": q} for _, fkey, q in quotes}
        note_text = (f"Research question: {asset.rq}\nMain result: {asset.result}\n"
                     f"Reader's view: {flavor or 'a plain fixture summary'}.")
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, markdown_hash, "
            "schema_id, schema_version, prompt_version, archetype, raw_note_json, note_text, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (note_id, run_id, ref.work_id, md.markdown_id, md.md_hash, schema_id, "1.1.0", "1.1.0", "empirical",
             json.dumps(raw, sort_keys=True), note_text, at))
        conn.execute("INSERT INTO note_fts (note_id, note_text) VALUES (?,?)", (note_id, note_text))
        claim_ids, span_ids = [], []
        for i, (ctype, fkey, q) in enumerate(quotes):
            claim_id = _id("claim", ref.key, tag, i)
            conn.execute(
                "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, work_id, claim_type, "
                "field_key, normalized_label, claim_text, status, epistemic_type, assertion_status, confidence, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (claim_id, note_id, run_id, ref.work_id, ctype, fkey, asset.topic, q, "found", "llm_extracted",
                 "stated", 0.9, at))
            conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text) VALUES (?,?,?)",
                         (claim_id, asset.topic, q))
            start = md.text.index(q)
            span_id = self.span(slug, ref, md, start, start + len(q), at, span_id=_id("span", ref.key, tag, i))
            conn.execute("INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?,?,?,?)",
                         (claim_id, span_id, 0, at))
            claim_ids.append(claim_id)
            span_ids.append(span_id)
        return {"note_id": note_id, "extraction_run_id": run_id, "claim_ids": claim_ids, "span_ids": span_ids}

    def references(self, ref: WorkRef, md: Md, at: str,
                   entries: Sequence[tuple[str, str, str | None, WorkRef | None, float | None]]) -> list[str]:
        """``entries``: (raw text, resolution_status, resolution_source, resolved work, confidence)."""
        ids = []
        for ordinal, (raw, status, source, target, conf) in enumerate(entries, start=1):
            rid = _id("ref", ref.key, ordinal)
            author, rest = raw.split(" (", 1)
            year = int(rest[:4])
            parsed = {"first_author": author.split(",")[0], "year": year,
                      "title": rest.split(". ", 2)[1] if ". " in rest else None,
                      "doi": None, "arxiv": None, "ordinal": ordinal, "continued_author": False}
            self.db[ref.slug].execute(
                "INSERT INTO reference_entries (reference_id, citing_work_id, raw_reference_text, parsed_fields_json, "
                "resolved_work_id, resolution_status, resolution_source, confidence, markdown_id, markdown_hash, "
                "section_label, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, ref.work_id, raw, json.dumps(parsed, sort_keys=True), target.work_id if target else None,
                 status, source, conf, md.markdown_id, md.md_hash, "References", at))
            ids.append(rid)
        return ids

    def edge(self, slug: str, src: WorkRef, dst: WorkRef, provenance: str, conf: float, ref_id: str | None,
             run_id: str, at: str) -> None:
        self.db[slug].execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, provenance, confidence, reference_id, "
            "run_id, created_at) VALUES (?,?,?,?,?,?,?)",
            (src.work_id, dst.work_id, provenance, conf, ref_id, run_id, at))

    def review(self, slug: str, item_type: str, target_type: str, target_id: str, payload: dict, at: str, *,
               status: str = "open", action: str | None = None, resolved_at: str | None = None,
               tag: str = "") -> str:
        item_id = _id("rq", slug, item_type, target_id, tag)
        self.db[slug].execute(
            "INSERT INTO review_queue (item_id, item_type, target_type, target_id, payload, status, action, "
            "created_at, resolved_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, item_type, target_type, target_id, json.dumps(payload, sort_keys=True), status, action,
             at, resolved_at))
        return item_id

    def citation_review(self, slug: str, citing: WorkRef, ref_id: str, raw: str, status: str, at: str, *,
                        action: str | None, reject_reason: str | None = None,
                        rejected_record: dict | None = None) -> str:
        author, rest = raw.split(" (", 1)
        payload = {"kind": "citation_resolution", "status": status, "citing_work_id": citing.work_id,
                   "reference_id": ref_id, "raw": raw, "title": rest.split(". ", 2)[1], "year": int(rest[:4]),
                   "first_author": author.split(",")[0], "doi": None, "arxiv": None, "candidates": [],
                   "rejected_record": rejected_record, "reject_reason": reject_reason,
                   "run_id": "run-20260705T120000Z-0000c1te"}
        return self.review(slug, "citation_resolution", "reference_entry", ref_id, payload, at,
                           status="resolved" if action else "open", action=action,
                           resolved_at="2026-07-09T10:00:00+00:00" if action else None)

    def plant(self, kind: str, ids: Mapping[str, Any], expect: Mapping[str, Any], note: str = "") -> None:
        self.fx.items.append(Planted(kind, dict(ids), dict(expect), note))

    def answer(self, slug: str, key: str, at: str, cites: Sequence[tuple[WorkRef, Sequence[str]]],
               run_id: str | None = None) -> tuple[str, str]:
        question, text = ANSWERS_TEXT[key]
        answer_id = _id("ans", slug, key)
        citations = [{"work_id": w.work_id, "title": w.title, "year": w.year, "span_ids": list(spans),
                      "quote": None, "section": "body", "epistemic_type": "llm_extracted",
                      "assertion_status": "stated"} for w, spans in cites]
        envelope = {
            "answer_id": answer_id, "question": question, "query_type": "factual",
            "answer_category": "source_grounded", "answer_text": text, "citations": citations,
            "recommendations": [], "cited_work_ids": [w.work_id for w, _ in cites],
            "cited_span_ids": [s for _, spans in cites for s in spans], "insufficient_evidence": False,
            "retrieved_item_ids": [s for _, spans in cites for s in spans], "warnings": [],
            "mode": "project_only", "llm_provenance": None,
        }
        trace = {
            "trace_version": 1, "answer_id": answer_id, "question": question, "created_at": at,
            "seedgraph_version": "0.0.0-fixture", "boost_constants": {},
            "spec": {"question": question, "protocol_hint": "factual", "phrases": [], "work_tokens": [],
                     "concept_tokens": [], "work_id": None, "section": None, "claim_type": None, "lens": None,
                     "access_class": None, "mode": "project_only"},
            "path": "standard", "outcome": "retrieval_only", "degrade_reasons": [],
            "retrieved_count": len(envelope["retrieved_item_ids"]), "dispositions_exact": True,
            "candidates": [], "shown_evidence": None, "prompt_version": None, "prompt_sha256": None,
            "neighborhood": None,
        }
        base = Path("projects") / slug / ("runs/" + run_id + "/answers" if run_id else "answers")
        for suffix, obj in ((".json", envelope), (".trace.json", trace)):
            _write(self.home / base / f"{answer_id}{suffix}",
                   json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))
        return answer_id, (base / f"{answer_id}.json").as_posix()

    def close(self) -> None:
        for conn in [self.cache, *self.db.values()]:
            conn.commit()
            conn.close()


# --- the lens ----------------------------------------------------------------

LENS_ID = "plot_layout_v1"
LENS_YAML = """\
lens_id: plot_layout_v1
name: Plot Layout
scope: project
description: >
  Record how a fixture paper's panel is built: units, periods and treatment timing.

object_type: assumption

positive_anchors:
  - panel
  - treatment timing
  - balanced sample

negative_anchors:
  - literature summary

output_schema:
  unit_of_observation: string
  periods: string
  treatment_timing: string
  evidence_span_ids:
    type: array
  confidence: number
  notes: string

evidence_policy:
  require_evidence_span: true
  allow_inferred: true
  inferred_requires_explanation: true
  record_not_found: true
"""
#: ``LensDefinition.definition_hash()`` of :data:`LENS_YAML` under v2's lens code, frozen
#: here (a test re-derives it while the v2 code exists).
LENS_DEFINITION_HASH = "a0c5e0e5e66d26badc82249e135d0a0af4a57f613a1de2b7a9e9dee158595f61"


# --- the planted universe ----------------------------------------------------

def build_v2_home(root: Path) -> V2Fixture:
    """Write the synthetic v2 home into ``root`` (created; must be absent or empty)."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to build a v2 fixture into a non-empty folder: {root}")
    root.mkdir(parents=True, exist_ok=True)
    b = _Builder(root)
    try:
        _populate(b)
    finally:
        b.close()
    return b.fx


def _populate(b: _Builder) -> None:  # noqa: PLR0915 - one linear script of the fixture
    A = ASSETS
    T0 = "2026-07-01T09:00:00+00:00"

    # Cache: PDFs and conversions -------------------------------------------
    for key, at in [("overlap", "2026-07-01T09:00:00+00:00"), ("dbl", "2026-07-01T09:01:00+00:00"),
                    ("dbl_copy", "2026-07-01T09:02:00+00:00"), ("mix", "2026-07-01T09:03:00+00:00"),
                    ("shared", "2026-07-01T09:04:00+00:00"), ("foreign", "2026-07-01T09:05:00+00:00"),
                    ("victim", "2026-07-01T09:06:00+00:00"), ("noconv", "2026-07-01T09:07:00+00:00"),
                    ("lattice", "2026-07-01T09:08:00+00:00"), ("ambig", "2026-07-01T09:09:00+00:00"),
                    ("noid", "2026-07-01T09:10:00+00:00"), ("moji", "2026-07-01T09:11:00+00:00"),
                    ("mojititle", "2026-07-01T09:12:00+00:00"), ("wrongpdf", "2026-07-01T09:13:00+00:00"),
                    ("malcached", "2026-07-01T09:14:00+00:00"), ("maluncached", "2026-07-01T09:15:00+00:00"),
                    ("port_a", "2026-07-01T09:16:00+00:00"), ("port_b", "2026-07-01T09:17:00+00:00"),
                    ("auction", "2026-07-01T09:18:00+00:00"), ("unconv", "2026-07-01T09:19:00+00:00"),
                    ("fake", "2026-07-01T09:20:00+00:00"), ("unheld", "2026-07-01T09:21:00+00:00"),
                    ("scan", "2026-07-01T09:23:00+00:00")]:
        b.pdf(A[key], at, acquisition="open_access_fetch" if key in ("lattice", "victim") else "upload",
              access="open_access" if key in ("lattice", "victim") else "user_supplied_private")
    b.pdf(A["external"], "2026-07-01T09:22:00+00:00", acquisition="folder_import")

    for key in ["overlap", "dbl", "dbl_copy", "mix", "shared", "foreign", "victim", "lattice", "ambig", "noid",
                "moji", "mojititle", "wrongpdf", "malcached", "maluncached", "port_a", "port_b", "auction",
                "unheld", "scan"]:
        b.convert(A[key], "2026-07-02T08:00:00+00:00", "2026-07-02T08:05:00+00:00")
    _, _ = b.convert(A["fake"], "2026-07-02T08:00:00+00:00", "2026-07-02T08:01:00+00:00", variant="fake")
    b.convert(A["external"], "2026-07-02T08:00:00+00:00", "2026-07-02T08:00:01+00:00", variant="external")
    failed_noconv, _ = b.convert(A["noconv"], "2026-07-02T08:00:00+00:00", "2026-07-02T08:02:00+00:00",
                                 status="failed")
    failed_unconv, _ = b.convert(A["unconv"], "2026-07-02T08:00:00+00:00", "2026-07-02T08:02:00+00:00",
                                 status="failed")
    pending, _ = b.convert(A["overlap"], "2026-07-20T08:00:00+00:00", None, status="pending", tag="leftover")
    md = b.mds

    # Provider cache ----------------------------------------------------------
    for provider, rkey, response, at in [
        ("crossref", "by_doi:doi=10.5555/sg.malformed-1",
         {"doi": "10.5555/sg.malformed-1", "title": A["malcached"].title, "year": 2013,
          "authors": list(A["malcached"].authors)}, "2026-06-30T23:00:00+00:00"),
        ("openalex", "by_doi:doi=10.5555/sg.overlap",
         {"openalex_id": "W9100000001", "doi": "10.5555/sg.overlap", "title": A["overlap"].title, "year": 2019,
          "authors": list(A["overlap"].authors)}, "2026-06-30T23:01:00+00:00"),
        ("semantic_scholar", "by_doi:doi=10.5555/sg.wrong", [], "2026-06-30T23:02:00+00:00"),
        ("crossref", "by_title:title=a theory of lattice rents|year=2011",
         [{"doi": "10.5555/sg.lattice", "title": A["lattice"].title, "year": 2011}], "2026-06-30T23:03:00+00:00"),
    ]:
        b.cache.execute("INSERT INTO provider_cache (provider, request_key, response, fetched_at) VALUES (?,?,?,?)",
                        (provider, rkey, json.dumps(response, sort_keys=True), at))

    # Projects ------------------------------------------------------------------
    for slug, at in [("second_test", "2026-07-03T22:40:00+00:00"), ("land_value", "2026-09-14T13:50:00+00:00"),
                     ("test_proj", "2026-07-01T12:00:00+00:00"), ("oct2026test", "2026-10-14T14:00:00+00:00"),
                     ("dbg_probe", "2026-07-19T19:30:00+00:00"), ("dbg_probe2", "2026-07-19T20:05:00+00:00")]:
        b.project(slug, at)

    ST, LV, TP, OCT = "second_test", "land_value", "test_proj", "oct2026test"
    V = "Journal of Fixture Studies"

    # Overlapping paper: held in second_test and test_proj, diverging notes ---------
    ov_st = b.held(ST, "overlap", "overlap", at="2026-07-03T23:30:00+00:00", year=2019, venue=V,
                   doi="10.5555/sg.overlap", openalex="W9100000001", seed=True, reason="seed_document")
    ov_tp = b.held(TP, "overlap", "overlap", at="2026-07-01T12:10:00+00:00", year=2019, venue=V,
                   doi="10.5555/sg.overlap")
    n_ov_st = b.note(ov_st, md["overlap"], "2026-07-06T00:05:00+00:00", flavor="pruning is the main lever")
    n_ov_tp = b.note(ov_tp, md["overlap"], "2026-07-08T10:00:00+00:00",
                     flavor="soil quality matters more than pruning")
    b.plant("overlap_diverging_notes",
            {"work_keys": [ov_st.key, ov_tp.key], "note_ids": [n_ov_st["note_id"], n_ov_tp["note_id"]],
             "markdown_id": md["overlap"].markdown_id,
             "span_ids": {"second_test": n_ov_st["span_ids"], "test_proj": n_ov_tp["span_ids"]}},
            {"merged": True, "current_note_id": n_ov_tp["note_id"], "version_order": [n_ov_st["note_id"],
             n_ov_tp["note_id"]], "parity_class": "current_elsewhere", "graph": "diff_in_diff",
             "manual_spans_dedupe_to": [min(a, c) for a, c in zip(n_ov_st["span_ids"], n_ov_tp["span_ids"])]},
            "same DOI in two projects; test_proj's note is newer, so it becomes current in the DiD graph")

    # Cited-only paper in both projects ---------------------------------------------
    cited_st = b.work(ST, "cited", title="Survey of Irrigation Contracts", year=2008, authors=("Tunde Okafor",),
                      venue="Journal of Made-Up Agronomy", doi="10.5555/sg.cited", inclusion="metadata_only",
                      reason="citation_walk", at="2026-07-04T14:40:00+00:00")
    cited_tp = b.work(TP, "cited", title="Survey of Irrigation Contracts", year=2008, authors=("Tunde Okafor",),
                      doi="10.5555/sg.cited", inclusion="metadata_only", reason="citation_walk",
                      at="2026-07-01T12:20:00+00:00")

    # Reference lists of the overlapping paper (both parsed from the same markdown) ---
    refs_st = b.references(ov_st, md["overlap"], "2026-07-04T14:39:00+00:00", [
        (REF_CITED, "resolved", "doi", cited_st, 1.0),
        (REF_AMBIG, "ambiguous", None, None, None),
        (REF_SUSPECT, "suspect", None, None, None),
    ])
    refs_tp = b.references(ov_tp, md["overlap"], "2026-07-01T12:19:00+00:00", [
        (REF_CITED, "resolved", "doi", cited_tp, 1.0),
        (REF_AMBIG, "ambiguous", None, None, None),
        (REF_SUSPECT, "suspect", None, None, None),
        (REF_TP_ONLY, "unresolved", None, None, None),
    ])
    run_st, run_tp = "run-20260704T151200Z-0000f1x7", "run-20260701T124500Z-0000f1x7"
    b.edge(ST, ov_st, cited_st, "provider_reference", 1.0, None, run_st, "2026-07-04T14:39:41+00:00")
    b.edge(ST, ov_st, cited_st, "parsed_bibliography", 0.9, refs_st[0], run_st, "2026-07-04T14:39:42+00:00")
    b.edge(TP, ov_tp, cited_tp, "parsed_bibliography", 0.95, refs_tp[0], run_tp, "2026-07-01T12:09:27+00:00")
    b.plant("cited_only", {"work_keys": [cited_st.key, cited_tp.key], "doi": "10.5555/sg.cited"},
            {"carried": "cited_only", "merged": True, "graph_status": {"diff_in_diff": "frontier"},
             "citation_links_dedupe": {"parsed_bibliography": 0.95, "provider_reference": 1.0}},
            "metadata-only in both projects, target of edges from the overlapping paper")

    # Citation review decisions -----------------------------------------------------
    rej = b.citation_review(ST, ov_st, refs_st[2], REF_SUSPECT, "suspect", "2026-07-05T12:00:00+00:00",
                            action="reject", reject_reason="low_title_overlap",
                            rejected_record={"doi": "10.5555/sg.unrelated", "title": "Stone Arches of Old Bridges",
                                             "year": 1999})
    amb = b.citation_review(ST, ov_st, refs_st[1], REF_AMBIG, "ambiguous", "2026-07-05T12:01:00+00:00",
                            action="reject")
    tp_mapped = b.citation_review(TP, ov_tp, refs_tp[1], REF_AMBIG, "ambiguous", "2026-07-02T12:00:00+00:00",
                                  action="dismissed")
    tp_unmapped = b.citation_review(TP, ov_tp, refs_tp[3], REF_TP_ONLY, "unresolved", "2026-07-02T12:01:00+00:00",
                                    action="dismissed")

    # Double import within second_test: one PDF, two works, non-conflicting ids --------
    dbl_a = b.held(ST, "dbl_a", "dbl", at="2026-07-03T23:40:00+00:00", year=2014, venue=V,
                   doi="10.5555/sg.double")
    dbl_b = b.held(ST, "dbl_b", "dbl", at="2026-07-10T09:00:00+00:00", year=2014, openalex="W9100000002")
    n_dbl_a = b.note(dbl_a, md["dbl"], "2026-07-06T00:10:00+00:00", tag="a", flavor="first import")
    n_dbl_b = b.note(dbl_b, md["dbl"], "2026-07-11T00:10:00+00:00", tag="b", flavor="second import")
    dup_open = b.review(ST, "duplicate_candidate", "work", dbl_b.work_id,
                        {"kind": "duplicate_candidate", "reason": "title_collision",
                         "conflicting_work_ids": [dbl_a.work_id],
                         "incoming": {"openalex": "W9100000002", "title": dbl_b.title, "year": 2014,
                                      "authors": list(A["dbl"].authors)}}, "2026-07-10T09:00:00+00:00")
    b.plant("double_import", {"work_keys": [dbl_a.key, dbl_b.key], "source_file_id": dbl_a.source_file_id,
                              "note_ids": [n_dbl_a["note_id"], n_dbl_b["note_id"]]},
            {"merged": True, "merge_key": "pdf", "duplicate_decision": "rule",
             "version_order": [n_dbl_a["note_id"], n_dbl_b["note_id"]], "parity_class": "merged"},
            "same PDF, one work has a DOI and the other an OpenAlex id: no conflict, so a rule merge")
    # The re-downloaded copy (different PDF bytes) is held by no project: an unheld cached PDF.

    # Shared PDF with conflicting DOIs: a true mix-up -------------------------------
    mix_a = b.held(ST, "mix_a", "mix", at="2026-07-03T23:50:00+00:00", year=2017, venue=V,
                   doi="10.5555/sg.mix-a")
    mix_b = b.held(LV, "mix_b", "mix", at="2026-09-14T13:55:00+00:00", year=1987,
                   title="Ledger of Medieval Wheat Tariffs", authors=("Joana Ferreira",), doi="10.5555/sg.mix-b",
                   venue="Pretend Historical Review")
    n_mix_a = b.note(mix_a, md["mix"], "2026-07-06T00:20:00+00:00")
    n_mix_b = b.note(mix_b, md["mix"], "2026-09-14T15:00:00+00:00")
    refs_mix_b = b.references(mix_b, md["mix"], "2026-09-14T15:27:00+00:00",
                              [(REF_TOLLS, "ambiguous", None, None, None)])
    lv_open = b.citation_review(LV, mix_b, refs_mix_b[0], REF_TOLLS, "ambiguous", "2026-09-14T15:28:00+00:00",
                                action=None)
    b.plant("shared_pdf_conflicting_dois",
            {"work_keys": [mix_a.key, mix_b.key], "source_file_id": mix_a.source_file_id,
             "note_ids": [n_mix_a["note_id"], n_mix_b["note_id"]]},
            {"merged": False, "attention": "shared_pdf", "miss_side": mix_b.key,
             "pdf_check": {mix_a.key: "pass", mix_b.key: "miss"},
             "hides_notes": {mix_a.key: 0, mix_b.key: 1}, "parity_class": "hidden"},
            "two DOIs, one PDF; the PDF is mix_a's paper, mix_b's title is nowhere in it")

    # Shared PDF where one side has no id ------------------------------------------
    sh_a = b.held(TP, "shared_a", "shared", at="2026-07-01T12:30:00+00:00", year=2016, venue=V,
                  doi="10.5555/sg.shared")
    sh_b = b.held(OCT, "shared_b", "shared", at="2026-10-14T14:06:00+00:00", year=2016)
    b.note(sh_a, md["shared"], "2026-07-02T01:00:00+00:00")
    n_sh_b = b.note(sh_b, md["shared"], "2026-10-14T14:18:00+00:00")
    b.plant("shared_pdf_one_side_no_id", {"work_keys": [sh_a.key, sh_b.key], "source_file_id": sh_a.source_file_id,
                                          "note_ids": [n_sh_b["note_id"]]},
            {"merged": True, "merge_key": "pdf", "duplicate_decision": "rule", "graph": None},
            "the no-id work shares its PDF with an identified work in a non-conflicting group")

    # Work pointing at another PDF's markdown ---------------------------------------
    victim = b.held(ST, "victim", "victim", at="2026-07-03T23:55:00+00:00", year=2020, venue=V,
                    doi="10.5555/sg.victim")
    b.note(victim, md["victim"], "2026-07-06T00:30:00+00:00")
    foreign = b.held(ST, "foreign", "foreign", at="2026-07-04T00:00:00+00:00", year=2012, venue=V,
                     doi="10.5555/sg.foreign", md_key="victim")
    n_foreign = b.note(foreign, md["victim"], "2026-07-06T00:40:00+00:00", tag="on-foreign-md")
    b.plant("foreign_markdown",
            {"work_key": foreign.key, "note_id": n_foreign["note_id"], "pointed_markdown_id": md["victim"].markdown_id,
             "own_markdown_id": md["foreign"].markdown_id},
            {"relink_to": md["foreign"].markdown_id, "version_label": "other_markdown", "parity_class": "relinked"},
            "the bridge row names the victim paper's markdown; the work's own PDF has a good markdown")
    noconv = b.held(LV, "noconv", "noconv", at="2026-09-14T14:00:00+00:00", year=2010, venue=V,
                    doi="10.5555/sg.foreign-noconv", md_key="victim")
    n_noconv = b.note(noconv, md["victim"], "2026-09-14T15:05:00+00:00", tag="on-foreign-md")
    b.plant("foreign_markdown_unconverted",
            {"work_key": noconv.key, "note_id": n_noconv["note_id"], "pointed_markdown_id": md["victim"].markdown_id,
             "source_file_id": noconv.source_file_id},
            {"queue_reason": "relink_failed", "version_label": "other_markdown", "parity_class": "relinked"},
            "foreign markdown pointer, and the work's own PDF only has a failed conversion")
    b.plant("conversion_failed", {"conversion_run_ids": [failed_noconv, failed_unconv],
                                  "source_file_id": f"sf_{b.pdfs['unconv']}"},
            {"carried": True, "queue_reason": "not_converted"},
            "failed marker runs are carried; the PDF without any markdown goes on the conversion queue")

    # Identified paper and a no-id work that matches it by title + year --------------
    lattice = b.held(ST, "lattice", "lattice", at="2026-07-04T00:10:00+00:00", year=2011, venue=V,
                     doi="10.5555/sg.lattice", openalex="W9100000003")
    b.note(lattice, md["lattice"], "2026-07-06T00:50:00+00:00")
    refs_lattice = b.references(lattice, md["lattice"], "2026-07-04T14:39:30+00:00",
                                [(REF_OVERLAP, "resolved", "title_year", ov_st, 0.92)])
    b.edge(ST, lattice, ov_st, "parsed_bibliography", 0.92, refs_lattice[0], run_st, "2026-07-04T14:39:43+00:00")
    noid_match = b.work(LV, "noid_match", title="A theory of lattice rents.", year=2011,
                        authors=("H. Marchetti",), inclusion="metadata_only", reason="citation_walk",
                        at="2026-09-14T14:05:00+00:00")
    b.plant("noid_title_year_match", {"work_key": noid_match.key, "matches": lattice.key},
            {"merged": True, "merge_key": "title_year", "duplicate_decision": "rule",
             "graph_status": {"land_value": "frontier"}},
            "no-id frontier work; normalize_title + year equal one identified paper only")

    # Ambiguous no-id title match ---------------------------------------------------
    mk1 = b.work(ST, "market_1", title="Notes on Matching Market Design", year=2015, authors=("Sana Qureshi",),
                 doi="10.5555/sg.market-1", inclusion="metadata_only", reason="citation_walk",
                 at="2026-07-04T00:20:00+00:00")
    mk2 = b.work(LV, "market_2", title="Notes on Matching Market Design", year=2015, authors=("Sana Qureshi",),
                 doi="10.5555/sg.market-2", inclusion="metadata_only", reason="citation_walk",
                 at="2026-09-14T14:10:00+00:00")
    ambig = b.held(ST, "noid_ambig", "ambig", at="2026-07-04T00:25:00+00:00", year=2015)
    b.note(ambig, md["ambig"], "2026-07-06T01:00:00+00:00")
    b.plant("noid_ambiguous", {"work_key": ambig.key, "candidates": [mk1.key, mk2.key]},
            {"merged": False, "attention": "identity_check", "reason": "ambiguous_title_year"},
            "held no-id work whose title + year match two identified papers with different DOIs")

    # No-id paper matching nothing (with a near-miss candidate) -----------------------
    near = b.work(LV, "near_miss", title="Hedonic Prices of Lighthouse Services in Coastal Towns", year=2006,
                  authors=("Tomas Varga",), doi="10.5555/sg.lighthouse", inclusion="metadata_only",
                  reason="citation_walk", at="2026-09-14T14:15:00+00:00")
    noid = b.held(ST, "noid_none", "noid", at="2026-07-04T00:30:00+00:00", year=2004)
    n_noid = b.note(noid, md["noid"], "2026-07-06T01:10:00+00:00")
    b.plant("noid_unmatched", {"work_key": noid.key, "near_miss": near.key, "note_id": n_noid["note_id"]},
            {"merged": False, "attention": "identity_check", "reason": "no_match",
             "paper_id_key": f"pdf:{b.pdfs['noid']}"},
            "held no-id work; nothing matches, the nearest title is a different year and wording")

    # Mojibake markdown and a mojibake title ----------------------------------------
    moji = b.held(ST, "moji", "moji", at="2026-07-04T00:40:00+00:00", year=2018, venue=V,
                  title="Specification Tests for Efficient Estimators of Difference Models", doi="10.5555/sg.moji")
    b.note(moji, md["moji"], "2026-07-06T01:20:00+00:00")
    b.plant("mojibake_markdown", {"work_key": moji.key, "markdown_id": md["moji"].markdown_id},
            {"pdf_check": "pass", "matched_on": "folded_variant"},
            "markdown and PDF text use Marker's lookalikes: Ö=fi, '¢ '=ffi, ¤=ff; the title is clean")
    mojit = b.held(TP, "moji_title", "mojititle", at="2026-07-01T12:40:00+00:00", year=2021, venue=V,
                   title="IdentiÖcation under ConÖnement Constraints", doi="10.5555/sg.moji-title")
    b.note(mojit, md["mojititle"], "2026-07-02T01:10:00+00:00")
    b.plant("mojibake_title", {"work_key": mojit.key}, {"pdf_check": "pass", "matched_on": "folded_variant"},
            "the v2 title carries Ö for fi; the PDF and markdown are clean")

    # Wrong-PDF paper -----------------------------------------------------------------
    wrong = b.held(LV, "wrong", "wrongpdf", at="2026-09-14T14:20:00+00:00", year=2009, venue=V,
                   title="Glacier Retreat and Alpine Tourism Revenue", authors=("Bruno Keller",),
                   doi="10.5555/sg.wrong")
    n_wrong = b.note(wrong, md["wrongpdf"], "2026-09-14T15:10:00+00:00")
    b.plant("wrong_pdf", {"work_key": wrong.key, "note_id": n_wrong["note_id"]},
            {"attention": "pdf_title_miss", "hides_notes": 1, "pdf_check": "miss", "parity_class": "hidden"},
            "the PDF (and its markdown) is a different paper; no title word, no DOI")

    # Malformed titles -------------------------------------------------------------
    malc = b.held(ST, "malformed_cached", "malcached", at="2026-07-04T00:50:00+00:00", year=2013,
                  title="draft_final_v3.pdf", doi="10.5555/sg.malformed-1")
    b.note(malc, md["malcached"], "2026-07-06T01:30:00+00:00")
    b.plant("malformed_title_cached", {"work_key": malc.key},
            {"title": A["malcached"].title, "title_source": "provider_cache", "attention": None},
            "filename-like title; v2's provider_cache holds a crossref by_doi response with the real title")
    malu = b.held(TP, "malformed_uncached", "maluncached", at="2026-07-01T12:50:00+00:00", year=2007,
                  title="9f8e7d6c5b4a39281706f5e4d3c2b1a0", doi="10.5555/sg.malformed-2")
    b.note(malu, md["maluncached"], "2026-07-02T01:20:00+00:00")
    b.plant("malformed_title_uncached", {"work_key": malu.key},
            {"attention": "malformed_title", "title_source": None},
            "hash-like title and no provider_cache row for its DOI")
    malf = b.work(ST, "malformed_frontier", title="scan_0042_refs.pdf", year=2011, authors=("Ola Berg",),
                  doi="10.5555/sg.malformed-3", inclusion="metadata_only", reason="citation_walk",
                  at="2026-07-04T00:55:00+00:00")
    b.plant("malformed_title_unheld", {"work_key": malf.key},
            {"attention": None, "counted_in": "titles.malformed_cited_only"},
            "filename-like title on a frontier paper with no PDF: left to the stage-2 metadata refresh")

    # Duplicate verdicts in test_proj -------------------------------------------------
    port_a = b.held(TP, "port_a", "port_a", at="2026-07-01T13:00:00+00:00", year=2018, venue=V,
                    doi="10.5555/sg.port")
    port_b = b.held(TP, "port_b", "port_b", at="2026-07-01T13:05:00+00:00", year=2018)
    b.note(port_a, md["port_a"], "2026-07-02T01:30:00+00:00")
    b.note(port_b, md["port_b"], "2026-07-02T01:35:00+00:00")
    refs_port = b.references(port_a, md["port_a"], "2026-07-01T13:10:00+00:00",
                             [(REF_PORT, "suspect", None, None, None)])
    port_dec = b.citation_review(TP, port_a, refs_port[0], REF_PORT, "suspect", "2026-07-02T12:02:00+00:00",
                                 action="dismissed", reject_reason="refwork_doi_denylist",
                                 rejected_record={"doi": "10.5555/sg.denied", "title": "Harbour queues", "year": 2016})
    nd = b.review(TP, "duplicate_candidate", "work", port_b.work_id,
                  {"kind": "duplicate_candidate", "reason": "title_collision", "conflicting_work_ids": [port_a.work_id],
                   "incoming": {"title": port_b.title, "year": 2018, "authors": list(A["port_b"].authors)}},
                  "2026-07-01T13:05:00+00:00", status="resolved", action="dismissed",
                  resolved_at="2026-07-03T09:00:00+00:00")
    b.plant("duplicate_verdict_different", {"item_id": nd, "work_keys": [port_a.key, port_b.key]},
            {"duplicate_decision": "different", "decided_by": "researcher", "merged": False,
             "rule_skipped": "title_year", "attention": "identity_check"},
            "v2 action 'dismissed' on a resolved duplicate_candidate = 'not duplicates'; the no-id work would "
            "otherwise merge by title + year")
    auc_a = b.held(TP, "auction_pub", "auction", at="2026-07-01T13:20:00+00:00", year=2021, venue=V,
                   doi="10.5555/sg.pub", arxiv="2101.00001")
    b.note(auc_a, md["auction"], "2026-07-02T01:40:00+00:00")
    auc_b = b.work(TP, "auction_pre", title=A["auction"].title, year=2021, authors=A["auction"].authors,
                   doi="10.5555/sg.pre", arxiv="2101.00001", column_only=("arxiv",), inclusion="metadata_only",
                   reason="citation_walk", at="2026-07-01T13:25:00+00:00")
    same = b.review(TP, "duplicate_candidate", "work", auc_b.work_id,
                    {"kind": "duplicate_candidate", "reason": "id_disagreement", "conflicting_work_ids": [auc_a.work_id],
                     "incoming": {"doi": "10.5555/sg.pre", "arxiv": "2101.00001", "title": auc_b.title, "year": 2021,
                                  "authors": list(A["auction"].authors)}},
                    "2026-07-01T13:25:00+00:00", status="resolved", action="merge",
                    resolved_at="2026-07-03T09:05:00+00:00")
    b.plant("duplicate_verdict_same", {"item_id": same, "work_keys": [auc_a.key, auc_b.key]},
            {"duplicate_decision": "same", "decided_by": "researcher", "carried": "subject_of_decision"},
            "preprint/published pair (shared arXiv id, two DOIs); the researcher resolved it as 'merge'")
    b.plant("citation_review_decided",
            {"items": [[ST, rej, "reject"], [ST, amb, "reject"], [TP, tp_mapped, "dismissed"],
                       [TP, tp_unmapped, "dismissed"], [TP, port_dec, "dismissed"]],
             "reference_ids": {"second_test": refs_st, "test_proj": refs_tp}},
            {"chosen_list": {"overlap": ST}, "mapped": {tp_mapped: refs_st[1]}, "unmapped": [tp_unmapped],
             "decision": "reject"},
            "decided citation reviews; test_proj's decision on the overlap list maps onto second_test's list by "
            "raw text + ordinal, one has no counterpart")
    b.plant("citation_review_open", {"item": [LV, lv_open]}, {"carried": False})

    # Unconverted, fake converter, external import, scan, unheld ----------------------
    unconv = b.held(TP, "unconv", "unconv", at="2026-07-01T13:30:00+00:00", year=2011, venue=V,
                    doi="10.5555/sg.unconverted", md_key="")
    fake = b.held(OCT, "fake", "fake", at="2026-10-14T14:10:00+00:00", year=2022, venue=V, doi="10.5555/sg.fake")
    b.note(fake, md["fake"], "2026-10-14T14:20:00+00:00")
    b.plant("fake_converter_markdown", {"work_key": fake.key, "markdown_id": md["fake"].markdown_id},
            {"attention": "test_converter_markdown", "carried": True})
    ext = b.held(LV, "external", "external", at="2026-09-14T14:25:00+00:00", year=2019, venue=V,
                 doi="10.5555/sg.external", seed=True, reason="seed_document")
    b.note(ext, md["external"], "2026-09-14T15:15:00+00:00")
    b.plant("external_markdown_no_markers", {"work_key": ext.key, "markdown_id": md["external"].markdown_id},
            {"first_pages": "first_8000_chars", "pdf_check": "pass"})
    scan = b.held(LV, "scan", "scan", at="2026-09-14T14:30:00+00:00", year=2015, venue=V, doi="10.5555/sg.scan")
    b.note(scan, md["scan"], "2026-09-14T15:20:00+00:00")
    b.plant("pdf_no_text_layer", {"work_key": scan.key, "source_file_id": scan.source_file_id},
            {"pdf_check_pdf_only": "unknown", "pdf_check": "pass"})
    b.plant("unheld_pdf", {"source_file_id": f"sf_{b.pdfs['unheld']}", "markdown_id": md["unheld"].markdown_id},
            {"held_paper": True, "title": None, "attention": "metadata_needed", "pdf_check": "unknown"})
    b.plant("unheld_pdf_second_copy", {"source_file_id": f"sf_{b.pdfs['dbl_copy']}",
                                       "markdown_id": md["dbl_copy"].markdown_id},
            {"held_paper": True, "title": None, "attention": "metadata_needed"},
            "a second download of the double-imported paper that no project points at")
    b.plant("conversion_pending", {"conversion_run_id": pending, "source_file_id": ov_st.source_file_id},
            {"carried": False})

    # Excluded work in the land_value graph, unreferenced works not carried ------------
    lv_excl = b.work(LV, "excluded", title="Toll Roads in the Bronze Age Imagination", year=2003,
                     authors=("Ida Brandt",), doi="10.5555/sg.excluded", inclusion="excluded",
                     reason="beyond_generation_1_cite_parse", at="2026-09-14T14:35:00+00:00")
    b.edge(LV, wrong, lv_excl, "provider_reference", 1.0, None, "run-20260914T160300Z-0000e1c1",
           "2026-09-14T15:27:18+00:00")
    b.plant("excluded_in_graph", {"work_key": lv_excl.key}, {"graph_status": {"land_value": "excluded"},
                                                            "carried": "cited_only"})
    oct_unref = b.work(OCT, "unreferenced", title="Kelp Harvest Cycles", year=2001, authors=("Rua Tane",),
                       doi="10.5555/sg.oct-unref", openalex="W9100000009", inclusion="metadata_only",
                       reason="citation_walk", at="2026-10-14T14:12:00+00:00")
    tp_excl = b.work(TP, "excluded", title="Front matter", year=None, inclusion="excluded",
                     reason="parse_artifact:no-title stub", at="2026-07-01T13:40:00+00:00")
    b.plant("metadata_only_unreferenced", {"work_key": oct_unref.key, "also": [tp_excl.key]},
            {"carried": False, "reason": "unreferenced_metadata_only_in_non_graph_project"})
    b.plant("held_unconverted", {"work_key": unconv.key}, {"queue_reason": "not_converted"})

    # Concept constraints --------------------------------------------------------
    for slug, kind, a, c, source, at in [
        (ST, "must_link", "difference in differences", "diff-in-diff", "llm_merge", "2026-07-08T00:00:00+00:00"),
        (ST, "must_link", "parallel trends", "common trends", "llm_merge", "2026-07-08T00:00:01+00:00"),
        (ST, "must_link", "event study", "staggered adoption", "user", "2026-07-08T00:00:02+00:00"),
        (ST, "must_link", "synthetic control", "synthetic controls", "user", "2026-07-08T00:00:03+00:00"),
        (TP, "must_link", "diff-in-diff", "difference in differences", "user", "2026-07-03T00:00:00+00:00"),
        (TP, "cannot_link", "staggered adoption", "event study", "user", "2026-07-03T00:00:01+00:00"),
        (TP, "cannot_link", "matching", "synthetic control", "user", "2026-07-03T00:00:02+00:00"),
    ]:
        b.db[slug].execute("INSERT INTO concept_constraints (kind, label_a, label_b, source, created_at) "
                           "VALUES (?,?,?,?,?)", (kind, a, c, source, at))
    b.plant("concept_llm_merge", {"slug": ST, "pair": ["common trends", "parallel trends"]},
            {"kind": "merge", "decided_by": "agent", "state": "pending_apply"})
    b.plant("concept_user_must_link", {"slug": ST, "pair": ["synthetic control", "synthetic controls"]},
            {"kind": "merge", "decided_by": "researcher", "state": "pending_apply"})
    b.plant("concept_user_cannot_link", {"slug": TP, "pair": ["matching", "synthetic control"]},
            {"kind": "never_merge", "decided_by": "researcher", "state": "pending_apply"})
    b.plant("concept_contradiction", {"pair": ["event study", "staggered adoption"],
                                      "projects": {ST: "must_link", TP: "cannot_link"}},
            {"attention": "concept_contradiction", "rows_kept": 2})
    b.plant("concept_agent_and_researcher_same_pair", {"pair": ["diff-in-diff", "difference in differences"],
                                                        "projects": {ST: "llm_merge", TP: "user"}},
            {"kind": "merge", "decided_by": "researcher", "rows": 1})
    # Dropped concept tables + test_proj's resolved concept-merge reviews (not carried twice).
    b.db[ST].execute("INSERT INTO concepts (concept_id, normalized_label, canonical_label, concept_type, "
                     "created_at, updated_at) VALUES (?,?,?,?,?,?)",
                     ("concept_" + _h("c1")[:32], "orchard yield", "orchard yield", "outcome", T0, T0))
    b.db[ST].execute("INSERT INTO concept_aliases (concept_id, alias_label, fold_reason, created_at) "
                     "VALUES (?,?,?,?)", ("concept_" + _h("c1")[:32], "orchard yields", "exact_key", T0))
    b.db[ST].execute("INSERT INTO claim_concepts (claim_id, concept_id, work_id, created_at) VALUES (?,?,?,?)",
                     (n_ov_st["claim_ids"][1], "concept_" + _h("c1")[:32], ov_st.work_id, T0))
    b.review(TP, "concept_merge_candidate", "Concept", "concept_" + _h("c2")[:32],
             {"kind": "concept_merge_candidate", "canonical_label": "difference in differences",
              "member_label": "diff-in-diff", "canonical_concept_id": "concept_" + _h("c2")[:32],
              "member_concept_id": "concept_" + _h("c3")[:32], "canonical_surface": "difference in differences",
              "member_surface": "diff-in-diff", "confidence": None, "run_id": "run-20260702T000000Z-0000c0nc"},
             "2026-07-02T00:00:00+00:00", status="resolved", action="approve", resolved_at="2026-07-03T00:00:00+00:00")

    # Lens with outputs (second_test) ----------------------------------------------
    _write(b.home / "projects" / ST / "lenses" / f"{LENS_ID}.yaml", LENS_YAML.encode("utf-8"))
    b.db[ST].execute("INSERT INTO lenses (lens_id, name, scope, object_type, status, yaml_path, definition_hash, "
                     "definition_yaml, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (LENS_ID, "Plot Layout", "project", "assumption", "draft",
                      f"lenses/{LENS_ID}.yaml", LENS_DEFINITION_HASH, None, "2026-07-12T00:00:00+00:00",
                      "2026-07-12T00:00:00+00:00"))
    lens_outputs = []
    for ref, status in [(ov_st, "found"), (lattice, "not_found")]:
        mdr = md["overlap"] if ref is ov_st else md["lattice"]
        asset = A["overlap"] if ref is ov_st else A["lattice"]
        run_id = _id("extr", ref.key, "lens")
        at = "2026-07-12T01:00:00+00:00"
        b.db[ST].execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, schema_id, lens_id, "
            "schema_version, prompt_version, model_name, provider, access_mode, run_status, created_at, "
            "lens_definition_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, ref.work_id, mdr.markdown_id, mdr.md_hash, None, LENS_ID, "lens_schema_v1", "lens_extraction_v1",
             "fixture-model", "fixture", "local", "success", at, LENS_DEFINITION_HASH))
        claim_id = None
        span_ids: list[str] = []
        if status == "found":
            claim_id = _id("claim", ref.key, "lens")
            q = asset.sample
            start = mdr.text.index(q)
            span = b.span(ST, ref, mdr, start, start + len(q), at, span_id=_id("span", ref.key, "lens"))
            span_ids = [span]
            b.db[ST].execute(
                "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, work_id, claim_type, "
                "field_key, normalized_label, claim_text, status, epistemic_type, assertion_status, confidence, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (claim_id, None, run_id, ref.work_id, "data", "plot_layout_v1[0]", "plot layout", q, "found",
                 "llm_extracted", "stated", 0.8, at))
            # v2's lens runner always indexes the claim it writes (lenses/runner.py)
            b.db[ST].execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text) VALUES (?,?,?)",
                             (claim_id, "plot layout", q))
            b.db[ST].execute("INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?,?,?,?)",
                             (claim_id, span, 0, at))
        lo_id = _id("lensout", ref.key)
        fields = {"status": status, "epistemic_type": "llm_extracted", "assertion_status": "stated",
                  "matched_anchor": "plot" if claim_id else None,
                  "candidate_quote": asset.sample if claim_id else None, "evidence_span_ids": span_ids,
                  "note": None}
        b.db[ST].execute(
            "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, work_id, status, claim_id, "
            "confidence, fields_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (lo_id, run_id, LENS_ID, ref.work_id, status, claim_id, 0.8 if claim_id else None,
             json.dumps(fields, sort_keys=True), at))
        lens_outputs.append(lo_id)
    lens_record = b.review(ST, "lens_record", "lens_output", lens_outputs[1],
                           {"kind": "lens_record", "lens_id": LENS_ID, "status": "not_found"},
                           "2026-07-12T01:00:00+00:00")
    b.plant("lens_with_outputs", {"lens_id": LENS_ID, "definition_hash": LENS_DEFINITION_HASH,
                                  "lens_output_ids": lens_outputs, "yaml": f"projects/{ST}/lenses/{LENS_ID}.yaml"},
            {"lens_definitions": 1, "graph_lenses": {"diff_in_diff": [LENS_ID]}, "lens_outputs_carried": 2})

    # Unmatched-upload review items, dropped tables -----------------------------------
    up_open = b.review(ST, "unmatched_upload", "source_file", f"sf_{b.pdfs['unheld']}",
                       {"kind": "unmatched_upload", "file_hash": b.pdfs["unheld"]}, "2026-07-15T00:00:00+00:00")
    b.plant("open_review_items", {"items": [[ST, dup_open], [ST, lens_record], [ST, up_open], [LV, lv_open]]},
            {"carried": False})
    b.db[ST].execute("INSERT INTO llm_usage_events (usage_event_id, task_type, provider, model, input_tokens, "
                     "output_tokens, created_at) VALUES (?,?,?,?,?,?,?)",
                     ("ue_" + _h("ue1")[:32], "extraction", "fixture", "fixture-model", 1200, 300, T0))
    b.db[ST].execute("INSERT INTO llm_key_refs (provider, access_mode, key_source, key_reference, created_at) "
                     "VALUES (?,?,?,?,?)", ("fixture", "api_key", "environment", "FIXTURE_API_KEY", T0))
    b.db[ST].execute("INSERT INTO audit_records (audit_id, audit_type, subject_type, subject_id, created_at) "
                     "VALUES (?,?,?,?,?)", ("audit_" + _h("a1")[:32], "claim_audit", "claim",
                                            n_ov_st["claim_ids"][0], T0))
    edge_id = "pge_" + _h("e1")[:32]
    b.db[ST].execute("INSERT INTO project_graph_edges (edge_id, source_node_type, source_node_id, target_node_type, "
                     "target_node_id, edge_type, epistemic_type, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     (edge_id, "Work", ov_st.work_id, "Work", lattice.work_id, "shares_concept", "deterministic", T0))
    b.db[ST].execute("INSERT INTO edge_spans (edge_id, span_id) VALUES (?,?)", (edge_id, n_ov_st["span_ids"][0]))
    b.plant("dropped_tables", {"tables": ["llm_usage_events", "llm_key_refs", "audit_records", "project_graph_edges",
                                          "edge_spans", "concepts", "concept_aliases", "claim_concepts"]},
            {"carried": False})

    # Saved answers ------------------------------------------------------------------
    for slug, key, at, cites in [
        (ST, "orchard", "2026-07-15T10:00:00+00:00", [(ov_st, n_ov_st["span_ids"][:1])]),
        (ST, "credit", "2026-07-16T10:00:00+00:00", [(foreign, n_foreign["span_ids"][:1])]),
        (TP, "ports", "2026-07-04T10:00:00+00:00", [(b.fx.works[port_a.key], [])]),
    ]:
        answer_id, path = b.answer(slug, key, at, cites)
        b.plant("saved_answer", {"slug": slug, "answer_id": answer_id, "path": path},
                {"export": f"v2_answers/{slug}/{answer_id}.md"})
    answer_id, path = b.answer(ST, "runscoped", "2026-07-17T10:00:00+00:00", [(noid, n_noid["span_ids"][:1])],
                               run_id="run-20260717T100000Z-0000a5c0")
    b.plant("run_scoped_answer", {"slug": ST, "answer_id": answer_id, "path": path}, {"carried": False})

    # dbg_probe / dbg_probe2: synthetic leftovers, no PDFs, soft markdown refs -------
    for slug in ("dbg_probe", "dbg_probe2"):
        conn = b.db[slug]
        at = "2026-07-19T19:31:00+00:00"
        for wid, title, status in [("w_a", "Probe paper A", "included"), ("w_b", "Probe paper B", "included"),
                                   ("w_c", "Probe cited C", "metadata_only"), ("w_d", "Probe excluded D", "excluded")]:
            b.work(slug, wid, title=title, year=2000, at=at, inclusion=status, reason=None, work_id=wid)
        conn.execute("INSERT INTO document_sections (section_id, markdown_id, markdown_hash, source_file_id, "
                     "source_file_hash, work_id, level, ordinal, section_kind, start_char, end_char, "
                     "section_parser_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     ("sec_a", "md_probe", "probe", "sf_probe", "probe", "w_a", 1, 0, "body", 0, 40, "sp1", at))
        conn.execute("INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
                     "schema_version, prompt_version, run_status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     ("r_a", "w_a", "md_probe", "probe", "v1", "p1", "success", at))
        conn.execute("INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, markdown_hash, "
                     "schema_version, prompt_version, raw_note_json, note_text, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ("n_a", "r_a", "w_a", "md_probe", "probe", "v1", "p1", "{}", "probe note text", at))
        conn.execute("INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, work_id, "
                     "claim_type, field_key, status, epistemic_type, assertion_status, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ("c_a", "n_a", "r_a", "w_a", "identification_assumption", "assumptions[0]", "found",
                      "llm_extracted", "stated", at))
        conn.execute("INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
                     "source_file_hash, work_id, section_id, start_char, end_char, exact_quote, quote_hash, "
                     "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     ("s_a", "md_probe", "probe", "sf_probe", "probe", "w_a", "sec_a", 0, 6, "probe!",
                      _quote_hash("probe!"), at))
        conn.execute("INSERT INTO claim_spans (claim_id, span_id, created_at) VALUES (?,?,?)", ("c_a", "s_a", at))
        conn.execute("INSERT INTO citation_edges (source_work_id, target_work_id, provenance, confidence, run_id, "
                     "created_at) VALUES (?,?,?,?,?,?)", ("w_a", "w_c", "provider_reference", 1.0, "cite1", at))
    b.plant("dbg_project", {"slugs": ["dbg_probe", "dbg_probe2"]},
            {"carried": False, "reason": "synthetic probe projects, no PDFs"})
