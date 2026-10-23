"""The migration plan (design §6.1 step 4): a pure function from snapshot to plan.

:func:`build_plan` turns a :class:`~seedgraph.migrate.snapshot.Snapshot` into a
:class:`MigrationPlan`: every library row already in v3 shape (sorted by primary
key), each graph's membership and lenses, the blobs to copy, the attention items to
raise, a :class:`Ledger` giving every v2 row its disposition, report sections, and
errors. Nothing here touches disk except reading the v2 markdown/PDF blobs the
snapshot points at (title checks, anchor checks). The same snapshot gives an equal
plan.

Rules (design §4, §6, §7.3, plus the rulings recorded in the S6 report):

* identity by :func:`seedgraph.library.identity.plan_identity` over every carried v2
  work; every cached PDF joins the library (unheld ones as papers with no metadata);
* second_test -> graph ``did`` (test_proj's held papers folded in as members),
  land_value -> ``land-value``; other projects are library-only; ``dbg_*`` carry nothing;
* notes are versions ordered by ``(created_at, project precedence, note_id)``;
* sections: newest parser version per markdown, then the highest-precedence project;
  spans deduped by id (auto) or by position (manual, canonical = smallest id);
* one reference list per paper; decisions on other lists mapped by raw text + ordinal;
* rule merges go to ``paper_merge_members``; researcher verdicts on papers that stay
  separate go to ``duplicate_decisions``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from seedgraph.anchor import assert_invariant
from seedgraph.library.identity import (
    MERGE_ID_TYPES, IdentityPlan, PlannedPaper, WorkRecord, WorkRef, mint_paper_id, normalize_ids,
    plan_identity, project_rank,
)
from seedgraph.project.identity import normalize_title
from seedgraph.semantic.merge import normalize as normalize_label

from . import defects
from .defects import AttentionSpec
from .snapshot import Snapshot

#: v2 project -> graph slug (design §6.2).
GRAPH_PROJECTS: dict[str, str] = {"second_test": "did", "land_value": "land-value"}
#: v2 project whose held papers fold into a graph as members.
FOLD_INTO: dict[str, str] = {"test_proj": "did"}
STATUS_MAP = {"included": "member", "metadata_only": "frontier", "excluded": "excluded"}
STATUS_PRIORITY = {"member": 0, "excluded": 1, "frontier": 2}
#: v2 tables the migration drops on purpose (design §2.1 "Dropped, not carried").
DROPPED_TABLES = frozenset({"llm_usage_events", "llm_key_refs", "audit_records", "project_graph_edges",
                            "edge_spans", "concepts", "concept_aliases", "claim_concepts"})
FTS_TABLES = ("span_fts", "claim_fts", "note_fts")
BOOKKEEPING_TABLES = frozenset({"schema_migrations", "sqlite_sequence"})
ID_TYPE_ALIASES = {"semantic_scholar": "s2", "semantic_scholar_id": "s2"}
WORK_ID_COLUMNS = (("doi", "doi"), ("openalex", "openalex_id"), ("arxiv", "arxiv_id"),
                   ("s2", "semantic_scholar_id"), ("ssrn", "ssrn_id"))
CITATION_PROVENANCE = frozenset({"provider_reference", "parsed_bibliography", "manual_override"})
DUP_SAME, DUP_DIFFERENT = frozenset({"merge"}), frozenset({"reject", "dismissed"})
CITE_REJECT = frozenset({"reject", "dismissed"})

# v3 column lists (library schema 0001-0007), in table order.
COLS: dict[str, tuple[str, ...]] = {
    "papers": ("paper_id", "title", "title_source", "authors_json", "year", "venue", "abstract", "oa_status",
               "created_at", "origin"),
    "identifiers": ("paper_id", "id_type", "id_value", "source", "confidence", "origin"),
    "source_files": ("source_file_id", "file_hash", "file_type", "access_class", "storage_uri",
                     "acquisition_method", "created_at", "byte_size", "updated_at"),
    "conversion_runs": ("conversion_run_id", "source_file_id", "source_file_hash", "converter_name",
                        "converter_package", "converter_version", "python_version", "config_json",
                        "conversion_fingerprint", "llm_enabled", "llm_service", "llm_model", "llm_prompt_version",
                        "conversion_epistemic_type", "run_status", "warnings_json", "error", "markdown_hash",
                        "created_at", "completed_at"),
    "markdown_documents": ("markdown_id", "conversion_run_id", "source_file_id", "markdown_hash", "storage_uri",
                           "conversion_status", "byte_size", "created_at"),
    "cache_events": ("cache_event_id", "object_type", "object_id", "event_type", "detail_json", "created_at"),
    "paper_files": ("paper_id", "source_file_id", "markdown_id", "role", "pdf_check_status", "pdf_check_json",
                    "attached_at", "origin"),
    "document_sections": ("section_id", "markdown_id", "markdown_hash", "source_file_id", "source_file_hash",
                          "parent_section_id", "level", "ordinal", "heading_text", "heading_path", "section_kind",
                          "start_char", "end_char", "page_start", "page_end", "section_parser_version",
                          "created_at", "origin"),
    "evidence_spans": ("span_id", "markdown_id", "markdown_hash", "source_file_id", "source_file_hash",
                       "section_id", "start_char", "end_char", "exact_quote", "quote_hash", "norm_version",
                       "span_kind", "page_start", "page_end", "access_class", "anchor_status", "created_at",
                       "origin"),
    "extraction_runs": ("extraction_run_id", "paper_id", "markdown_id", "markdown_hash", "schema_id", "lens_id",
                        "schema_version", "prompt_version", "model_name", "provider", "access_mode", "temperature",
                        "access_class", "external_full_text", "run_status", "input_tokens", "output_tokens",
                        "estimated_cost", "run_id", "created_at", "extraction_mode", "parent_extraction_run_id",
                        "chunk_index", "chunk_count", "chunk_section_ids", "lens_definition_hash", "origin"),
    "structured_notes": ("note_id", "extraction_run_id", "paper_id", "markdown_id", "markdown_hash", "schema_id",
                         "schema_version", "prompt_version", "archetype", "access_class", "raw_note_json",
                         "note_text", "created_at", "version_no", "version_label", "origin"),
    "extracted_claims": ("claim_id", "structured_note_id", "extraction_run_id", "paper_id", "claim_type",
                         "claim_subtype", "field_key", "normalized_label", "claim_text", "status", "epistemic_type",
                         "assertion_status", "inferred_explanation", "confidence", "access_class", "created_at",
                         "source_chunk_index", "source_extraction_run_id", "origin"),
    "claim_spans": ("id", "claim_id", "span_id", "rank", "created_at", "origin"),
    "reference_entries": ("reference_id", "citing_paper_id", "raw_reference_text", "parsed_fields_json",
                          "resolved_paper_id", "resolution_status", "resolution_source", "confidence",
                          "markdown_id", "markdown_hash", "section_label", "ordinal", "created_at", "origin"),
    "citation_links": ("source_paper_id", "target_paper_id", "provenance", "confidence", "reference_id",
                       "created_at", "origin"),
    "citation_review_decisions": ("reference_id", "decision", "rejected_record_json", "reason", "decided_at",
                                  "origin"),
    "duplicate_decisions": ("paper_a", "paper_b", "verdict", "decided_by", "reason", "decided_at", "origin"),
    "paper_merge_members": ("paper_id", "v2_project", "v2_work_id", "joined_by_key_type", "joined_by_key_value",
                            "joined_with_project", "joined_with_work_id", "origin"),
    "lens_definitions": ("lens_id", "name", "object_type", "status", "definition_hash", "definition_yaml",
                         "created_at", "origin"),
    "lens_outputs": ("lens_output_id", "extraction_run_id", "lens_id", "paper_id", "status", "claim_id",
                     "confidence", "fields_json", "access_class", "created_at", "origin"),
    "concept_decisions": ("decision_id", "kind", "label_a", "label_b", "decided_by", "state", "note", "origin",
                          "decided_at"),
    "conversion_queue": ("source_file_id", "reason", "queued_at", "origin"),
    "v2_id_map": ("v2_project", "v2_table", "v2_id", "v3_table", "v3_id"),
}
#: Primary keys used to sort each table's rows (inserts happen in key order, §4.5).
PK: dict[str, tuple[str, ...]] = {
    "papers": ("paper_id",), "identifiers": ("id_type", "id_value"), "source_files": ("source_file_id",),
    "conversion_runs": ("conversion_run_id",), "markdown_documents": ("markdown_id",),
    "cache_events": ("cache_event_id",), "paper_files": ("paper_id", "source_file_id"),
    "document_sections": ("section_id",), "evidence_spans": ("span_id",),
    "extraction_runs": ("extraction_run_id",), "structured_notes": ("note_id",),
    "extracted_claims": ("claim_id",), "claim_spans": ("id",), "reference_entries": ("reference_id",),
    "citation_links": ("source_paper_id", "target_paper_id", "provenance"),
    "citation_review_decisions": ("reference_id",), "duplicate_decisions": ("paper_a", "paper_b"),
    "paper_merge_members": ("v2_project", "v2_work_id"), "lens_definitions": ("lens_id", "definition_hash"),
    "lens_outputs": ("lens_output_id",), "concept_decisions": ("decision_id",),
    "conversion_queue": ("source_file_id",), "v2_id_map": ("v2_project", "v2_table", "v2_id"),
}


def _row(src: Mapping[str, Any], table: str, **over: Any) -> dict[str, Any]:
    return {c: over[c] if c in over else src.get(c) for c in COLS[table]}


def _sort_key(value: Any) -> tuple[int, Any]:
    return (0, "") if value is None else (1, value)


# --- the ledger ------------------------------------------------------------------------

@dataclass(frozen=True)
class Disposition:
    """What happened to one v2 row: ``carried``, ``merged`` (folded into another carried
    row), ``not_carried`` (with a reason) or ``rebuilt`` (FTS content)."""

    outcome: str
    reason: str | None = None
    v3_table: str | None = None
    v3_id: str | None = None


@dataclass
class Ledger:
    """Disposition of every v2 row, keyed by ``(source, table, row key)``; source is the
    project slug or ``'cache'``; the row key is the v2 primary key (``|``-joined)."""

    entries: dict[tuple[str, str, str], Disposition] = field(default_factory=dict)

    def record(self, source: str, table: str, key: Any, outcome: str, reason: str | None = None,
               v3_table: str | None = None, v3_id: str | None = None) -> None:
        k = (source, table, str(key))
        if k not in self.entries:
            self.entries[k] = Disposition(outcome, reason, v3_table, v3_id)

    def get(self, source: str, table: str, key: Any) -> Disposition | None:
        return self.entries.get((source, table, str(key)))

    def summary(self) -> dict[str, dict[str, dict[str, int]]]:
        out: dict[str, dict[str, dict[str, int]]] = {}
        for (source, table, _), d in sorted(self.entries.items()):
            label = d.outcome if d.reason is None else f"{d.outcome}:{d.reason}"
            bucket = out.setdefault(source, {}).setdefault(table, {})
            bucket[label] = bucket.get(label, 0) + 1
        return out

    def unaccounted(self) -> list[tuple[str, str, str]]:
        return sorted(k for k, d in self.entries.items() if d.outcome == "unaccounted")


def row_key(db_pk: tuple[str, ...], row: Mapping[str, Any]) -> str:
    if db_pk:
        return "|".join(str(row[c]) for c in db_pk)
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --- plan values -----------------------------------------------------------------------

@dataclass
class GraphPlan:
    slug: str
    source_project: str
    display_name: str
    created_at: str
    origin: str
    papers: list[dict[str, Any]]   # paper_id, status, is_seed, reason, added_at, origin (sorted by paper_id)
    lenses: list[dict[str, Any]]   # lens_id, chosen_at, origin


@dataclass(frozen=True)
class BlobCopy:
    kind: str     # 'pdf' | 'markdown'
    src: str      # relative to the v2 cache root
    dest: str     # relative to the library folder
    sha256: str
    size: int


@dataclass
class MigrationPlan:
    taken_at: str
    library: dict[str, list[dict[str, Any]]]
    provider_cache: list[dict[str, Any]]
    graphs: list[GraphPlan]
    blobs: list[BlobCopy]
    attention: list[AttentionSpec]
    ledger: Ledger
    report: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    identity: IdentityPlan

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self.library.get(table, [])

    def graph(self, slug: str) -> GraphPlan | None:
        return next((g for g in self.graphs if g.slug == slug), None)


# --- builder ---------------------------------------------------------------------------

@dataclass
class _Paper:
    paper_id: str
    members: tuple[WorkRef, ...]
    planned: PlannedPaper | None
    title: str | None = None
    title_source: str | None = None
    dois: tuple[str, ...] = ()
    primary_sf: str | None = None
    primary_wsf: dict | None = None
    markdown: defects.MarkdownChoice | None = None
    check: defects.PaperCheck | None = None
    pdf_shas: tuple[str, ...] = ()   # every PDF of the paper's members


class _Builder:
    def __init__(self, s: Snapshot, graph_names: Mapping[str, str] | None):
        self.s = s
        self.graph_names = dict(graph_names or {})
        self.ledger = Ledger()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.report: dict[str, Any] = {}
        self.lib: dict[str, list[dict[str, Any]]] = {t: [] for t in COLS}
        self.attention: list[AttentionSpec] = []
        self.idmap: dict[tuple[str, str, str], tuple[str, str]] = {}
        self.slugs = list(s.projects)
        self.carried_projects = [p for p in self.slugs if not p.startswith("dbg_")]
        c = s.cache
        self.sf_by_id = {r["source_file_id"]: r for r in c.rows("source_files")}
        self.sf_by_hash = {r["file_hash"]: r for r in c.rows("source_files")}
        self.pdf_blobs = {f.sha256 for f in s.pdf_files}
        self.md_blobs = {f.relpath.rsplit("/", 1)[-1][:-3] for f in s.markdown_files if f.relpath.endswith(".md")}
        self.runs_by_id = {r["conversion_run_id"]: r for r in c.rows("conversion_runs")}
        self.carried_runs = {k for k, r in self.runs_by_id.items() if r["run_status"] != "pending"}
        self.md_all = {r["markdown_id"]: r for r in c.rows("markdown_documents")}
        self.md_by_id = {k: r for k, r in self.md_all.items()
                         if r["conversion_run_id"] in self.carried_runs and r["markdown_hash"] in self.md_blobs}
        self.md_by_sf: dict[str, list[dict]] = defaultdict(list)
        for r in self.md_by_id.values():
            self.md_by_sf[r["source_file_id"]].append(r)
        # per project indexes
        self.works: dict[str, dict[str, dict]] = {}
        self.docs: dict[str, dict[str, dict]] = {}
        self.wsf: dict[str, dict[str, list[dict]]] = {}
        self.idents: dict[str, dict[str, list[dict]]] = {}
        for slug in self.slugs:
            db = s.projects[slug]
            self.works[slug] = {r["work_id"]: r for r in db.rows("works")}
            self.docs[slug] = {r["work_id"]: r for r in db.rows("project_documents")}
            by = defaultdict(list)
            for r in db.rows("work_source_files"):
                by[r["work_id"]].append(r)
            self.wsf[slug] = by
            ib = defaultdict(list)
            for r in db.rows("identifiers"):
                ib[r["work_id"]].append(r)
            self.idents[slug] = ib

    # helpers -----------------------------------------------------------------------
    def rank(self, slug: str) -> tuple[int, str]:
        return project_rank(slug)

    def valid_wsf(self, slug: str, wid: str) -> list[dict]:
        out = []
        for r in self.wsf[slug].get(wid, []):
            sf = self.sf_by_id.get(r["source_file_id"]) or self.sf_by_hash.get(r["file_hash"])
            if sf is not None and sf["file_hash"] in self.pdf_blobs:
                out.append(r)
        return sorted(out, key=lambda r: (r["file_hash"], r["work_source_file_id"]))

    def is_held(self, slug: str, wid: str) -> bool:
        return bool(self.valid_wsf(slug, wid))

    def map_id(self, slug: str, table: str, v2_id: str, v3_table: str, v3_id: str) -> None:
        self.idmap.setdefault((slug, table, str(v2_id)), (v3_table, str(v3_id)))

    def attn(self, kind: str, subject: dict, paper_id: str | None, detail: str | None, hides: bool = False):
        self.attention.append(AttentionSpec(kind, subject, paper_id, detail, hides))

    # 1. which works are carried ---------------------------------------------------------
    def carried_works(self) -> dict[str, set[str]]:
        carried: dict[str, set[str]] = {}
        self.dup_items: dict[str, list[dict]] = {}
        for slug in self.carried_projects:
            ws = self.works[slug]
            sel = {w for w in ws if self.is_held(slug, w)}
            if slug in GRAPH_PROJECTS:
                sel |= {w for w in self.docs[slug] if w in ws}
            items = [r for r in self.s.projects[slug].rows("review_queue")
                     if r["item_type"] == "duplicate_candidate" and r["status"] == "resolved"
                     and r["action"] in (DUP_SAME | DUP_DIFFERENT)]
            self.dup_items[slug] = items
            for it in items:
                sel |= {w for w in self.dup_sides(slug, it)[1] if w in ws}
                t = self.dup_sides(slug, it)[0]
                if t:
                    sel.add(t)
            carried[slug] = sel
        # targets of carried citation links and resolved references, to a fixpoint
        changed = True
        while changed:
            changed = False
            for slug in self.carried_projects:
                db, sel, ws = self.s.projects[slug], carried[slug], self.works[slug]
                for e in db.rows("citation_edges"):
                    if e["source_work_id"] in sel and e["target_work_id"] in ws and e["target_work_id"] not in sel:
                        sel.add(e["target_work_id"])
                        changed = True
                for r in db.rows("reference_entries"):
                    t = r["resolved_work_id"]
                    if r["citing_work_id"] in sel and t and t in ws and t not in sel:
                        sel.add(t)
                        changed = True
        return carried

    def dup_sides(self, slug: str, item: dict) -> tuple[str | None, list[str]]:
        payload = json.loads(item["payload"] or "{}")
        ws = self.works[slug]
        target = item["target_id"] if item["target_type"] == "work" and item["target_id"] in ws else None
        if target is None:  # v2 'merge' may have collapsed the target: find it by the incoming ids
            incoming = payload.get("incoming") or {}
            want = set(normalize_ids([(t, incoming.get(t)) for t in ("doi", "openalex", "arxiv") if incoming.get(t)]))
            if want:
                for wid in sorted(ws):
                    if want & set(self.work_ids(slug, wid)):
                        target = wid
                        break
        conflicting = [w for w in (payload.get("conflicting_work_ids") or []) if w in ws and w != target]
        return target, conflicting

    def work_ids(self, slug: str, wid: str) -> tuple[tuple[str, str], ...]:
        w = self.works[slug][wid]
        raw = [(t, w.get(col)) for t, col in WORK_ID_COLUMNS if w.get(col)]
        raw += [(ID_TYPE_ALIASES.get(r["id_type"], r["id_type"]), r["id_value"]) for r in self.idents[slug].get(wid, [])]
        return normalize_ids(raw)

    # 2. identity ------------------------------------------------------------------------
    def identity(self) -> None:
        self.carried = self.carried_works()
        records = []
        for slug in self.carried_projects:
            for wid in sorted(self.carried[slug]):
                w = self.works[slug][wid]
                raw = [(t, w.get(col)) for t, col in WORK_ID_COLUMNS if w.get(col)]
                raw += [(ID_TYPE_ALIASES.get(r["id_type"], r["id_type"]), r["id_value"])
                        for r in self.idents[slug].get(wid, [])]
                records.append(WorkRecord(
                    project=slug, work_id=wid, title=w.get("canonical_title"), year=w.get("year"),
                    authors=w.get("authors"), venue=w.get("venue"), abstract=w.get("abstract"),
                    ids=tuple(raw), pdf_hashes=tuple(sorted({r["file_hash"] for r in self.valid_wsf(slug, wid)})),
                ))
        never = []
        for slug in self.carried_projects:
            for it in self.dup_items[slug]:
                if it["action"] in DUP_DIFFERENT:
                    target, conflicting = self.dup_sides(slug, it)
                    if target and target in self.carried[slug]:
                        never += [(WorkRef(slug, target), WorkRef(slug, c)) for c in conflicting
                                  if c in self.carried[slug]]
        self.ident = plan_identity(records, never)
        # report facts (acceptance totals): carried works with no strong id, and papers held
        # (a PDF attached) in more than one v2 project
        self.report["no_strong_id_works"] = sorted(
            [r.project, r.work_id] for r in records
            if not any(t in MERGE_ID_TYPES for t, _ in normalize_ids(r.ids)))
        held_in = []
        self.paper_of: dict[WorkRef, _Paper] = {}
        self.papers: list[_Paper] = []
        for pp in self.ident.papers:
            p = _Paper(pp.paper_id, pp.members, pp)
            self.papers.append(p)
            for m in pp.members:
                self.paper_of[WorkRef(*m)] = p
            projects = sorted({m.project for m in pp.members if self.is_held(m.project, m.work_id)},
                              key=self.rank)
            if len(projects) > 1:
                held_in.append({"paper_id": pp.paper_id, "projects": projects})
        self.report["papers_held_in_several_projects"] = sorted(held_in, key=lambda r: r["paper_id"])
        # every cached PDF held by no carried work becomes a held paper with no metadata
        held_hashes = {h for pp in self.ident.papers for h in pp.pdf_hashes}
        taken = {pp.paper_id for pp in self.ident.papers}
        self.unheld: list[_Paper] = []
        for sf in sorted(self.s.cache.rows("source_files"), key=lambda r: r["file_hash"]):
            if sf["file_hash"] in held_hashes:
                continue
            minted = mint_paper_id(f"pdf:{sf['file_hash']}", taken)
            taken.add(minted.paper_id)
            p = _Paper(minted.paper_id, (), None, primary_sf=sf["source_file_id"], pdf_shas=(sf["file_hash"],))
            self.unheld.append(p)
            if minted.collided:
                self.attn("identity_collision", {"key": minted.key}, p.paper_id, f"derived id collided; key {minted.key}")
        for c in self.ident.collisions:
            self.attn("identity_collision", {"key": c.key}, c.paper_id, f"derived id collided; key {c.key}")

    def paper(self, slug: str, wid: str | None) -> _Paper | None:
        return self.paper_of.get(WorkRef(slug, wid)) if wid else None

    # 3. papers, identifiers, files, defects --------------------------------------------
    def build_papers(self) -> None:
        provider_rows = self.s.cache.rows("provider_cache")
        merged_titles, repaired = [], []
        for p in self.papers:
            pp = p.planned
            assert pp is not None
            recs = [self.works[m.project][m.work_id] for m in pp.members]
            p.dois = tuple(v for t, v in pp.ids if t == "doi")
            p.title, p.title_source = pp.title, ("v2" if pp.title else None)
            reason = defects.malformed_reason(p.title)
            if reason:
                fix = defects.title_from_provider_cache(p.dois, provider_rows)
                if fix:
                    repaired.append({"paper_id": p.paper_id, "old": p.title, "new": fix.title,
                                     "provider": fix.provider, "request_key": fix.request_key})
                    p.title, p.title_source = fix.title, "provider_cache"
                else:
                    self.attn("malformed_title", {"paper_id": p.paper_id}, p.paper_id,
                              f"malformed title ({reason}): {p.title!r}; no cached metadata for its DOI")
            authors = pp.authors
            self.lib["papers"].append({
                "paper_id": p.paper_id, "title": p.title, "title_source": p.title_source,
                "authors_json": _authors_json(authors), "year": pp.year, "venue": pp.venue,
                "abstract": pp.abstract, "oa_status": next((r.get("oa_status") for r in recs if r.get("oa_status")), None),
                "created_at": recs[0]["created_at"], "origin": f"v2:{pp.members[0].project}",
            })
            if len(pp.members) > 1:
                merged_titles.append({"paper_id": p.paper_id, "title": p.title,
                                      "members": [list(m) for m in pp.members]})
            # primary PDF: the precedence-first member holding one, smallest hash
            shas = []
            for m in pp.members:
                for r in self.valid_wsf(m.project, m.work_id):
                    shas.append(r["file_hash"])
                    if p.primary_wsf is None:
                        p.primary_wsf = r
                        p.primary_sf = self.sf_by_hash[r["file_hash"]]["source_file_id"]
            p.pdf_shas = tuple(sorted(set(shas)))
        for p in self.unheld:
            sf = self.sf_by_id[p.primary_sf]
            self.lib["papers"].append({
                "paper_id": p.paper_id, "title": None, "title_source": None, "authors_json": None, "year": None,
                "venue": None, "abstract": None, "oa_status": None, "created_at": sf["created_at"],
                "origin": "migration",
            })
            self.attn("metadata_needed", {"paper_id": p.paper_id}, p.paper_id,
                      f"cached PDF {sf['file_hash'][:12]} is held by no v2 project; title unknown")
        self.report["titles_repaired"] = repaired
        self.report["merged_papers"] = merged_titles
        self.identifiers()
        self.files()

    def identifiers(self) -> None:
        owner: dict[tuple[str, str], str] = {}
        conflicts = []
        for p in self.papers:
            pp = p.planned
            for t, v in pp.ids:
                if (t, v) in owner:
                    conflicts.append({"id_type": t, "id_value": v, "kept_on": owner[(t, v)], "dropped_from": p.paper_id})
                    continue
                owner[(t, v)] = p.paper_id
                source, conf, origin = "v2_works_column", None, f"v2:{pp.members[0].project}"
                found = False
                for m in pp.members:
                    for r in self.idents[m.project].get(m.work_id, []):
                        if (t, v) in normalize_ids([(ID_TYPE_ALIASES.get(r["id_type"], r["id_type"]), r["id_value"])]):
                            source, conf, origin, found = r["resolution_source"], r["confidence"], f"v2:{m.project}", True
                            break
                    if found:
                        break
                if not found:
                    origin = next((f"v2:{m.project}" for m in pp.members if (t, v) in self.work_ids(m.project, m.work_id)),
                                  origin)
                self.lib["identifiers"].append({"paper_id": p.paper_id, "id_type": t, "id_value": v, "source": source,
                                                "confidence": conf, "origin": origin})
        self.report["identifier_conflicts"] = conflicts
        self.id_owner = owner

    def files(self) -> None:
        c = self.s.cache
        for r in c.rows("source_files"):
            self.lib["source_files"].append(_row(r, "source_files"))
            self.ledger.record("cache", "source_files", r["source_file_id"], "carried", None, "source_files",
                               r["source_file_id"])
            if r["file_hash"] not in self.pdf_blobs:
                self.warnings.append(f"PDF blob missing for {r['source_file_id']}")
        for r in c.rows("conversion_runs"):
            if r["conversion_run_id"] in self.carried_runs:
                self.lib["conversion_runs"].append(_row(r, "conversion_runs"))
                self.ledger.record("cache", "conversion_runs", r["conversion_run_id"], "carried", None,
                                   "conversion_runs", r["conversion_run_id"])
            else:
                self.ledger.record("cache", "conversion_runs", r["conversion_run_id"], "not_carried", "pending_run")
        for r in c.rows("markdown_documents"):
            if r["markdown_id"] in self.md_by_id:
                self.lib["markdown_documents"].append(_row(r, "markdown_documents"))
                self.ledger.record("cache", "markdown_documents", r["markdown_id"], "carried", None,
                                   "markdown_documents", r["markdown_id"])
            else:
                self.ledger.record("cache", "markdown_documents", r["markdown_id"], "not_carried", "blob_missing")
                self.warnings.append(f"markdown blob missing for {r['markdown_id']}")
        for r in c.rows("cache_events"):
            self.lib["cache_events"].append(_row(r, "cache_events"))
            self.ledger.record("cache", "cache_events", r["cache_event_id"], "carried", None, "cache_events",
                               r["cache_event_id"])
        for r in c.rows("provider_cache"):
            self.ledger.record("cache", "provider_cache", r["provider_cache_id"], "carried", None,
                               "provider_cache.db", r["provider_cache_id"])

        # each held paper's markdown, title check, file row
        relinks, not_converted, test_conv, secondary = [], [], [], []
        queue: dict[str, str] = {}
        for p in [*self.papers, *self.unheld]:
            if p.primary_sf is None:
                continue
            pointed = p.primary_wsf["markdown_id"] if p.primary_wsf else None
            choice = defects.choose_markdown(p.primary_sf, pointed, self.md_by_sf, self.md_by_id, self.runs_by_id)
            p.markdown = choice
            if choice.repair in ("relinked", "relink_failed", "pointer_missing"):
                relinks.append({"paper_id": p.paper_id, "repair": choice.repair, "from_markdown_id": pointed,
                                "to_markdown_id": choice.markdown_id, "foreign": choice.foreign})
            if choice.repair == "relink_failed":
                queue[p.primary_sf] = "relink_failed"
            if choice.repair == "test_converter" or (choice.test_converter_markdown_id and
                                                     choice.test_converter_markdown_id == pointed):
                test_conv.append(p.paper_id)
                self.attn("test_converter_markdown", {"paper_id": p.paper_id}, p.paper_id,
                          f"markdown {choice.test_converter_markdown_id} came from a test/fake converter; "
                          "it is carried but not used as the paper's conversion")
            md_text = None
            if choice.markdown_id:
                md_text = self.s.markdown_text(self.md_by_id[choice.markdown_id]["markdown_hash"])
            else:
                not_converted.append({"paper_id": p.paper_id, "title": p.title, "source_file_id": p.primary_sf})
            sha = self.sf_by_id[p.primary_sf]["file_hash"]
            # A malformed title (not repaired) says nothing about the PDF: check on the DOI only.
            title = None if defects.malformed_reason(p.title) else p.title
            p.check = defects.check_paper(title, p.dois, md_text, lambda sha=sha: self.s.pdf_first_pages(sha))
            if p.planned is not None and len(p.pdf_shas) > 1:
                secondary.append({"paper_id": p.paper_id, "primary": sha,
                                  "secondary": [h for h in p.pdf_shas if h != sha]})
            self.lib["paper_files"].append({
                "paper_id": p.paper_id, "source_file_id": p.primary_sf, "markdown_id": choice.markdown_id,
                "role": "primary", "pdf_check_status": p.check.status, "pdf_check_json": p.check.json,
                "attached_at": p.primary_wsf["created_at"] if p.primary_wsf else self.sf_by_id[p.primary_sf]["created_at"],
                "origin": f"v2:{p.members[0].project}" if p.members else "migration",
            })
        for sf_id, sf in sorted(self.sf_by_id.items()):
            if sf_id not in queue and not any(not defects.is_test_converter(self.runs_by_id.get(m["conversion_run_id"]))
                                              for m in self.md_by_sf.get(sf_id, [])):
                queue[sf_id] = "not_converted"
        for sf_id, reason in sorted(queue.items()):
            self.lib["conversion_queue"].append({"source_file_id": sf_id, "reason": reason,
                                                 "queued_at": self.s.taken_at, "origin": "migration"})
        self.report.update(relinks=relinks, not_converted=not_converted, test_converter_papers=test_conv,
                           secondary_pdfs=secondary)
        self.title_check_attention()

    def title_check_attention(self) -> None:
        checks = {p.paper_id: p for p in [*self.papers, *self.unheld] if p.check}
        covered = set()
        shared_report = []
        for sp in self.ident.shared_pdfs:
            shared_report.append({"pdf_sha256": sp.pdf_hash, "paper_ids": list(sp.paper_ids)})
            for pid in sp.paper_ids:
                p = checks.get(pid)
                miss = bool(p and p.check.status == "miss" and self.sf_by_id[p.primary_sf]["file_hash"] == sp.pdf_hash)
                if miss:
                    covered.add(pid)
                self.attn("shared_pdf", {"pdf_sha256": sp.pdf_hash, "paper_id": pid}, pid,
                          json.dumps({"papers": list(sp.paper_ids), "title_check": p.check.status if p else None},
                                     sort_keys=True), hides=miss)
        weak, miss = [], []
        for pid, p in checks.items():
            if p.check.status == "weak":
                weak.append(pid)
            if p.check.status == "miss":
                miss.append(pid)
                if pid not in covered:
                    self.attn("pdf_title_miss", {"paper_id": pid, "source_file_id": p.primary_sf}, pid,
                              f"title {p.title!r} not found in its PDF or markdown: this PDF may not be this paper",
                              hides=True)
        self.report.update(shared_pdfs=shared_report, title_check_weak=sorted(weak), title_check_miss=sorted(miss))
        # "check these" (identity), minus a redundant no_match on a paper already listed
        skipped = {ci.paper_id for ci in self.ident.check_items if ci.reason == "never_merge_skipped"}
        seen = set()
        for ci in self.ident.check_items:
            if ci.reason == "no_match" and ci.paper_id in skipped:
                continue
            subj = {"reason": ci.reason, "paper_id": ci.paper_id, "key_type": ci.key_type, "key_value": ci.key_value}
            key = json.dumps(subj, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            evidence = {"works": [list(w) for w in ci.works], "title": ci.title, "year": ci.year,
                        "candidates": [{"paper_id": c.paper_id, "title": c.title, "year": c.year,
                                        "similarity": c.similarity, "shared_keys": [list(k) for k in c.shared_keys]}
                                       for c in ci.candidates]}
            self.attn("identity_check", subj, ci.paper_id, json.dumps(evidence, sort_keys=True, ensure_ascii=False))

    # 4. works ledger, merge members, graphs ------------------------------------------------
    def works_and_graphs(self) -> None:
        rm_first: dict[WorkRef, tuple] = {}
        for rm in self.ident.rule_merges:
            for me, other in ((rm.work_a, rm.work_b), (rm.work_b, rm.work_a)):
                rm_first.setdefault(WorkRef(*me), (rm.key_type, rm.key_value, WorkRef(*other)))
        for p in self.papers:
            if len(p.members) < 2:
                continue
            for m in p.members:
                kt, kv, other = rm_first[m]
                self.lib["paper_merge_members"].append({
                    "paper_id": p.paper_id, "v2_project": m.project, "v2_work_id": m.work_id,
                    "joined_by_key_type": kt, "joined_by_key_value": kv, "joined_with_project": other.project,
                    "joined_with_work_id": other.work_id, "origin": "migration"})
        for slug in self.slugs:
            for wid in sorted(self.works[slug]):
                p = self.paper(slug, wid)
                if p is not None:
                    outcome = "merged" if len(p.members) > 1 else "carried"
                    reason = rm_first[WorkRef(slug, wid)][0] if outcome == "merged" else None
                    self.ledger.record(slug, "works", wid, outcome, reason, "papers", p.paper_id)
                    self.map_id(slug, "works", wid, "papers", p.paper_id)
                elif slug.startswith("dbg_"):
                    self.ledger.record(slug, "works", wid, "not_carried", "dbg_project")
                else:
                    self.ledger.record(slug, "works", wid, "not_carried",
                                       "unreferenced_metadata_only_in_non_graph_project")
            for r in self.s.projects[slug].rows("identifiers"):
                p = self.paper(slug, r["work_id"])
                key = r["identifier_id"]
                if p is None:
                    self.ledger.record(slug, "identifiers", key, "not_carried", "work_not_carried")
                    continue
                norm = normalize_ids([(ID_TYPE_ALIASES.get(r["id_type"], r["id_type"]), r["id_value"])])
                if norm and self.id_owner.get(norm[0]) == p.paper_id:
                    self.ledger.record(slug, "identifiers", key, "carried", None, "identifiers", ":".join(norm[0]))
                elif norm:
                    self.ledger.record(slug, "identifiers", key, "not_carried", "identifier_kept_on_another_paper")
                else:
                    self.ledger.record(slug, "identifiers", key, "not_carried", "unknown_id_type")
            for wid, rows in self.wsf[slug].items():
                p = self.paper(slug, wid)
                for r in rows:
                    key = r["work_source_file_id"]
                    if p is None:
                        self.ledger.record(slug, "work_source_files", key, "not_carried", "work_not_carried")
                    elif p.primary_wsf is r:
                        self.ledger.record(slug, "work_source_files", key, "carried", None, "paper_files",
                                           f"{p.paper_id}|{p.primary_sf}")
                    elif r["file_hash"] not in self.pdf_blobs:
                        self.ledger.record(slug, "work_source_files", key, "not_carried", "pdf_blob_missing")
                    elif p.primary_sf and self.sf_by_id[p.primary_sf]["file_hash"] == r["file_hash"]:
                        self.ledger.record(slug, "work_source_files", key, "merged", "same_paper_same_pdf",
                                           "paper_files", f"{p.paper_id}|{p.primary_sf}")
                    else:
                        self.ledger.record(slug, "work_source_files", key, "not_carried", "secondary_pdf_of_paper")
        self.graphs = [self.graph_plan(src, slug) for src, slug in GRAPH_PROJECTS.items() if src in self.s.projects]

    def graph_plan(self, src: str, slug: str) -> GraphPlan:
        rows: dict[str, dict] = {}
        notes = []

        def offer(pid: str, cand: dict) -> None:
            cur = rows.get(pid)
            if cur is None or STATUS_PRIORITY[cand["status"]] < STATUS_PRIORITY[cur["status"]]:
                if cur is not None:
                    cand["is_seed"] = max(cand["is_seed"], cur["is_seed"])
                rows[pid] = cand
            else:
                cur["is_seed"] = max(cur["is_seed"], cand["is_seed"])

        for wid in sorted(self.docs[src], key=lambda w: w):
            doc = self.docs[src][wid]
            p = self.paper(src, wid)
            if p is None:
                self.ledger.record(src, "project_documents", wid, "not_carried", "work_not_carried")
                continue
            status = STATUS_MAP.get(doc["inclusion_status"], "frontier")
            if status == "member" and not p.primary_sf:
                notes.append({"paper_id": p.paper_id, "note": "included in v2 without a PDF; carried as frontier"})
                status = "frontier"
            offer(p.paper_id, {"paper_id": p.paper_id, "status": status, "is_seed": int(bool(doc["is_seed"])),
                               "reason": doc["inclusion_reason"], "added_at": doc["created_at"], "origin": f"v2:{src}"})
            self.ledger.record(src, "project_documents", wid, "carried", None, f"graph:{slug}.graph_papers", p.paper_id)
        folded = []
        for fold_src, target in FOLD_INTO.items():
            if target != slug or fold_src not in self.s.projects:
                continue
            for wid in sorted(self.docs[fold_src]):
                doc = self.docs[fold_src][wid]
                p = self.paper(fold_src, wid)
                if p is None or not self.is_held(fold_src, wid):
                    continue
                cur = rows.get(p.paper_id)
                if cur is not None and cur["status"] in ("member", "excluded"):
                    if cur["status"] == "excluded":
                        notes.append({"paper_id": p.paper_id, "note": f"{fold_src} paper excluded in {slug}; kept excluded"})
                    self.ledger.record(fold_src, "project_documents", wid, "merged", f"already_in_graph_{slug}",
                                       f"graph:{slug}.graph_papers", p.paper_id)
                    continue
                rows[p.paper_id] = {"paper_id": p.paper_id, "status": "member", "is_seed": 0,
                                    "reason": doc["inclusion_reason"], "added_at": doc["created_at"],
                                    "origin": f"v2:{fold_src}"}
                folded.append({"paper_id": p.paper_id, "title": p.title, "was": cur["status"] if cur else None})
                self.ledger.record(fold_src, "project_documents", wid, "carried", f"folded_into_{slug}",
                                   f"graph:{slug}.graph_papers", p.paper_id)
        self.report.setdefault("folded_in", {})[slug] = sorted(folded, key=lambda r: r["paper_id"])
        self.report.setdefault("graph_notes", {})[slug] = notes
        yaml_doc = self.s.project_yaml.get(src, {})
        created = yaml_doc.get("created_at") or min((d["created_at"] for d in self.docs[src].values()),
                                                    default=self.s.taken_at)
        # lenses run by the source project
        ran = sorted({r["lens_id"] for r in self.s.projects[src].rows("extraction_runs") if r.get("lens_id")})
        lens_rows = {r["lens_id"]: r for r in self.s.projects[src].rows("lenses")}
        lenses = [{"lens_id": lid, "chosen_at": (lens_rows.get(lid) or {}).get("created_at") or created,
                   "origin": f"v2:{src}"} for lid in ran]
        return GraphPlan(slug=slug, source_project=src, display_name=self.graph_names.get(slug, slug),
                         created_at=str(created), origin=f"v2:{src}",
                         papers=[rows[k] for k in sorted(rows)], lenses=lenses)

    def remaining_project_documents(self) -> None:
        for slug in self.slugs:
            for wid in self.docs[slug]:
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "project_documents", wid, "not_carried", "dbg_project")
                else:
                    self.ledger.record(slug, "project_documents", wid, "not_carried", "project_not_a_graph")

    # 5. text: sections and spans ---------------------------------------------------------
    def text(self) -> None:
        self.span_remap: dict[tuple[str, str], str] = {}
        counts = defaultdict(int)
        # sections ---------------------------------------------------------------------
        by_md: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for slug in self.slugs:
            for r in self.s.projects[slug].rows("document_sections"):
                key = r["section_id"]
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "document_sections", key, "not_carried", "dbg_project")
                elif r["markdown_id"] not in self.md_by_id:
                    self.ledger.record(slug, "document_sections", key, "not_carried", "markdown_not_carried")
                    self.warnings.append(f"section {key} ({slug}) is on markdown {r['markdown_id']} that is not carried")
                    counts["sections_markdown_missing"] += 1
                else:
                    by_md[r["markdown_id"]].append((slug, r))
        cmp_cols = [c for c in COLS["document_sections"] if c not in ("created_at", "origin")]
        for md_id in sorted(by_md):
            entries = by_md[md_id]
            best_v = max(_parser_version(r["section_parser_version"]) for _, r in entries)
            at_best = [(s, r) for s, r in entries if _parser_version(r["section_parser_version"]) == best_v]
            winner = min({s for s, _ in at_best}, key=self.rank)
            win_rows = {r["section_id"]: r for s, r in at_best if s == winner}
            for sid in sorted(win_rows):
                r = win_rows[sid]
                self.lib["document_sections"].append(_row(r, "document_sections", origin=f"v2:{winner}"))
                self.ledger.record(winner, "document_sections", sid, "carried", None, "document_sections", sid)
                counts["sections_carried"] += 1
            for s, r in entries:
                if s == winner and r["section_id"] in win_rows:
                    continue
                sid = r["section_id"]
                if _parser_version(r["section_parser_version"]) != best_v:
                    self.ledger.record(s, "document_sections", sid, "not_carried", "older_parser_version")
                    counts["sections_older_parser"] += 1
                    continue
                w = win_rows.get(sid)
                if w is not None and all(w.get(c) == r.get(c) for c in cmp_cols):
                    self.ledger.record(s, "document_sections", sid, "merged", "same_section", "document_sections", sid)
                    counts["sections_duplicate"] += 1
                else:
                    self.errors.append(f"section conflict on {md_id}: {sid} from {s} differs from {winner}'s "
                                       f"({r['section_parser_version']})")
                    self.ledger.record(s, "document_sections", sid, "not_carried", "section_conflict")
                    counts["sections_conflict"] += 1
        # spans ----------------------------------------------------------------------
        auto: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        manual: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
        for slug in self.slugs:
            for r in self.s.projects[slug].rows("evidence_spans"):
                sid = r["span_id"]
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "evidence_spans", sid, "not_carried", "dbg_project")
                elif r["markdown_id"] not in self.md_by_id:
                    self.ledger.record(slug, "evidence_spans", sid, "not_carried", "markdown_not_carried")
                    counts["spans_markdown_missing"] += 1
                elif r["span_kind"] == "manual":
                    manual[(r["markdown_id"], r["start_char"], r["end_char"], r["span_kind"])].append((slug, r))
                else:
                    auto[sid].append((slug, r))
        span_cmp = [c for c in COLS["evidence_spans"] if c not in ("created_at", "origin")]
        chosen: dict[str, tuple[str, dict]] = {}
        for sid in sorted(auto):
            entries = sorted(auto[sid], key=lambda e: self.rank(e[0]))
            ws, wr = entries[0]
            chosen[sid] = (ws, wr)
            self.span_remap[(ws, sid)] = sid
            self.ledger.record(ws, "evidence_spans", sid, "carried", None, "evidence_spans", sid)
            for s, r in entries[1:]:
                self.span_remap[(s, sid)] = sid
                if all(wr.get(c) == r.get(c) for c in span_cmp):
                    self.ledger.record(s, "evidence_spans", sid, "merged", "same_auto_span", "evidence_spans", sid)
                    counts["spans_auto_duplicate"] += 1
                else:
                    self.errors.append(f"auto span conflict: {sid} from {s} differs from {ws}'s")
                    self.ledger.record(s, "evidence_spans", sid, "not_carried", "span_conflict")
        for key in sorted(manual, key=lambda k: (k[0], k[1], k[2], k[3])):
            entries = sorted(manual[key], key=lambda e: (e[1]["span_id"], self.rank(e[0])))
            ws, wr = entries[0]
            canon = wr["span_id"]
            if canon in chosen:
                self.errors.append(f"manual span id {canon} collides with another span")
                continue
            chosen[canon] = (ws, wr)
            for s, r in entries:
                self.span_remap[(s, r["span_id"])] = canon
                if (s, r["span_id"]) == (ws, canon):
                    self.ledger.record(s, "evidence_spans", canon, "carried", None, "evidence_spans", canon)
                else:
                    self.ledger.record(s, "evidence_spans", r["span_id"], "merged", "same_position",
                                       "evidence_spans", canon)
                    self.map_id(s, "evidence_spans", r["span_id"], "evidence_spans", canon)
                    counts["spans_manual_duplicate"] += 1
        for sid in sorted(chosen):
            s, r = chosen[sid]
            md = self.md_by_id[r["markdown_id"]]
            text = self.s.markdown_text(md["markdown_hash"])
            try:
                assert_invariant(text or "", SimpleNamespace(start_char=r["start_char"], end_char=r["end_char"],
                                                             exact_quote=r["exact_quote"], quote_hash=r["quote_hash"]))
            except ValueError as exc:
                self.errors.append(f"anchor check failed for span {sid} ({s}): {exc}")
            self.lib["evidence_spans"].append(_row(r, "evidence_spans", origin=f"v2:{s}"))
            counts["spans_carried"] += 1
        self.report["text_counts"] = dict(sorted(counts.items()))

    # 6. notes, claims, anchors, lens outputs ---------------------------------------------
    def notes(self) -> None:
        runs, notes, claims, cspans, louts = [], [], [], {}, []
        hidden_versions = []
        for slug in self.slugs:
            db = self.s.projects[slug]
            for table, idcol in (("extraction_runs", "extraction_run_id"), ("structured_notes", "note_id"),
                                 ("extracted_claims", "claim_id"), ("lens_outputs", "lens_output_id")):
                for r in db.rows(table):
                    p = self.paper(slug, r["work_id"])
                    if p is None:
                        self.ledger.record(slug, table, r[idcol], "not_carried",
                                           "dbg_project" if slug.startswith("dbg_") else "work_not_carried")
                        continue
                    self.ledger.record(slug, table, r[idcol], "carried", None, table, r[idcol])
                    item = (slug, p, r)
                    {"extraction_runs": runs, "structured_notes": notes, "extracted_claims": claims,
                     "lens_outputs": louts}[table].append(item)
        for slug, p, r in runs:
            self.lib["extraction_runs"].append(_row(r, "extraction_runs", paper_id=p.paper_id, origin=f"v2:{slug}"))
        # versions per (paper, schema_id)
        groups: dict[tuple[str, Any], list] = defaultdict(list)
        for slug, p, r in notes:
            groups[(p.paper_id, r["schema_id"])].append((slug, p, r))
        self.current_note: dict[tuple[str, Any], tuple[str, str]] = {}
        for gkey in sorted(groups, key=lambda k: (k[0], _sort_key(k[1]))):
            items = sorted(groups[gkey], key=lambda e: (e[2]["created_at"], self.rank(e[0]), e[2]["note_id"]))
            cur = None
            for n, (slug, p, r) in enumerate(items, start=1):
                label = self.note_label(p, r)
                if label:
                    hidden_versions.append({"note_id": r["note_id"], "paper_id": p.paper_id, "label": label})
                else:
                    cur = (slug, r["note_id"])
                self.lib["structured_notes"].append(_row(r, "structured_notes", paper_id=p.paper_id, version_no=n,
                                                         version_label=label, origin=f"v2:{slug}"))
            if cur:
                self.current_note[gkey] = cur
        for slug, p, r in claims:
            self.lib["extracted_claims"].append(_row(r, "extracted_claims", paper_id=p.paper_id, origin=f"v2:{slug}"))
        carried_claims = {(slug, r["claim_id"]) for slug, _, r in claims}
        for slug in self.slugs:
            for r in self.s.projects[slug].rows("claim_spans"):
                key = r["id"]
                if (slug, r["claim_id"]) not in carried_claims:
                    self.ledger.record(slug, "claim_spans", key, "not_carried",
                                       "dbg_project" if slug.startswith("dbg_") else "claim_not_carried")
                    continue
                span = self.span_remap.get((slug, r["span_id"]))
                if span is None:
                    self.errors.append(f"anchor lost: claim {r['claim_id']} ({slug}) -> span {r['span_id']} not carried")
                    self.ledger.record(slug, "claim_spans", key, "not_carried", "span_not_carried")
                    continue
                k = (r["claim_id"], span)
                if k in cspans:
                    self.ledger.record(slug, "claim_spans", key, "merged", "same_anchor", "claim_spans", "|".join(k))
                    continue
                cspans[k] = (slug, r)
                self.ledger.record(slug, "claim_spans", key, "carried", None, "claim_spans", "|".join(k))
        for i, k in enumerate(sorted(cspans), start=1):
            slug, r = cspans[k]
            self.lib["claim_spans"].append({"id": i, "claim_id": k[0], "span_id": k[1], "rank": r["rank"],
                                            "created_at": r["created_at"], "origin": f"v2:{slug}"})
        for slug, p, r in louts:
            self.lib["lens_outputs"].append(_row(r, "lens_outputs", paper_id=p.paper_id, origin=f"v2:{slug}"))
        self.report["labelled_versions"] = hidden_versions

    def note_label(self, p: _Paper, note: dict) -> str | None:
        """``'other_markdown'`` when the note was made on markdown of a PDF that is not the
        paper's (§7.3). Unknown markdown gives no label (cannot tell)."""
        md = self.md_all.get(note["markdown_id"])
        if md is None:
            return None
        sf = self.sf_by_id.get(md["source_file_id"])
        if sf is None:
            return None
        return None if sf["file_hash"] in p.pdf_shas else "other_markdown"

    def current_elsewhere(self) -> None:
        out = []
        for g in self.graphs:
            members = {r["paper_id"]: r for r in g.papers if r["status"] == "member"}
            for (pid, schema), (slug, note_id) in sorted(self.current_note.items(), key=lambda kv: (kv[0][0], _sort_key(kv[0][1]))):
                if pid in members and slug != g.source_project:
                    out.append({"graph": g.slug, "paper_id": pid, "schema_id": schema, "note_id": note_id,
                                "from_project": slug, "folded_in": members[pid]["origin"] != f"v2:{g.source_project}"})
        self.report["current_elsewhere"] = out

    # 7. references, citations, decisions -------------------------------------------------
    def other_markdown(self, p: _Paper, entries: list[dict]) -> bool:
        """A reference list parsed from markdown of a PDF that is not one of the paper's."""
        for r in entries:
            md = self.md_all.get(r["markdown_id"]) if r["markdown_id"] else None
            sf = self.sf_by_id.get(md["source_file_id"]) if md else None
            if sf is not None:
                return sf["file_hash"] not in p.pdf_shas
        return False

    def references(self) -> None:
        self.bad_refs: set[tuple[str, str]] = set()
        lists: dict[str, list[tuple[str, str, list[dict]]]] = defaultdict(list)
        for slug in self.slugs:
            by_work = defaultdict(list)
            for r in self.s.projects[slug].rows("reference_entries"):
                by_work[r["citing_work_id"]].append(r)
            for wid, entries in sorted(by_work.items()):
                p = self.paper(slug, wid)
                if p is None:
                    for r in entries:
                        self.ledger.record(slug, "reference_entries", r["reference_id"], "not_carried",
                                           "dbg_project" if slug.startswith("dbg_") else "work_not_carried")
                    continue
                entries = sorted(entries, key=lambda r: (_parsed_ordinal(r), r["created_at"], r["reference_id"]))
                if self.other_markdown(p, entries):
                    # parsed from another PDF's markdown: another paper's bibliography, never carried
                    for r in entries:
                        self.ledger.record(slug, "reference_entries", r["reference_id"], "not_carried",
                                           "other_markdown_list")
                        self.bad_refs.add((slug, r["reference_id"]))
                    self.report.setdefault("reference_lists_other_markdown", []).append(
                        {"paper_id": p.paper_id, "project": slug, "work_id": wid, "entries": len(entries)})
                    continue
                lists[p.paper_id].append((slug, wid, entries))
        self.ref_remap: dict[tuple[str, str], str | None] = {}
        chosen_report, unmapped_refs = [], 0
        paper_by_id = {p.paper_id: p for p in self.papers}
        for pid in sorted(lists):
            p = paper_by_id[pid]
            current = p.markdown.markdown_id if p.markdown else None
            cands = sorted(lists[pid], key=lambda c: (not any(r["markdown_id"] == current for r in c[2]) or current is None,
                                                      self.rank(c[0]), c[1]))
            cslug, cwid, centries = cands[0]
            from_current = current is not None and any(r["markdown_id"] == current for r in centries)
            chosen_report.append({"paper_id": pid, "project": cslug, "work_id": cwid, "from_current_markdown": from_current,
                                  "entries": len(centries)})
            index_full, index_raw = {}, defaultdict(list)
            for ordinal, r in enumerate(centries, start=1):
                rid = r["reference_id"]
                tgt = self.paper(cslug, r["resolved_work_id"])
                if r["resolved_work_id"] and tgt is None:
                    self.warnings.append(f"reference {rid}: resolved work {r['resolved_work_id']} not carried")
                self.lib["reference_entries"].append(_row(
                    r, "reference_entries", citing_paper_id=pid, resolved_paper_id=tgt.paper_id if tgt else None,
                    ordinal=ordinal, origin=f"v2:{cslug}"))
                self.ledger.record(cslug, "reference_entries", rid, "carried", None, "reference_entries", rid)
                self.ref_remap[(cslug, rid)] = rid
                norm = normalize_title(r["raw_reference_text"] or "")
                index_full[(norm, ordinal)] = rid
                index_raw[norm].append(rid)
            for slug, wid, entries in cands[1:]:
                for ordinal, r in enumerate(entries, start=1):
                    norm = normalize_title(r["raw_reference_text"] or "")
                    target = index_full.get((norm, ordinal))
                    if target is None and len(index_raw.get(norm, [])) == 1:
                        target = index_raw[norm][0]
                    self.ref_remap[(slug, r["reference_id"])] = target
                    if target:
                        self.ledger.record(slug, "reference_entries", r["reference_id"], "merged",
                                           "mapped_onto_chosen_list", "reference_entries", target)
                        self.map_id(slug, "reference_entries", r["reference_id"], "reference_entries", target)
                    else:
                        unmapped_refs += 1
                        self.ledger.record(slug, "reference_entries", r["reference_id"], "not_carried",
                                           "other_projects_list_no_counterpart")
        self.report["reference_lists"] = chosen_report
        self.report["reference_entries_unmapped"] = unmapped_refs
        self.citations()
        self.reviews()

    def citations(self) -> None:
        best: dict[tuple[str, str, str], tuple] = {}
        dropped = defaultdict(int)
        for slug in self.slugs:
            for e in self.s.projects[slug].rows("citation_edges"):
                key = e["edge_id"]
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "citation_edges", key, "not_carried", "dbg_project")
                    continue
                sp, tp = self.paper(slug, e["source_work_id"]), self.paper(slug, e["target_work_id"])
                if sp is None or tp is None:
                    dropped["end_not_carried"] += 1
                    self.ledger.record(slug, "citation_edges", key, "not_carried", "end_not_carried")
                    continue
                if sp.paper_id == tp.paper_id:
                    dropped["self_citation_after_merge"] += 1
                    self.ledger.record(slug, "citation_edges", key, "not_carried", "self_citation_after_merge")
                    continue
                if e["provenance"] not in CITATION_PROVENANCE:
                    dropped["unknown_provenance"] += 1
                    self.ledger.record(slug, "citation_edges", key, "not_carried", "unknown_provenance")
                    continue
                if e["provenance"] == "parsed_bibliography" and (slug, e["reference_id"]) in self.bad_refs:
                    dropped["parsed_from_other_markdown"] += 1
                    self.ledger.record(slug, "citation_edges", key, "not_carried", "parsed_from_other_markdown")
                    continue
                ref = self.ref_remap.get((slug, e["reference_id"])) if e["reference_id"] else None
                k = (sp.paper_id, tp.paper_id, e["provenance"])
                cand = (-(e["confidence"] or 0.0), ref is None, ref or "", self.rank(slug), key, slug, e, ref)
                if k not in best or cand[:5] < best[k][:5]:
                    best[k] = cand
        winners = {}
        for k in sorted(best):
            c = best[k]
            slug, e, ref = c[5], c[6], c[7]
            winners[(slug, c[4])] = k
            self.lib["citation_links"].append({"source_paper_id": k[0], "target_paper_id": k[1], "provenance": k[2],
                                               "confidence": e["confidence"], "reference_id": ref,
                                               "created_at": e["created_at"], "origin": f"v2:{slug}"})
        for slug in self.carried_projects:
            for e in self.s.projects[slug].rows("citation_edges"):
                if self.ledger.get(slug, "citation_edges", e["edge_id"]):
                    continue
                sp, tp = self.paper(slug, e["source_work_id"]), self.paper(slug, e["target_work_id"])
                k = (sp.paper_id, tp.paper_id, e["provenance"])
                outcome = "carried" if (slug, e["edge_id"]) in winners else "merged"
                self.ledger.record(slug, "citation_edges", e["edge_id"], outcome,
                                   None if outcome == "carried" else "deduped", "citation_links", "|".join(k))
                self.map_id(slug, "citation_edges", e["edge_id"], "citation_links", "|".join(k))
        self.report["citation_edges_dropped"] = dict(sorted(dropped.items()))

    def reviews(self) -> None:
        decisions: dict[str, tuple] = {}
        unmapped, collapsed = [], []
        dup_rows: dict[tuple[str, str], dict] = {}
        dup_report = []
        for slug in self.slugs:
            for it in self.s.projects[slug].rows("review_queue"):
                key = it["item_id"]
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "review_queue", key, "not_carried", "dbg_project")
                    continue
                if it["status"] != "resolved":
                    self.ledger.record(slug, "review_queue", key, "not_carried", "open_review_item")
                    continue
                payload = json.loads(it["payload"] or "{}")
                at = it["resolved_at"] or it["created_at"]
                if it["item_type"] == "citation_resolution":
                    if it["action"] not in CITE_REJECT:
                        self.ledger.record(slug, "review_queue", key, "not_carried", "reflected_in_reference_entry")
                        continue
                    rid = it["target_id"] if it["target_type"] == "reference_entry" else payload.get("reference_id")
                    mapped = self.ref_remap.get((slug, rid))
                    if mapped is None:
                        unmapped.append({"project": slug, "item_id": key, "reference_id": rid,
                                         "raw": payload.get("raw")})
                        self.ledger.record(slug, "review_queue", key, "not_carried", "unmapped_citation_decision")
                        continue
                    direct = mapped == rid
                    cand = (not direct, self.rank(slug), at, key, slug, it, payload)
                    if mapped in decisions:
                        loser = max(decisions[mapped], cand, key=lambda c: c[:4])
                        winner = min(decisions[mapped], cand, key=lambda c: c[:4])
                        decisions[mapped] = winner
                        collapsed.append({"project": loser[4], "item_id": loser[3], "onto_reference_id": mapped})
                    else:
                        decisions[mapped] = cand
                elif it["item_type"] == "duplicate_candidate":
                    if it["action"] not in (DUP_SAME | DUP_DIFFERENT):
                        self.ledger.record(slug, "review_queue", key, "not_carried",
                                           "graph_exclusion_in_project_documents" if it["action"] == "exclude"
                                           else "unsupported_duplicate_action")
                        continue
                    verdict = "same" if it["action"] in DUP_SAME else "different"
                    target, conflicting = self.dup_sides(slug, it)
                    pa = self.paper(slug, target)
                    pbs = [self.paper(slug, c) for c in conflicting]
                    if verdict == "same" and (pa is None or not pbs) and (pa or any(pbs)):
                        # v2's 'merge' already collapsed the pair into one work: nothing left to decide
                        survivor = pa or next(pb for pb in pbs if pb)
                        dup_report.append({"item_id": key, "verdict": verdict, "satisfied_by_v2_merge": True,
                                           "paper_id": survivor.paper_id})
                        self.ledger.record(slug, "review_queue", key, "merged", "satisfied_by_v2_merge", "papers",
                                           survivor.paper_id)
                        continue
                    if pa is None or not pbs or any(pb is None for pb in pbs):
                        unmapped.append({"project": slug, "item_id": key, "kind": "duplicate_candidate"})
                        self.ledger.record(slug, "review_queue", key, "not_carried", "unmapped_duplicate_decision")
                        continue
                    outcome = None
                    for pb in pbs:
                        if pb.paper_id == pa.paper_id:
                            if verdict == "different":
                                self.errors.append(f"researcher 'not duplicates' {key} ({slug}) was merged anyway")
                            dup_report.append({"item_id": key, "verdict": verdict, "satisfied_by_merge": True,
                                               "paper_id": pa.paper_id})
                            outcome = outcome or ("merged", "satisfied_by_merge", "papers", pa.paper_id)
                            continue
                        a, b = sorted((pa.paper_id, pb.paper_id))
                        if (a, b) not in dup_rows:
                            dup_rows[(a, b)] = {"paper_a": a, "paper_b": b, "verdict": verdict,
                                                "decided_by": "researcher",
                                                "reason": f"v2 {slug} duplicate_candidate action={it['action']}",
                                                "decided_at": at, "origin": f"v2:{slug}"}
                        dup_report.append({"item_id": key, "verdict": verdict, "paper_a": a, "paper_b": b})
                        outcome = ("carried", None, "duplicate_decisions", f"{a}|{b}")
                        self.map_id(slug, "review_queue", key, "duplicate_decisions", f"{a}|{b}")
                    self.ledger.record(slug, "review_queue", key, *outcome)
                elif it["item_type"] == "concept_merge_candidate":
                    self.ledger.record(slug, "review_queue", key, "not_carried", "reflected_in_concept_constraints")
                else:
                    self.ledger.record(slug, "review_queue", key, "not_carried", f"resolved_{it['item_type']}_not_carried")
        for rid in sorted(decisions):
            _, _, at, key, slug, it, payload = decisions[rid]
            rec = payload.get("rejected_record")
            self.lib["citation_review_decisions"].append({
                "reference_id": rid, "decision": "reject",
                "rejected_record_json": json.dumps(rec, sort_keys=True, ensure_ascii=False) if rec is not None else None,
                "reason": payload.get("reject_reason"), "decided_at": at, "origin": f"v2:{slug}"})
            self.ledger.record(slug, "review_queue", key, "carried", None, "citation_review_decisions", rid)
            self.map_id(slug, "review_queue", key, "citation_review_decisions", rid)
        for c in collapsed:
            self.ledger.record(c["project"], "review_queue", c["item_id"], "merged", "mapped_onto_existing_decision",
                               "citation_review_decisions", c["onto_reference_id"])
            self.map_id(c["project"], "review_queue", c["item_id"], "citation_review_decisions", c["onto_reference_id"])
        for k in sorted(dup_rows):
            self.lib["duplicate_decisions"].append(dup_rows[k])
        self.report["unmapped_decisions"] = unmapped
        self.report["citation_decisions_collapsed"] = collapsed
        self.report["duplicate_decisions"] = dup_report

    # 8. lenses and concepts -------------------------------------------------------------
    def lenses(self) -> None:
        defs: dict[tuple[str, str], dict] = {}
        for slug in self.slugs:
            for r in self.s.projects[slug].rows("lenses"):
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "lenses", r["lens_id"], "not_carried", "dbg_project")
                    continue
                k = (r["lens_id"], r["definition_hash"])
                files = self.s.lens_files.get(slug, {})
                yaml_path = (r.get("yaml_path") or "").replace("\\", "/")
                text = (r.get("definition_yaml") or files.get(yaml_path)
                        or files.get("lenses/" + yaml_path.rsplit("/", 1)[-1]))
                if text is None:
                    self.warnings.append(f"lens {r['lens_id']} ({slug}): YAML text not found")
                if k in defs:
                    self.ledger.record(slug, "lenses", r["lens_id"], "merged", "same_lens_version", "lens_definitions",
                                       "|".join(k))
                    continue
                defs[k] = {"lens_id": r["lens_id"], "name": r["name"], "object_type": r["object_type"],
                           "status": r["status"], "definition_hash": r["definition_hash"], "definition_yaml": text,
                           "created_at": r["created_at"], "origin": f"v2:{slug}"}
                self.ledger.record(slug, "lenses", r["lens_id"], "carried", None, "lens_definitions", "|".join(k))
        for k in sorted(defs):
            self.lib["lens_definitions"].append(defs[k])

    def newest(self, entries: list) -> tuple:
        """The most recent decision; ties go to the higher-precedence project."""
        return max(sorted(entries, key=lambda e: self.rank(e[0])), key=lambda e: e[4]["created_at"])

    def concepts(self) -> None:
        rows: dict[str, tuple] = {}
        by_pair: dict[tuple[str, str], list] = defaultdict(list)
        for slug in self.slugs:
            for r in self.s.projects[slug].rows("concept_constraints"):
                key = r["constraint_id"]
                if slug.startswith("dbg_"):
                    self.ledger.record(slug, "concept_constraints", key, "not_carried", "dbg_project")
                    continue
                kind = {"must_link": "merge", "cannot_link": "never_merge"}.get(r["kind"])
                a, b = sorted((normalize_label(r["label_a"] or ""), normalize_label(r["label_b"] or "")))
                if kind is None or not a or a == b:
                    self.ledger.record(slug, "concept_constraints", key, "not_carried", "unusable_constraint")
                    continue
                by = "researcher" if r["source"] == "user" else "agent"
                by_pair[(a, b)].append((slug, key, kind, by, r))
        contradictions = []
        overridden = []
        for pair in sorted(by_pair):
            entries = by_pair[pair]
            researcher = [e for e in entries if e[3] == "researcher"]
            pool = researcher or entries
            kinds = sorted({e[2] for e in pool})
            if len(kinds) > 1 and researcher:
                newest = self.newest(pool)
                contradictions.append({"label_a": pair[0], "label_b": pair[1], "newest_kind": newest[2],
                                       "newest_project": newest[0], "newest_at": newest[4]["created_at"],
                                       "decisions": [{"project": e[0], "kind": e[2], "at": e[4]["created_at"]}
                                                     for e in pool]})
                self.attn("concept_contradiction", {"label_a": pair[0], "label_b": pair[1]}, None,
                          f"researcher decisions disagree on {pair[0]!r} / {pair[1]!r}; newest is "
                          f"{newest[2]} ({newest[0]}, {newest[4]['created_at']})")
            for kind in kinds:
                group = [e for e in pool if e[2] == kind]
                win = self.newest(group)
                did = "cd_" + hashlib.sha256(f"{kind}|{pair[0]}|{pair[1]}".encode()).hexdigest()[:20]
                rows[did] = (win, kind, pair)
                for e in group:
                    out = "carried" if e is win else "merged"
                    self.ledger.record(e[0], "concept_constraints", e[1], out, None if e is win else "same_decision",
                                       "concept_decisions", did)
                    self.map_id(e[0], "concept_constraints", str(e[1]), "concept_decisions", did)
            for e in entries:
                if e in pool:
                    continue
                if e[2] in kinds:  # the researcher made the same decision: one row, decided_by researcher
                    did = "cd_" + hashlib.sha256(f"{e[2]}|{pair[0]}|{pair[1]}".encode()).hexdigest()[:20]
                    self.ledger.record(e[0], "concept_constraints", e[1], "merged", "same_decision_by_researcher",
                                       "concept_decisions", did)
                    self.map_id(e[0], "concept_constraints", str(e[1]), "concept_decisions", did)
                else:
                    overridden.append({"project": e[0], "constraint_id": e[1], "label_a": pair[0], "label_b": pair[1]})
                    self.ledger.record(e[0], "concept_constraints", e[1], "not_carried", "agent_overridden_by_researcher")
        for did in sorted(rows):
            (slug, _, kind, by, r), _, pair = rows[did]
            self.lib["concept_decisions"].append({
                "decision_id": did, "kind": kind, "label_a": pair[0], "label_b": pair[1], "decided_by": by,
                "state": "pending_apply", "note": None if by == "researcher" else f"v2 bulk merge (source={r['source']})",
                "origin": f"v2:{slug}", "decided_at": r["created_at"]})
        self.report["concept_contradictions"] = contradictions
        self.report["concept_agent_overridden"] = overridden
        # v2 rows reflected in a decision (carried, or merged into one), by their own project,
        # kind and maker: lets acceptance check each project's v2 counts separately
        per_project: dict[str, dict[str, int]] = {s: {} for s in self.carried_projects}
        for entries in by_pair.values():
            for slug, key, kind, by, _ in entries:
                d = self.ledger.get(slug, "concept_constraints", key)
                if d is not None and d.outcome in ("carried", "merged"):
                    bucket = per_project.setdefault(slug, {})
                    bucket[f"{kind}:{by}"] = bucket.get(f"{kind}:{by}", 0) + 1
        self.report["concept_v2_rows"] = {s: dict(sorted(v.items())) for s, v in sorted(per_project.items())}

    # 9. remaining tables ----------------------------------------------------------------
    def finalize_ledger(self) -> None:
        for source, db in [("cache", self.s.cache), *self.s.projects.items()]:
            fts_names = [t for t in db.tables if t in FTS_TABLES]
            for table, rows in db.tables.items():
                if table in DROPPED_TABLES:
                    outcome, reason = "not_carried", "dropped_table"
                elif table in FTS_TABLES or any(table.startswith(f + "_") for f in fts_names):
                    outcome, reason = ("rebuilt", None) if not source.startswith("dbg_") else ("not_carried", "dbg_project")
                elif table in BOOKKEEPING_TABLES:
                    outcome, reason = "not_carried", "v2_schema_bookkeeping"
                elif source.startswith("dbg_"):
                    outcome, reason = "not_carried", "dbg_project"
                else:
                    outcome, reason = "unaccounted", None
                pk = db.pk.get(table, ())
                for r in rows:
                    self.ledger.record(source, table, row_key(pk, r), outcome, reason)

    def blobs(self) -> list[BlobCopy]:
        out = []
        sizes = {f.relpath: f for f in [*self.s.pdf_files, *self.s.markdown_files]}
        for r in sorted(self.s.cache.rows("source_files"), key=lambda r: r["file_hash"]):
            rel = f"pdfs/{r['file_hash']}.pdf"
            f = sizes.get(rel)
            if f is not None:
                out.append(BlobCopy("pdf", rel, rel, f.sha256, f.size))
        for md_id in sorted(self.md_by_id):
            rel = f"markdown/{self.md_by_id[md_id]['markdown_hash']}.md"
            f = sizes.get(rel)
            if f is not None:
                out.append(BlobCopy("markdown", rel, rel, f.sha256, f.size))
        return out

    def build(self) -> MigrationPlan:
        self.identity()
        self.build_papers()
        self.works_and_graphs()
        self.remaining_project_documents()
        self.text()
        self.notes()
        self.current_elsewhere()
        self.references()
        self.lenses()
        self.concepts()
        self.finalize_ledger()
        for (slug, table, v2_id), (t3, id3) in sorted(self.idmap.items()):
            self.lib["v2_id_map"].append({"v2_project": slug, "v2_table": table, "v2_id": v2_id,
                                          "v3_table": t3, "v3_id": id3})
        for table, rows in self.lib.items():
            pk = PK[table]
            rows.sort(key=lambda r: tuple(_sort_key(r[c]) for c in pk))
        attention: dict[str, AttentionSpec] = {}
        for a in sorted(self.attention, key=AttentionSpec.sort_key):
            attention.setdefault(json.dumps([a.kind, a.subject], sort_keys=True), a)
        self.report["projects"] = {s: ("graph:" + GRAPH_PROJECTS[s] if s in GRAPH_PROJECTS else
                                       "folded:" + FOLD_INTO[s] if s in FOLD_INTO else
                                       "not_carried" if s.startswith("dbg_") else "library_only")
                                   for s in self.slugs}
        self.report["identity"] = {
            "rule_merges": [{"work_a": list(m.work_a), "work_b": list(m.work_b), "key_type": m.key_type,
                             "key_value": m.key_value} for m in self.ident.rule_merges],
            "blocked": [{"work_a": list(b.work_a), "work_b": list(b.work_b), "key_type": b.key_type,
                         "key_value": b.key_value, "conflict_types": list(b.conflict_types)} for b in self.ident.blocked],
            "skipped_never_merge": [{"work_a": list(m.work_a), "work_b": list(m.work_b), "key_type": m.key_type,
                                     "key_value": m.key_value} for m in self.ident.skipped_never_merge],
        }
        self.report["answers"] = {"saved": self.s.answer_files, "run_scoped_not_carried": self.s.run_answer_files}
        return MigrationPlan(
            taken_at=self.s.taken_at, library=self.lib,
            provider_cache=[dict(r) for r in sorted(self.s.cache.rows("provider_cache"),
                                                    key=lambda r: r["provider_cache_id"])],
            graphs=self.graphs, blobs=self.blobs(), attention=list(attention.values()), ledger=self.ledger,
            report=self.report, errors=self.errors, warnings=self.warnings, identity=self.ident,
        )


def _authors_json(authors: str | None) -> str | None:
    if authors is None or (isinstance(authors, str) and not authors.strip()):
        return None
    try:
        value = json.loads(authors)
    except (TypeError, ValueError):
        value = [authors]
    if isinstance(value, str):
        value = [value]
    return json.dumps(value, ensure_ascii=False)


def _parser_version(v: str | None) -> tuple[int, str]:
    m = re.search(r"(\d+)$", v or "")
    return (int(m.group(1)) if m else -1, v or "")


def _parsed_ordinal(r: Mapping[str, Any]) -> tuple[int, int]:
    try:
        o = json.loads(r.get("parsed_fields_json") or "{}").get("ordinal")
        if isinstance(o, int):
            return (0, o)
    except (TypeError, ValueError, AttributeError):
        pass
    return (1, 0)


def build_plan(snapshot: Snapshot, *, graph_names: Mapping[str, str] | None = None) -> MigrationPlan:
    """The pure plan builder (an internal seam; tested directly and via run_migration)."""
    return _Builder(snapshot, graph_names).build()
