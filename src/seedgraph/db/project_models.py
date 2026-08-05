"""ORM-mapping-only SQLModel classes for project.db (phase_5).

These mirror the authoritative schema in
``db/schema/project/0002_project_model.sql`` (works, identifiers,
project_documents) plus a mapping for the phase_0-owned ``review_queue`` table.

**Decision D6 (binding).** These classes are ORM MAPPING ONLY. They are *never*
used to author or evolve schema — no code path calls
``SQLModel.metadata.create_all`` against a project.db. Schema is applied solely by
the numbered ``.sql`` migrations via the ``migrate`` step
(:mod:`seedgraph.db.migrations`). The skipped ``test_phase5_schema_parity`` test
asserts this metadata equals the migrated schema and that ``create_all`` is never
called to author schema.

Identity conventions (D1): ``works.work_id`` is a mutable-identity uuid row
(``"work_" + uuid4().hex``); ``identifiers.identifier_id`` is a pure-join INTEGER
PK (decision 7/44).
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import JSON, Column, Index, UniqueConstraint
from sqlmodel import Field, SQLModel


class Work(SQLModel, table=True):
    """A flat canonical scholarly object (decisions 1/25/41).

    ``authors`` is a JSON array of strings; ``title_hash`` is the indexed fuzzy
    matcher (``sha1(normalize_title(title))``) that NEVER auto-merges. Mirrors the
    ``works`` table in 0002_project_model.sql.
    """

    __tablename__ = "works"

    work_id: str = Field(primary_key=True)
    canonical_title: Optional[str] = None
    title_hash: Optional[str] = Field(default=None, index=True)
    authors: Optional[list[str]] = Field(default=None, sa_column=Column(JSON))
    venue: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = Field(default=None, index=True)
    arxiv_id: Optional[str] = Field(default=None, index=True)
    openalex_id: Optional[str] = Field(default=None, index=True)
    semantic_scholar_id: Optional[str] = Field(default=None, index=True)
    ssrn_id: Optional[str] = Field(default=None, index=True)
    created_at: str
    # Build D ch10 (0017_works_abstract_oa_status.sql) — mapping only (D6).
    # abstract is LOCAL-ONLY (CONTENT_ACCESS_POLICY.md:42; excluded from every
    # export tier by the D8 allowlist); oa_status is the free-text OA color
    # (gold/green/hybrid/bronze/diamond/closed) powering the triage split.
    abstract: Optional[str] = None
    oa_status: Optional[str] = None


class Identifier(SQLModel, table=True):
    """External-id -> work resolver row (pure join table; decision 7/44).

    ``UNIQUE(id_type, id_value)`` guarantees an external id maps to exactly one
    work in-project. Mirrors the ``identifiers`` table in 0002_project_model.sql.
    """

    __tablename__ = "identifiers"
    __table_args__ = (UniqueConstraint("id_type", "id_value", name="uq_identifiers_type_value"),)

    identifier_id: Optional[int] = Field(default=None, primary_key=True)
    work_id: str = Field(foreign_key="works.work_id", index=True)
    id_type: str
    id_value: str
    resolution_source: Optional[str] = None
    confidence: Optional[float] = None


class ProjectDocument(SQLModel, table=True):
    """Corpus-membership row (decision 12 — no project_id; PK = work_id, 1:1).

    ``inclusion_status`` is the settled closed enum (included|metadata_only|
    excluded; decision 13/59). ``access_status`` mirrors the resolved access_class
    and is ALWAYS NULL this phase (decision 11/30 — never derived from membership).
    Mirrors the ``project_documents`` table in 0002_project_model.sql.
    """

    __tablename__ = "project_documents"

    work_id: str = Field(primary_key=True, foreign_key="works.work_id")
    inclusion_status: str = Field(index=True)
    inclusion_reason: Optional[str] = None
    is_seed: int = Field(default=0, index=True)
    access_status: Optional[str] = None
    user_note: Optional[str] = None
    created_at: str
    updated_at: str


class WorkSourceFile(SQLModel, table=True):
    """ORM mapping for the phase_5b-owned ``work_source_files`` bridge (decision 3).

    Mirrors ``db/schema/project/0003_acquisition.sql`` (D6 — mapping only, never
    ``create_all``-authored; a test asserts ORM metadata == migrated schema).
    ``UNIQUE(work_id)`` is the single-primary guarantee (must-fix #3): exactly one
    bridge row per work, so ``resolve_work_markdown`` (acquisition/bridge.py) is a
    deterministic plain ``WHERE work_id = ?`` read — NO ``role`` / ``manifestation``
    column (decision 1 deferral; NEW-A). ``source_file_id`` / ``markdown_id`` are
    SOFT xrefs into ``cache.db`` anchored on content hashes (D1) — no cross-DB FK.
    """

    __tablename__ = "work_source_files"

    work_source_file_id: Optional[int] = Field(default=None, primary_key=True)
    work_id: str = Field(foreign_key="works.work_id", unique=True)  # ONE row per work
    source_file_id: str
    file_hash: str = Field(index=True)
    markdown_id: Optional[str] = None
    markdown_hash: Optional[str] = Field(default=None, index=True)
    acquisition_method: str
    created_at: str
    updated_at: str


class ReviewQueueItem(SQLModel, table=True):
    """ORM mapping for the phase_0-owned ``review_queue`` table (decision 80).

    This phase does NOT author or alter ``review_queue`` (it is created in
    project/0001_foundation.sql); this class only maps the existing table so the
    review surface (:mod:`seedgraph.project.review`) can read/write it via the ORM.
    The polymorphic ``payload`` is a JSON string validated per ``item_type`` by the
    ``ReviewPayload`` discriminated union at the app boundary before persistence.
    """

    __tablename__ = "review_queue"

    item_id: str = Field(primary_key=True)
    item_type: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    payload: Optional[str] = None
    status: str = Field(default="open", index=True)
    action: Optional[str] = None
    created_at: str
    resolved_at: Optional[str] = None


# ===========================================================================
# phase_2 — citation layer (ORM MAPPING ONLY; mirrors project/0004_citation_graph.sql)
# ===========================================================================
# D6 (binding): the numbered .sql is the SOLE schema authority. These two classes
# are ORM mapping only and are NEVER used to author/evolve schema (no create_all);
# ``test_migration_parity_orm_equals_migrated_schema`` (tests/test_phase_2.py)
# asserts column / type / nullability / PK / index / UNIQUE / FK parity against the
# migrated 0004 schema.


class ReferenceEntry(SQLModel, table=True):
    """ORM mapping for ``reference_entries`` (project.db) — mirrors 0004 §4.1.

    Created now, POPULATED LATER by the deferred parsed-bibliography phase
    (phase_3b); phase_2 leaves it EMPTY. ``reference_id`` is a prefixed-opaque
    ``'ref_<uuid>'`` because it is CROSS-REFERENCED (``citation_edges.reference_id``
    + ``review_queue.target_id``; decision 44). The three explicit indexes
    (``citing_work_id`` / ``resolved_work_id`` / ``resolution_status``) mirror the
    migration's ``idx_reference_entries_*``.
    """

    __tablename__ = "reference_entries"

    reference_id: str = Field(primary_key=True)
    citing_work_id: str = Field(foreign_key="works.work_id", index=True)
    raw_reference_text: str
    parsed_fields_json: Optional[str] = None
    resolved_work_id: Optional[str] = Field(
        default=None, foreign_key="works.work_id", index=True
    )
    resolution_status: str = Field(default="unresolved", index=True)
    resolution_source: Optional[str] = None
    confidence: Optional[float] = None
    markdown_id: Optional[str] = None
    markdown_hash: Optional[str] = None
    section_label: Optional[str] = None
    created_at: str


class CitationEdge(SQLModel, table=True):
    """ORM mapping for ``citation_edges`` (project.db) — mirrors 0004 §4.2.

    Populated by phase_2's offline provider-edge projection. ``edge_id`` is a pure
    INTEGER row id (decision 44 — internal-only, never cross-referenced).
    ``UNIQUE(source_work_id, target_work_id, edge_type, provenance, run_id)`` is the
    union-storage + dedup key; the three explicit indexes
    (``source_work_id`` / ``target_work_id`` / ``run_id``) mirror the migration's
    ``idx_citation_edges_*``. ``target_work_id`` is NEVER NULL (decision 59) — edges
    connect only existing works (D3), so the FK is real by construction.
    """

    __tablename__ = "citation_edges"
    __table_args__ = (
        UniqueConstraint(
            "source_work_id",
            "target_work_id",
            "edge_type",
            "provenance",
            "run_id",
            name="uq_citation_edges_dedup",
        ),
    )

    edge_id: Optional[int] = Field(default=None, primary_key=True)
    source_work_id: str = Field(foreign_key="works.work_id", index=True)
    target_work_id: str = Field(foreign_key="works.work_id", index=True)
    edge_type: str = Field(default="cites")
    provenance: str
    confidence: float
    reference_id: Optional[str] = Field(
        default=None, foreign_key="reference_entries.reference_id"
    )
    run_id: str = Field(index=True)
    created_at: str


# ===========================================================================
# phase_3 — evidence spans (ORM MAPPING ONLY; mirrors project/0005_evidence_spans.sql)
# ===========================================================================
# D6 (binding): the numbered 0005 .sql is the SOLE schema authority. These two
# classes are ORM mapping only and are NEVER used to author/evolve schema (no
# create_all); ``test_migration_orm_metadata_parity`` (tests/test_phase_3.py)
# asserts column / type / nullability / PK / index / UNIQUE parity against the
# migrated 0005 schema. Cross-db refs (markdown_id / source_file_id pointing at
# cache.db) and section_id / parent_section_id are SOFT (non-FK) refs (D1 / §4.6);
# only work_id is a real intra-project FK.


class DocumentSection(SQLModel, table=True):
    """ORM mapping for ``document_sections`` (project.db) — mirrors 0005 §4.1.

    Deterministic ``section_id = 'sec_' + sha256(markdown_hash|ordinal)[:16]``
    (rebuild-safe). ``UNIQUE(markdown_id, ordinal)`` mirrors the migration; the two
    explicit indexes (``markdown_id`` / ``work_id``) mirror ``ix_sections_markdown`` /
    ``ix_sections_work``. ``parent_section_id`` is a SOFT self-ref (deterministic id),
    NOT FK-enforced, so re-parse never trips a foreign key.
    """

    __tablename__ = "document_sections"
    __table_args__ = (
        UniqueConstraint("markdown_id", "ordinal", name="uq_sections_markdown_ordinal"),
    )

    section_id: str = Field(primary_key=True)
    markdown_id: str = Field(index=True)  # ix_sections_markdown
    markdown_hash: str
    source_file_id: str
    source_file_hash: str
    work_id: str = Field(foreign_key="works.work_id", index=True)  # ix_sections_work
    parent_section_id: Optional[str] = None
    level: int
    ordinal: int
    heading_text: Optional[str] = None
    heading_path: Optional[str] = None
    section_kind: str = Field(default="body")
    start_char: int
    end_char: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_parser_version: str
    created_at: str


class EvidenceSpan(SQLModel, table=True):
    """ORM mapping for ``evidence_spans`` (project.db) — mirrors 0005 §4.2.

    ``exact_quote`` is AUTHORITATIVE (NOT NULL); ``quote_hash = sha256(NFC(exact_quote))``.
    Four explicit indexes mirror the migration: ``markdown_id`` (``ix_spans_markdown``),
    ``work_id`` (``ix_spans_work``), ``quote_hash`` (``ix_spans_quotehash``), and the
    composite ``(markdown_id, span_kind)`` (``ix_spans_md_kind``). ``section_id`` is a
    SOFT ref re-resolved on index (no FK); only ``work_id`` is a real FK.
    """

    __tablename__ = "evidence_spans"
    __table_args__ = (
        Index("ix_spans_md_kind", "markdown_id", "span_kind"),
    )

    span_id: str = Field(primary_key=True)
    markdown_id: str = Field(index=True)  # ix_spans_markdown
    markdown_hash: str
    source_file_id: str
    source_file_hash: str
    work_id: str = Field(foreign_key="works.work_id", index=True)  # ix_spans_work
    section_id: Optional[str] = None
    start_char: int
    end_char: int
    exact_quote: str
    quote_hash: str = Field(index=True)  # ix_spans_quotehash
    norm_version: str = Field(default="nfc-1")
    span_kind: str = Field(default="manual")
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    access_class: str = Field(default="user_supplied_private")
    anchor_status: str = Field(default="anchored")
    created_at: str
