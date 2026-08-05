"""Project-scope ORM mapping (decision D6 — MAPPING ONLY).

These SQLModel classes MIRROR the schema authored by the numbered ``.sql``
migrations; ``create_all()`` is NEVER used to author or evolve the schema (a
migration-parity test asserts ORM metadata == migrated schema). This module holds
phase_4's contribution — ``ExtractionRun``, ``StructuredNote``, ``ExtractedClaim``
(born in ``schema/project/0007_notes.sql``).

NOT mapped here:
  * ``claim_spans`` — born in this phase's ``0007_notes.sql`` (§4.4) but kept a
    pure junction with NO ORM class; phase_4 INSERTs junction rows via raw SQL in
    the same transaction as the claim + ``ensure_span`` (D7).
  * ``works`` / ``evidence_spans`` / ``document_sections`` — owned by phase_5 /
    phase_3; referenced by ``foreign_key=`` strings, resolved lazily.

All classes register into the shared ``SQLModel.metadata``, so other table-defining
phases add their own mapping-only classes (here or a sibling module) without
re-authoring schema.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field, SQLModel


class ExtractionRun(SQLModel, table=True):
    """ORM mirror of ``extraction_runs`` (plan §4.1) — per-unit provenance, append-only."""

    # ``model_name`` mirrors the SQL column but trips Pydantic's protected
    # ``model_`` namespace guard; opt out (no Pydantic model attrs are shadowed).
    model_config = {"protected_namespaces": ()}

    __tablename__ = "extraction_runs"

    extraction_run_id: str = Field(primary_key=True)
    work_id: str = Field(foreign_key="works.work_id")
    markdown_id: str
    markdown_hash: str
    # nullable: lens-origin runs set lens_id + lens_definition_hash and leave
    # schema_id NULL (sibling schema, mutually exclusive w/ schema_id; decision 22).
    schema_id: str | None = None
    lens_id: str | None = None
    schema_version: str
    prompt_version: str
    model_name: str | None = None
    provider: str | None = None
    access_mode: str | None = None
    temperature: float | None = None
    access_class: str = "user_supplied_private"
    external_full_text: int = 0
    run_status: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    run_id: str | None = None
    created_at: str
    # --- phase_4b chunk-provenance columns (mirror 0008_chunked.sql, D6) -------
    # 'whole' (phase_4) | 'chunked_map' (one per chunk) | 'chunked_reduce' (the
    # merged note's run). NOT NULL DEFAULT 'whole' so pre-existing phase_4 rows
    # back-fill; ORM default matches the SQL constant default.
    extraction_mode: str = "whole"
    # map runs point at their reduce run; NULL for 'whole' / 'chunked_reduce'.
    parent_extraction_run_id: str | None = None
    chunk_index: int | None = None  # 0-based; NULL unless 'chunked_map'
    chunk_count: int | None = None  # total chunks in the plan; NULL for 'whole'
    chunk_section_ids: str | None = None  # JSON array of covered document_sections ids
    # --- phase_6 lens version stamp (mirror 0009_lenses.sql ALTER, D6) ---------
    # NULL for default-schema (phase_4) runs; a lens run sets lens_id +
    # lens_definition_hash — together with markdown_hash the idempotency/staleness
    # key (§7). Mirrored so ORM metadata == the post-0009 migrated extraction_runs
    # (parity asserted by test_phase_6.test_lens_migration; the phase_4/4b parity
    # tests now apply migrations through 0009).
    lens_definition_hash: str | None = None


class StructuredNote(SQLModel, table=True):
    """ORM mirror of ``structured_notes`` (plan §4.2) — thin header, 1:1 with a run."""

    __tablename__ = "structured_notes"

    note_id: str = Field(primary_key=True)
    extraction_run_id: str = Field(
        foreign_key="extraction_runs.extraction_run_id", unique=True
    )
    work_id: str = Field(foreign_key="works.work_id")
    markdown_id: str
    markdown_hash: str
    schema_id: str | None = None  # nullable: lens-origin runs set lens_id not schema_id
    schema_version: str
    prompt_version: str
    archetype: str | None = None
    access_class: str = "user_supplied_private"
    raw_note_json: str
    note_text: str
    created_at: str


class ExtractedClaim(SQLModel, table=True):
    """ORM mirror of ``extracted_claims`` (plan §4.3) — one row per field/element.

    Carries BOTH ``epistemic_type`` (NOT NULL) and ``assertion_status`` (nullable)
    as independent columns (D2). No scalar ``evidence_span_id`` (decision 70).
    """

    __tablename__ = "extracted_claims"

    claim_id: str = Field(primary_key=True)
    # nullable: lens-origin claims mint no structured_notes header
    structured_note_id: str | None = Field(
        default=None, foreign_key="structured_notes.note_id"
    )
    extraction_run_id: str = Field(foreign_key="extraction_runs.extraction_run_id")
    work_id: str = Field(foreign_key="works.work_id")
    claim_type: str
    claim_subtype: str | None = None
    field_key: str
    normalized_label: str | None = None
    claim_text: str | None = None
    status: str
    epistemic_type: str
    assertion_status: str | None = None
    inferred_explanation: str | None = None
    confidence: float | None = None
    access_class: str = "user_supplied_private"
    created_at: str
    # --- phase_4b chunk-provenance columns (mirror 0008_chunked.sql, D6) -------
    # which map chunk produced this claim (NULL for whole/merged-scalar) + the map
    # run it was extracted in (audit back-link). Both nullable, additive.
    source_chunk_index: int | None = None
    source_extraction_run_id: str | None = None


# ===========================================================================
# phase_6 — project lenses. ORM MAPPING ONLY (D6): these mirror the schema
# authored by ``schema/project/0009_lenses.sql``; create_all() is NOT the
# authoring path (a migration-parity test asserts ORM metadata == migrated
# schema). The 0009 ``ALTER extraction_runs ADD lens_definition_hash`` column IS
# mirrored on ``ExtractionRun`` above (per the binding); accordingly the
# phase_4/phase_4b parity tests apply migrations through 0009 and
# test_phase_6.test_lens_migration asserts ORM == migrated for extraction_runs.
# ===========================================================================


class Lens(SQLModel, table=True):
    """ORM mirror of ``lenses`` (plan §4.1) — thin registry over the on-disk YAML.

    While ``status IN ('draft','calibrating')`` the lens is mutable: ``definition_hash``
    refreshes from the YAML and ``definition_yaml`` stays NULL. On first full run the
    lens promotes to ``active`` and ``definition_yaml`` is snapshotted once
    (immutability-on-use, decision 50). Mapping-only (D6)."""

    __tablename__ = "lenses"

    lens_id: str = Field(primary_key=True)
    name: str
    scope: str = "project"
    object_type: str
    status: str = "draft"
    yaml_path: str
    definition_hash: str
    definition_yaml: str | None = None
    created_at: str
    updated_at: str


class LensOutput(SQLModel, table=True):
    """ORM mirror of ``lens_outputs`` (plan §4.2) — one generic JSON container row
    per (work, record), incl. explicit ``not_found`` / ``not_applicable`` rows.

    ``fields_json`` round-trips the FULL validated output_schema instance verbatim
    (found AND not-found shapes). ``claim_id`` is NULL for not_found / not_applicable
    / ambiguous / extraction_failed; ``access_class`` is denormalized fail-closed
    (decision 60/76). Mapping-only (D6)."""

    __tablename__ = "lens_outputs"

    lens_output_id: str = Field(primary_key=True)
    extraction_run_id: str = Field(foreign_key="extraction_runs.extraction_run_id")
    lens_id: str = Field(foreign_key="lenses.lens_id")
    work_id: str = Field(foreign_key="works.work_id")
    status: str
    claim_id: str | None = Field(default=None, foreign_key="extracted_claims.claim_id")
    confidence: float | None = None
    fields_json: str
    access_class: str = "user_supplied_private"
    created_at: str


# ===========================================================================
# phase_7 — semantic graph overlay. ORM MAPPING ONLY (D6): these six classes
# mirror the schema authored by ``schema/project/0010_semantic_overlay.sql``;
# create_all() is NEVER the authoring path. ``test_orm_migration_parity``
# (tests/test_phase_7.py) asserts ORM metadata == migrated schema for all six.
# No project_id column anywhere (one project per project.db; decision 12).
# ===========================================================================


class Concept(SQLModel, table=True):
    """ORM mirror of ``concepts`` (plan §4.1) — canonical concept node.

    ``concept_id`` is DETERMINISTIC (``'concept::' || normalized_label``);
    ``normalized_label`` is the UNIQUE merge_key. ``concept_type`` reuses the
    single ``ClaimType`` vocab (decision 48). ``definition`` is full-text-derived
    (nullable; dropped on non-shareable export). ``access_class`` is denormalized,
    fail-closed = MAX over contributing ``extracted_claims.access_class``.
    Closed-vocab CHECKs on ``status`` / ``epistemic_type`` are SQL-enforced; the
    ORM mirrors columns/PK/UNIQUE only (D6)."""

    __tablename__ = "concepts"

    concept_id: str = Field(primary_key=True)
    normalized_label: str = Field(unique=True)
    canonical_label: str
    concept_type: str
    definition: Optional[str] = None
    paper_frequency: int = 0
    # Build B (0015): anti-stopword IDF weight — mapping-only mirror (D6).
    weight: float = 0.0
    status: str = "auto"
    epistemic_type: str = "deterministic"
    access_class: str = "user_supplied_private"
    run_id: Optional[str] = None
    created_at: str
    updated_at: str


class ConceptAlias(SQLModel, table=True):
    """ORM mirror of ``concept_aliases`` (plan §4.1) — TRUE-interchangeability fold.

    A surface label folded into a concept (``fold_reason`` ∈
    {exact_key, acronym, llm_proposed_reviewed}). Related-but-distinct terms are
    NEVER aliases — they become typed ``project_graph_edges``. ``alias_id`` is a
    pure-join INTEGER PK; ``UNIQUE(concept_id, alias_label)`` mirrors the
    migration. CHECKs on ``fold_reason`` / ``epistemic_type`` are SQL-enforced."""

    __tablename__ = "concept_aliases"
    __table_args__ = (
        UniqueConstraint("concept_id", "alias_label", name="uq_concept_aliases_concept_label"),
    )

    alias_id: Optional[int] = Field(default=None, primary_key=True)
    concept_id: str = Field(foreign_key="concepts.concept_id")
    alias_label: str
    fold_reason: str
    epistemic_type: str = "deterministic"
    run_id: Optional[str] = None
    created_at: str


class ClaimConcept(SQLModel, table=True):
    """ORM mirror of ``claim_concepts`` (plan §4.1) — Claim→Concept junction.

    High-volume deterministic junction (INTEGER PK). ``epistemic_type`` is
    ``deterministic`` (label match) or ``llm_extracted`` (LLM fold). Every concept
    has ≥1 row here — no orphan concept is created. ``UNIQUE(claim_id,
    concept_id)`` + the two named indexes mirror the migration."""

    __tablename__ = "claim_concepts"
    __table_args__ = (
        UniqueConstraint("claim_id", "concept_id", name="uq_claim_concepts_claim_concept"),
        Index("ix_claim_concepts_concept", "concept_id"),
        Index("ix_claim_concepts_work", "work_id"),
    )

    claim_concept_id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: str = Field(foreign_key="extracted_claims.claim_id")
    concept_id: str = Field(foreign_key="concepts.concept_id")
    work_id: str = Field(foreign_key="works.work_id")
    epistemic_type: str = "deterministic"
    confidence: Optional[float] = None
    run_id: Optional[str] = None
    created_at: str


class ProjectGraphEdge(SQLModel, table=True):
    """ORM mirror of ``project_graph_edges`` (plan §4.2) — constrained polymorphic
    interpretive overlay (deterministic citation edges stay in ``citation_edges``).

    SQLite cannot FK a polymorphic column; endpoint integrity is enforced at the
    single application insert path (``semantic.edges.insert_edge``) + a doctor
    scan. ``epistemic_type`` is the ONLY provenance column (D2 — no
    ``assertion_status`` on edges). CHECKs on node types + epistemic_type are
    SQL-enforced; the UNIQUE (incl. ``run_id``) + two named indexes mirror the
    migration."""

    __tablename__ = "project_graph_edges"
    __table_args__ = (
        UniqueConstraint(
            "source_node_type",
            "source_node_id",
            "target_node_type",
            "target_node_id",
            "edge_type",
            "run_id",
            name="uq_pge_dedup",
        ),
        Index("ix_pge_src", "source_node_type", "source_node_id"),
        Index("ix_pge_tgt", "target_node_type", "target_node_id"),
    )

    edge_id: str = Field(primary_key=True)
    source_node_type: str
    source_node_id: str
    target_node_type: str
    target_node_id: str
    edge_type: str
    epistemic_type: str
    confidence: Optional[float] = None
    # Build B (0015): co-occurrence shared-work count — mapping-only mirror (D6).
    shared_count: Optional[int] = None
    access_class: str = "user_supplied_private"
    run_id: Optional[str] = None
    created_at: str


class EdgeSpan(SQLModel, table=True):
    """ORM mirror of ``edge_spans`` (plan §4.1) — span-bound interpretive-edge
    junction. Shape DEFINED (decision 70) but NOT populated this phase. Composite
    PK ``(edge_id, span_id)``; FK ``edge_id`` → project_graph_edges (ON DELETE
    CASCADE), ``span_id`` → evidence_spans."""

    __tablename__ = "edge_spans"

    edge_id: str = Field(primary_key=True, foreign_key="project_graph_edges.edge_id")
    span_id: str = Field(primary_key=True, foreign_key="evidence_spans.span_id")


class ConceptConstraint(SQLModel, table=True):
    """ORM mirror of ``concept_constraints`` (plan §4.1) — STICKY human decisions.

    ``must_link`` / ``cannot_link`` pairs over normalized labels, consulted BEFORE
    every canon run so a resolved decision never flips on corpus growth. CHECK on
    ``kind`` is SQL-enforced; ``UNIQUE(kind, label_a, label_b)`` mirrors the
    migration. INTEGER PK (pure-join row)."""

    __tablename__ = "concept_constraints"
    __table_args__ = (
        UniqueConstraint("kind", "label_a", "label_b", name="uq_concept_constraints"),
    )

    constraint_id: Optional[int] = Field(default=None, primary_key=True)
    kind: str
    label_a: str
    label_b: str
    source: str = "user"
    created_at: str


# ===========================================================================
# phase_9 — evaluation. ORM MAPPING ONLY (D6): this class mirrors the schema
# authored by ``schema/project/0012_evaluation.sql``; create_all() is NEVER the
# authoring path. ``test_phase_9_audit.test_audit_records_orm_migration_parity``
# asserts ORM metadata == migrated schema for ``audit_records``.
#
# Boundary (phase_9 §4): audit_records only MEASURES (records the grade + the
# accept/reject/edit decision); it NEVER mutates graph state (that is
# review_queue's job, decisions 56/71/80). ``access_class`` defaults fail-closed
# to the most-restrictive ``user_supplied_private`` (decision 60/76); the python
# stamping rule in ``eval/audit.py`` propagates it. ``subject_id`` is a deliberate
# soft polymorphic ref over heterogeneous subjects — NO cross-table FK.
# ===========================================================================


class AuditRecord(SQLModel, table=True):
    """ORM mirror of ``audit_records`` (phase_9 §4) — append-only graded verdicts.

    Every column / PK / index mirrors ``0012_evaluation.sql`` plus the additive
    ``payload`` column from ``0016_audit_payload.sql`` (D6). The three
    non-unique indexes (``ix_audit_open`` over ``(status, audit_type)``,
    ``ix_audit_batch`` over ``sample_batch_id``, ``ix_audit_subject`` over
    ``(subject_type, subject_id)``) are declared in ``__table_args__`` so the
    parity test's index column-set comparison matches the migration exactly. No
    UNIQUE constraints, no ``subject_fingerprint``/``is_stale`` (staleness deferred),
    no ``project_id`` (one project.db == one project, decision 12/79)."""

    __tablename__ = "audit_records"
    __table_args__ = (
        Index("ix_audit_open", "status", "audit_type"),
        Index("ix_audit_batch", "sample_batch_id"),
        Index("ix_audit_subject", "subject_type", "subject_id"),
    )

    audit_id: str = Field(primary_key=True)
    audit_type: str
    subject_type: str
    subject_id: str
    run_id: Optional[str] = None
    sample_batch_id: Optional[str] = None
    sample_seed: Optional[str] = None
    verdict: Optional[str] = None
    severity: Optional[str] = None
    problem: Optional[str] = None
    recommended_fix: Optional[str] = None
    decision: Optional[str] = None
    edit_payload: Optional[str] = None
    reviewer: Optional[str] = None
    status: str = "open"
    access_class: str = "user_supplied_private"
    created_at: str
    resolved_at: Optional[str] = None
    # 0016 (Build C chunk 9, D10/decision 56): JSON canon decision log for
    # audit_type='canon_decision_log' rows; NULL for every other audit_type.
    # Distinct from edit_payload (reserved for decision='edit' semantics).
    payload: Optional[str] = None
