-- project.db — Phase 2 (Citation Graph MVP). Authoritative schema (decision D6).
--
-- Creates the durable citation layer tables: `reference_entries` (created now,
-- POPULATED LATER by phase_3b) and `citation_edges` (populated by phase_2's
-- offline provider-edge projection). Both live in project.db (one project per
-- file — NO project_id column; decisions 12/79).
--
-- Foreign keys target `works(work_id)`, authored by the project-model phase in
-- project/0002_*.sql (lower-numbered migration applied first). SQLite resolves a
-- foreign key's parent table at DML time, not at CREATE TABLE time, so these
-- CREATEs apply cleanly regardless of whether `works` exists yet; enforcement
-- (foreign_keys=ON, set by db/connection.py) bites only on row writes.
--
-- This file is the SINGLE SOURCE OF TRUTH for these tables (D6). The
-- ReferenceEntry / CitationEdge SQLModel classes are ORM mapping ONLY and must
-- mirror this schema; create_all() never authors/evolves it.

-- 4.1 — reference_entries: created now, populated in the parsed-bibliography
-- phase (phase_3b). Its reference_id is a prefixed-opaque 'ref_<uuid>' because it
-- is CROSS-REFERENCED (citation_edges.reference_id + review_queue.target_id);
-- decision 44 overrides the lenient decision 29 here.
CREATE TABLE reference_entries (
    reference_id        TEXT PRIMARY KEY,                          -- 'ref_<uuid>'
    citing_work_id      TEXT NOT NULL,                             -- FK -> works.work_id
    raw_reference_text  TEXT NOT NULL,                             -- PRIVATE (full-text-derived); kept verbatim for recall
    parsed_fields_json  TEXT,                                      -- {first_author, year, title, doi, arxiv} best-effort
    resolved_work_id    TEXT,                                      -- nullable FK -> works.work_id
    resolution_status   TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (resolution_status IN ('resolved', 'unresolved', 'ambiguous', 'suspect')),
    resolution_source   TEXT
        CHECK (resolution_source IS NULL OR resolution_source IN ('doi', 'arxiv', 'title_year')),
    confidence          REAL
        CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    markdown_id         TEXT,                                      -- cache.db markdown_documents.markdown_id (soft xref; set by parsed phase)
    markdown_hash       TEXT,                                      -- content anchor for staleness (decision 2/24; set by parsed phase)
    section_label       TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (citing_work_id)   REFERENCES works(work_id),
    FOREIGN KEY (resolved_work_id) REFERENCES works(work_id)
);

CREATE INDEX idx_reference_entries_citing   ON reference_entries(citing_work_id);
CREATE INDEX idx_reference_entries_resolved ON reference_entries(resolved_work_id);
CREATE INDEX idx_reference_entries_status   ON reference_entries(resolution_status);

-- 4.2 — citation_edges: union storage + dedup on
-- (source, target, edge_type, provenance, run_id). The AUTHORITATIVE edge per
-- (source, target) per run = highest-authority provenance
-- (manual_override > parsed_bibliography > provider_reference). Phase_2 writes
-- only `provider_reference`; the seam is in place for the parsed layer (phase_3b).
-- target_work_id is NEVER NULL (decision 59) — edges connect only existing works
-- (D3), so the FK is real by construction and edges never dangle.
CREATE TABLE citation_edges (
    edge_id          INTEGER PRIMARY KEY,                          -- pure edge row id (decision 44: internal-only, NEVER cross-referenced)
    source_work_id   TEXT NOT NULL,                                -- FK -> works.work_id (citing)
    target_work_id   TEXT NOT NULL,                                -- FK -> works.work_id (cited); NEVER NULL (decision 59)
    edge_type        TEXT NOT NULL DEFAULT 'cites',                -- 'cites' only this phase
    provenance       TEXT NOT NULL
        CHECK (provenance IN ('provider_reference', 'parsed_bibliography', 'manual_override')),
    confidence       REAL NOT NULL
        CHECK (confidence >= 0.0 AND confidence <= 1.0),           -- provider=1.0
    reference_id     TEXT,                                         -- nullable FK -> reference_entries (NULL for provider edges; set by parsed phase)
    run_id           TEXT NOT NULL,                                -- pipeline run scope (decision 16/35)
    created_at       TEXT NOT NULL,
    UNIQUE (source_work_id, target_work_id, edge_type, provenance, run_id),
    FOREIGN KEY (source_work_id) REFERENCES works(work_id),
    FOREIGN KEY (target_work_id) REFERENCES works(work_id),
    FOREIGN KEY (reference_id)   REFERENCES reference_entries(reference_id)
);

CREATE INDEX idx_citation_edges_source ON citation_edges(source_work_id);
CREATE INDEX idx_citation_edges_target ON citation_edges(target_work_id);
CREATE INDEX idx_citation_edges_run    ON citation_edges(run_id);
