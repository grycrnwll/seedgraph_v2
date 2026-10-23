-- library.db 0004 — extraction runs, versioned notes, claims, anchors and their FTS
-- (v3 design §2.1 "notes", §6.6).
--
-- v2 columns (project/0007 + 0008 chunk columns + 0009 lens_definition_hash) with
-- work_id -> paper_id. markdown_id on runs and notes stays a SOFT ref as in v2: it
-- is provenance (markdown_hash is the anchor), so a historical note survives even
-- when that markdown is no longer carried.
--
-- structured_notes adds version_no (1..n per paper_id, schema_id) and
-- version_label (NULL | 'wrong_pdf' | 'other_markdown'). Labelled versions are
-- never current and hidden from agent search.
-- Reserved for stage 3, NOT created here: note_overlays, overlay_versions.

CREATE TABLE extraction_runs (
    extraction_run_id        TEXT PRIMARY KEY,
    paper_id                 TEXT NOT NULL REFERENCES papers(paper_id),
    markdown_id              TEXT NOT NULL,        -- SOFT ref -> markdown_documents
    markdown_hash            TEXT NOT NULL,
    schema_id                TEXT,                 -- NULL for lens-origin runs
    lens_id                  TEXT,                 -- SOFT ref -> lens_definitions (lens_id, lens_definition_hash)
    schema_version           TEXT NOT NULL,
    prompt_version           TEXT NOT NULL,
    model_name               TEXT,
    provider                 TEXT,
    access_mode              TEXT,
    temperature              REAL,
    access_class             TEXT NOT NULL DEFAULT 'user_supplied_private',
    external_full_text       INTEGER NOT NULL DEFAULT 0,
    run_status               TEXT NOT NULL,
    input_tokens             INTEGER,
    output_tokens            INTEGER,
    estimated_cost           REAL,
    run_id                   TEXT,
    created_at               TEXT NOT NULL,
    extraction_mode          TEXT NOT NULL DEFAULT 'whole',
    parent_extraction_run_id TEXT,
    chunk_index              INTEGER,
    chunk_count              INTEGER,
    chunk_section_ids        TEXT,
    lens_definition_hash     TEXT,
    origin                   TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX idx_extr_paper  ON extraction_runs (paper_id, schema_id, created_at);
CREATE INDEX idx_extr_lens   ON extraction_runs (paper_id, lens_id, created_at);
CREATE INDEX idx_extr_parent ON extraction_runs (parent_extraction_run_id);

CREATE TABLE structured_notes (
    note_id           TEXT PRIMARY KEY,
    extraction_run_id TEXT NOT NULL UNIQUE REFERENCES extraction_runs(extraction_run_id),
    paper_id          TEXT NOT NULL REFERENCES papers(paper_id),
    markdown_id       TEXT NOT NULL,               -- SOFT ref
    markdown_hash     TEXT NOT NULL,
    schema_id         TEXT,
    schema_version    TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    archetype         TEXT,
    access_class      TEXT NOT NULL DEFAULT 'user_supplied_private',
    raw_note_json     TEXT NOT NULL,
    note_text         TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    version_no        INTEGER NOT NULL CHECK (version_no >= 1),
    version_label     TEXT CHECK (version_label IS NULL OR version_label IN ('wrong_pdf', 'other_markdown')),
    origin            TEXT NOT NULL DEFAULT 'library',
    UNIQUE (paper_id, schema_id, version_no)
);
CREATE INDEX idx_note_paper ON structured_notes (paper_id, schema_id, version_no);

-- The current note: the highest version_no per (paper_id, schema_id) among the
-- versions whose version_label IS NULL. Replaces v2's CURRENT_CLAIMS_CTE.
CREATE VIEW note_current AS
SELECT n.*
FROM structured_notes n
WHERE n.version_label IS NULL
  AND n.version_no = (
      SELECT MAX(o.version_no) FROM structured_notes o
      WHERE o.paper_id = n.paper_id
        AND o.schema_id IS n.schema_id
        AND o.version_label IS NULL
  );

CREATE TABLE extracted_claims (
    claim_id                 TEXT PRIMARY KEY,
    structured_note_id       TEXT REFERENCES structured_notes(note_id),  -- NULL for lens-origin claims
    extraction_run_id        TEXT NOT NULL REFERENCES extraction_runs(extraction_run_id),
    paper_id                 TEXT NOT NULL REFERENCES papers(paper_id),
    claim_type               TEXT NOT NULL,
    claim_subtype            TEXT,
    field_key                TEXT NOT NULL,
    normalized_label         TEXT,
    claim_text               TEXT,
    status                   TEXT NOT NULL,
    epistemic_type           TEXT NOT NULL,
    assertion_status         TEXT,
    inferred_explanation     TEXT,
    confidence               REAL,
    access_class             TEXT NOT NULL DEFAULT 'user_supplied_private',
    created_at               TEXT NOT NULL,
    source_chunk_index       INTEGER,
    source_extraction_run_id TEXT,
    origin                   TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX idx_claim_note       ON extracted_claims (structured_note_id);
CREATE INDEX idx_claim_run        ON extracted_claims (extraction_run_id);
CREATE INDEX idx_claim_paper_type ON extracted_claims (paper_id, claim_type);

-- Quote anchors, as v2 (span ids remapped by the migration's dedupe, §4.4).
CREATE TABLE claim_spans (
    id         INTEGER PRIMARY KEY,
    claim_id   TEXT NOT NULL REFERENCES extracted_claims(claim_id),
    span_id    TEXT NOT NULL REFERENCES evidence_spans(span_id),
    rank       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    origin     TEXT NOT NULL DEFAULT 'library',
    UNIQUE (claim_id, span_id)
);
CREATE INDEX idx_claim_spans_span ON claim_spans (span_id);

-- FTS5 as v2 (project/0007).
CREATE VIRTUAL TABLE claim_fts USING fts5(
    claim_id UNINDEXED, normalized_label, claim_text,
    tokenize = "unicode61 remove_diacritics 2"
);
CREATE VIRTUAL TABLE note_fts USING fts5(
    note_id UNINDEXED, note_text,
    tokenize = "unicode61 remove_diacritics 2"
);
