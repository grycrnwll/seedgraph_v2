-- project.db migration 0005 — Phase 3 "Evidence Spans" (authoritative schema, decision D6).
--
-- Authors the NEW Phase-3 tables: document_sections, evidence_spans, and the
-- span_fts FTS5 virtual table (populated). The claim<->span junction claim_spans
-- and the claim_fts / note_fts indexes are OWNED BY phase_4 (0007_notes.sql); this
-- migration applies first so 0007's claim_spans.span_id FK into evidence_spans
-- resolves. Claims attach to spans ONLY through that junction — no scalar span_id
-- link column anywhere (decision 70); phase_4 INSERTs a claim row + ensure_span()
-- + claim_spans row atomically in one transaction via db.adapter.raw_conn (D7).
--
-- Applies on top of phase_0 (0001_foundation.sql) and the project_model phase
-- (works / identifiers / project_documents in project/0002_*.sql): it REFERENCES
-- works(work_id) by exact name, so 0002 must already be applied (the migration
-- runner applies in numeric order, so this holds at apply time).
--
-- Schema-authority rules honored:
--   * No CREATE TABLE in Python / no create_all() authoring (D6).
--   * PRAGMA foreign_keys=ON is set by db/connection.py; cross-DB refs
--     (markdown_id, source_file_id pointing at cache.db) are SOFT (non-FK) by
--     construction — SQLite cannot enforce cross-database FKs. section_id and
--     parent_section_id are SOFT self/soft refs so re-parse never trips a FK.
--   * Content-addressed anchor (D1): cache markdown_id == "md_" + sha256(bytes)
--     and source_file_id == "sf_" + sha256(bytes); the denormalized
--     markdown_hash / source_file_hash columns equal those ids minus the prefix
--     and are the integrity surface doctor verifies.
--   * Offsets are Python str code-point indices (NOT UTF-8 byte offsets) into the
--     decoded markdown string — a documented producer/consumer contract.

-- 4.1 document_sections — deterministic ATX-heading sectioning substrate (NEW).
CREATE TABLE document_sections (
    section_id             TEXT    PRIMARY KEY,   -- deterministic 'sec_' + sha256(markdown_hash|ordinal)[:16]
    markdown_id            TEXT    NOT NULL,       -- SOFT cross-db ref -> cache.markdown_documents.markdown_id
    markdown_hash          TEXT    NOT NULL,       -- denormalized content anchor (staleness; section_id key)
    source_file_id         TEXT    NOT NULL,       -- denormalized cache.markdown_documents.source_file_id
    source_file_hash       TEXT    NOT NULL,       -- denormalized lineage anchor (survives cache GC)
    work_id                TEXT    NOT NULL REFERENCES works(work_id),
    parent_section_id      TEXT,                   -- SOFT self-ref (deterministic id), not FK-enforced
    level                  INTEGER NOT NULL,       -- 0 = preamble/no-heading, 1..6 = ATX depth
    ordinal                INTEGER NOT NULL,       -- 0-based sequence within the document
    heading_text           TEXT,                   -- NULL for preamble (private-by-default content)
    heading_path           TEXT,                   -- breadcrumb e.g. '2 Identification > 2.1 Assumptions'
    section_kind           TEXT    NOT NULL DEFAULT 'body',  -- body|abstract|references|appendix|acknowledgments
    start_char             INTEGER NOT NULL,       -- half-open [start,end) code-point offsets into markdown
    end_char               INTEGER NOT NULL,
    page_start             INTEGER,                -- best-effort from in-markdown delimiters, else NULL
    page_end               INTEGER,
    section_parser_version TEXT    NOT NULL,       -- staleness key for sectioning
    created_at             TEXT    NOT NULL,
    UNIQUE (markdown_id, ordinal)
);
CREATE INDEX ix_sections_work     ON document_sections(work_id);
CREATE INDEX ix_sections_markdown ON document_sections(markdown_id);

-- 4.2 evidence_spans — the atomic accountability unit (NEW).
-- exact_quote is AUTHORITATIVE (NOT NULL); quote_hash = sha256(NFC(exact_quote)).
CREATE TABLE evidence_spans (
    span_id          TEXT    PRIMARY KEY,          -- manual: new_id('span'); auto: deterministic auto_span_id(...)
    markdown_id      TEXT    NOT NULL,             -- SOFT cross-db ref -> cache.markdown_documents.markdown_id
    markdown_hash    TEXT    NOT NULL,             -- denormalized content anchor (staleness / rebuild-safe join)
    source_file_id   TEXT    NOT NULL,             -- denormalized lineage (cache.markdown_documents.source_file_id)
    source_file_hash TEXT    NOT NULL,             -- denormalized lineage anchor (staleness survives cache GC)
    work_id          TEXT    NOT NULL REFERENCES works(work_id),
    section_id       TEXT,                          -- SOFT ref -> document_sections.section_id (re-resolved on index)
    start_char       INTEGER NOT NULL,             -- half-open [start,end) code-point offsets
    end_char         INTEGER NOT NULL,
    exact_quote      TEXT    NOT NULL,             -- AUTHORITATIVE verbatim slice: markdown[start:end]
    quote_hash       TEXT    NOT NULL,             -- sha256(NFC(exact_quote)), lowercase hex
    norm_version     TEXT    NOT NULL DEFAULT 'nfc-1',
    span_kind        TEXT    NOT NULL DEFAULT 'manual',  -- paragraph|section|manual
    page_start       INTEGER,                      -- best-effort, NULL when no pagination delimiters
    page_end         INTEGER,
    access_class     TEXT    NOT NULL DEFAULT 'user_supplied_private',  -- fail-closed; stamped at create
    anchor_status    TEXT    NOT NULL DEFAULT 'anchored',  -- anchored|stale|orphaned
    created_at       TEXT    NOT NULL
);
CREATE INDEX ix_spans_markdown  ON evidence_spans(markdown_id);
CREATE INDEX ix_spans_work      ON evidence_spans(work_id);
CREATE INDEX ix_spans_quotehash ON evidence_spans(quote_hash);
CREATE INDEX ix_spans_md_kind   ON evidence_spans(markdown_id, span_kind);

-- 4.3 FTS5 — standalone (content-bearing) tables, project.db only.
-- unicode61 + no porter/stemming keeps `mixing` / `ergodicity` / `Assumption 2`
-- matchable verbatim; detail defaults to `full` so phrase queries work. Ids are
-- carried UNINDEXED for join-back and for clean batch invalidation by markdown_id
-- (DELETE FROM span_fts WHERE markdown_id=? then re-insert; no triggers).
CREATE VIRTUAL TABLE span_fts USING fts5(
    quote_text,
    span_id     UNINDEXED,
    markdown_id UNINDEXED,   -- enables DELETE FROM span_fts WHERE markdown_id=? (re-index)
    work_id     UNINDEXED,
    section_id  UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2',
    prefix   = '2 3'
);
-- claim_fts / note_fts are OWNED BY phase_4 (0007_notes.sql), NOT created here.
