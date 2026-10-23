-- library.db 0003 — sections, spans and span_fts (v3 design §2.1 "text").
--
-- v2's DDL (project/0005) keyed by markdown_id with work_id DROPPED: text belongs
-- to a markdown, not a paper. Papers reach their text through
-- paper_files.markdown_id. markdown_id / source_file_id are now real FKs (same file).
-- Offsets are Python str code-point indices into the decoded markdown.

CREATE TABLE document_sections (
    section_id             TEXT    PRIMARY KEY,   -- deterministic 'sec_' + sha256(markdown_hash|ordinal)[:16]
    markdown_id            TEXT    NOT NULL REFERENCES markdown_documents(markdown_id),
    markdown_hash          TEXT    NOT NULL,
    source_file_id         TEXT    NOT NULL REFERENCES source_files(source_file_id),
    source_file_hash       TEXT    NOT NULL,
    parent_section_id      TEXT,                   -- SOFT self-ref
    level                  INTEGER NOT NULL,
    ordinal                INTEGER NOT NULL,
    heading_text           TEXT,
    heading_path           TEXT,
    section_kind           TEXT    NOT NULL DEFAULT 'body',
    start_char             INTEGER NOT NULL,
    end_char               INTEGER NOT NULL,
    page_start             INTEGER,
    page_end               INTEGER,
    section_parser_version TEXT    NOT NULL,
    created_at             TEXT    NOT NULL,
    origin                 TEXT    NOT NULL DEFAULT 'library',
    UNIQUE (markdown_id, ordinal)
);
CREATE INDEX ix_sections_markdown ON document_sections (markdown_id);

CREATE TABLE evidence_spans (
    span_id          TEXT    PRIMARY KEY,          -- manual: random; auto: deterministic auto_span_id(...)
    markdown_id      TEXT    NOT NULL REFERENCES markdown_documents(markdown_id),
    markdown_hash    TEXT    NOT NULL,
    source_file_id   TEXT    NOT NULL REFERENCES source_files(source_file_id),
    source_file_hash TEXT    NOT NULL,
    section_id       TEXT,                          -- SOFT ref -> document_sections.section_id
    start_char       INTEGER NOT NULL,
    end_char         INTEGER NOT NULL,
    exact_quote      TEXT    NOT NULL,
    quote_hash       TEXT    NOT NULL,
    norm_version     TEXT    NOT NULL DEFAULT 'nfc-1',
    span_kind        TEXT    NOT NULL DEFAULT 'manual',  -- paragraph|section|manual
    page_start       INTEGER,
    page_end         INTEGER,
    access_class     TEXT    NOT NULL DEFAULT 'user_supplied_private',  -- provenance only (ADR-0012)
    anchor_status    TEXT    NOT NULL DEFAULT 'anchored',
    created_at       TEXT    NOT NULL,
    origin           TEXT    NOT NULL DEFAULT 'library'
);
CREATE INDEX ix_spans_markdown  ON evidence_spans (markdown_id);
CREATE INDEX ix_spans_quotehash ON evidence_spans (quote_hash);
CREATE INDEX ix_spans_md_kind   ON evidence_spans (markdown_id, span_kind);
CREATE INDEX ix_spans_manual_key ON evidence_spans (markdown_id, start_char, end_char, span_kind);

-- FTS5 as v2 (unicode61, remove_diacritics 2, prefix 2 3, no stemming) without work_id.
CREATE VIRTUAL TABLE span_fts USING fts5(
    quote_text,
    span_id     UNINDEXED,
    markdown_id UNINDEXED,
    section_id  UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2',
    prefix   = '2 3'
);
