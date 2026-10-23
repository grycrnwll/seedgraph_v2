-- library.db 0002 — papers, identifiers, paper_files and the two derived views
-- (v3 design §2.1 "papers" and "files").
--
-- `origin` (every non-verbatim table): 'v2:<slug>' or 'migration' for rows the
-- migration creates; 'library' (the default) for rows written later by library code.

CREATE TABLE papers (
    paper_id     TEXT PRIMARY KEY,               -- 'lp_' + sha256(key)[:20]; minted once, never re-derived (§4.1)
    title        TEXT,                           -- NULL for an unheld cached PDF with no metadata
    title_source TEXT,                           -- e.g. 'v2', 'provider_cache', 'user'
    authors_json TEXT,                           -- JSON array of strings
    year         INTEGER,
    venue        TEXT,
    abstract     TEXT,
    oa_status    TEXT,
    created_at   TEXT NOT NULL,
    origin       TEXT NOT NULL DEFAULT 'library'
);

CREATE TABLE identifiers (
    paper_id   TEXT NOT NULL REFERENCES papers(paper_id),
    id_type    TEXT NOT NULL CHECK (id_type IN ('doi', 'openalex', 'arxiv', 's2', 'ssrn')),
    id_value   TEXT NOT NULL,
    source     TEXT,
    confidence REAL,
    origin     TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (id_type, id_value)
);
CREATE INDEX ix_identifiers_paper ON identifiers (paper_id);

-- Many-to-many: one PDF may be linked to several papers (the flagged shared-PDF
-- case, §7.3). At most one primary file per paper.
CREATE TABLE paper_files (
    paper_id         TEXT NOT NULL REFERENCES papers(paper_id),
    source_file_id   TEXT NOT NULL REFERENCES source_files(source_file_id),
    markdown_id      TEXT REFERENCES markdown_documents(markdown_id),  -- the paper's current markdown
    role             TEXT NOT NULL DEFAULT 'primary' CHECK (role IN ('primary')),
    pdf_check_status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (pdf_check_status IN ('pass', 'weak', 'miss', 'unknown')),
    pdf_check_json   TEXT,
    attached_at      TEXT NOT NULL,
    origin           TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (paper_id, source_file_id)
);
CREATE UNIQUE INDEX ux_paper_files_primary ON paper_files (paper_id) WHERE role = 'primary';
CREATE INDEX ix_paper_files_source ON paper_files (source_file_id);
CREATE INDEX ix_paper_files_markdown ON paper_files (markdown_id);

-- Held vs cited-only is derived, never stored (§2.1 notes).
CREATE VIEW paper_status AS
SELECT p.paper_id,
       CASE WHEN EXISTS (SELECT 1 FROM paper_files f WHERE f.paper_id = p.paper_id)
            THEN 'held' ELSE 'cited_only' END AS status
FROM papers p;

-- paper -> primary file -> current markdown -> the converter_version of the run
-- that produced it. Markdown/run columns are NULL until the file is converted.
CREATE VIEW paper_converter AS
SELECT f.paper_id,
       f.source_file_id,
       f.markdown_id,
       m.conversion_run_id,
       r.converter_name,
       r.converter_version
FROM paper_files f
LEFT JOIN markdown_documents m ON m.markdown_id = f.markdown_id
LEFT JOIN conversion_runs r ON r.conversion_run_id = m.conversion_run_id
WHERE f.role = 'primary';
