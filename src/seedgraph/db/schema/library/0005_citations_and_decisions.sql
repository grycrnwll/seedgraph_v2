-- library.db 0005 — reference lists, citation links, citation review and duplicate
-- decisions (v3 design §2.1 "citations" and "identity", §6.7, §6.8).

-- One reference list per paper (§6.7); `ordinal` is the entry's order in it.
-- v2 columns (post project/0014) with citing_work_id -> citing_paper_id,
-- resolved_work_id -> resolved_paper_id. markdown_id stays a SOFT ref.
CREATE TABLE reference_entries (
    reference_id       TEXT PRIMARY KEY,
    citing_paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
    raw_reference_text TEXT NOT NULL,
    parsed_fields_json TEXT,
    resolved_paper_id  TEXT REFERENCES papers(paper_id),
    resolution_status  TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (resolution_status IN ('resolved', 'unresolved', 'ambiguous', 'suspect')),
    resolution_source  TEXT
        CHECK (resolution_source IS NULL OR resolution_source IN ('doi', 'arxiv', 'title_year', 'manual_override')),
    confidence         REAL
        CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    markdown_id        TEXT,
    markdown_hash      TEXT,
    section_label      TEXT,
    ordinal            INTEGER,
    created_at         TEXT NOT NULL,
    origin             TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX idx_reference_entries_citing   ON reference_entries (citing_paper_id, ordinal);
CREATE INDEX idx_reference_entries_resolved ON reference_entries (resolved_paper_id);
CREATE INDEX idx_reference_entries_status   ON reference_entries (resolution_status);

-- Library facts: v2's run_id and edge_type are dropped. Deduped on
-- (source, target, provenance), which is the primary key.
CREATE TABLE citation_links (
    source_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    target_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    provenance      TEXT NOT NULL
        CHECK (provenance IN ('provider_reference', 'parsed_bibliography', 'manual_override')),
    confidence      REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reference_id    TEXT REFERENCES reference_entries(reference_id),
    created_at      TEXT NOT NULL,
    origin          TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (source_paper_id, target_paper_id, provenance)
);
CREATE INDEX idx_citation_links_target ON citation_links (target_paper_id);

CREATE TABLE citation_review_decisions (
    reference_id         TEXT PRIMARY KEY REFERENCES reference_entries(reference_id),
    decision             TEXT NOT NULL CHECK (decision IN ('reject')),
    rejected_record_json TEXT,
    reason               TEXT,
    decided_at           TEXT NOT NULL,
    origin               TEXT NOT NULL DEFAULT 'library'
);

-- paper_a < paper_b. The two sides are SOFT refs (no FK): a rule merge records two
-- v2 works that became one paper, so a side need not be a live papers row.
CREATE TABLE duplicate_decisions (
    paper_a    TEXT NOT NULL,
    paper_b    TEXT NOT NULL,
    verdict    TEXT NOT NULL CHECK (verdict IN ('same', 'different')),
    decided_by TEXT NOT NULL CHECK (decided_by IN ('researcher', 'rule')),
    reason     TEXT,
    decided_at TEXT NOT NULL,
    origin     TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (paper_a, paper_b),
    CHECK (paper_a < paper_b)
);
CREATE INDEX ix_duplicate_decisions_b ON duplicate_decisions (paper_b);
