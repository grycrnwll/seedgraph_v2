-- graph.db 0001 — one context graph (v3 design §1.2, §3).
-- Holds only graph-owned rows; paper ids and lens ids refer to the library SOFTLY.
-- Readers ATTACH the library read-only to join (seedgraph.library.graphs).
-- The in-graph citation view is derived through that ATTACH, never stored.
-- Reserved, NOT created here: literatures, literature_members (stage 4);
-- traversal_records, traversal_events (stage 5).

CREATE TABLE schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- key/value: slug (fixed), display_name (renameable), created_at, origin.
CREATE TABLE graph_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Members must be held papers (enforced by library/graphs.py, which can see the library).
CREATE TABLE graph_papers (
    paper_id TEXT PRIMARY KEY,
    status   TEXT NOT NULL CHECK (status IN ('member', 'frontier', 'excluded')),
    is_seed  INTEGER NOT NULL DEFAULT 0 CHECK (is_seed IN (0, 1)),
    reason   TEXT,
    added_at TEXT NOT NULL,
    origin   TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX ix_graph_papers_status ON graph_papers (status);

CREATE TABLE graph_lenses (
    lens_id   TEXT PRIMARY KEY,
    chosen_at TEXT NOT NULL,
    origin    TEXT NOT NULL DEFAULT 'library'
);
