-- library.db 0011 — traversal records and the session log (v3 stage 5 design §2, §3.1-3.7).
--
-- Records, events and sessions are stored once in the library, not per graph; a graph
-- lists its records through record_graphs (§2.2). This supersedes the graph.db reservation
-- of traversal_records / traversal_events (stage 1 §3), which created no tables.
--
-- Conventions as stage 1-4: TEXT ids, origin, explicit ordering (traversal_events.seq is
-- assigned MAX + 1 inside the writing transaction, one order across the whole library, so
-- a session family's interleaved log needs no rowid tie-break). Timestamps are UTC
-- ISO-8601 text in one fixed format so they compare as strings.
--
-- Item tags (found / opened / cited) are derived from events and evidence, never stored
-- (seedgraph.records.path). "In progress" and "path without answer" are display only:
-- both are state = 'open' (§3.5). No seed rows: an empty library stays replaceable.

-- One row per MCP server process (stdio: one process per client connection).
CREATE TABLE record_connections (
    connection_id  TEXT PRIMARY KEY,                     -- the process worker_id ('mcp-' + hex)
    client         TEXT,                                 -- from the initialize handshake; NULL = unknown
    started_at     TEXT NOT NULL,                        -- its first logged call
    closed_at      TEXT,                                 -- set when stdin ends
    meta_keys_json TEXT NOT NULL DEFAULT '[]',           -- identity probe: sorted union of MCP _meta keys
    origin         TEXT NOT NULL DEFAULT 'library'
);

-- One row per stream (§3.1). Named: keyed by (client, name), survives restarts; a helper
-- ('<parent>/<suffix>') points to its parent. Connection: keyed by the connection, named
-- by its start date and time (local, 'YYYY-MM-DD HH:MM', '-2' on a same-minute collision).
CREATE TABLE record_sessions (
    session_id        TEXT PRIMARY KEY,                  -- 'ss_' + hash of the key
    key_kind          TEXT NOT NULL CHECK (key_kind IN ('named', 'connection')),
    client            TEXT,
    name              TEXT NOT NULL,                     -- the session name, or the start date/time label
    connection_id     TEXT REFERENCES record_connections(connection_id),  -- connection streams only
    parent_session_id TEXT REFERENCES record_sessions(session_id),        -- helper streams only
    first_at          TEXT NOT NULL,
    last_at           TEXT NOT NULL,
    origin            TEXT NOT NULL DEFAULT 'library',
    CHECK ((key_kind = 'connection') = (connection_id IS NOT NULL)),
    CHECK (parent_session_id IS NULL OR key_kind = 'named')
);
CREATE UNIQUE INDEX ux_record_sessions_label ON record_sessions (name) WHERE key_kind = 'connection';
CREATE INDEX ix_record_sessions_parent ON record_sessions (parent_session_id);

CREATE TABLE traversal_records (
    record_id         TEXT PRIMARY KEY,                  -- 'tr_' + time-ordered hex
    session_id        TEXT NOT NULL REFERENCES record_sessions(session_id),  -- the owning (parent) stream
    state             TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'answered')),
    opened_at         TEXT NOT NULL,
    last_event_at     TEXT NOT NULL,
    answered_at       TEXT,
    question          TEXT,
    answer            TEXT,
    judgement         TEXT,                              -- the agent's own view, kept apart
    evidence_json     TEXT,                              -- frozen at answer time (§3.4)
    flags_json        TEXT NOT NULL DEFAULT '[]',        -- the citation check, frozen at answer time
    counts_json       TEXT NOT NULL DEFAULT '{}',        -- searches, opened, cited
    unlogged_calls    INTEGER NOT NULL DEFAULT 0 CHECK (unlogged_calls >= 0),
    starred           INTEGER NOT NULL DEFAULT 0 CHECK (starred IN (0, 1)),
    annotation        TEXT,
    annotated_at      TEXT,
    record_version    INTEGER NOT NULL DEFAULT 1,        -- readers tolerate later fields (additive rule)
    seedgraph_version TEXT NOT NULL,
    origin            TEXT NOT NULL DEFAULT 'library',
    CHECK ((state = 'answered') = (answered_at IS NOT NULL))
);
-- The open record of a stream: at most one (§3.5).
CREATE UNIQUE INDEX ux_traversal_records_open ON traversal_records (session_id) WHERE state = 'open';
CREATE INDEX ix_traversal_records_opened ON traversal_records (opened_at, record_id);

-- Content is immutable; record_id changes at most once, when a helper records its own
-- answer and its events move out of the parent's open record (§3.1). session_id is the
-- stream that made the call (a helper's own), so the path can say which helper it was.
CREATE TABLE traversal_events (
    event_id      TEXT PRIMARY KEY,                      -- 'te_' + time-ordered hex
    record_id     TEXT NOT NULL REFERENCES traversal_records(record_id),
    session_id    TEXT NOT NULL REFERENCES record_sessions(session_id),
    seq           INTEGER NOT NULL UNIQUE CHECK (seq >= 1),  -- one library-wide order
    at            TEXT NOT NULL,
    tool          TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('search', 'open_paper', 'open_text', 'open_concept')),
    connection_id TEXT NOT NULL REFERENCES record_connections(connection_id),
    args_json     TEXT NOT NULL,
    items_json    TEXT NOT NULL,                         -- ids, ranges and short previews only
    origin        TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX ix_traversal_events_record ON traversal_events (record_id, seq);
CREATE INDEX ix_traversal_events_session ON traversal_events (session_id, seq);

-- Which graphs a record belongs to (§3.6, ticket 07 #5). slug is a SOFT ref: a record
-- outlives its graph's deletion.
CREATE TABLE record_graphs (
    record_id TEXT NOT NULL REFERENCES traversal_records(record_id),
    slug      TEXT NOT NULL,
    how       TEXT NOT NULL CHECK (how IN ('searched', 'cited_member')),
    origin    TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (record_id, slug)
);
CREATE INDEX ix_record_graphs_slug ON record_graphs (slug, record_id);
