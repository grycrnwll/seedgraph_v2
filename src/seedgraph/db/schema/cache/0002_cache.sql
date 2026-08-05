-- cache.db migration 0002 — Corpus Cache MVP (phase_1).
-- Authoritative schema (decision D6): every CREATE TABLE / index / CHECK for
-- this phase's NEW cache objects lives here. SQLModel classes in
-- seedgraph/cache/models.py MIRROR this file for ORM use only — create_all is
-- never the authoring path (a phase_1 test asserts ORM metadata == this schema).
--
-- Applies on top of cache/0001_baseline.sql (which already created
-- schema_migrations + source_files). source_files is NEVER recreated here; the
-- two columns the phase-1 data model adds (byte_size, updated_at) are introduced
-- via ALTER TABLE. All FKs are intra-cache.db so PRAGMA foreign_keys=ON
-- (db/connection.py) is genuine enforcement; there are NO cross-database FKs.
--
-- ID/hash conventions (D1): immutable content-defined blobs are content-addressed
-- (source_file_id == 'sf_'||file_hash, markdown_id == 'md_'||markdown_hash); the
-- file_hash / markdown_hash columns are kept as explicit duplicates for query
-- clarity. conversion_runs is a MUTABLE identity row -> opaque 'conv_'||uuid4hex.
-- cache_events is a pure log -> INTEGER PRIMARY KEY AUTOINCREMENT.

-- --- source_files: add the two columns phase_1 needs (do NOT recreate) --------
-- byte_size: recorded by ingest (doc 04 §7). NOT NULL needs a constant default
-- because SQLite ALTER ... ADD COLUMN cannot add a NOT NULL column without one;
-- ingest always supplies the real size on INSERT.
ALTER TABLE source_files ADD COLUMN byte_size INTEGER NOT NULL DEFAULT 0;
-- updated_at: bumped ONLY on access_class reconciliation (§4). Added nullable
-- (vs. the plan's NOT NULL) because SQLite cannot ALTER-ADD a NOT NULL TEXT
-- column with a non-constant default, and an empty-string sentinel for a
-- timestamp is worse than NULL; ingest sets updated_at = created_at on INSERT.
ALTER TABLE source_files ADD COLUMN updated_at TEXT;

-- --- conversion_runs: full Marker provenance (doc 04 §8 / doc 11 §4/§5) --------
CREATE TABLE conversion_runs (
    conversion_run_id         TEXT PRIMARY KEY,            -- 'conv_' + uuid4hex (mutable identity row, D1)
    source_file_id            TEXT NOT NULL REFERENCES source_files(source_file_id),
    source_file_hash          TEXT NOT NULL,               -- denormalized anchor + dedup key
    converter_name            TEXT NOT NULL,               -- 'marker'
    converter_package         TEXT NOT NULL,               -- 'marker-pdf'
    converter_version         TEXT NOT NULL,               -- captured at runtime (importlib.metadata)
    python_version            TEXT NOT NULL,               -- captured at runtime (platform.python_version)
    config_json               TEXT NOT NULL,               -- canonical JSON (sorted keys), incl. paginate_output
    conversion_fingerprint    TEXT NOT NULL,               -- sha256 of conversion-identity tuple (incl. paginate_output)
    llm_enabled               INTEGER NOT NULL DEFAULT 0,
    llm_service               TEXT,                        -- captured iff llm_enabled
    llm_model                 TEXT,
    llm_prompt_version        TEXT,
    conversion_epistemic_type TEXT NOT NULL                -- doc 11 §5 enum
        CHECK (conversion_epistemic_type IN (
            'local_deterministic_conversion',
            'local_conversion_with_external_llm_assist',
            'local_conversion_with_local_llm_assist')),
    run_status                TEXT NOT NULL                -- 'pending' | 'success' | 'failed'
        CHECK (run_status IN ('pending', 'success', 'failed')),
    warnings_json             TEXT,                        -- JSON array of validation signals
    error                     TEXT,                        -- set on fatal validation / backend exception
    markdown_hash             TEXT,                        -- sha256 hex of produced markdown (set on SUCCESS only). Records this run's output hash so dedup can resolve run -> markdown by CONTENT (not via the shared content-addressed markdown_documents row, whose single pointer cannot represent the many runs/fingerprints that yield byte-identical markdown).
    created_at                TEXT NOT NULL,               -- ISO-8601 UTC
    completed_at              TEXT
);
-- Dedup lookup index (NOT unique — append-only allows many same-fingerprint runs
-- to coexist; canonical run = latest success by created_at, §4 must-fix).
CREATE INDEX ix_conv_dedup  ON conversion_runs (source_file_hash, conversion_fingerprint, run_status, created_at);
CREATE INDEX ix_conv_source ON conversion_runs (source_file_id);

-- --- markdown_documents: content-addressed markdown + cross-scope anchor -------
CREATE TABLE markdown_documents (
    markdown_id        TEXT PRIMARY KEY,                   -- 'md_' + sha256_hex(markdown_bytes); id IS the hash (D1); CROSS-DB ANCHOR
    conversion_run_id  TEXT NOT NULL REFERENCES conversion_runs(conversion_run_id),
    source_file_id     TEXT NOT NULL REFERENCES source_files(source_file_id),
    markdown_hash      TEXT NOT NULL,                      -- sha256 hex of markdown bytes (explicit duplicate of the id-hash)
    storage_uri        TEXT NOT NULL,                      -- POSIX path RELATIVE to cache root
    conversion_status  TEXT NOT NULL                       -- only successful conversions get a markdown row
        CHECK (conversion_status IN ('success')),
    byte_size          INTEGER NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_md_hash   ON markdown_documents (markdown_hash, created_at);
CREATE INDEX ix_md_source ON markdown_documents (source_file_id);

-- --- cache_events: append-only audit log --------------------------------------
-- event_type vocab (intentionally NOT a CHECK — the audit log stays open so a new
-- event kind never forces a schema migration): source_file_ingested,
-- source_file_deduped, access_class_reconciled, conversion_started,
-- conversion_succeeded, conversion_failed, conversion_reused, markdown_written.
CREATE TABLE cache_events (
    cache_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_type    TEXT NOT NULL
        CHECK (object_type IN ('source_file', 'conversion_run', 'markdown_document')),
    object_id      TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    detail_json    TEXT,                                   -- optional JSON detail
    created_at     TEXT NOT NULL
);
CREATE INDEX ix_cache_events_object ON cache_events (object_type, object_id);
