-- library.db 0001 — meta tables + stage 0's cache tables (v3 design §1.1, §2.1).
-- Authoritative schema (decision D6): numbered .sql only, no create_all.
--
-- source_files / conversion_runs / markdown_documents / cache_events are stage 0's
-- cache.db DDL VERBATIM (cache/0001 + cache/0002, in that order, so the column
-- order matches cache.db exactly and rows copy across unchanged). They therefore
-- carry no `origin` column. The FKs among them are now genuine library FKs.

CREATE TABLE schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- key/value: origin ('local' | 'migration'), dirty ('0' | '1'), migration_run_id, v2_home.
-- Any non-migration write sets dirty = '1' (library/store.py); the migration swap
-- refuses to replace a dirty library (§1.3).
CREATE TABLE library_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
INSERT INTO library_meta (key, value) VALUES ('dirty', '0');

-- Excluded from the canonical dump (§4.5).
CREATE TABLE migration_runs (
    run_id         TEXT PRIMARY KEY,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    v2_fingerprint TEXT,
    dump_sha256    TEXT,
    code_version   TEXT
);

-- Traceability v2 id -> v3 id (used by parity, §8.3).
CREATE TABLE v2_id_map (
    v2_project TEXT NOT NULL,
    v2_table   TEXT NOT NULL,
    v2_id      TEXT NOT NULL,
    v3_table   TEXT NOT NULL,
    v3_id      TEXT NOT NULL,
    PRIMARY KEY (v2_project, v2_table, v2_id)
);
CREATE INDEX ix_v2_id_map_v3 ON v2_id_map (v3_table, v3_id);

-- ---------------------------------------------------------------------------
-- stage 0 cache.db DDL (cache/0001_baseline.sql + cache/0002_cache.sql),
-- verbatim from here to the end of this file.
-- ---------------------------------------------------------------------------

CREATE TABLE source_files (
    source_file_id     TEXT PRIMARY KEY,            -- "sf_" + sha256_hex(file_bytes); id IS the content hash (D1)
    file_hash          TEXT NOT NULL UNIQUE,        -- sha256 hex; explicit duplicate of the id-hash for query clarity / dedup (doc 10 Phase 2)
    file_type          TEXT,
    access_class       TEXT NOT NULL DEFAULT 'user_supplied_private'
                       CHECK (access_class IN ('open_access','user_supplied_private',
                                               'metadata_only','licensed_future','unknown')),
    storage_uri        TEXT,                        -- POSIX path RELATIVE to cache root (decision 17)
    acquisition_method TEXT,
    created_at         TEXT NOT NULL
);

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
