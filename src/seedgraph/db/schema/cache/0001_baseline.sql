-- cache.db baseline (Phase 0). Authoritative schema — the single source of
-- truth for these tables (decision D6). Only source_files is genuine Phase-0
-- cache substrate; conversion_runs / markdown_documents land in cache/0002_*.sql.

CREATE TABLE schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

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
