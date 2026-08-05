-- project.db — work_source_files (Phase 5b). Authoritative schema (decision D6).
--
-- The load-bearing work->markdown bridge (decision 3): the single most missing
-- table. ONE row per work — UNIQUE(work_id) — so the downstream `work -> markdown`
-- read (resolve_work_markdown, §6.3) is deterministic (must-fix #3). There is NO
-- `role` / `manifestation_kind` column (decision 1 deferral; binding resolution of
-- NEW-A): consumers (phase_3b, phase_6) resolve via a plain `WHERE work_id = ?`,
-- NEVER a `role='primary'` filter. Multi-manifestation is a future migration.
--
-- `work_id` REFERENCES works(work_id) — works is created by project/0002 (phase_5);
-- this migration is numbered above it so it applies on top. NO cross-DB FK:
-- source_file_id / markdown_id point into cache.db and are SOFT xrefs anchored on
-- the content hashes (D1: id == 'sf_'/'md_' + sha256). Integrity is the content
-- hashes + the application-level reconcile (phase_5b §4.5), not an FK SQLite cannot
-- enforce.

CREATE TABLE work_source_files (
    work_source_file_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- pure join row (decision 44)
    work_id            TEXT NOT NULL UNIQUE                 -- ONE bridge (primary) row per work (must-fix #3)
                       REFERENCES works(work_id) ON DELETE CASCADE,
    source_file_id     TEXT NOT NULL,        -- cache.db source_files.source_file_id == 'sf_'+file_hash (SOFT xref; D1)
    file_hash          TEXT NOT NULL,        -- == source_file_id minus the 'sf_' prefix; explicit duplicate for query clarity (D1)
    markdown_id        TEXT,                 -- cache.db markdown_documents.markdown_id == 'md_'+markdown_hash (set after conversion; D1)
    markdown_hash      TEXT,                 -- == markdown_id minus the 'md_' prefix; explicit duplicate (D1). Cross-DB joins anchor on markdown_id
    acquisition_method TEXT NOT NULL,        -- 'open_access_fetch'|'manual_upload'|'already_cached_local'
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL         -- bumped on markdown backfill / re-acquire
);

CREATE INDEX ix_wsf_markdown_hash ON work_source_files(markdown_hash);
CREATE INDEX ix_wsf_file_hash     ON work_source_files(file_hash);
