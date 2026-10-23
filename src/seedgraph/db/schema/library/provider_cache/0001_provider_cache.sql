-- library/provider_cache.db 0001 (v3 design §1.1). A separate, disposable TTL cache,
-- excluded from the canonical dump. The provider_cache table below is stage 0's
-- cache/0003_provider_cache.sql DDL verbatim.

CREATE TABLE schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE provider_cache (
    provider_cache_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- pure log/cache row (decision 44)
    provider     TEXT NOT NULL,            -- 'openalex'|'crossref'|'unpaywall'|'semantic_scholar'|'core'
    request_key  TEXT NOT NULL,            -- PINNED grammar (§6.5); e.g. 'by_doi:doi=10.x/y'
                                           --   | 'referenced_works:src=openalex=w..' | 'oa_pdf_candidates:src=doi=..'
    response     TEXT,                     -- raw provider JSON; MAY include non-shareable fields; export projects ONLY the allowlist (D8)
    fetched_at   TEXT NOT NULL,            -- ISO-8601 UTC
    ttl_seconds  INTEGER NOT NULL DEFAULT 2592000,  -- 30-day TTL (decision 39)
    UNIQUE(provider, request_key)
);

CREATE INDEX ix_provider_cache_lookup ON provider_cache(provider, request_key, fetched_at);
