-- cache.db — provider_cache (Phase 5b). Authoritative schema (decision D6).
--
-- The provider-response cache; this phase is the SOLE writer (decisions 39/64/73).
-- phase_2 ATTACHes it read-only and recovers each work's reference list offline.
-- The `request_key` is a PINNED cross-phase contract (phase_5b §6.5), not an
-- opaque internal string. 30-day TTL; a negative (empty) hit is cached so a known
-- miss is not re-queried within the TTL.
--
-- D8 shareability is decided at the FIELD level (vocab.PROVIDER_SHAREABLE_FIELDS),
-- NOT by any row-level column here: the `response` payload MAY hold non-shareable
-- provider fields (abstracts / snippets / TDM / licensed fields / raw blobs)
-- locally; every export projects ONLY the SHAREABLE allowlist. No full-text PDF
-- ever enters this table. There is deliberately NO `access_class` column — a
-- row-level shareability flag would be dead weight (lazy-but-correct, §4.1).
--
-- numbering: cache/0001 (phase_0 baseline) and cache/0002 (phase_1 markdown/
-- conversion) precede this; gaps are tolerated by the runner.

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
