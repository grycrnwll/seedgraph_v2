-- project.db — Project Model MVP (phase_5). Authoritative schema (decision D6).
--
-- Sole schema source of truth for this phase's NEW tables: works, identifiers,
-- project_documents. Applies on top of project/0001_foundation.sql (phase_0,
-- which already owns schema_migrations / review_queue / llm_usage_events /
-- llm_key_refs — NONE of which are recreated or altered here).
--
-- One project per project.db file (decision 12): there is deliberately NO
-- project_id column anywhere (decision 79; slug == project identity, kept in
-- project.yaml). No cache.db tables and no cross-database references are
-- introduced — project.db is fully self-contained this phase.
--
-- The SQLModel classes in db/project_models.py are ORM-MAPPING-ONLY mirrors of
-- the tables below and never author/evolve schema (no create_all) — D6.

-- works: flat canonical scholarly object (decisions 1/25/41 — authors JSON +
-- venue live on the work; no manifestation/author/venue side tables).
CREATE TABLE works (
    work_id             TEXT PRIMARY KEY,         -- 'work_' + uuid4().hex (mutable identity row; D1)
    canonical_title     TEXT,
    title_hash          TEXT,                     -- sha1(normalize_title(title)); fuzzy matcher, NEVER auto-merges (decision 13/64)
    authors             TEXT,                     -- JSON array of strings (SQLModel JSON column)
    venue               TEXT,
    year                INTEGER,
    doi                 TEXT,
    arxiv_id            TEXT,
    openalex_id         TEXT,
    semantic_scholar_id TEXT,
    ssrn_id             TEXT,
    created_at          TEXT NOT NULL             -- UTC ISO8601
);
CREATE INDEX ix_works_doi        ON works(doi);
CREATE INDEX ix_works_arxiv      ON works(arxiv_id);
CREATE INDEX ix_works_openalex   ON works(openalex_id);
CREATE INDEX ix_works_s2         ON works(semantic_scholar_id);
CREATE INDEX ix_works_ssrn       ON works(ssrn_id);
CREATE INDEX ix_works_title_hash ON works(title_hash);

-- identifiers: pure join table (INTEGER PK; decision 7/44). external-id -> work
-- resolver. UNIQUE(id_type, id_value) guarantees an external id maps to exactly
-- one work in-project; the merge flow (project/identity.py §6) queries existing
-- identifiers first and never blind-inserts a colliding id, so this UNIQUE is
-- never violated (cross-work collisions route to review instead).
-- id_type is validated at the app boundary by the IdType Literal
-- (doi|openalex|arxiv|s2|ssrn); left without a DB CHECK so new provider id kinds
-- can land in a later phase without a schema break.
CREATE TABLE identifiers (
    identifier_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id           TEXT NOT NULL REFERENCES works(work_id) ON DELETE CASCADE,
    id_type           TEXT NOT NULL,             -- doi|openalex|arxiv|s2|ssrn (IdType enum)
    id_value          TEXT NOT NULL,
    resolution_source TEXT,                      -- 'user' this phase; provider names arrive in phase_5b
    confidence        REAL,
    UNIQUE(id_type, id_value)
);
CREATE INDEX ix_identifiers_work ON identifiers(work_id);

-- project_documents: corpus membership (NO project_id; decision 12). PK = work_id
-- => exactly one membership row per work (the 1:1 invariant). access_status is the
-- mirror of the resolved access_class (decision 11/30) and is ALWAYS NULL this
-- phase (fail-closed/private; NEVER derived from inclusion_status). inclusion_status
-- is the settled closed enum (doc 01 §6 / decisions 13/59) — enforced by CHECK so
-- the DB itself rejects any value outside the three sanctioned ones.
CREATE TABLE project_documents (
    work_id          TEXT PRIMARY KEY REFERENCES works(work_id) ON DELETE CASCADE,
    inclusion_status TEXT NOT NULL
        CHECK (inclusion_status IN ('included','metadata_only','excluded')),
    inclusion_reason TEXT,                       -- seed_document|citation_walk|user_added|unavailable_full_text|user_excluded (open vocab)
    is_seed          INTEGER NOT NULL DEFAULT 0, -- seed tagging (0/1)
    access_status    TEXT,                       -- mirror of resolved access_class; ALWAYS NULL this phase (decision 11/30)
    user_note        TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX ix_projdoc_status ON project_documents(inclusion_status);
CREATE INDEX ix_projdoc_seed   ON project_documents(is_seed);
