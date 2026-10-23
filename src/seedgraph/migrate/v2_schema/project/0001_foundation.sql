-- project.db foundation (Phase 0). Authoritative schema (decision D6). Only the
-- LLM substrate + review queue; works / identifiers / project_documents land in
-- project/0002_*.sql (project_model phase).

CREATE TABLE schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE review_queue (                           -- generalized (decisions 34/56/80)
    item_id     TEXT PRIMARY KEY,                     -- "rq_" + uuid4hex
    item_type   TEXT NOT NULL,
    target_type TEXT,
    target_id   TEXT,
    payload     TEXT,                                 -- JSON
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved')),
    action      TEXT,
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE llm_usage_events (                        -- doc 13 §11; project_id dropped (decision 12)
    usage_event_id      TEXT PRIMARY KEY,              -- "ue_" + uuid4hex
    run_id              TEXT,
    task_type           TEXT NOT NULL,
    provider            TEXT,
    model               TEXT,
    access_mode         TEXT,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    estimated_cost      REAL,                          -- NULL until pricing exists (decision 37)
    unit_price_json     TEXT,                          -- snapshot of price used, if any
    source_access_class TEXT,
    external_full_text  INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL
);

CREATE TABLE llm_key_refs (                            -- doc 13 §12 — references only, never secrets
    provider          TEXT NOT NULL,
    access_mode       TEXT,
    key_source        TEXT,                            -- environment | system_keyring
    key_reference     TEXT,                            -- env var name or keyring entry name
    env_var           TEXT,
    created_at        TEXT NOT NULL,
    last_validated_at TEXT,
    PRIMARY KEY (provider, key_reference)
);
