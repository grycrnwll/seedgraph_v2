-- library.db 0008 — the one agent work protocol (v3 stage 2 design §1.7).
--
-- Batches, their items (one per subject), work packages, package claims (a worker's
-- time-limited hold on one package), submissions and the append-only batch event feed;
-- the text-quality gate cache (§5); extracted_claims.plain_words (§4.3) and claim_fts
-- recreated with a plain_words column (FTS5 cannot ALTER).
--
-- Conventions as stage 1: TEXT ids, origin, explicit ordering (claim_no and seq are
-- assigned MAX + 1 inside the writing transaction; no rowid tie-breaks). Timestamps
-- are UTC ISO-8601 text in one fixed format so they compare as strings. Pause state:
-- 'stopped' and 'usage_limit' are stored on batches; 'claim expired', 'waiting' and
-- 'working' are derived from package_claims (seedgraph.work.state).

CREATE TABLE batches (
    batch_id            TEXT PRIMARY KEY,                    -- 'b_' + time-ordered hex
    task_type           TEXT NOT NULL,
    task_version        TEXT NOT NULL,
    selection_query     TEXT,                                -- the query text, if one was given
    selection_json      TEXT NOT NULL,                       -- the resolved, frozen subject ids
    graph_slug          TEXT,
    options_json        TEXT NOT NULL DEFAULT '{}',
    instructions_sha256 TEXT,
    labels_sha256       TEXT,
    pause_kind          TEXT CHECK (pause_kind IS NULL OR pause_kind IN ('stopped', 'usage_limit')),
    stop_mode           TEXT CHECK (stop_mode IS NULL OR stop_mode IN ('finish', 'now')),
    resets_at           TEXT,
    pause_detail        TEXT,
    paused_at           TEXT,
    created_at          TEXT NOT NULL,
    created_by_client   TEXT,
    origin              TEXT NOT NULL DEFAULT 'library',
    CHECK (stop_mode IS NULL OR pause_kind IS 'stopped')
);
CREATE INDEX ix_batches_created ON batches (created_at, batch_id);

CREATE TABLE batch_items (
    batch_id                 TEXT NOT NULL REFERENCES batches(batch_id),
    ordinal                  INTEGER NOT NULL CHECK (ordinal >= 1),
    subject_kind             TEXT NOT NULL DEFAULT 'paper',
    subject_id               TEXT NOT NULL,                  -- SOFT ref (a paper id for 'paper')
    state                    TEXT NOT NULL DEFAULT 'queued'
        CHECK (state IN ('queued', 'held_back', 'preserved', 'failed', 'in_progress',
                         'done', 'stale', 'conflict')),
    markdown_id              TEXT,                           -- SOFT ref
    markdown_hash            TEXT,
    expected_current_note_id TEXT,                           -- SOFT ref
    part_count               INTEGER,
    result_ref               TEXT,
    reason                   TEXT,
    attention_item_id        TEXT,                           -- SOFT ref
    updated_at               TEXT,
    origin                   TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (batch_id, ordinal),
    UNIQUE (batch_id, subject_kind, subject_id)
);
CREATE INDEX ix_batch_items_subject ON batch_items (subject_kind, subject_id);

CREATE TABLE work_packages (
    packet_id   TEXT PRIMARY KEY,                            -- sha256 of the binding
    batch_id    TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    part        INTEGER NOT NULL CHECK (part >= 1),
    part_count  INTEGER NOT NULL CHECK (part_count >= 1),
    start_char  INTEGER,
    end_char    INTEGER,
    est_tokens  INTEGER,
    text_path   TEXT,
    state       TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'claimed', 'saved', 'void')),
    attempt     INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    created_at  TEXT NOT NULL,
    origin      TEXT NOT NULL DEFAULT 'library',
    FOREIGN KEY (batch_id, ordinal) REFERENCES batch_items(batch_id, ordinal),
    CHECK (part <= part_count)
);
CREATE INDEX ix_work_packages_pick ON work_packages (batch_id, state, ordinal, part);

CREATE TABLE package_claims (
    claim_id       TEXT PRIMARY KEY,
    packet_id      TEXT NOT NULL REFERENCES work_packages(packet_id),
    claim_no       INTEGER NOT NULL CHECK (claim_no >= 1),   -- 1..n per package, the claim order
    worker_id      TEXT NOT NULL,
    worker_kind    TEXT NOT NULL CHECK (worker_kind IN ('interactive', 'background', 'ollama')),
    client         TEXT,
    model          TEXT,
    claimed_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    released_at    TEXT,
    release_reason TEXT CHECK (release_reason IS NULL OR release_reason IN
                               ('submitted', 'expired', 'released', 'stopped', 'limit')),
    acknowledged   INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
    origin         TEXT NOT NULL DEFAULT 'library',
    UNIQUE (packet_id, claim_no),
    CHECK ((released_at IS NULL) = (release_reason IS NULL))
);
-- The live claim is the row with released_at IS NULL: at most one per package.
CREATE UNIQUE INDEX ux_package_claims_live ON package_claims (packet_id) WHERE released_at IS NULL;
CREATE INDEX ix_package_claims_expiry ON package_claims (released_at, expires_at);
CREATE INDEX ix_package_claims_worker ON package_claims (worker_id);

-- The note is stored once (payload_json); only the latest accepted row per package is used.
CREATE TABLE package_submissions (
    submission_id  TEXT PRIMARY KEY,
    packet_id      TEXT NOT NULL REFERENCES work_packages(packet_id),
    claim_id       TEXT NOT NULL REFERENCES package_claims(claim_id),
    submission_no  INTEGER NOT NULL CHECK (submission_no >= 1),  -- 1..n per package
    received_at    TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    outcome        TEXT NOT NULL CHECK (outcome IN ('accepted', 'needs_revision', 'invalid')),
    feedback_json  TEXT,
    usage_json     TEXT,
    model          TEXT,
    origin         TEXT NOT NULL DEFAULT 'library',
    UNIQUE (packet_id, submission_no)
);

-- Append-only feed (stage 3 activity view, stage 6 batch page).
CREATE TABLE batch_events (
    batch_id    TEXT NOT NULL REFERENCES batches(batch_id),
    seq         INTEGER NOT NULL CHECK (seq >= 1),
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    packet_id   TEXT,
    worker_id   TEXT,
    detail_json TEXT,
    origin      TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (batch_id, seq)
);

-- Text-quality gate cache (§5.3); checker_version forces a re-assessment.
CREATE TABLE text_quality (
    markdown_id     TEXT PRIMARY KEY,                        -- SOFT ref -> markdown_documents
    status          TEXT NOT NULL CHECK (status IN ('pass', 'garbled', 'no_text')),
    score_json      TEXT NOT NULL,
    checker_version TEXT NOT NULL,
    assessed_at     TEXT NOT NULL,
    origin          TEXT NOT NULL DEFAULT 'library'
);

-- §4.3: one or two sentences stating the claim without the paper's notation.
ALTER TABLE extracted_claims ADD COLUMN plain_words TEXT;

-- FTS5 cannot ALTER: drop, recreate with plain_words, repopulate (explicit columns).
DROP TABLE claim_fts;
CREATE VIRTUAL TABLE claim_fts USING fts5(
    claim_id UNINDEXED, normalized_label, claim_text, plain_words,
    tokenize = "unicode61 remove_diacritics 2"
);
INSERT INTO claim_fts (claim_id, normalized_label, claim_text, plain_words)
SELECT claim_id, normalized_label, claim_text, plain_words FROM extracted_claims ORDER BY claim_id;
