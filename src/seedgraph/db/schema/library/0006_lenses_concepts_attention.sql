-- library.db 0006 — lenses, concept decisions, attention items, conversion queue
-- (v3 design §2.1, §6.9, §6.10, §7).
-- Reserved for stage 4, NOT created here: concepts, concept_members, concept_links,
-- claim_concepts.

-- Keyed by lens_id + definition_hash (= lens version); the YAML text lives here.
CREATE TABLE lens_definitions (
    lens_id         TEXT NOT NULL,
    name            TEXT NOT NULL,
    object_type     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'draft',
    definition_hash TEXT NOT NULL,
    definition_yaml TEXT,
    created_at      TEXT NOT NULL,
    origin          TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (lens_id, definition_hash)
);

-- v2 columns with work_id -> paper_id. lens_id is a SOFT ref (lens_definitions has a
-- composite key; the version is extraction_runs.lens_definition_hash).
CREATE TABLE lens_outputs (
    lens_output_id    TEXT PRIMARY KEY,
    extraction_run_id TEXT NOT NULL REFERENCES extraction_runs(extraction_run_id),
    lens_id           TEXT NOT NULL,
    paper_id          TEXT NOT NULL REFERENCES papers(paper_id),
    status            TEXT NOT NULL,
    claim_id          TEXT REFERENCES extracted_claims(claim_id),
    confidence        REAL,
    fields_json       TEXT NOT NULL,
    access_class      TEXT NOT NULL DEFAULT 'user_supplied_private',
    created_at        TEXT NOT NULL,
    origin            TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX idx_lens_outputs_lens  ON lens_outputs (lens_id, status);
CREATE INDEX idx_lens_outputs_paper ON lens_outputs (paper_id);
CREATE INDEX idx_lens_outputs_run   ON lens_outputs (extraction_run_id);

-- decision_id = hash(kind + pair); labels normalized with label_a < label_b.
CREATE TABLE concept_decisions (
    decision_id TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('merge', 'never_merge', 'link')),
    label_a     TEXT NOT NULL,
    label_b     TEXT NOT NULL,
    decided_by  TEXT NOT NULL CHECK (decided_by IN ('researcher', 'agent')),
    state       TEXT NOT NULL DEFAULT 'pending_apply'
        CHECK (state IN ('pending_apply', 'applied', 'undone')),
    note        TEXT,
    origin      TEXT NOT NULL DEFAULT 'library',
    decided_at  TEXT NOT NULL,
    CHECK (label_a < label_b)
);
CREATE INDEX ix_concept_decisions_pair ON concept_decisions (label_a, label_b);

-- item_id = hash(kind + subject). An open item with hides_notes = 1 hides the
-- paper's notes from agent search (§2.1 notes).
CREATE TABLE attention_items (
    item_id      TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    paper_id     TEXT REFERENCES papers(paper_id),
    subject_json TEXT NOT NULL,
    detail       TEXT,
    hides_notes  INTEGER NOT NULL DEFAULT 0 CHECK (hides_notes IN (0, 1)),
    status       TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    resolution   TEXT,
    created_at   TEXT NOT NULL,
    resolved_at  TEXT,
    origin       TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX ix_attention_paper ON attention_items (paper_id, status);
CREATE INDEX ix_attention_kind  ON attention_items (kind, status);

CREATE TABLE conversion_queue (
    source_file_id TEXT PRIMARY KEY REFERENCES source_files(source_file_id),
    reason         TEXT NOT NULL CHECK (reason IN ('not_converted', 'relink_failed')),
    queued_at      TEXT NOT NULL,
    origin         TEXT NOT NULL DEFAULT 'library'
);
