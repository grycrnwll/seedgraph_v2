-- library.db 0009 — researcher edits as an overlay on an extraction (v3 stage 3 design
-- §4.1, ADR-0011).
--
-- Adds tables and redefines one view; changes no stage 1/2 row. With no overlays,
-- note_current is exactly the stage 1 definition (now also named note_latest), so a
-- migration dump has no new rows.
--
-- overlay_versions is the source of truth: each version stores the COMPLETE op set
-- (ops_json) and the researcher notes; revert writes a new version. The
-- interpretation_* tables are derived from the head version and exist only for
-- overlaid papers (seedgraph.notes.overlay.rebuild). Everything is keyed by stable ids
-- (paper_id, base extraction_run_id, claim fingerprints inside ops_json) so a later
-- export/replay stays possible.

CREATE TABLE note_overlays (
    overlay_id             TEXT PRIMARY KEY,                 -- 'ov_' + hex
    paper_id               TEXT NOT NULL REFERENCES papers(paper_id),
    schema_id              TEXT,
    base_note_id           TEXT NOT NULL REFERENCES structured_notes(note_id),  -- = head version's base
    base_extraction_run_id TEXT NOT NULL,
    head_version_no        INTEGER NOT NULL CHECK (head_version_no >= 1),
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    origin                 TEXT NOT NULL DEFAULT 'library',
    UNIQUE (paper_id, schema_id)
);
CREATE INDEX ix_note_overlays_base ON note_overlays (base_note_id);

CREATE TABLE overlay_versions (
    overlay_id             TEXT NOT NULL REFERENCES note_overlays(overlay_id),
    version_no             INTEGER NOT NULL CHECK (version_no >= 1),
    base_note_id           TEXT NOT NULL,                    -- SOFT ref (history)
    base_extraction_run_id TEXT NOT NULL,
    ops_json               TEXT NOT NULL,                    -- the complete op set at this version
    researcher_notes       TEXT,
    action                 TEXT NOT NULL CHECK (action IN ('edit', 'hide', 'unhide', 'add',
                               'remove_added', 'notes', 'revert', 'accept_extraction')),
    summary                TEXT,
    reverted_to            INTEGER,
    created_at             TEXT NOT NULL,
    origin                 TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (overlay_id, version_no)
);

-- Derived: one row per claim of the current interpretation of an overlaid paper,
-- hidden claims included (hidden = 1) for the researcher's "show hidden" toggle.
CREATE TABLE interpretation_claims (
    overlay_id       TEXT NOT NULL REFERENCES note_overlays(overlay_id),
    claim_id         TEXT NOT NULL,          -- the base claim id, or 'rc_' + hex for an added claim
    paper_id         TEXT NOT NULL,
    version_no       INTEGER NOT NULL,
    claim_type       TEXT NOT NULL,
    claim_subtype    TEXT,
    field_key        TEXT NOT NULL,          -- the base field key; 'added[i]' for an added claim
    normalized_label TEXT,
    claim_text       TEXT,
    plain_words      TEXT,
    status           TEXT NOT NULL,
    trust            TEXT NOT NULL CHECK (trust IN ('paper_states', 'agent_inferred', 'researcher')),
    edited           INTEGER NOT NULL DEFAULT 0 CHECK (edited IN (0, 1)),
    hidden           INTEGER NOT NULL DEFAULT 0 CHECK (hidden IN (0, 1)),
    kind             TEXT NOT NULL CHECK (kind IN ('agent', 'quote', 'assertion')),
    span_id          TEXT,
    PRIMARY KEY (overlay_id, claim_id)
);
CREATE INDEX ix_interp_claims_paper ON interpretation_claims (paper_id);

-- Derived: the note text recomputed from the interpretation plus the researcher notes.
CREATE TABLE interpretation_notes (
    paper_id  TEXT NOT NULL,
    schema_id TEXT,
    note_text TEXT NOT NULL,
    PRIMARY KEY (paper_id, schema_id)
);

-- "Concepts out of date": stage 4 refreshes only these papers' concepts and clears them.
CREATE TABLE concept_refresh (
    paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id),
    reason   TEXT NOT NULL,
    since    TEXT NOT NULL
);

-- note_latest: the stage 1 note_current (highest unlabelled version). Extraction
-- freshness and the note save conflict check read this.
DROP VIEW note_current;
CREATE VIEW note_latest AS
SELECT n.*
FROM structured_notes n
WHERE n.version_label IS NULL
  AND n.version_no = (
      SELECT MAX(o.version_no) FROM structured_notes o
      WHERE o.paper_id = n.paper_id
        AND o.schema_id IS n.schema_id
        AND o.version_label IS NULL
  );

-- note_current: the current interpretation's note. For an overlaid (paper, schema) whose
-- base is unlabelled, the overlay's base (the pin); otherwise note_latest.
CREATE VIEW note_current AS
SELECT n.*
FROM structured_notes n
JOIN note_overlays o ON o.base_note_id = n.note_id
WHERE n.version_label IS NULL
UNION ALL
SELECT l.*
FROM note_latest l
WHERE NOT EXISTS (
    SELECT 1 FROM note_overlays o JOIN structured_notes b ON b.note_id = o.base_note_id
    WHERE o.paper_id = l.paper_id AND o.schema_id IS l.schema_id AND b.version_label IS NULL
);

-- current_claims: what every claim reader uses. Overlaid papers: the interpretation
-- (hidden claims excluded); all others: the claims of note_current with derived trust.
CREATE VIEW current_claims AS
SELECT ic.claim_id, ic.paper_id, n.note_id, n.schema_id, ic.claim_type, ic.claim_subtype,
       ic.field_key, ic.normalized_label, ic.claim_text, ic.plain_words, ic.status,
       ic.trust, ic.edited, ic.kind, ic.span_id, o.overlay_id, ic.version_no
FROM interpretation_claims ic
JOIN note_overlays o ON o.overlay_id = ic.overlay_id
JOIN note_current n ON n.note_id = o.base_note_id
WHERE ic.hidden = 0
UNION ALL
SELECT c.claim_id, c.paper_id, n.note_id, n.schema_id, c.claim_type, c.claim_subtype,
       c.field_key, c.normalized_label, c.claim_text, c.plain_words, c.status,
       CASE WHEN c.status = 'found' AND s.span_id IS NOT NULL
            THEN 'paper_states' ELSE 'agent_inferred' END AS trust,
       0 AS edited, 'agent' AS kind, s.span_id, NULL AS overlay_id, NULL AS version_no
FROM extracted_claims c
JOIN note_current n ON n.note_id = c.structured_note_id
LEFT JOIN claim_spans s ON s.id = (
    SELECT cs.id FROM claim_spans cs WHERE cs.claim_id = c.claim_id
    ORDER BY cs.rank, cs.span_id LIMIT 1)
WHERE NOT EXISTS (SELECT 1 FROM note_overlays o WHERE o.base_note_id = n.note_id);
