-- library.db 0010 — concepts as label + sense (v3 stage 4 design §1.2, §2.3).
--
-- Stored, authoritative rows: concept_decisions (rebuilt: SQLite cannot ALTER a CHECK;
-- every carried row is copied unchanged in meaning), concept_senses, sense_assignments,
-- ambiguous_labels, concept_vectors, concept_reviews, concept_build_state.
-- Derived rows (rebuilt deterministically by seedgraph.concepts.refresh, never edited by
-- hand): claim_concepts, concepts, concept_members, concept_links.
--
-- No seed rows are inserted here: the migration inserts its carried concept_decisions
-- after the migrations run, and a seed row could collide with one. concepts.refresh
-- inserts the seed never-merge rows and the seed ambiguous labels (insert-or-ignore).
--
-- Every new concept_decisions column is nullable or defaulted, so the migration's
-- nine-column insert keeps working.

CREATE TABLE concept_decisions_new (
    decision_id  TEXT PRIMARY KEY,           -- 'cd_' + hash of kind + subjects
    kind         TEXT NOT NULL
        CHECK (kind IN ('merge', 'link', 'never_merge', 'never_connect', 'unrelated')),
    family       TEXT,                       -- NULL = every family where both labels exist
    family_b     TEXT,                       -- side b's family when a link crosses families (NULL = family)
    label_a      TEXT NOT NULL,
    sense_a      TEXT,                       -- NULL = the default sense (s0); for a veto: every sense
    label_b      TEXT NOT NULL,
    sense_b      TEXT,
    name         TEXT,                       -- merge only: the chosen concept name (one of the labels)
    decided_by   TEXT NOT NULL CHECK (decided_by IN ('researcher', 'agent', 'rule')),
    state        TEXT NOT NULL DEFAULT 'pending_apply'
        CHECK (state IN ('applied', 'pending_apply', 'queued', 'undone', 'vetoed', 'needs_rejudge')),
    note         TEXT,
    quote_a_json TEXT,                       -- {paper_id, span_id, claim_fingerprint}
    quote_b_json TEXT,
    confidence   TEXT CHECK (confidence IS NULL OR confidence IN ('high', 'medium', 'low')),
    veto_reason  TEXT,
    batch_id     TEXT,
    packet_id    TEXT,
    model        TEXT,
    origin       TEXT NOT NULL DEFAULT 'library',
    decided_at   TEXT NOT NULL,
    CHECK (label_a < label_b
           OR (label_a = label_b AND (COALESCE(sense_a, '') < COALESCE(sense_b, '')
               OR (COALESCE(sense_a, '') = COALESCE(sense_b, '')
                   AND COALESCE(family, '') < COALESCE(family_b, '')))))
);
INSERT INTO concept_decisions_new (decision_id, kind, label_a, label_b, decided_by, state, note, origin,
                                   decided_at)
SELECT decision_id, kind, label_a, label_b, decided_by, state, note, origin, decided_at
FROM concept_decisions ORDER BY decision_id;
DROP INDEX ix_concept_decisions_pair;
DROP TABLE concept_decisions;
ALTER TABLE concept_decisions_new RENAME TO concept_decisions;
CREATE INDEX ix_concept_decisions_pair ON concept_decisions (label_a, label_b);
CREATE INDEX ix_concept_decisions_state ON concept_decisions (state, kind);

-- Rows exist only for split or sense-checked labels; a label with no row has the
-- implicit s0, unconfirmed. 'undone' records an undone split (assignments kept).
CREATE TABLE concept_senses (
    family       TEXT NOT NULL,
    label_key    TEXT NOT NULL,
    sense_id     TEXT NOT NULL,              -- 's0' (confirmed single sense) or 's1'..'sn'
    gloss        TEXT,
    decided_by   TEXT NOT NULL CHECK (decided_by IN ('researcher', 'agent', 'rule')),
    state        TEXT NOT NULL CHECK (state IN ('default', 'confirmed', 'queued', 'undone')),
    decision_ref TEXT,
    origin       TEXT NOT NULL DEFAULT 'library',
    decided_at   TEXT NOT NULL,
    PRIMARY KEY (family, label_key, sense_id)
);

-- Which sense a claim belongs to under a split label, keyed by the claim fingerprint
-- (canonical JSON, seedgraph.concepts.model.claim_fingerprint), never by claim_id.
CREATE TABLE sense_assignments (
    family            TEXT NOT NULL,
    label_key         TEXT NOT NULL,
    paper_id          TEXT NOT NULL,
    claim_fingerprint TEXT NOT NULL,
    sense_id          TEXT NOT NULL,
    decided_by        TEXT NOT NULL CHECK (decided_by IN ('researcher', 'agent', 'rule')),
    decided_at        TEXT NOT NULL,
    PRIMARY KEY (family, label_key, paper_id, claim_fingerprint)
);
CREATE INDEX ix_sense_assignments_paper ON sense_assignments (paper_id);

-- The ambiguous-label list (§2.3): library-wide rows (scope = '') plus per-context-graph
-- overrides (scope = the graph slug). effect 'ambiguous' adds a pattern; 'single_sense'
-- (graph scope only) marks a library-ambiguous pattern as single-sense in that graph.
CREATE TABLE ambiguous_labels (
    pattern    TEXT NOT NULL,
    scope      TEXT NOT NULL DEFAULT '',
    match      TEXT NOT NULL CHECK (match IN ('label', 'head')),
    effect     TEXT NOT NULL DEFAULT 'ambiguous' CHECK (effect IN ('ambiguous', 'single_sense')),
    note       TEXT,
    decided_by TEXT NOT NULL CHECK (decided_by IN ('researcher', 'agent', 'rule')),
    origin     TEXT NOT NULL DEFAULT 'library',
    created_at TEXT NOT NULL,
    PRIMARY KEY (pattern, scope),
    CHECK (scope <> '' OR effect = 'ambiguous')
);

-- §5: claim-context vectors, only if embeddings are adopted; the table always exists.
CREATE TABLE concept_vectors (
    text_sha256 TEXT NOT NULL,
    model       TEXT NOT NULL,
    dims        INTEGER NOT NULL,
    vector      BLOB NOT NULL,               -- float32 little-endian
    created_at  TEXT NOT NULL,
    PRIMARY KEY (text_sha256, model)
);

-- §11: the stratified review sample's marks.
CREATE TABLE concept_reviews (
    sample_id   TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    mark        TEXT NOT NULL CHECK (mark IN ('right', 'wrong', 'should_be_link', 'should_be_split')),
    note        TEXT,
    marked_at   TEXT NOT NULL,
    PRIMARY KEY (sample_id, decision_id)
);

-- What each paper's current claims looked like at the last refresh (§1.5).
CREATE TABLE concept_build_state (
    paper_id      TEXT PRIMARY KEY,
    claims_sha256 TEXT NOT NULL,
    built_at      TEXT NOT NULL
);

-- Derived ---------------------------------------------------------------------------------

CREATE TABLE claim_concepts (
    claim_id      TEXT NOT NULL,
    paper_id      TEXT NOT NULL,
    family        TEXT NOT NULL,
    label_key     TEXT NOT NULL,
    sense_id      TEXT,                      -- NULL = needs sense (split label, unassigned claim)
    concept_id    TEXT,                      -- NULL when the sense is NULL
    supports      INTEGER NOT NULL CHECK (supports IN (0, 1)),
    agent_visible INTEGER NOT NULL CHECK (agent_visible IN (0, 1)),
    PRIMARY KEY (claim_id)
);
CREATE INDEX ix_claim_concepts_concept ON claim_concepts (concept_id);
CREATE INDEX ix_claim_concepts_label ON claim_concepts (family, label_key);
CREATE INDEX ix_claim_concepts_paper ON claim_concepts (paper_id);

CREATE TABLE concepts (
    concept_id        TEXT PRIMARY KEY,      -- 'cn_' + hash of family + smallest member sense key
    family            TEXT NOT NULL,
    name              TEXT NOT NULL,
    gloss             TEXT,
    paper_count       INTEGER NOT NULL,
    claim_count       INTEGER NOT NULL,
    quote_count       INTEGER NOT NULL,
    supported         INTEGER NOT NULL CHECK (supported IN (0, 1)),
    weight            REAL,                  -- log(N_papers / paper_count); NULL when unsupported
    needs_sense_check INTEGER NOT NULL CHECK (needs_sense_check IN (0, 1))
);
CREATE INDEX ix_concepts_family ON concepts (family, supported);

CREATE TABLE concept_members (
    concept_id TEXT NOT NULL,
    family     TEXT NOT NULL,
    label_key  TEXT NOT NULL,
    sense_id   TEXT NOT NULL,
    PRIMARY KEY (family, label_key, sense_id)
);
CREATE INDEX ix_concept_members_concept ON concept_members (concept_id);

CREATE TABLE concept_links (
    concept_a   TEXT NOT NULL,
    concept_b   TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    note        TEXT,
    decided_by  TEXT NOT NULL,
    PRIMARY KEY (concept_a, concept_b, decision_id),
    CHECK (concept_a < concept_b)
);
