-- project.db — phase_9 evaluation. Authoritative schema (decision D6) for the
-- ONE new table this phase owns: audit_records — the durable, APPEND-ONLY home
-- for every graded human/agent verdict (doc 09 §4; governing decision 62).
--
-- Boundary (phase_9 §4): audit_records only MEASURES — it records the grade
-- (verdict/severity/problem/recommended_fix) and the accept/reject/edit
-- validation decision; it NEVER mutates graph state (that is review_queue's job,
-- decisions 56/71/80). A single concept-merge therefore has at most one audit row
-- (the grade) and at most one review_queue resolution (the fix) — no double-record.
--
-- Notes:
--   * subject_id is a DELIBERATE soft polymorphic reference over heterogeneous
--     subjects (claim|span|concept_merge|answer|citation_edge|work|source_file|
--     markdown|identifier|reference_entry) — NO cross-table / cross-database FK.
--     PRAGMA foreign_keys stays ON for the rest of the scope (db/connection.py).
--   * No project_id column (one project.db == one project, decision 12/79).
--   * No subject_fingerprint / is_stale columns — staleness is deferred; the
--     single reproducibility record is sample_batch_id + sample_seed (phase_9 §2).
--   * access_class is fail-closed to the most-restrictive default
--     'user_supplied_private' (decision 60/76; propagated by eval/audit.py §7).
--     Intentionally NO column-level access_class CHECK constraint here (plan §4
--     authors none); enforcement is the python stamping rule, not a constraint.
--   * audit_type / subject_type / verdict / severity / decision are SOFT TEXT
--     vocab (the closed sets live in eval/audit.py), not SQL CHECKs — the subject
--     set is intentionally polymorphic and grows with producers (r2 gap-12).

CREATE TABLE audit_records (
    audit_id        TEXT PRIMARY KEY,        -- 'audit_' || uuid4().hex (decision 65)
    audit_type      TEXT NOT NULL,           -- soft vocab (phase_9 §4); every value has a producer
    subject_type    TEXT NOT NULL,           -- soft polymorphic kind of the audited row
    subject_id      TEXT NOT NULL,           -- soft ref to the audited row (NO cross-table FK)
    run_id          TEXT,                    -- build run the subject belongs to (nullable)
    sample_batch_id TEXT,                    -- groups one §11 draw / release sample
    sample_seed     TEXT,                    -- RNG seed for this batch (single reproducibility record)
    verdict         TEXT,                    -- graded judgment (vocab depends on audit_type)
    severity        TEXT,                    -- low|medium|high|critical
    problem         TEXT,                    -- free text (doc 09 §4)
    recommended_fix TEXT,                    -- free text (doc 09 §4)
    decision        TEXT,                    -- accept|reject|edit (validation action, decision 80)
    edit_payload    TEXT,                    -- JSON of edited fields when decision='edit'
    reviewer        TEXT,                    -- 'human' | agent id | email
    status          TEXT NOT NULL DEFAULT 'open',                    -- open|resolved
    access_class    TEXT NOT NULL DEFAULT 'user_supplied_private',   -- propagated, fail-closed (decision 60/76)
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);

CREATE INDEX ix_audit_open    ON audit_records(status, audit_type);
CREATE INDEX ix_audit_batch   ON audit_records(sample_batch_id);
CREATE INDEX ix_audit_subject ON audit_records(subject_type, subject_id);
