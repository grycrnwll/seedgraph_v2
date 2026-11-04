-- library.db 0012 — spot-check samples and marks (v3 stage 5 design §5.2-5.5).
--
-- A spot-check sample is about 20 agent claims drawn least-marked cell first
-- (seedgraph.spotcheck.sample). Each item is keyed by the immutable extracted_claims id,
-- which an overlay never changes, so a mark always judges the agent's original claim.
-- "Fixed" is derived, not stored: an overlay_versions row for the paper after marked_at
-- whose ops touch the claim's fingerprint. A changed mark replaces the old one in place
-- (the last mark counts). No seed rows.

CREATE TABLE spot_check_samples (
    sample_id    TEXT PRIMARY KEY,                       -- 'sc_' + hash of seed, size and ids
    seed         INTEGER NOT NULL,
    size         INTEGER NOT NULL CHECK (size >= 1),
    filters_json TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    origin       TEXT NOT NULL DEFAULT 'library'
);
CREATE INDEX ix_spot_check_samples_created ON spot_check_samples (created_at, sample_id);

CREATE TABLE spot_check_items (
    sample_id         TEXT NOT NULL REFERENCES spot_check_samples(sample_id),
    claim_id          TEXT NOT NULL,                     -- SOFT ref -> extracted_claims
    ordinal           INTEGER NOT NULL CHECK (ordinal >= 1),
    paper_id          TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,
    claim_type        TEXT NOT NULL,                     -- the original claim_type
    model_key         TEXT NOT NULL,                     -- model, '<provider>, model not recorded' or 'model not recorded'
    claim_fingerprint TEXT NOT NULL,                     -- stage 3 notes.identity: finds the claim in a later note
    mark              TEXT CHECK (mark IS NULL OR mark IN ('right', 'partly', 'wrong')),
    reasons_json      TEXT NOT NULL DEFAULT '[]',
    note              TEXT,
    marked_at         TEXT,
    origin            TEXT NOT NULL DEFAULT 'library',
    PRIMARY KEY (sample_id, claim_id),
    UNIQUE (sample_id, ordinal),
    CHECK ((mark IS NULL) = (marked_at IS NULL))
);
CREATE INDEX ix_spot_check_items_claim ON spot_check_items (claim_id);
CREATE INDEX ix_spot_check_items_cell ON spot_check_items (claim_type, model_key);

-- Raw counts per (claim type, model key) over the latest mark of each claim. Per-type and
-- per-model figures and the weighted overall line are computed from these rows
-- (seedgraph.spotcheck.accuracy); partly right is counted apart, never as half.
CREATE VIEW spot_check_accuracy AS
WITH latest AS (
    SELECT claim_id, claim_type, model_key, mark,
           ROW_NUMBER() OVER (PARTITION BY claim_id ORDER BY marked_at DESC, sample_id DESC) AS rn
    FROM spot_check_items
    WHERE mark IS NOT NULL
)
SELECT claim_type,
       model_key,
       SUM(mark = 'right')  AS n_right,
       SUM(mark = 'partly') AS n_partly,
       SUM(mark = 'wrong')  AS n_wrong,
       COUNT(*)             AS n_marked
FROM latest
WHERE rn = 1
GROUP BY claim_type, model_key;
