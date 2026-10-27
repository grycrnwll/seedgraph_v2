-- graph.db 0002 — literatures (v3 design §6.2). Written by seedgraph.library.literatures.
-- One row per regroup in literature_runs; the latest run is the current grouping.
-- citation_sha256 fingerprints the member list AND the in-graph citation edges, so a
-- run is stale when either changes. Unplaced papers are not stored: they are the
-- graph's members that sit in no literature of the latest run.

CREATE TABLE literature_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm       TEXT NOT NULL,
    resolution      REAL NOT NULL,
    chosen_by       TEXT NOT NULL CHECK (chosen_by IN ('rule', 'researcher')),
    rule_met        INTEGER NOT NULL CHECK (rule_met IN (0, 1)),
    seed            INTEGER NOT NULL,
    modularity      REAL,               -- standard (gamma = 1) modularity, comparable across rungs
    ari_mean        REAL,               -- mean ARI of stability seeds vs seed; NULL when not run
    nmi_mean        REAL,
    stability_seeds TEXT NOT NULL DEFAULT '',  -- comma-separated
    citation_sha256 TEXT NOT NULL,
    built_at        TEXT NOT NULL
);

-- Names: agent_name/agent_description come from the naming task (agent inferred);
-- researcher_name overrides and survives regrouping. Both are carried to the next run's
-- literature with member Jaccard >= 0.5 (carried_from names the source literature).
CREATE TABLE literatures (
    literature_id     TEXT PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES literature_runs(run_id) ON DELETE CASCADE,
    ordinal           INTEGER NOT NULL,
    size              INTEGER NOT NULL,
    agent_name        TEXT,
    agent_description TEXT,
    researcher_name   TEXT,
    named_at          TEXT,
    carried_from      TEXT,
    UNIQUE (run_id, ordinal)
);

CREATE TABLE literature_members (
    literature_id TEXT NOT NULL REFERENCES literatures(literature_id) ON DELETE CASCADE,
    paper_id      TEXT NOT NULL,
    PRIMARY KEY (literature_id, paper_id)
);
CREATE INDEX ix_literature_members_paper ON literature_members (paper_id);
