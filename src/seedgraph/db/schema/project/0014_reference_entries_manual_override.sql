-- project.db — Build A chunk 8. Widen reference_entries.resolution_source CHECK
-- to admit 'manual_override' (the human-verdict provenance written by the
-- citation_resolution review applier). SQLite cannot ALTER a CHECK, so this is a
-- table recreate — the FIRST in the repo. Mechanism notes (binding):
--
-- * FK toggling lives INSIDE this script. The runner (db/migrations.py) applies
--   each migration as a bare conn.executescript(sql) on a connection opened with
--   foreign_keys=ON; with FKs ON, `DROP TABLE reference_entries` performs an
--   implicit DELETE that violates citation_edges.reference_id on any populated
--   DB. Python's executescript commits any open transaction and then runs these
--   statements in AUTOCOMMIT, so the PRAGMA below is effective (it would be a
--   no-op inside a transaction).
-- * Consequently the recreate is NOT atomic: statements commit one by one.
--   Non-atomicity is handled by RE-RUNNABILITY, not pretended away: the leading
--   `DROP TABLE IF EXISTS reference_entries_new` lets a mid-script crash re-run
--   cleanly from the top (the runner writes the schema_migrations ledger row
--   only after the whole script completes, so an interrupted migration is
--   retried whole).
-- * Rename order is safe: nothing references reference_entries_new, so SQLite's
--   rename-time FK-clause fixups never touch citation_edges, whose FK names
--   `reference_entries` and lands on the renamed table.
-- * Verification (PRAGMA foreign_key_check empty + the three idx_* present on a
--   fresh AND a populated DB) lives in tests/test_migrations.py, not here — a
--   PRAGMA inside executescript returns rows but cannot fail the script.
--
-- DDL below is IDENTICAL to project/0004_citation_graph.sql §4.1 except the
-- widened resolution_source CHECK. ORM (db/project_models.ReferenceEntry) needs
-- no change: CHECK constraints are not part of the mapping (D6 parity unaffected).

PRAGMA foreign_keys=OFF;

DROP TABLE IF EXISTS reference_entries_new;

CREATE TABLE reference_entries_new (
    reference_id        TEXT PRIMARY KEY,                          -- 'ref_<uuid>'
    citing_work_id      TEXT NOT NULL,                             -- FK -> works.work_id
    raw_reference_text  TEXT NOT NULL,                             -- PRIVATE (full-text-derived); kept verbatim for recall
    parsed_fields_json  TEXT,                                      -- {first_author, year, title, doi, arxiv} best-effort
    resolved_work_id    TEXT,                                      -- nullable FK -> works.work_id
    resolution_status   TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (resolution_status IN ('resolved', 'unresolved', 'ambiguous', 'suspect')),
    resolution_source   TEXT
        CHECK (resolution_source IS NULL OR resolution_source IN ('doi', 'arxiv', 'title_year', 'manual_override')),
    confidence          REAL
        CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    markdown_id         TEXT,                                      -- cache.db markdown_documents.markdown_id (soft xref)
    markdown_hash       TEXT,                                      -- content anchor for staleness (decision 2/24)
    section_label       TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (citing_work_id)   REFERENCES works(work_id),
    FOREIGN KEY (resolved_work_id) REFERENCES works(work_id)
);

INSERT INTO reference_entries_new
    SELECT reference_id, citing_work_id, raw_reference_text, parsed_fields_json,
           resolved_work_id, resolution_status, resolution_source, confidence,
           markdown_id, markdown_hash, section_label, created_at
    FROM reference_entries;

DROP TABLE reference_entries;

ALTER TABLE reference_entries_new RENAME TO reference_entries;

CREATE INDEX idx_reference_entries_citing   ON reference_entries(citing_work_id);
CREATE INDEX idx_reference_entries_resolved ON reference_entries(resolved_work_id);
CREATE INDEX idx_reference_entries_status   ON reference_entries(resolution_status);

PRAGMA foreign_keys=ON;
