-- project.db — Phase 6 (project-specific lenses). Authoritative schema (D6) for
-- this phase's NEW tables/columns: the lens registry, the generic per-record
-- lens-output container, and the lens-version stamp added to extraction_runs.
--
-- Applies ON TOP OF phase_0 (0001) + the lower-numbered project migrations that
-- create the tables referenced here by exact name:
--   works                         -> phase_5  (project_model)
--   work_source_files             -> phase_5b (acquisition_resolution; read-only here)
--   evidence_spans / claim_fts    -> phase_3 / phase_4 (read/append-only here)
--   extraction_runs               -> phase_4  (ALTERed below; owned jointly per §12)
--   extracted_claims / claim_spans-> phase_4  (CONSUMED only — NEVER altered here, decision 70 / §12)
--
-- No cache.db changes. No cross-database FK (cross-scope refs resolved at runtime
-- by read-only ATTACH + content-hash join, decisions 2/24/42). PRAGMA
-- foreign_keys=ON is set by db/connection.py for this scope (decision 8).

-- 4.1  NEW: lenses — thin registry over the on-disk YAML source-of-truth.
--   While status IN ('draft','calibrating') the lens is mutable: definition_hash
--   refreshes in place from the YAML, prior runs go stale (never deleted), and
--   definition_yaml stays NULL. On first full run the lens promotes to 'active'
--   and definition_yaml is snapshotted once (frozen reproducibility, decision 50);
--   further edits require a new lens_id (immutability-on-use).
CREATE TABLE lenses (
    lens_id            TEXT PRIMARY KEY,             -- authored slug+version, e.g. 'regularity_conditions_v1'
    name               TEXT NOT NULL,
    scope              TEXT NOT NULL DEFAULT 'project',
    object_type        TEXT NOT NULL,               -- e.g. 'assumption'
    status             TEXT NOT NULL DEFAULT 'draft',-- draft|calibrating|active|archived (vocab.LensStatus)
    yaml_path          TEXT NOT NULL,               -- POSIX path relative to projects/{slug}/
    definition_hash    TEXT NOT NULL,               -- sha256 hex of canonical resolved YAML (the version key)
    definition_yaml    TEXT,                        -- NULL while draft/calibrating; snapshotted once on promotion to active
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

-- 4.2  NEW: lens_outputs — one generic JSON container row per (work, record),
--   including explicit status='not_found' / 'not_applicable' rows. fields_json
--   round-trips the FULL validated output_schema instance verbatim (found AND
--   not-found shapes preserved exactly). access_class is denormalized and
--   fail-closed (decision 60/76) so lens outputs are private-by-default and the
--   default-deny export allowlist withholds them; only the lens YAML definition
--   (metadata-class) is shareable.
CREATE TABLE lens_outputs (
    lens_output_id     TEXT PRIMARY KEY,            -- new_id('lensout')
    extraction_run_id  TEXT NOT NULL REFERENCES extraction_runs(extraction_run_id),
    lens_id            TEXT NOT NULL REFERENCES lenses(lens_id),
    work_id            TEXT NOT NULL REFERENCES works(work_id),
    status             TEXT NOT NULL,               -- found|not_found|ambiguous|not_applicable|extraction_failed (vocab.FieldStatus)
    claim_id           TEXT REFERENCES extracted_claims(claim_id),  -- NULL for not_found / not_applicable
    confidence         REAL,
    fields_json        TEXT NOT NULL,               -- JSON1 instance of the lens output_schema, verbatim
    access_class       TEXT NOT NULL DEFAULT 'user_supplied_private',  -- vocab enforced in Python only (decisions 14/30/48); no SQL CHECK (matches 0007/0010/0012)
    created_at         TEXT NOT NULL
);
CREATE INDEX idx_lens_outputs_lens ON lens_outputs(lens_id, status);
CREATE INDEX idx_lens_outputs_work ON lens_outputs(work_id);
CREATE INDEX idx_lens_outputs_run  ON lens_outputs(extraction_run_id);

-- 4.3  ALTER: extraction_runs — add the lens version stamp. NULL for default-
--   schema (phase_4) runs. A lens run sets lens_id + lens_definition_hash and,
--   together with the existing markdown_hash column, these form the
--   idempotency/staleness key (§7). This is the ONLY ALTER this phase owns;
--   extracted_claims / claim_spans are NEVER altered here (§12).
ALTER TABLE extraction_runs ADD COLUMN lens_definition_hash TEXT;  -- NULL for default-schema runs

-- 4.4  review_queue item type 'lens_record'. ambiguous / extraction_failed lens
--   records and unanchorable quotes route to review_queue with
--   item_type='lens_record' (a sanctioned polymorphic-queue extension, decision
--   80). review_queue (phase_0) has a free-text item_type column with no CHECK
--   constraint, so NO DDL is required to admit this value — it is documented
--   here as the authoritative sanctioned value for this phase.
