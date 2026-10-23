-- project.db — Phase 4 (default note extractor). Authoritative schema (decision D6).
--
-- Creates the default-note substrate: per-unit extraction provenance, the thin
-- note header, the typed claims, and the two standalone FTS5 indexes. The
-- numbered .sql is the SINGLE source of truth (D6); db/models_project.py mirrors
-- it as ORM-mapping-only (no create_all authorship).
--
-- Cross-phase references (by exact name; created by lower-numbered migrations):
--   * works(work_id)            -> phase_5 (project model)
--   * evidence_spans / document_sections / span_fts -> phase_3 (evidence spans)
--   * claim_spans               -> OWNED BY this phase (§4.4). Created below AFTER
--                                  extracted_claims; its span_id FK references
--                                  evidence_spans, born in 0005 (applied first).
--
-- access_class ownership (GAP-5, decision r2-4 / decisions 30/60/76): the
-- `access_class TEXT NOT NULL DEFAULT 'user_supplied_private'` columns on
-- structured_notes and extracted_claims are BORN here (birth-DDL); phase_4 is the
-- sole owner and stamps them most-restrictive at write. No later phase ALTERs
-- these tables to add the column.
--
-- D2: extracted_claims carries BOTH epistemic_type (6-tier origin, NOT NULL) and
-- assertion_status (stated|inferred|NULL) as independent columns.
--
-- Closed vocabularies (run_status, status, epistemic_type, assertion_status,
-- access_class) are enforced in the Python layer via vocab.py StrEnums — the
-- single source of truth (decisions 14/30/48). No SQL CHECK is added here so the
-- vocab cannot drift between two authorities; the column comments enumerate the
-- sanctioned values.

-- 4.1 extraction_runs — per-unit extraction provenance; append-only.
CREATE TABLE extraction_runs (
  extraction_run_id  TEXT PRIMARY KEY,                  -- new_id('extr')
  work_id            TEXT NOT NULL REFERENCES works(work_id),
  markdown_id        TEXT NOT NULL,                     -- soft ref -> cache.db markdown_documents.markdown_id (no cross-db FK)
  markdown_hash      TEXT NOT NULL,                     -- denormalized fingerprint (staleness anchor)
  schema_id          TEXT,                              -- 'default_research_note_v1'; NULL for lens-origin runs (mutually exclusive w/ lens_id, decision 22 / phase_6 §4.3)
  lens_id            TEXT,                              -- NULL here; reserved for phase_6 (mutually exclusive w/ schema_id)
  schema_version     TEXT NOT NULL,
  prompt_version     TEXT NOT NULL,
  model_name         TEXT,                              -- NULL on no-LLM/skip
  provider           TEXT,
  access_mode        TEXT,                              -- api_key | local | disabled
  temperature        REAL,
  access_class       TEXT NOT NULL DEFAULT 'user_supplied_private',
  external_full_text INTEGER NOT NULL DEFAULT 0,        -- did source text leave the machine
  run_status         TEXT NOT NULL,                     -- success | extraction_failed | skipped_no_llm | skipped_policy | skipped_oversize | skipped_budget
  input_tokens       INTEGER,
  output_tokens      INTEGER,
  estimated_cost     REAL,
  run_id             TEXT,                              -- optional build run_id (manifest), decision 78
  created_at         TEXT NOT NULL
);
CREATE INDEX idx_extr_work ON extraction_runs(work_id, schema_id, created_at);

-- 4.2 structured_notes — thin header, 1:1 with a successful run; container of
-- claims. A row exists IFF the run succeeded (no `status` column — failures
-- write no note). access_class born here (decision r2-4).
CREATE TABLE structured_notes (
  note_id            TEXT PRIMARY KEY,                  -- new_id('note')
  extraction_run_id  TEXT NOT NULL UNIQUE REFERENCES extraction_runs(extraction_run_id),
  work_id            TEXT NOT NULL REFERENCES works(work_id),
  markdown_id        TEXT NOT NULL,
  markdown_hash      TEXT NOT NULL,
  schema_id          TEXT,                              -- nullable: lens-origin runs set lens_id not schema_id (no default schema header)
  schema_version     TEXT NOT NULL,
  prompt_version     TEXT NOT NULL,
  archetype          TEXT,                              -- derived from populated fields (decision 9)
  access_class       TEXT NOT NULL DEFAULT 'user_supplied_private',
  raw_note_json      TEXT NOT NULL,                     -- lossless validated LLM JSON
  note_text          TEXT NOT NULL,                     -- flattened summary, source for note_fts
  created_at         TEXT NOT NULL
);
CREATE INDEX idx_note_work ON structured_notes(work_id, schema_id, created_at);

-- 4.3 extracted_claims — one row per schema field / array element (not-found is a
-- row). epistemic_type + assertion_status are independent (D2). No scalar
-- evidence_span_id (decision 70) — claim<->span is the many-to-many claim_spans
-- junction owned by phase_3. access_class born here (decision r2-4).
CREATE TABLE extracted_claims (
  claim_id            TEXT PRIMARY KEY,                 -- new_id('claim')
  structured_note_id  TEXT REFERENCES structured_notes(note_id),  -- nullable: lens-origin claims mint no structured_notes header
  extraction_run_id   TEXT NOT NULL REFERENCES extraction_runs(extraction_run_id),
  work_id             TEXT NOT NULL REFERENCES works(work_id),
  claim_type          TEXT NOT NULL,                    -- doc 05 §6 vocab incl. {identification_assumption,regularity_condition,general_assumption,data,setting,...}
  claim_subtype       TEXT,                             -- ONLY method_type / result_type / limitation_type (decision 47)
  field_key           TEXT NOT NULL,                    -- schema field path, e.g. 'assumptions[1]'
  normalized_label    TEXT,
  claim_text          TEXT,                             -- nullable (not_found rows)
  status              TEXT NOT NULL,                    -- found | not_found | ambiguous | not_applicable | extraction_failed
  epistemic_type      TEXT NOT NULL,                    -- 6-tier origin: deterministic|metadata_resolved|llm_extracted|llm_inferred|user_supplied|user_validated (D2)
  assertion_status    TEXT,                             -- author assertion: stated|inferred|NULL; INDEPENDENT of epistemic_type (D2)
  inferred_explanation TEXT,                            -- required when assertion_status='inferred'
  confidence          REAL,
  access_class        TEXT NOT NULL DEFAULT 'user_supplied_private',
  created_at          TEXT NOT NULL
);
CREATE INDEX idx_claim_note ON extracted_claims(structured_note_id);
CREATE INDEX idx_claim_work_type ON extracted_claims(work_id, claim_type);

-- 4.4 claim_spans — canonical claim<->span junction (decision 70). OWNED here
-- (moved from phase_3 per RECONCILIATION). Surrogate PK; both FKs are real and
-- enforced: claim_id -> extracted_claims (created just above, same migration),
-- span_id -> evidence_spans (born in 0005, applied before 0007). phase_4 INSERTs
-- the junction row in the same transaction as the claim + ensure_span() (D7).
CREATE TABLE claim_spans (
  id         INTEGER PRIMARY KEY,                       -- surrogate (pure junction row)
  claim_id   TEXT NOT NULL REFERENCES extracted_claims(claim_id),
  span_id    TEXT NOT NULL REFERENCES evidence_spans(span_id),
  rank       INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  UNIQUE (claim_id, span_id)
);
CREATE INDEX idx_claim_spans_span ON claim_spans(span_id);

-- 4.5 FTS5 (standalone, unicode61 — decisions 10/54). No porter/stemming so exact
-- scholarly terms survive. No triggers — reindex is application-managed in the
-- same write transaction (delete-by-id + insert via db/fts.py). span_fts already
-- exists from phase_3 and is NOT recreated here.
CREATE VIRTUAL TABLE IF NOT EXISTS claim_fts USING fts5(
  claim_id UNINDEXED, normalized_label, claim_text,
  tokenize = "unicode61 remove_diacritics 2"
);
CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
  note_id UNINDEXED, note_text,
  tokenize = "unicode61 remove_diacritics 2"
);
