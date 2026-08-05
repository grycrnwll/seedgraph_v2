-- Phase 4b — Chunked / Section Extraction: chunk/section map-reduce provenance.
--
-- Authoritative schema delta for phase_4b (decision D6 — the numbered .sql is the
-- single source of truth; ORM classes only MIRROR this, never author it via
-- create_all). Implements plan §4.1 verbatim. This migration adds NO new table:
-- phase_4b reuses the phase_4 artifact tables (extraction_runs / structured_notes
-- / extracted_claims / claim_spans) and the phase_3 span tables. The only schema
-- change is ADDITIVE, NULLABLE, backward-compatible chunk-provenance COLUMNS on
-- two existing phase_4 tables, plus two supporting indexes.
--
-- Ordering: this migration (version 0008) applies on top of phase_0 (0001) and
-- all lower-numbered project migrations, INCLUDING the phase_4 extraction-notes
-- migration that CREATEs `extraction_runs` and `extracted_claims`. It only ALTERs
-- those tables; it never re-CREATEs them (decision D6 ownership: phase_4 owns the
-- birth-DDL incl. the access_class columns; phase_4b only appends provenance).
--
-- `extraction_mode` distinguishes phase-4 whole-doc runs from phase-4b map/reduce
-- runs. NOT NULL DEFAULT 'whole' so pre-existing phase_4 rows back-fill correctly
-- and SQLite's ADD COLUMN constant-default rule is satisfied.

ALTER TABLE extraction_runs ADD COLUMN extraction_mode TEXT NOT NULL DEFAULT 'whole';
    -- 'whole' (phase 4) | 'chunked_map' (one per chunk) | 'chunked_reduce' (the merged note's run)
ALTER TABLE extraction_runs ADD COLUMN parent_extraction_run_id TEXT;
    -- map runs point at their reduce run; NULL for 'whole' and 'chunked_reduce'
ALTER TABLE extraction_runs ADD COLUMN chunk_index INTEGER;     -- 0-based; NULL unless 'chunked_map'
ALTER TABLE extraction_runs ADD COLUMN chunk_count INTEGER;     -- total chunks in the plan; NULL for 'whole'
ALTER TABLE extraction_runs ADD COLUMN chunk_section_ids TEXT;  -- JSON array of document_sections.section_id covered by the chunk

ALTER TABLE extracted_claims ADD COLUMN source_chunk_index INTEGER;       -- which map chunk produced this claim (NULL for whole/merged-scalar)
ALTER TABLE extracted_claims ADD COLUMN source_extraction_run_id TEXT;    -- the map run this claim was extracted in (audit back-link)

CREATE INDEX IF NOT EXISTS idx_extr_parent ON extraction_runs(parent_extraction_run_id);
CREATE INDEX IF NOT EXISTS idx_extr_mode   ON extraction_runs(work_id, extraction_mode, created_at);
