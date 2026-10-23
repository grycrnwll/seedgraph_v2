-- project.db — Build C chunk 9 (D10; decision 56's flip-detection mandate).
-- Add a nullable JSON `payload` column to audit_records: the per-build canon
-- decision log written by semantic/audit.record_canon_decision_log
-- (audit_type='canon_decision_log', subject_type='concept_set',
-- subject_id=concept_set_hash) and diffed by compare_canon_decision_logs.
--
-- Deliberately NOT edit_payload — that column is reserved for decision='edit'
-- grading semantics (eval/audit.record_verdict). Additive + nullable, no
-- backfill: safe on populated databases; every existing consumer ignores it.
-- ORM mirror: db/models_project.py::AuditRecord.payload (parity-tested by
-- test_phase_9_audit.test_audit_records_orm_migration_parity).

ALTER TABLE audit_records ADD COLUMN payload TEXT;
