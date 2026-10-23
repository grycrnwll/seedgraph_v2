-- project.db — Build D chunk 10 (D-10; gap scan §4.3). Last of the program's
-- four project migrations (0014 A, 0015 B, 0016 C, 0017 D); merges after C's
-- 0016 (the runner applies only versions strictly greater than
-- MAX(schema_migrations.version)).
--
-- Two nullable columns on `works` powering the missing-paper triage:
--
--   abstract  — the running abstract, rehydrated provider-side from OpenAlex's
--               abstract_inverted_index (providers/openalex.py _work_to_record).
--               LOCAL-ONLY per CONTENT_ACCESS_POLICY.md:42; excluded from every
--               export tier by the existing D8 allowlist (vocab.py
--               PROVIDER_SHAREABLE_FIELDS), which this build does not touch.
--   oa_status — free-text OA color (gold/green/hybrid/bronze/diamond/closed),
--               deliberately NOT an enum/CHECK (decision-1 style: display/triage
--               vocabulary stays app-side). Separates "OA exists, a re-run or
--               bigger budget will get it" (available_open_access) from
--               "genuinely paywalled".
--
-- Additive + nullable, no backfill: safe on populated databases. NOTE: the
-- provider chain caches SHAPED records, so pre-existing provider_cache entries
-- lack the two new fields — harmless (columns are nullable; both are backfilled
-- empty-only on the next fresh provider fetch of the work).
-- ORM mirror: db/project_models.py::Work (parity-tested by
-- tests/test_phase_5.py::test_phase5_schema_parity + tests/test_migrations.py).

ALTER TABLE works ADD COLUMN abstract TEXT;
ALTER TABLE works ADD COLUMN oa_status TEXT;
