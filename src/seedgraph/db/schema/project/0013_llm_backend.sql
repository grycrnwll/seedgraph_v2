-- 0013 LLM backend (Track 1): additive executor telemetry on llm_usage_events.
--
-- The policy-gated executor logs one usage row per call with status / error /
-- latency / request id / retry count, plus FULL sha256 hex hashes of the prompt
-- and response. NO prompt/response BODY is ever stored — only the hashes
-- (cross-cutting decision #5); there is deliberately no text-excerpt column.
--
-- llm_usage_events is a raw-SQL-only table (no SQLAlchemy ORM model), so these
-- ALTERs have ZERO ORM-parity impact. All new columns are nullable except
-- retry_count (NOT NULL DEFAULT 0). No access_class CHECK is added here, so the
-- vocab/SQL parity test is unaffected.

ALTER TABLE llm_usage_events ADD COLUMN status                TEXT;
ALTER TABLE llm_usage_events ADD COLUMN error_code            TEXT;
ALTER TABLE llm_usage_events ADD COLUMN latency_ms            INTEGER;
ALTER TABLE llm_usage_events ADD COLUMN request_id            TEXT;
ALTER TABLE llm_usage_events ADD COLUMN prompt_hash           TEXT;   -- sha256 hex, never a body
ALTER TABLE llm_usage_events ADD COLUMN response_hash         TEXT;   -- sha256 hex, never a body
ALTER TABLE llm_usage_events ADD COLUMN retry_count           INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_usage_events ADD COLUMN pricing_snapshot_date TEXT;
