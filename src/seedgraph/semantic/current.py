"""Shared read-time selection of activated note and lens interpretations."""

#: Read-time "current claims" filter (stale-claims fix). ``structured_notes`` is
#: append-only with NO supersession flag — a re-extraction (prompt/schema bump,
#: changed markdown, or ``--force``) APPENDS a new note for the same
#: ``(work_id, schema_id)`` and the old generation is superseded implicitly. The
#: write-time rule (``extraction.runner.current_note``) pins the live code
#: constants (SCHEMA_VERSION/PROMPT_VERSION), which a read-time filter cannot
#: use: a work extracted under an older prompt and never re-extracted must still
#: contribute. So read-time current = the LATEST note per ``(work_id,
#: schema_id)`` by ``(created_at, rowid)`` — exactly the row the writer would
#: have superseded last. Lens-origin claims (``structured_note_id IS NULL``,
#: run carries ``lens_id``) get the analogous rule: latest run per
#: ``(work_id, lens_id)``.
#: ponytail: lens currency here is latest-run-per-(work, lens) only — the
#: definition-hash + live-markdown staleness walk stays in ``lenses/results.py``
#: (it needs a cross-db cache read this SQL-level filter can't do). Note-less
#: claims from non-lens runs have no supersession container and pass through
#: unfiltered (also keeps direct-claim test fixtures valid).
CURRENT_CLAIMS_CTE = """
WITH current_notes AS (
    SELECT note_id FROM (
        SELECT note_id, ROW_NUMBER() OVER (
            PARTITION BY work_id, schema_id
            ORDER BY created_at DESC, rowid DESC
        ) AS rn FROM structured_notes
    ) WHERE rn = 1
),
current_lens_runs AS (
    SELECT extraction_run_id FROM (
        SELECT extraction_run_id, ROW_NUMBER() OVER (
            PARTITION BY work_id, lens_id
            ORDER BY created_at DESC, rowid DESC
        ) AS rn FROM extraction_runs WHERE lens_id IS NOT NULL
    ) WHERE rn = 1
),
current_claims AS (
    SELECT ec.*
    FROM extracted_claims ec
    LEFT JOIN extraction_runs er ON er.extraction_run_id = ec.extraction_run_id
    WHERE (ec.structured_note_id IS NOT NULL
           AND ec.structured_note_id IN (SELECT note_id FROM current_notes))
       OR (ec.structured_note_id IS NULL AND er.lens_id IS NOT NULL
           AND ec.extraction_run_id
               IN (SELECT extraction_run_id FROM current_lens_runs))
       OR (ec.structured_note_id IS NULL AND er.lens_id IS NULL)
)
"""
