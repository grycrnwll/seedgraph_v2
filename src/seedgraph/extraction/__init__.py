"""Phase 4 — default field-agnostic note extraction.

Produces, for every converted paper that FITS the routed model context window, a
durable evidence-linked structured note: a thin ``structured_notes`` header (1:1
with an ``extraction_runs`` row), typed ``extracted_claims`` (one row per schema
field / array element — not-found is a row), and verbatim-anchored evidence spans
linked through the ``claim_spans`` junction (born in this phase's 0007_notes.sql).

Public surface (see ``runner``/``schema``/``validator``/``normalize``):
    extract_note, validate_note, normalize_note, resolve_source_access_class,
    current_note, DefaultNoteV1, SCHEMA_ID, SCHEMA_VERSION, PROMPT_VERSION.
"""

from __future__ import annotations
