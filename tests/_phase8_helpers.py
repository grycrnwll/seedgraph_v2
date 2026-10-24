"""Shared offline v2 project fixture (kept from the phase_8 suite, whose answer
harness was deleted in stage 2).

A tiny ``project.db`` (~3 works + spans + claims + notes + citation_edges + one
``metadata_only`` work). Fully offline.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from seedgraph.project import service


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_conn(handle) -> sqlite3.Connection:
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def build_fixture_project(slug: str = "ph8"):
    """Create a project and populate the offline fixture corpus. Returns the handle.

    Works:
      * ``work_a`` (included)  — spans on "Assumption 2 / parallel trends" (open) and a
        ``user_supplied_private`` span ("secret identification trick").
      * ``work_b`` (included)  — span on "rank condition" (open).
      * ``work_c`` (metadata_only) — NO spans; cited by work_a (recommendation target).
    """
    handle = service.create_project(slug)
    conn = _raw_conn(handle)

    def add_work(work_id, title, status="included", year=2020):
        conn.execute(
            "INSERT INTO works (work_id, canonical_title, year, created_at) VALUES (?,?,?,?)",
            (work_id, title, year, _now()),
        )
        conn.execute(
            "INSERT INTO project_documents (work_id, inclusion_status, is_seed, created_at, "
            "updated_at) VALUES (?,?,?,?,?)",
            (work_id, status, 0, _now(), _now()),
        )

    def add_run(run_id, work_id, access_class="open_access"):
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_version, prompt_version, access_class, run_status, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, work_id, "md_" + work_id, "h", "v1", "p1", access_class, "success", _now()),
        )

    def add_note(note_id, run_id, work_id, text, access_class="open_access"):
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_version, prompt_version, access_class, raw_note_json, "
            "note_text, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (note_id, run_id, work_id, "md_" + work_id, "h", "v1", "p1", access_class, "{}",
             text, _now()),
        )
        conn.execute("INSERT INTO note_fts(note_id, note_text) VALUES (?,?)", (note_id, text))

    def add_section(section_id, work_id, kind="body", heading="Identification"):
        conn.execute(
            "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, "
            "source_file_id, source_file_hash, work_id, level, ordinal, heading_text, "
            "heading_path, section_kind, start_char, end_char, section_parser_version, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (section_id, "md_" + work_id, "h", "sf", "sfh", work_id, 1, 0, heading, heading,
             kind, 0, 100, "sp1", _now()),
        )

    def add_claim(claim_id, note_id, run_id, work_id, claim_type, label, text,
                  epistemic="llm_extracted", assertion="stated", access_class="open_access"):
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
            "work_id, claim_type, field_key, normalized_label, claim_text, status, "
            "epistemic_type, assertion_status, access_class, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (claim_id, note_id, run_id, work_id, claim_type, "f", label, text, "found",
             epistemic, assertion, access_class, _now()),
        )
        conn.execute(
            "INSERT INTO claim_fts(claim_id, normalized_label, claim_text) VALUES (?,?,?)",
            (claim_id, label, text),
        )

    def add_span(span_id, work_id, quote, access_class="open_access", section_id=None):
        conn.execute(
            "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
            "source_file_hash, work_id, section_id, start_char, end_char, exact_quote, "
            "quote_hash, access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (span_id, "md_" + work_id, "h", "sf", "sfh", work_id, section_id, 0, len(quote),
             quote, "qh_" + span_id, access_class, _now()),
        )
        conn.execute(
            "INSERT INTO span_fts(quote_text, span_id, markdown_id, work_id, section_id) "
            "VALUES (?,?,?,?,?)",
            (quote, span_id, "md_" + work_id, work_id, section_id),
        )

    def link(claim_id, span_id):
        conn.execute(
            "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?,?,0,?)",
            (claim_id, span_id, _now()),
        )

    add_work("work_a", "Paper A: Difference-in-Differences")
    add_work("work_b", "Paper B: Linear IV")
    add_work("work_c", "Paper C: Foundational (metadata only)", status="metadata_only")
    # work_x was EXCLUDED *after* extraction, so it still RETAINS a citable span (s_x).
    # The §11 faithfulness invariant must keep it out of every citation set despite the
    # retained span — exercised non-vacuously by the safety/guard tests.
    add_work("work_x", "Paper X: Excluded after extraction", status="excluded")
    add_run("r_a", "work_a")
    add_run("r_b", "work_b")
    add_run("r_a_priv", "work_a", access_class="user_supplied_private")

    add_section("sec_a", "work_a", kind="body", heading="Identification")

    add_note("n_a", "r_a", "work_a", "Paper A discusses parallel trends and Assumption 2.")
    add_note("n_b", "r_b", "work_b", "Paper B discusses the rank condition for identification.")

    add_claim("c_a", "n_a", "r_a", "work_a", "identification_assumption", "parallel trends",
              "The parallel trends assumption is central to the design.")
    add_claim("c_b", "n_b", "r_b", "work_b", "regularity_condition", "rank condition",
              "The rank condition must hold for identification.")

    add_span("s_a", "work_a", "We invoke Assumption 2: parallel trends hold across groups.",
             section_id="sec_a")
    add_span("s_b", "work_b", "The rank condition ensures point identification.")
    add_span("s_priv", "work_a", "The secret identification trick relies on hidden data.",
             access_class="user_supplied_private")
    # Excluded work_x retains a span sharing the "across groups" phrase with s_a, so a
    # grounded query reaches BOTH — pre-fix the excluded work was citable; post-fix the
    # retrieval/guard filter keeps it out while the included work_a is still cited.
    add_span("s_x", "work_x", "Across groups, the retracted finding no longer holds.")

    link("c_a", "s_a")
    link("c_b", "s_b")

    conn.execute(
        "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, provenance, "
        "confidence, run_id, created_at) VALUES ('work_a','work_c','cites',"
        "'provider_reference',1.0,'cite1',?)",
        (_now(),),
    )
    conn.commit()
    conn.close()
    return handle


def empty_fts_project(slug: str = "ph8empty"):
    """A migrated project with works but NO extracted spans/claims/notes (empty FTS)."""
    handle = service.create_project(slug)
    conn = _raw_conn(handle)
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, created_at) VALUES "
        "('work_x','Unextracted Paper',2021,?)",
        (_now(),),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, created_at, "
        "updated_at) VALUES ('work_x','included',1,?,?)",
        (_now(), _now()),
    )
    conn.commit()
    conn.close()
    return handle


