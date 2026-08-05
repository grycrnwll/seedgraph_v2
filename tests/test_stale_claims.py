"""Stale-claims fix — concepts build reads CURRENT claims only.

``structured_notes`` is append-only with no supersession flag: a re-extraction
(prompt/schema bump, changed markdown, ``--force``) APPENDS a new note per
``(work_id, schema_id)`` and the old generation is superseded implicitly.
Before the fix, ``build_concepts`` consumed ``extracted_claims`` unfiltered, so
after a prompt-version re-extraction the old and new generations mixed (live
evidence: 124 of 223 concept links on test_proj came from replaced 1.0.0
notes). These tests pin the read-time currency rule in
``semantic.concepts._CURRENT_CLAIMS_CTE``:

  * note-origin claims count iff their note is the LATEST for its
    ``(work_id, schema_id)`` by ``(created_at, rowid)``;
  * lens-origin claims (``structured_note_id IS NULL``) count iff their run is
    the LATEST for its ``(work_id, lens_id)``;
  * note-less non-lens claims have no supersession container and pass through.

Fully offline (no LLM: ``profile=None`` → NullProposer).
"""

from __future__ import annotations

import math
import sqlite3

import seedgraph.db.migrations as migrations
from seedgraph.semantic import build_semantic_overlay

T1 = "2026-01-01T00:00:00+00:00"
T2 = "2026-01-02T00:00:00+00:00"

SCHEMA_ID = "default_research_note_v1"


def _conn(tmp_path) -> sqlite3.Connection:
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(tmp_path / "project.db"))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    return conn


def _add_work(conn, work_id):
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
        (work_id, f"Paper {work_id}", T1),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
        "created_at, updated_at) VALUES (?, 'included', 0, ?, ?)",
        (work_id, T1, T1),
    )


def _add_run(conn, run_id, work_id, *, prompt_version="1.0.0", created_at=T1,
             lens_id=None):
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_id, lens_id, schema_version, prompt_version, "
        "access_class, run_status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, work_id, "md_" + work_id, "h",
         None if lens_id else SCHEMA_ID, lens_id, "1.0", prompt_version,
         "open_access", "success", created_at),
    )


def _add_note(conn, note_id, run_id, work_id, *, prompt_version="1.0.0",
              created_at=T1):
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, "
        "markdown_id, markdown_hash, schema_id, schema_version, prompt_version, "
        "access_class, raw_note_json, note_text, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (note_id, run_id, work_id, "md_" + work_id, "h", SCHEMA_ID, "1.0",
         prompt_version, "open_access", "{}", "note text", created_at),
    )


def _add_claim(conn, claim_id, run_id, work_id, label, *, note_id=None):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, structured_note_id, "
        "extraction_run_id, work_id, claim_type, field_key, normalized_label, "
        "claim_text, status, epistemic_type, access_class, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, note_id, run_id, work_id, "method", "f", label, label,
         "found", "llm_extracted", "open_access", T1),
    )


def _note_generation(conn, work_id, gen, *, prompt_version, created_at, labels):
    """One extraction generation: run + note + one claim per label. Returns
    the claim_ids."""
    run_id = f"extr_{work_id}_{gen}"
    note_id = f"note_{work_id}_{gen}"
    _add_run(conn, run_id, work_id, prompt_version=prompt_version,
             created_at=created_at)
    _add_note(conn, note_id, run_id, work_id, prompt_version=prompt_version,
              created_at=created_at)
    claim_ids = []
    for i, label in enumerate(labels):
        cid = f"c_{work_id}_{gen}_{i}"
        _add_claim(conn, cid, run_id, work_id, label, note_id=note_id)
        claim_ids.append(cid)
    return claim_ids


def _concept_labels(conn) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT normalized_label FROM concepts").fetchall()}


def _linked_claim_ids(conn) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT claim_id FROM claim_concepts").fetchall()}


def test_superseded_note_claims_produce_no_concept_links(tmp_path):
    """Two note generations for one work: only the current (latest) generation
    produces concepts/links; the replaced generation contributes NOTHING."""
    conn = _conn(tmp_path)
    _add_work(conn, "w1")
    old_ids = _note_generation(
        conn, "w1", "g1", prompt_version="1.0.0", created_at=T1,
        labels=["difference in differences", "old only concept"],
    )
    new_ids = _note_generation(
        conn, "w1", "g2", prompt_version="1.1.0", created_at=T2,
        labels=["difference in differences", "new only concept"],
    )
    # A second work still on the OLD prompt version and never re-extracted:
    # currency is latest-per-work, NOT match-the-live-prompt-constant, so its
    # sole note must still contribute.
    _add_work(conn, "w2")
    w2_ids = _note_generation(
        conn, "w2", "g1", prompt_version="1.0.0", created_at=T1,
        labels=["w2 concept"],
    )
    conn.commit()

    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)

    assert _concept_labels(conn) == {
        "difference in differences", "new only concept", "w2 concept",
    }
    # Only current-generation claims link; the old generation is absent.
    assert _linked_claim_ids(conn) == set(new_ids) | set(w2_ids)
    assert _linked_claim_ids(conn).isdisjoint(old_ids)
    # discusses edges derive from claim_concepts → same filtered set.
    edges = conn.execute(
        "SELECT source_node_id, target_node_id FROM project_graph_edges "
        "WHERE edge_type='discusses'"
    ).fetchall()
    assert ("w1", "concept::old only concept") not in edges
    assert ("w1", "concept::difference in differences") in edges
    assert ("w1", "concept::new only concept") in edges
    assert ("w2", "concept::w2 concept") in edges
    conn.close()


def test_latest_note_wins_created_at_tie_by_rowid(tmp_path):
    """Two generations with an identical created_at: the later-inserted note
    (higher rowid) is current — deterministic, no generation mixing."""
    conn = _conn(tmp_path)
    _add_work(conn, "w1")
    _note_generation(conn, "w1", "g1", prompt_version="1.0.0",
                     created_at=T1, labels=["first concept"])
    _note_generation(conn, "w1", "g2", prompt_version="1.1.0",
                     created_at=T1, labels=["second concept"])
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    assert _concept_labels(conn) == {"second concept"}
    conn.close()


def test_idf_n_total_and_paper_frequency_use_current_claims(tmp_path):
    """weight = log(N_total / paper_frequency) is computed over the SAME
    current-claims view as the links — a replaced generation inflates
    neither the numerator nor the per-concept link count."""
    conn = _conn(tmp_path)
    _add_work(conn, "w1")
    _note_generation(conn, "w1", "g1", prompt_version="1.0.0", created_at=T1,
                     labels=["shared thing", "old only concept"])
    _note_generation(conn, "w1", "g2", prompt_version="1.1.0", created_at=T2,
                     labels=["shared thing", "w1 special"])
    _add_work(conn, "w2")
    _note_generation(conn, "w2", "g1", prompt_version="1.1.0", created_at=T2,
                     labels=["shared thing"])
    conn.commit()

    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)

    rows = {
        label: (pf, weight, cid)
        for cid, label, pf, weight in conn.execute(
            "SELECT concept_id, normalized_label, paper_frequency, weight "
            "FROM concepts"
        ).fetchall()
    }
    assert set(rows) == {"shared thing", "w1 special"}
    # N_total = 2 works with current claims; shared in both => weight 0.
    assert rows["shared thing"][0] == 2
    assert rows["shared thing"][1] == 0.0
    assert rows["w1 special"][0] == 1
    assert abs(rows["w1 special"][1] - math.log(2)) < 1e-12
    # Exactly one link per current claim of the shared concept (w1 g2 + w2),
    # not three (the g1 duplicate contributes nothing).
    n_links = conn.execute(
        "SELECT COUNT(*) FROM claim_concepts WHERE concept_id = ?",
        (rows["shared thing"][2],),
    ).fetchone()[0]
    assert n_links == 2
    conn.close()


def test_lens_rerun_claims_excluded(tmp_path):
    """Lens-origin claims (no note header) follow the analogous rule: only the
    latest run per (work_id, lens_id) contributes."""
    conn = _conn(tmp_path)
    _add_work(conn, "w1")
    _add_run(conn, "lensrun_1", "w1", created_at=T1, lens_id="lens_1")
    _add_claim(conn, "c_lens_old", "lensrun_1", "w1", "old lens concept")
    _add_run(conn, "lensrun_2", "w1", created_at=T2, lens_id="lens_1")
    _add_claim(conn, "c_lens_new", "lensrun_2", "w1", "new lens concept")
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    assert _concept_labels(conn) == {"new lens concept"}
    assert _linked_claim_ids(conn) == {"c_lens_new"}
    conn.close()


def test_noteless_non_lens_claims_pass_through(tmp_path):
    """A claim on a non-lens run with no note header has no supersession
    container and still contributes (pins direct-claim fixture behavior)."""
    conn = _conn(tmp_path)
    _add_work(conn, "w1")
    _add_run(conn, "extr_bare", "w1", created_at=T1)
    _add_claim(conn, "c_bare", "extr_bare", "w1", "bare concept")
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    assert _concept_labels(conn) == {"bare concept"}
    assert _linked_claim_ids(conn) == {"c_bare"}
    conn.close()
