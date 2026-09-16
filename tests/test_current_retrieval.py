"""Replacement history is inspectable but not an ordinary interpretation."""
import sqlite3

from _phase8_helpers import build_fixture_project, make_spec
from test_stale_claims import _add_run, _add_note, _add_claim
from seedgraph.answer import answer, retrieve
from seedgraph.semantic.query import concept_detail, concept_provenance, concept_provenance_counts


def test_search_filters_history_before_limits_and_preserves_independent_spans():
    h = build_fixture_project("current_search")
    with sqlite3.connect(h.db_path) as conn:
        # Existing fixture notes have schema NULL; use that same container.
        for gen, label, year in (("old", "obsoleteword", "2027"), ("new", "currentword", "2028")):
            _add_run(conn, f"r_{gen}", "work_a", created_at=year)
            _add_note(conn, f"n_{gen}", f"r_{gen}", "work_a", created_at=year)
            conn.execute("UPDATE structured_notes SET schema_id=NULL, note_text=? WHERE note_id=?",
                         (f"sharedterm {label}", f"n_{gen}"))
            _add_claim(conn, f"c_{gen}", f"r_{gen}", "work_a", f"sharedterm {label}", note_id=f"n_{gen}")
            conn.execute("INSERT INTO claim_fts(claim_id,normalized_label,claim_text) VALUES (?,?,?)",
                         (f"c_{gen}", label, f"sharedterm {label}"))
            conn.execute("INSERT INTO note_fts(note_id,note_text) VALUES (?,?)",
                         (f"n_{gen}", f"sharedterm {label}"))
        conn.commit()
        spec = make_spec("sharedterm")
        assert [x.claim_id for x in retrieve.claims(conn, spec, 1)] == ["c_new"]
        assert [x.note_id for x in retrieve.notes(conn, spec, 1)] == ["n_new"]
        assert retrieve.retrieve(conn, h.slug, make_spec("obsoleteword"), 40) == []
        retained = retrieve.spans(conn, make_spec("across groups"), 10)
        assert [x.span_id for x in retained] == ["s_a"]
        assert retained[0].claim_id is None
        assert conn.execute("SELECT note_id FROM structured_notes WHERE note_id='n_old'").fetchone()
        conn.execute("INSERT INTO concepts(concept_id,normalized_label,canonical_label,concept_type,created_at,updated_at) "
                     "VALUES ('concept::shared','shared','shared','method','2028','2028')")
        for claim_id in ("c_old", "c_new"):
            conn.execute("INSERT INTO claim_concepts(claim_id,concept_id,work_id,created_at) "
                         "VALUES (?,'concept::shared','work_a','2028')", (claim_id,))
        assert [x.claim_id for x in retrieve.concept_lookup(conn, make_spec("shared", concept_tokens=["shared"]))] == ["c_new"]
        assert [x["claim_id"] for x in concept_detail(conn, "concept::shared")["claims"]] == ["c_new"]
        assert [x["claim_id"] for x in concept_provenance(conn, "concept::shared")] == ["c_new"]
        assert concept_provenance_counts(conn, ["concept::shared"])["concept::shared"]["claims"] == 1
    env, _ = answer("across groups", h, no_llm=True)
    assert env.citations and env.citations[0].epistemic_type != "llm_extracted"
