"""Concept model (stage 4 design §1): families, grounding, handles. Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.concepts import families, model
from seedgraph.concepts.refresh import refresh
from seedgraph.library import attention, store
from seedgraph.notes import overlay

from ._world import World, a, m


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def claim_row(lib, cid):
    return lib.conn.execute("SELECT * FROM claim_concepts WHERE claim_id = ?", (cid,)).fetchone()


# --- families -------------------------------------------------------------------------------------

def test_families_fence_by_family_not_claim_type():
    assert families.family_of("identification_assumption") == "assumption"
    assert families.family_of("regularity_condition") == "assumption"
    assert families.family_of("model") == families.family_of("method") == "method"
    assert families.family_of("research_question") == "paper_scalar"
    assert not families.is_concept_family("paper_scalar")
    assert families.family_of("something_else") is None


# --- grounding ------------------------------------------------------------------------------------

def test_anchored_claims_support_and_inferred_claims_are_listed_without_support(lib, w):
    ids = w.note("lp_p1", [a("parallel trends"), a("no anticipation", quote=False, status="inferred")])
    refresh(lib)
    assert claim_row(lib, ids["parallel trends"])["supports"] == 1
    inferred = claim_row(lib, ids["no anticipation"])
    assert inferred["supports"] == 0  # listed, labelled by trust, supports nothing
    concept = w.concept(w.concept_of("assumption", "no anticipation"))
    assert (concept["supported"], concept["paper_count"], concept["claim_count"]) == (0, 0, 1)
    pt = w.concept(w.concept_of("assumption", "parallel trends"))
    assert (pt["supported"], pt["paper_count"], pt["quote_count"]) == (1, 1, 1)


def test_hidden_claims_do_not_count(lib, w):
    ids = w.note("lp_p1", [a("parallel trends"), a("common shocks")])
    overlay.save_edit(lib, "lp_p1", {"op": "hide", "target": {"claim_id": ids["common shocks"]}},
                      expected_version=0)
    refresh(lib)
    assert claim_row(lib, ids["common shocks"]) is None
    assert w.concept_of("assumption", "common shocks") is None


def test_researcher_assertion_supports(lib, w):
    w.note("lp_p1", [a("parallel trends")])
    overlay.save_edit(lib, "lp_p1", {"op": "add", "kind": "assertion", "claim_type": "general_assumption",
                                     "normalized_label": "bounded widgets",
                                     "claim_text": "Widgets are bounded (invented)."}, expected_version=0)
    refresh(lib)
    cid = w.concept_of("assumption", "bounded widgets")
    assert w.concept(cid)["supported"] == 1


def test_agent_visibility_drops_papers_with_hidden_notes(lib, w):
    ids = w.note("lp_p1", [a("parallel trends")])
    ids2 = w.note("lp_p2", [a("parallel trends")])
    attention.raise_item(lib, "wrong_pdf", {"paper_id": "lp_p2"}, paper_id="lp_p2", detail="invented",
                         hides_notes=True)
    refresh(lib)
    assert claim_row(lib, ids["parallel trends"])["agent_visible"] == 1
    assert claim_row(lib, ids2["parallel trends"])["agent_visible"] == 0


def test_paper_scalars_and_not_found_rows_carry_no_concept(lib, w):
    ids = w.note("lp_p1", [{"type": "research_question", "label": "do widgets grow", "text": "Do widgets grow?"},
                           a("empty", status="not_found", quote=False)])
    refresh(lib)
    assert claim_row(lib, ids["do widgets grow"]) is None
    assert claim_row(lib, ids["empty"]) is None


# --- handles --------------------------------------------------------------------------------------

def test_handle_round_trip_and_surface_forms_normalize():
    assert model.handle(model.SenseKey("assumption", "parallel trends")) == "assumption/parallel trends"
    assert model.handle(model.SenseKey("assumption", "identification", "s2")) == "assumption/identification~s2"
    assert model.parse_handle("assumption/Parallel-Trends") == ("assumption", "parallel trends", None)
    with pytest.raises(ValueError):
        model.parse_handle("no family")


def test_handles_resolve_after_merge_and_split(lib, w):
    w.note("lp_p1", [a("parallel trends"), a("common trends"), m("identification")])
    w.note("lp_p2", [m("identification", "Invented: the system has a unique solution.")])
    refresh(lib)
    before = model.resolve(lib.conn, "assumption/common trends")
    assert before.kind == "concept"

    w.decide("merge", "common trends", "parallel trends", family="assumption", name="parallel trends")
    refresh(lib)
    r1 = model.resolve(lib.conn, "assumption/common trends")
    r2 = model.resolve(lib.conn, "assumption/parallel trends")
    assert r1.kind == r2.kind == "concept" and r1.concept_id == r2.concept_id
    assert w.concept(r1.concept_id)["name"] == "parallel trends"

    w.split("method", "identification", {"s1": "source of exogenous variation (invented gloss)",
                                         "s2": "unique solution of the system (invented gloss)"})
    refresh(lib)
    chooser = model.resolve(lib.conn, "method/identification")
    assert chooser.kind == "chooser"
    assert [s.handle for s in chooser.senses] == ["method/identification~s1", "method/identification~s2"]
    assert chooser.senses[0].gloss.startswith("source of exogenous")
    s2 = model.resolve(lib.conn, "method/identification~s2")
    assert s2.kind == "concept" and s2.concept_id != chooser.senses[0].concept_id
    assert model.resolve(lib.conn, "method/never used").kind == "missing"


def test_concept_name_is_always_a_member_label(lib, w):
    w.note("lp_p1", [a("parallel trends"), a("common trends")])
    w.decide("merge", "common trends", "parallel trends", family="assumption", name="an invented umbrella")
    refresh(lib)
    cid = w.concept_of("assumption", "common trends")
    assert w.concept(cid)["name"] in {"parallel trends", "common trends"}
