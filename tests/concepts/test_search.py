"""Concept search (stage 4 design §5.3-§5.4): keyword hits aggregated to senses, meaning
search over sense vectors, and the silent keyword fallback. Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.concepts import refresh as rf
from seedgraph.concepts import search as cs
from seedgraph.concepts import vectors as vx
from seedgraph.concepts.model import SenseKey
from seedgraph.library import store

from ._embed import DownEmbedder, FakeEmbedder, index_fts
from ._world import World, a, m


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def _homonyms(w, lib):
    """'identification' used two ways, plus a same-sense partner for the causal reading."""
    i1 = w.note("sp_1", [a("identification", "The causal effect is identified by a policy shock to treatment.")])
    i2 = w.note("sp_2", [a("identification", "The system of equations has a unique solution under a full gadget rank.")])
    w.note("sp_3", [a("quasi experiment", "A policy shock to treatment reveals the causal effect.")])
    w.note("sp_4", [a("full gadget rank", "The matrix of instruments has full column rank so the system solves.")])
    w.split("assumption", "identification", {"s1": "causal", "s2": "unique solution"})
    w.assign("assumption", "identification", i1["identification"], "s1")
    w.assign("assumption", "identification", i2["identification"], "s2")
    index_fts(lib)
    rf.refresh(lib)


# --- keyword ------------------------------------------------------------------------------------

def test_keyword_hits_aggregate_to_senses_by_best_score(lib, w):
    w.note("sp_1", [a("gadget alignment", "Untreated outcome paths move in parallel."),
                    m("event study", "Leads and lags of treatment trace outcome paths.")])
    w.note("sp_2", [a("gadget alignment", "Absent treatment, both groups share outcome trends.")])
    w.note("sp_3", [a("no foresight", "Units do not react before treatment.", quote=False)])  # inferred
    index_fts(lib)
    rf.refresh(lib)
    hits = cs.find(lib, "parallel outcome trends", mode="keyword")
    assert [h.handle for h in hits] == ["assumption/gadget alignment", "method/event study"]
    top = hits[0]
    assert top.mode == "keyword" and top.claim_count == 2 and top.paper_count == 2
    assert top.score >= hits[1].score
    assert top.best_claim_text in {"Untreated outcome paths move in parallel.",
                                   "Absent treatment, both groups share outcome trends."}
    # unsupported (agent-inferred) claims are never hits; the family filter narrows
    assert cs.find(lib, "units react", mode="keyword") == []
    assert [h.handle for h in cs.find(lib, "outcome paths", mode="keyword", family="method")] == \
        ["method/event study"]


def test_keyword_query_syntax_is_inert(lib, w):
    w.note("sp_1", [m("ar(1) model", "An AR(1) model with a unit-root test.")])
    index_fts(lib)
    rf.refresh(lib)
    assert [h.handle for h in cs.find(lib, 'AR(1) "unit-root" OR -', mode="keyword")] == ["method/ar 1 model"]
    assert cs.find(lib, "  ", mode="keyword") == []


def test_a_sense_query_uses_its_label_and_distinctive_terms_and_skips_itself(lib, w):
    _homonyms(w, lib)
    hits = cs.find(lib, SenseKey("assumption", "identification", "s1"), mode="keyword")
    handles = [h.handle for h in hits]
    assert "assumption/identification~s1" not in handles
    # the claims' distinctive words (policy, shock, causal ...) find the partner; nothing
    # in the query matches the rank-condition claim
    assert "assumption/quasi experiment" in handles
    assert "assumption/full gadget rank" not in handles


# --- meaning ------------------------------------------------------------------------------------

def test_meaning_search_separates_homonym_senses(lib, w):
    _homonyms(w, lib)
    emb = FakeEmbedder()
    vx.embed_library(lib, emb)
    hits = cs.find(lib, SenseKey("assumption", "identification", "s1"), mode="meaning", embedder=emb)
    handles = [h.handle for h in hits]
    assert hits[0].mode == "meaning"
    assert handles[0] == "assumption/quasi experiment"
    assert handles.index("assumption/quasi experiment") < handles.index("assumption/identification~s2")
    text_hits = cs.find(lib, "unique solution of a system", mode="meaning", embedder=emb)
    assert text_hits[0].handle in {"assumption/identification~s2", "assumption/full gadget rank"}
    # the free-text query is embedded with the query prefix
    assert emb.embedded[-1] == "search_query: unique solution of a system"


# --- fallback -----------------------------------------------------------------------------------

def test_auto_is_keyword_until_embeddings_are_adopted(lib, w):
    _homonyms(w, lib)
    emb = FakeEmbedder()
    vx.embed_library(lib, emb)
    assert cs.adoption(lib.conn).adopted is False
    assert {h.mode for h in cs.find(lib, "policy shock", embedder=emb)} == {"keyword"}
    cs.set_adoption(lib, adopted=True, model=emb.model, report="invented/report.md")
    assert cs.adoption(lib.conn).model == emb.model
    assert {h.mode for h in cs.find(lib, "policy shock", embedder=emb)} == {"meaning"}


def test_ollama_down_falls_back_to_keyword_with_the_marker(lib, w):
    _homonyms(w, lib)
    vx.embed_library(lib, FakeEmbedder())
    cs.set_adoption(lib, adopted=True, model="fake-embed")
    for mode in ("auto", "meaning"):
        hits = cs.find(lib, "policy shock", mode=mode, embedder=DownEmbedder())
        assert hits and {h.mode for h in hits} == {"keyword"}
        assert hits[0].fallback == "unreachable"


def test_no_vectors_yet_falls_back_to_keyword(lib, w):
    _homonyms(w, lib)
    hits = cs.find(lib, "policy shock", mode="meaning", embedder=FakeEmbedder())
    assert {h.mode for h in hits} == {"keyword"} and hits[0].fallback == "no_vectors"
