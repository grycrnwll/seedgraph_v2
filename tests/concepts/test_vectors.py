"""Claim-context vectors (stage 4 design §5.2-§5.3): the embed text, the cache keyed by
text hash + model, sense vectors as the normalised mean of their claims. Invented text."""

from __future__ import annotations

import pytest

from seedgraph.concepts import refresh as rf
from seedgraph.concepts import vectors as vx
from seedgraph.concepts.model import SenseKey
from seedgraph.library import store

from ._embed import FakeEmbedder
from ._world import World, a, m


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def test_claim_context_carries_family_label_text_plain_words_and_a_distinct_quote():
    text = vx.claim_context("assumption", "gadget alignment", "Untreated paths move together.",
                            plain_words="Both groups would have changed alike.",
                            quote="We assume the groups share a common trend before treatment.")
    assert text == ("assumption: gadget alignment. Untreated paths move together. "
                    "Both groups would have changed alike. "
                    "Quote: We assume the groups share a common trend before treatment.")
    # a quote that repeats the claim text adds nothing
    assert vx.claim_context("method", "widget", "A widget is fit.", quote="a widget is fit") == \
        "method: widget. A widget is fit."


def test_vectors_are_cached_by_text_hash_and_model(lib):
    emb = FakeEmbedder()
    vecs, stats = vx.vectors_for(lib, ["alpha beta", "gamma delta"], emb)
    assert (stats.embedded, stats.cached) == (2, 0)
    assert emb.embedded == ["search_document: alpha beta", "search_document: gamma delta"]
    again, stats = vx.vectors_for(lib, ["gamma delta", "alpha beta", "epsilon"], emb)
    assert (stats.embedded, stats.cached) == (1, 2)
    assert emb.embedded[-1] == "search_document: epsilon"
    assert again["alpha beta"] == pytest.approx(vecs["alpha beta"], abs=1e-6)
    # another model, or the query prefix, is another key
    _, stats = vx.vectors_for(lib, ["alpha beta"], FakeEmbedder(model="other-embed"))
    assert stats.embedded == 1
    _, stats = vx.vectors_for(lib, ["alpha beta"], emb, prefix=vx.QUERY_PREFIX)
    assert stats.embedded == 1
    rows = lib.conn.execute("SELECT model, dims, length(vector) FROM concept_vectors ORDER BY model").fetchall()
    assert len(rows) == 5 and all(r[1] == 64 and r[2] == 64 * 4 for r in rows)


def test_a_changed_claim_text_reembeds_only_that_claim(lib, w):
    w.note("vp_1", [a("gadget alignment", "Untreated paths move together."), m("widget weighting")])
    rf.refresh(lib)
    emb = FakeEmbedder()
    first = vx.embed_library(lib, emb)
    assert (first.embedded, first.cached) == (2, 0)
    w.note("vp_1", [a("gadget alignment", "Untreated paths move together."),
                    m("widget weighting", "Widgets are reweighted by size.")])
    rf.refresh(lib)
    second = vx.embed_library(lib, emb)
    assert (second.embedded, second.cached) == (1, 1)


def test_sense_vectors_separate_the_senses_of_a_split_label(lib, w):
    ids1 = w.note("vp_1", [a("identification", "The causal effect is identified by the policy shock.")])
    ids2 = w.note("vp_2", [a("identification", "The system of equations has a unique solution by rank.")])
    w.note("vp_3", [a("quasi experiment", "A policy shock identifies the causal effect.")])
    w.split("assumption", "identification", {"s1": "causal", "s2": "unique solution"})
    w.assign("assumption", "identification", ids1["identification"], "s1")
    w.assign("assumption", "identification", ids2["identification"], "s2")
    rf.refresh(lib)
    emb = FakeEmbedder()
    contexts = vx.claim_contexts(lib.conn)
    vecs, _ = vx.vectors_for(lib, [c.text for c in contexts], emb)
    senses = vx.sense_vectors(contexts, vecs)
    s1, s2 = SenseKey("assumption", "identification", "s1"), SenseKey("assumption", "identification", "s2")
    partner = SenseKey("assumption", "quasi experiment")
    assert set(senses) == {s1, s2, partner}
    assert vx.dot(senses[s1], senses[s1]) == pytest.approx(1.0)
    assert vx.dot(senses[s1], senses[partner]) > vx.dot(senses[s1], senses[s2])


def test_float32_round_trip():
    v = [0.5, -1.25, 3.0]
    assert vx.unpack(vx.pack(v)) == v
