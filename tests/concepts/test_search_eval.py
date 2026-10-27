"""The embeddings adoption test (stage 4 design §5.5): case sets resolved from the
library, both modes ranked on the same state, metrics and the pass rule. Invented."""

from __future__ import annotations

import json

import pytest

from seedgraph.concepts import refresh as rf
from seedgraph.concepts import search_eval as ev
from seedgraph.library import store

from ._embed import FakeEmbedder, index_fts
from ._world import World, a


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


# --- metrics and the pass rule (pure) -------------------------------------------------------------

def _mode(ranks, probes):
    return ev.mode_metrics({f"P{i}": r for i, r in enumerate(ranks, 1)},
                           {f"S{i}": p for i, p in enumerate(probes, 1)})


SEPARATED = ev.ProbeOutcome(partner_rank=1, other_rank=5)
MIXED = ev.ProbeOutcome(partner_rank=7, other_rank=2)
OVERMERGED = ev.ProbeOutcome(partner_rank=2, other_rank=1)


def test_metrics_recall_mrr_and_separation():
    kw = _mode([1, 4, None, 12, None], [OVERMERGED, MIXED, SEPARATED, ev.ProbeOutcome(None, None)])
    assert kw.recall_at_10 == pytest.approx(2 / 5)
    assert kw.mrr == pytest.approx((1 + 1 / 4 + 0 + 1 / 12 + 0) / 5)
    assert kw.separation == pytest.approx(1 / 4)  # neither found: not separated
    assert kw.over_merged == ["S1"]
    assert ev.ProbeOutcome(partner_rank=3, other_rank=None).separated


def test_pass_rule_needs_all_three():
    kw = _mode([1, None, None, None, 20, 2, 3, 4, 5, 6], [MIXED] * 5)
    good = _mode([1, 2, 3, 4, None, 2, 3, 4, 5, 6], [SEPARATED] * 5)
    v = ev.pass_rule(kw, good)
    assert v.passed and v.new_partners == ["P2", "P3", "P4"] and v.needed_new == 3
    # 1. separation below 0.8
    v = ev.pass_rule(kw, _mode([1, 2, 3, 4, None, 2, 3, 4, 5, 6], [SEPARATED] * 3 + [MIXED] * 2))
    assert not v.passed and v.checks["separation"] is False
    # 2a. recall below keyword's
    v = ev.pass_rule(kw, _mode([1, 2, 3, 4, None, None, None, None, None, 6], [SEPARATED] * 5))
    assert not v.passed and v.checks["recall"] is False
    # 2b. only two partners keyword misses
    v = ev.pass_rule(kw, _mode([1, 2, 3, None, 30, 2, 3, 4, 5, 6], [SEPARATED] * 5))
    assert not v.passed and v.checks["new_partners"] is False
    # 3. an over-merge, even with separation at 0.8
    v = ev.pass_rule(kw, _mode([1, 2, 3, 4, None, 2, 3, 4, 5, 6], [SEPARATED] * 4 + [OVERMERGED]))
    assert not v.passed and v.checks["no_over_merge"] is False and v.checks["separation"] is True


def test_needed_new_partners_is_a_quarter_of_many_cases():
    many = _mode([None] * 20, [SEPARATED])
    assert ev.pass_rule(many, many).needed_new == 5


# --- the case sets ------------------------------------------------------------------------------

CASES = {
    "partners": [
        {"id": "P1", "a": "quasi experiment", "b": "as good as random", "kind": "synonym"},
        {"id": "P2", "a": "widget coverage", "b": "shared widget range", "kind": "synonym", "also": ["widget floor"]},
        {"id": "P3", "a": "no such label", "b": "widget coverage", "kind": "synonym"},
    ],
    "sense_probes": [
        {"id": "S1", "label": "identification", "sense_a": "policy shock source",
         "sense_b": "unique solution", "partner_of_a": "quasi experiment"},
        {"id": "S2", "label": "sprocket", "sense_a": "sprocket bias", "sense_b": "sprocket knob choice",
         "partner_of_b": "knob setting"},
    ],
}


def _library(w, lib):
    w.note("ev_1", [a("identification", "The causal effect is identified by a policy shock to treatment."),
                    a("quasi experiment", "A policy shock to treatment reveals the causal effect.")])
    w.note("ev_2", [a("identification", "The system of equations has a unique solution under a full gadget rank."),
                    a("full gadget rank", "The instrument matrix has full column rank so the system solves.")])
    w.note("ev_3", [a("as good as random", "A policy shock to treatment reveals the causal effect.")])
    w.note("ev_4", [a("widget coverage", "Every unit has a positive probability of each treatment."),
                    a("widget floor", "Each propensity score is bounded away from zero and one.")])
    w.note("ev_5", [{"type": "limitation", "label": "sprocket bias", "text": "Units self-select into treatment."},
                    {"type": "limitation", "label": "sprocket knob choice",
                     "text": "Results depend on the chosen smoothing bandwidth."},
                    {"type": "limitation", "label": "knob setting",
                     "text": "The smoothing bandwidth is chosen by cross validation."}])
    index_fts(lib)
    rf.refresh(lib)


def test_protocol_resolves_sets_by_the_ladder_and_reports_malformed_cases(lib, w):
    _library(w, lib)
    ident = {r[0]: r[1] for r in lib.conn.execute(
        "SELECT claim_id, claim_text FROM current_claims WHERE normalized_label = 'identification'")}
    causal = next(c for c, t in ident.items() if "policy" in t)
    system = next(c for c, t in ident.items() if "system" in t)
    pools = ev.gloss_pools(lib.conn, CASES)
    assert {c.claim_id for c in pools["S1"]} == {causal, system}  # head pool; S2 needs none
    assignments = {"S1": {causal: "a", system: "b"}}
    p = ev.build_protocol(lib.conn, CASES, assignments=assignments)
    assert [c.id for c in p.partners] == ["P1", "P2"]
    assert p.malformed == [{"id": "P3", "reason": "label 'no such label' has no supported claims"}]
    p2 = next(c for c in p.partners if c.id == "P2")
    assert p2.targets == ["shared widget range", "widget floor"] and p2.missing_targets == ["shared widget range"]
    s1 = next(x for x in p.probes if x.id == "S1")
    assert s1.query.claim_ids == (causal,) and s1.other.claim_ids == (system,)
    assert s1.query.how.startswith("gloss assignment") and s1.partner == "quasi experiment"
    s2 = next(x for x in p.probes if x.id == "S2")
    # partner_of_b: the query set is sense b
    assert s2.query.labels == ("sprocket knob choice",) and s2.other.labels == ("sprocket bias",)
    assert s2.query.how == "label 'sprocket knob choice'"
    # without an assignment the head-pool probe cannot be built
    p_none = ev.build_protocol(lib.conn, CASES)
    assert {"id": "S1", "reason": "sense_a 'policy shock source': no gloss assignment for the head pool"} \
        in p_none.malformed
    # a protocol round-trips through JSON (it is written before any run)
    assert ev.Protocol.from_json(json.loads(json.dumps(p.to_json()))) == p


def test_evaluate_ranks_both_modes_and_renders_the_verdict(lib, w):
    _library(w, lib)
    ident = {r[0]: r[1] for r in lib.conn.execute(
        "SELECT claim_id, claim_text FROM current_claims WHERE normalized_label = 'identification'")}
    assignments = {"S1": {c: ("a" if "policy" in t else "b") for c, t in ident.items()}}
    p = ev.build_protocol(lib.conn, CASES, assignments=assignments)
    emb = FakeEmbedder()
    result = ev.evaluate(lib, p, embedder=emb)
    kw, mn = result.modes["keyword"], result.modes["meaning"]
    assert kw.partner_ranks["P1"] is not None and mn.partner_ranks["P1"] == 1
    assert mn.probes["S1"].separated
    assert result.embed_stats.embedded > 0
    md = ev.render_markdown(result)
    assert "| P1 |" in md and "| S1 |" in md and "Verdict" in md
    assert result.verdict.passed is False  # two cases cannot find three new partners
    # keyword only, when no embedder: no verdict
    assert ev.evaluate(lib, p, embedder=None).verdict is None
