"""Cleanup of a stored bridge batch (stage 4 design §11, slice C11): the read-only
findings (conflicting verdicts between two concepts, link / unrelated rows inside one
concept, rows on a sense that no longer exists, merges contradicted by an unrelated row)
and the cleanup run that queues, undoes or re-judges them. Invented labels only."""

from __future__ import annotations

import pytest

from seedgraph.concepts import cleanup as cu
from seedgraph.concepts import refresh as rf
from seedgraph.concepts.model import decision_id
from seedgraph.library import store

from ._world import World, a

BATCH = "b_invented"
AT = "2026-01-02T00:00:00+00:00"


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def agent(lib, kind: str, la: str, lb: str, *, state: str = "applied", reason: str | None = None,
          conf: str = "medium", family: str = "assumption", sense_a: str | None = None,
          batch: str = BATCH) -> str:
    """An agent row of the invented batch (labels are ordered as the schema wants)."""
    if la > lb:
        la, lb = lb, la
    did = decision_id(kind, la, lb, family=family, sense_a=sense_a)
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO concept_decisions (decision_id, kind, family, label_a, sense_a, label_b, name, decided_by, "
            "state, confidence, veto_reason, batch_id, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'agent', ?, ?, ?, ?, ?)",
            (did, kind, family, la, sense_a, lb, la if kind == "merge" else None, state, conf, reason, batch, AT))
    return did


def state(lib, did: str) -> tuple:
    return tuple(lib.conn.execute("SELECT state, veto_reason FROM concept_decisions WHERE decision_id = ?",
                                  (did,)).fetchone())


LABELS = ("red widget", "blue widget", "gold gadget", "silver gadget", "plain sprocket", "fancy sprocket",
          "hub cog", "left cog", "right cog", "top cog")


@pytest.fixture
def batch(lib, w):
    """Two researcher-merged concepts {red, blue widget} and {plain, fancy sprocket}, a
    researcher-merged {hub, top cog}, agent merges joining left and right cog onto the hub,
    and the batch's verdicts:

    - red~gold link + blue~gold unrelated, both applied (a conflict, both applied);
    - plain~silver link queued + fancy~silver unrelated applied (a conflict);
    - red~blue link (inside one concept);
    - left~right unrelated inside the cog concept (contradicts the two agent merges);
    - left~top link: inside now, but a real link once the cog merges are queued.
    """
    for i, label in enumerate(LABELS):
        w.note(f"cu_p{i}", [a(label, f"Invented: the {label} condition holds.")])
    w.decide("merge", "red widget", "blue widget", family="assumption")
    w.decide("merge", "plain sprocket", "fancy sprocket", family="assumption")
    w.decide("merge", "hub cog", "top cog", family="assumption")
    ids = {
        "red_gold": agent(lib, "link", "red widget", "gold gadget"),
        "blue_gold": agent(lib, "unrelated", "blue widget", "gold gadget", conf="high"),
        "plain_silver": agent(lib, "link", "plain sprocket", "silver gadget", state="queued",
                              reason="low_confidence", conf="low"),
        "fancy_silver": agent(lib, "unrelated", "fancy sprocket", "silver gadget"),
        "red_blue": agent(lib, "link", "red widget", "blue widget"),
        "hub_left": agent(lib, "merge", "hub cog", "left cog"),
        "hub_right": agent(lib, "merge", "hub cog", "right cog"),
        "left_right": agent(lib, "unrelated", "left cog", "right cog"),
        "left_top": agent(lib, "link", "left cog", "top cog"),
        "single": agent(lib, "link", "gold gadget", "silver gadget"),
    }
    rf.refresh(lib)
    return ids


def test_findings_list_conflicting_concept_pairs_inside_rows_and_contradicted_merges(lib, batch):
    f = cu.findings(lib.conn, BATCH)
    pairs = sorted(sorted(c.decision_ids) for c in f.conflicts)
    assert pairs == sorted([sorted([batch["red_gold"], batch["blue_gold"]]),
                            sorted([batch["plain_silver"], batch["fancy_silver"]])])
    assert [c.both_applied for c in f.conflicts] == [True, False]  # both-applied first
    assert sorted(r["decision_id"] for r in f.inside) == sorted(
        [batch["red_blue"], batch["left_right"], batch["left_top"]])
    assert [(c.unrelated_id, sorted(c.merge_ids)) for c in f.contradicted] == [
        (batch["left_right"], sorted([batch["hub_left"], batch["hub_right"]]))]
    assert f.dead == []


def test_dry_run_plans_every_change_and_writes_nothing(lib, batch):
    before = [tuple(r) for r in lib.conn.execute("SELECT * FROM concept_decisions ORDER BY decision_id")]
    res = cu.run(lib, BATCH, apply=False)
    assert not res.applied
    assert {ch.decision_id: ch.rule for ch in res.changes} == {
        batch["hub_left"]: "contradicted_merge", batch["hub_right"]: "contradicted_merge",
        batch["red_blue"]: "inside_concept",
        batch["red_gold"]: "disagreement", batch["blue_gold"]: "disagreement",
        batch["plain_silver"]: "disagreement", batch["fancy_silver"]: "disagreement",
        batch["left_top"]: "disagreement",
    }
    assert [tuple(r) for r in lib.conn.execute("SELECT * FROM concept_decisions ORDER BY decision_id")] == before


def test_apply_queues_conflicts_and_contradicted_merges_and_undoes_inside_links(lib, w, batch):
    res = cu.run(lib, BATCH, apply=True, now=AT)
    assert res.applied
    for key in ("red_gold", "blue_gold", "plain_silver", "fancy_silver", "hub_left", "hub_right"):
        assert state(lib, batch[key]) == ("queued", "disagreement")
    assert state(lib, batch["red_blue"]) == ("undone", "cleanup:inside_concept")
    # the cog merges are queued, so left and right cog are apart; left~top is no longer
    # inside one concept (not undone) but disagrees with the queued hub~left merge
    assert w.concept_of("assumption", "left cog") != w.concept_of("assumption", "right cog")
    assert state(lib, batch["left_top"]) == ("queued", "disagreement")
    assert state(lib, batch["single"]) == ("applied", None)
    old = {ch.decision_id: (ch.old_state, ch.old_reason) for ch in res.changes}
    assert old[batch["plain_silver"]] == ("queued", "low_confidence")
    after = cu.findings(lib.conn, BATCH)
    assert (after.unsettled, after.inside, after.contradicted, after.dead) == ([], [], [], [])
    assert len(after.conflicts) == 3  # still disagree, but every row waits for the researcher


def test_a_second_run_changes_nothing(lib, batch):
    cu.run(lib, BATCH, apply=True, now=AT)
    assert cu.run(lib, BATCH, apply=False).changes == []


def test_expected_counts_guard_the_apply(lib, batch):
    with pytest.raises(cu.CleanupMismatch, match="disagreement"):
        cu.run(lib, BATCH, apply=True, expect={"disagreement": 99})
    assert state(lib, batch["red_gold"]) == ("applied", None)


def test_dead_sense_rows_and_falsely_vetoed_pairs_are_set_to_rejudge(lib, w, batch):
    w.note("cu_split", [a("twin cog", "Invented: the twin cog meshes."),
                        a("twin cog", "Invented: the twin cog is a spare.", start=900)])
    w.split("assumption", "twin cog", {"s1": "meshes", "s2": "spare"})
    claims = [r[0] for r in lib.conn.execute(
        "SELECT claim_id FROM current_claims WHERE paper_id = 'cu_split' ORDER BY claim_id")]
    w.assign("assumption", "twin cog", claims[0], "s1")
    w.assign("assumption", "twin cog", claims[1], "s2")
    dead = agent(lib, "link", "gold gadget", "twin cog")
    rf.refresh(lib)
    res = cu.run(lib, BATCH, apply=True, now=AT, false_vetoes=[("silver gadget", "gold gadget")])
    rules = {ch.decision_id: ch.rule for ch in res.changes}
    assert rules[dead] == "dead_sense" and rules[batch["single"]] == "false_veto"
    assert state(lib, dead) == ("needs_rejudge", "cleanup:label_split")
    assert state(lib, batch["single"]) == ("needs_rejudge", "cleanup:stoplist_refusal")


def test_residual_stays_clean_after_the_researcher_accepts_the_queued_merges(lib, w, batch):
    from seedgraph.concepts import researcher as rs
    cu.run(lib, BATCH, apply=True, now=AT)
    rs.settle(lib, batch["hub_left"], "accept")
    rs.settle(lib, batch["hub_right"], "accept")
    assert w.concept_of("assumption", "left cog") == w.concept_of("assumption", "right cog")
    f = cu.findings(lib.conn, BATCH)
    assert (f.unsettled, f.inside, f.contradicted, f.dead) == ([], [], [], [])
    assert sorted(r["decision_id"] for r in f.overridden) == sorted([batch["left_right"], batch["left_top"]])
    assert cu.run(lib, BATCH, apply=False).changes == []


def test_record_stores_the_changes_in_library_meta(lib, batch):
    res = cu.run(lib, BATCH, apply=True, now=AT)
    key = cu.record(lib, res)
    stored = cu.records(lib.conn, BATCH)
    assert [r["key"] for r in stored] == [key]
    assert stored[0]["counts"] == res.counts()
    assert len(stored[0]["changes"]) == len(res.changes)
