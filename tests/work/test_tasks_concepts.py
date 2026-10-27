"""The concept agent tasks on the one protocol (stage 4 design §3, §4; slice C5):
``concepts:merge``, ``concepts:bridge`` and ``concepts:name_literatures`` create, claim,
check and save through :mod:`seedgraph.work.routine`. Invented text only."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from concepts._world import World, a  # tests/ is on sys.path (rootdir conftest)

from seedgraph.work import routine, tasks
from seedgraph.work.claims import WorkerRef
from seedgraph.work.store import WorkError

from .conftest import T0, at

W1 = WorkerRef("interactive", "w1", client="claude")


@pytest.fixture
def w(lib):
    return World(lib)


def lexical_world(w: World) -> None:
    w.note("lp_p1", [a("gadget overlap", "Invented: every gadget has a positive chance of the widget."),
                     a("sprocket balance", "Invented: sprocket weights equalise the widget groups.")])
    w.note("lp_p2", [a("gadget overlap rule", "Invented: each gadget could have received the widget."),
                     a("sprocket balance rule", "Invented: sprocket balance holds after weighting.")])


def claim(lib, batch_id: str, n: int = 1):
    reply = routine.claim_next(lib, batch_id, W1, now=at(n))
    assert reply.status == "claimed", reply
    return reply.message


def package_text(msg) -> str:
    return Path(msg["text_path"]).read_bytes().decode("utf-8")


def items_of(text: str) -> dict[int, list[str]]:
    """``{item number: [refs]}`` from a package text."""
    out: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        hit = re.match(r"^## (\d+)\.", line)
        if hit:
            current = int(hit.group(1))
            out[current] = []
        elif current is not None:
            out[current] += re.findall(r"\b(c\d+):", line)
    return out


def test_concept_tasks_are_registered_without_importing_the_module(lib):
    tt = tasks.get_task(lib, "concepts:bridge")
    assert tt.name == "concepts:bridge" and tt.subject_kind == "concept_package"
    assert tasks.get_task(lib, "concepts:merge").name == "concepts:merge"
    with pytest.raises(WorkError, match="concept task"):
        tasks.get_task(lib, "concepts:tidy")


def test_merge_batch_packages_lexical_groups_with_claim_refs(lib, w):
    lexical_world(w)
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    assert [i.subject_kind for i in view.items] == ["concept_package"]
    assert view.items[0].subject_label == "assumption groups 1-2"
    msg = claim(lib, view.batch_id)
    text = package_text(msg)
    found = items_of(text)
    assert len(found) == 2 and all(len(refs) == 2 for refs in found.values())
    assert "gadget overlap rule" in text and "Invented: each gadget could have received the widget." in text
    assert "An invented paper lp_p2" in text
    instructions = Path(msg["instructions_path"]).read_text(encoding="utf-8")
    assert '"verdict"' in instructions and "never write quotes" in instructions.lower()
    assert "claim_fingerprint" not in text  # the frozen manifest stays out of the agent's text


def answer_all(text: str, **fields) -> dict:
    return {"items": [{"item": n, "confidence": "high", **fields} for n in items_of(text)]}


def test_merge_verdicts_save_decisions_and_refresh(lib, w):
    lexical_world(w)
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    msg = claim(lib, view.batch_id)
    text = package_text(msg)
    refs = items_of(text)
    first, second = sorted(refs)
    payload = {"items": [
        {"item": first, "verdict": "merge", "confidence": "high", "name": "gadget overlap"},
        {"item": second, "verdict": "link", "confidence": "high", "note": "Invented: the rule adds weighting.",
         "quote_a": refs[second][0], "quote_b": refs[second][1]},
    ]}
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=payload, model="m1", now=at(2))
    assert res.status == "accepted" and res.paper_status == "done", res.errors
    assert res.quotes_found == 0  # concept tasks cite refs: no anchor preview
    rows = {r["kind"]: r for r in lib.conn.execute(
        "SELECT * FROM concept_decisions WHERE decided_by = 'agent'").fetchall()}
    assert rows["merge"]["state"] == "applied" and rows["merge"]["name"] == "gadget overlap"
    assert rows["merge"]["batch_id"] == view.batch_id and rows["merge"]["model"] == "m1"
    link = rows["link"]
    assert link["note"] == "Invented: the rule adds weighting." and link["state"] == "applied"
    qa, qb = json.loads(link["quote_a_json"]), json.loads(link["quote_b_json"])
    assert {qa["paper_id"], qb["paper_id"]} == {"lp_p1", "lp_p2"} and qa["claim_fingerprint"]
    assert w.concept_of("assumption", "gadget overlap") == w.concept_of("assumption", "gadget overlap rule")
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0] == 1


@pytest.mark.parametrize("bad, message", [
    ({"verdict": "merge", "name": "sprocket"}, "name must be one of the item's labels"),
    ({"verdict": "link", "note": "Invented."}, "a link needs quote_a and quote_b"),
    ({"verdict": "same_sense"}, "not allowed for a pair"),
    ({"verdict": "split", "senses": []}, "split on a pair needs both sides to share one label"),
    ({"verdict": "link", "note": "x" * 201, "quote_a": "c1", "quote_b": "c2"}, "note is longer than 200"),
    ({"verdict": "maybe"}, "unknown verdict"),
])
def test_each_verdict_validates(lib, w, bad, message):
    lexical_world(w)
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    msg = claim(lib, view.batch_id)
    first, second = sorted(items_of(package_text(msg)))
    payload = {"items": [{"item": first, "confidence": "high", **bad},
                         {"item": second, "verdict": "unrelated", "confidence": "high"}]}
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=payload, now=at(2))
    assert res.status == "invalid"
    assert any(f"item {first}" in e and message in e for e in res.errors), res.errors


def test_every_item_answered_exactly_once_and_refs_belong_to_their_side(lib, w):
    lexical_world(w)
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    msg = claim(lib, view.batch_id)
    refs = items_of(package_text(msg))
    first, second = sorted(refs)
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": first, "verdict": "unrelated", "confidence": "high"},
        {"item": first, "verdict": "unrelated", "confidence": "high"}]}, now=at(2))
    assert res.status == "invalid"
    assert any("answered twice" in e for e in res.errors) and any(f"item {second}: not answered" in e
                                                                   for e in res.errors)
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": first, "verdict": "link", "confidence": "high", "note": "Invented.",
         "quote_a": refs[second][0], "quote_b": refs[first][1]},
        {"item": second, "verdict": "unrelated", "confidence": "high"}]}, now=at(3))
    assert any("quote_a" in e and "not a ref of side A" in e for e in res.errors), res.errors


def test_a_vetoed_merge_gets_feedback_naming_the_trigger_and_the_allowed_answers(lib, w):
    w.note("lp_p1", [a("sprocket shift v1", "Invented: sprocket shift of the first kind.")])
    w.note("lp_p2", [a("sprocket shift v2", "Invented: sprocket shift of the second kind.")])
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    msg = claim(lib, view.batch_id)
    (item,) = items_of(package_text(msg))
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "merge", "confidence": "high", "name": "sprocket shift v1"}]}, now=at(2))
    assert res.status == "invalid"
    assert res.errors == [f"item {item}: merge vetoed: the labels differ in a version word: 'sprocket shift v1' "
                          "has 'v1', 'sprocket shift v2' has 'v2'; answer link, unrelated or unsure"]
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "unrelated", "confidence": "high"}]}, now=at(3))
    assert res.status == "accepted"


def test_merge_or_link_on_an_ambiguous_label_must_compare_claims_from_both_sides(lib, w):
    w.note("lp_p1", [a("exogeneity", "Invented: the sprocket shock is unrelated to the widget.")])
    w.note("lp_p2", [a("exogeneity assumption", "Invented: widget draws ignore sprocket shocks.")])
    w.note("lp_p3", [a("exogeneity", "Invented: no sprocket feedback reaches the widget.")])
    with lib.write() as conn:  # keep the two labels apart (no rule fold) so they are candidates
        conn.execute("INSERT INTO ambiguous_labels (pattern, scope, match, effect, decided_by, created_at) "
                     "VALUES ('exogeneity assumption', '', 'label', 'ambiguous', 'researcher', '2026-01-01')")
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    msg = claim(lib, view.batch_id)
    (item,) = items_of(package_text(msg))
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "merge", "confidence": "high", "name": "exogeneity"}]}, now=at(2))
    assert res.status == "invalid"
    assert any("ambiguous label" in e and "quote_a and quote_b" in e for e in res.errors), res.errors


def test_the_select_hook_refuses_paper_arguments(lib, w):
    lexical_world(w)
    with pytest.raises(WorkError, match="chooses its own subjects"):
        routine.create_batch(lib, "concepts:merge", query="held", now=T0)


def test_a_library_with_no_candidates_selects_nothing(lib, w):
    w.note("lp_p1", [a("gadget overlap")])
    with pytest.raises(WorkError, match="selected no subjects"):
        routine.create_batch(lib, "concepts:merge", now=T0)


def test_bridge_batch_packages_sense_checks_and_pairs(lib, w):
    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t), a("sprocket independence", "Invented: cog lever flange pivots freely.")])
    w.note("lp_p2", [a("exogeneity", t + " Again."), a("sprocket exclusion", "Invented: cog lever flange stays.")])
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption"}, now=T0)
    labels = [i.subject_label for i in view.items]
    assert labels[0] == "assumption sense checks 1-1" and labels[1].startswith("assumption pairs 1-")
    msg = claim(lib, view.batch_id)
    text = package_text(msg)
    (item,) = items_of(text)
    refs = items_of(text)[item]
    assert len(refs) == 2  # both papers' claims are listed for the sense check
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "split", "confidence": "high",
         "senses": [{"gloss": "Invented: shock independence", "claims": [refs[0]]},
                    {"gloss": "Invented: draw independence", "claims": [refs[1]]}]}]}, now=at(2))
    assert res.status == "accepted", res.errors
    senses = lib.conn.execute("SELECT sense_id, gloss, decided_by, state FROM concept_senses "
                              "WHERE label_key = 'exogeneity' ORDER BY sense_id").fetchall()
    assert [tuple(r) for r in senses] == [("s1", "Invented: shock independence", "agent", "confirmed"),
                                         ("s2", "Invented: draw independence", "agent", "confirmed")]
    assert w.concept_of("assumption", "exogeneity", "s1") != w.concept_of("assumption", "exogeneity", "s2")
    assert lib.conn.execute("SELECT COUNT(*) FROM claim_concepts WHERE label_key = 'exogeneity' "
                            "AND sense_id IS NULL").fetchone()[0] == 0


def test_a_split_must_cover_every_listed_claim_once(lib, w):
    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t)])
    w.note("lp_p2", [a("exogeneity", t + " Again.")])
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption"}, now=T0)
    msg = claim(lib, view.batch_id)
    text = package_text(msg)
    (item,) = items_of(text)
    refs = items_of(text)[item]
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "split", "confidence": "high",
         "senses": [{"gloss": "one", "claims": [refs[0]]}, {"gloss": "two", "claims": [refs[0]]}]}]}, now=at(2))
    assert res.status == "invalid"
    assert any("cited twice" in e for e in res.errors) and any(f"{refs[1]} is not assigned" in e for e in res.errors)


def test_same_sense_confirms_the_label(lib, w):
    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t)])
    w.note("lp_p2", [a("exogeneity", t + " Again.")])
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption"}, now=T0)
    msg = claim(lib, view.batch_id)
    (item,) = items_of(package_text(msg))
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "same_sense", "confidence": "high"}]}, now=at(2))
    assert res.status == "accepted"
    cid = w.concept_of("assumption", "exogeneity")
    assert w.concept(cid)["needs_sense_check"] == 0


def test_bridge_decisions_are_never_judged_twice(lib, w):
    w.note("lp_p1", [a("sprocket independence", "Invented: cog lever flange pivots freely.")])
    w.note("lp_p2", [a("sprocket exclusion", "Invented: cog lever flange stays put.")])
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption"}, now=T0)
    msg = claim(lib, view.batch_id)
    (item,) = items_of(package_text(msg))
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": item, "verdict": "unrelated", "confidence": "high"}]}, now=at(2))
    assert res.status == "accepted"
    with pytest.raises(WorkError, match="selected no subjects"):
        routine.create_batch(lib, "concepts:bridge", options={"families": "assumption"}, now=at(3))


def test_unknown_task_options_are_refused(lib, w):
    lexical_world(w)
    with pytest.raises(WorkError, match="unknown batch option"):
        routine.create_batch(lib, "concepts:merge", options={"k": 3}, now=T0)
    with pytest.raises(WorkError, match="family"):
        routine.create_batch(lib, "concepts:bridge", options={"families": "widgets"}, now=T0)
