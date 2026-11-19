"""The acceptance commands and report on an invented library (design §10; slice R8): the
checklists are written once and kept, the report counts records, flags, paths without answer,
connections and the spot-check, the cross-check command compares a transcript with the log, and
the spot-check figure is the same after a fix. Invented text only."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from records._world import Agent, standard
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.library import store as libstore
from seedgraph.records import acceptance as acc
from seedgraph.spotcheck import create_sample, mark

from .test_transcript import T0, Writer

runner = CliRunner()


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    lib = libstore.open_library_at(h / "library")
    lib.close()
    return h


@pytest.fixture
def world(home):
    lib = libstore.open_library_at(home / "library")
    w = standard(lib)
    yield lib, *w
    lib.close()


def run(home, *args):
    return runner.invoke(app, ["records", *args, "--root", str(home)])


def stage(home):
    return acc.stage_dir(home)


def write_questions(home, rows):
    lines = ["questions:"]
    for r in rows:
        lines += [f"  - id: {r['id']}", f"    text: \"{r.get('text', 'an invented question')}\"",
                  f"    client: {r.get('client', 'claude')}", f"    helpers: {str(r.get('helpers', False)).lower()}",
                  f"    record: \"{r.get('record', '')}\""]
    (stage(home) / acc.QUESTIONS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def answered(lib, did, name, client="claude"):
    a = Agent(lib, conn_id=f"mcp-{name[:4]}0001", name=name, client=client)
    a.search()
    a.paper_show(did)
    return a, a.answer([{"paper_id": did.paper_id}], question=f"question for {name}")


# --- the first run ------------------------------------------------------------------------------

def test_the_first_run_writes_the_template_checklists_and_an_empty_report(home):
    res = run(home, "acceptance-report")
    assert res.exit_code == 0, res.output
    d = stage(home)
    for name in (acc.QUESTIONS_FILE, acc.CHECKLIST_FILE, acc.WALKTHROUGH_FILE, acc.REPORT_FILE, acc.REPORT_JSON,
                 acc.SPOTCHECK_MD):
        assert (d / name).is_file(), name
    assert "questions: not yet" in res.output and "wrote" in res.output
    assert not (d / acc.SPOTCHECK_JSON).exists()
    template = acc.load_questions(d / acc.QUESTIONS_FILE)
    assert [q.id for q in template] == ["q1", "q2", "q3", "q4", "q5"]
    assert [q.client for q in template].count("codex") == 1 and sum(q.helpers for q in template) == 1
    report = (d / acc.REPORT_FILE).read_text(encoding="utf-8")
    for n in range(1, 7):
        assert f"## {n}." in report
    walk = (d / acc.WALKTHROUGH_FILE).read_text(encoding="utf-8")
    assert "always allow" in walk and "- [ ] filtered the Records list by graph" in walk


def test_the_checklists_are_never_overwritten_and_new_records_are_appended(home, world):
    lib, w, did, ate = world
    _, one = answered(lib, did, "first-1400")
    run(home, "acceptance-report")
    write_questions(home, [{"id": "q1", "record": one.record_id}])
    res = run(home, "acceptance-report")
    assert res.exit_code == 0 and f"added {one.record_id}" in res.output
    path = stage(home) / acc.CHECKLIST_FILE
    text = path.read_text(encoding="utf-8")
    assert text.count("- [ ]") == len(acc.CHECKLIST_ITEMS)
    ticked = text.replace("- [ ] the question is right", "- [x] the question is right")
    path.write_text(ticked, encoding="utf-8")
    (stage(home) / acc.WALKTHROUGH_FILE).write_text("my own words\n- [x] done\n", encoding="utf-8")
    _, two = answered(lib, did, "second-1400")
    write_questions(home, [{"id": "q1", "record": one.record_id}, {"id": "q2", "record": two.record_id}])
    res = run(home, "acceptance-report")
    assert res.exit_code == 0 and f"added {two.record_id}" in res.output
    after = path.read_text(encoding="utf-8")
    assert after.startswith(ticked)                              # byte for byte, then the new section
    assert acc.parse_ticks(after) == {one.record_id: (1, 8), two.record_id: (0, 8)}
    assert (stage(home) / acc.WALKTHROUGH_FILE).read_text(encoding="utf-8") == "my own words\n- [x] done\n"
    assert json.loads((stage(home) / acc.REPORT_JSON).read_text(encoding="utf-8"))["walkthrough"] == {}
    run(home, "acceptance-report")
    assert path.read_text(encoding="utf-8") == after             # a third run changes nothing


def test_an_unknown_record_in_questions_is_named_not_fatal(home, world):
    run(home, "acceptance-report")
    write_questions(home, [{"id": "q1", "record": "tr_nope"}])
    res = run(home, "acceptance-report")
    assert res.exit_code == 0 and "q1: no record tr_nope" in res.output


# --- the report's figures -----------------------------------------------------------------------

def test_the_questions_pass_needs_five_ticked_records_a_clean_cross_check_and_a_codex_one(world, home):
    lib, w, did, ate = world
    qs, text, cross = [], acc.CHECKLIST_HEAD, {}
    for i in range(5):
        client = "codex" if i == 4 else "claude"
        _, rec = answered(lib, did, f"pass-{i}-1400", client=client)
        qs.append(acc.Question(f"q{i + 1}", f"question {i}", client, i == 2, rec.record_id))
        text, _ = acc.sync_checklist(text, [(acc.manage.get_record(lib, rec.record_id), f"q{i + 1}")])
        if client == "claude":
            cross[rec.record_id] = {"ok": True, "missing": [], "extra": [], "order": []}
    ticked = text.replace("- [ ]", "- [x]")

    def gather(**kw):
        return acc.gather(lib, home, checklist=kw.get("checklist", ticked), cross_checks=kw.get("cross", cross),
                          questions=qs)
    full = gather()
    assert full["questions_pass"] and full["questions_summary"]["checklists_complete"] == 5
    assert "**pass**: PASS" in acc.render(full)
    assert not gather(checklist=ticked.replace("- [x] the graphs are right", "- [ ] the graphs are right", 1))["questions_pass"]
    one = next(iter(cross))
    assert not gather(cross={**cross, one: {"ok": False, "missing": [["search", "x", None]], "extra": [], "order": []}})["questions_pass"]
    assert not gather(cross={k: v for k, v in cross.items() if k != one})["questions_pass"]
    assert not acc.gather(lib, home, checklist=ticked, cross_checks=cross, questions=qs[:4])["questions_pass"]


def test_the_report_lists_flags_paths_without_answer_and_the_identity_probe(world, home):
    lib, w, did, ate = world
    a = Agent(lib, name="flags-1400")
    a.search()
    a.answer([{"paper_id": ate.paper_id}])                       # never opened: flagged
    b = Agent(lib, conn_id="mcp-bbbb0002", name="gone-1400")
    b.search()
    from seedgraph.records import log
    log.close_connection(lib, b.conn_id, now=b.tick(1))
    run(home, "acceptance-report")
    data = json.loads((stage(home) / acc.REPORT_JSON).read_text(encoding="utf-8"))
    assert list(data["planted"]["flag_texts"]) == [acc.paths.FLAG_NOT_READ]
    assert [r["session"] for r in data["path_without_answer"]] == ["gone-1400"]
    assert {c["connection_id"] for c in data["identity"]["connections"]} == {"mcp-aaaa0001", "mcp-bbbb0002"}
    assert data["identity"]["sessions_by_kind"] == {"named": 2}
    text = (stage(home) / acc.REPORT_FILE).read_text(encoding="utf-8")
    assert "records showing \"path without answer\" now: 1" in text and "open or crashed" in text


# --- the spot-check -----------------------------------------------------------------------------

def test_the_spotcheck_figure_is_the_same_after_a_fix(home, world):
    lib, w, did, ate = world
    sample = create_sample(lib, n=4, seed=7)
    marks = ["right", "partly", "wrong", "right"]
    for item, value in zip(sample.items, marks):
        mark(lib, sample.sample_id, item.claim_id, value)
    first = run(home, "acceptance-report")
    assert first.exit_code == 0 and "marked 4 of 4" in first.output
    d = json.loads((stage(home) / acc.SPOTCHECK_JSON).read_text(encoding="utf-8"))
    assert d["all_marked"] and d["figure"] == {"right": 2, "partly": 1, "wrong": 1} and d["fixed"] == []
    assert d["unchanged_after_fix"] is None and len(d["snapshots"]) == 1 and d["seed"] == 7
    wrong = next(i for i in sample.items if i.claim_id == next(
        s.claim_id for s, m in zip(sample.items, marks) if m == "wrong"))
    paper = did if wrong.paper_id == did.paper_id else ate
    w.edit(paper, wrong.claim_id, 0, claim_text="A corrected claim text.")
    again = run(home, "acceptance-report")
    assert again.exit_code == 0 and "fixed 1" in again.output
    d = json.loads((stage(home) / acc.SPOTCHECK_JSON).read_text(encoding="utf-8"))
    assert [f["claim_id"] for f in d["fixed"]] == [wrong.claim_id]
    assert d["figure"] == {"right": 2, "partly": 1, "wrong": 1}
    assert d["unchanged_after_fix"] is True and [s["n_fixed"] for s in d["snapshots"]] == [0, 1]
    md = (stage(home) / acc.SPOTCHECK_MD).read_text(encoding="utf-8")
    assert "figure unchanged after a fix: yes" in md and "| claim type | model | items |" in md
    run(home, "acceptance-report")                                # unchanged state adds no snapshot
    assert len(json.loads((stage(home) / acc.SPOTCHECK_JSON).read_text(encoding="utf-8"))["snapshots"]) == 2


def test_a_moved_figure_after_a_fix_is_called_out():
    zero = {"right": 0, "partly": 0, "wrong": 0}
    snaps = [{"at": "t1", "n_marked": 4, "n_fixed": 0, "figure": {"right": 2}, "overall": {"right": None}},
             {"at": "t2", "n_marked": 4, "n_fixed": 1, "figure": {"right": 3}, "overall": {"right": None}}]
    assert acc.figure_unchanged_after_fix(snaps) is False
    assert acc.figure_unchanged_after_fix(snaps[:1]) is None
    shown = [dict(s, figure=zero) for s in snaps]
    text = acc.render_spotcheck({
        "sample_id": "sc_x", "seed": 1, "items": 4, "size": 4, "filters": {}, "marked": 4, "figure": zero,
        "fixed": [{}], "unchanged_after_fix": False, "cells": [], "snapshots": shown,
        "accuracy": {"overall": {"summary": "s"}, "by_type": [], "by_model": [], "cells": []}})
    assert "NO (the figure moved)" in text


# --- check-transcript ---------------------------------------------------------------------------

def transcript(tmp_path, rid, *, drop_show=False, extra=False):
    w = Writer()
    w.call("search", {"query": "parallel trends", "session": "xc-1400"}, {"hits": []}, t=1)
    if not drop_show:
        w.call("paper_show", {"paper_id": "lp_did0000000000000001", "session": "xc-1400"}, {"ok": 1}, t=3)
    if extra:
        w.call("search", {"query": "never logged", "session": "xc-1400"}, {"hits": []}, t=4)
    w.call("record_answer", {"question": "q", "answer": "a", "evidence": [], "session": "xc-1400"},
           {"record_id": rid}, t=10)
    path = tmp_path / ("t.jsonl" if not (drop_show or extra) else "bad.jsonl")
    w.write(path)
    return path


def test_check_transcript_passes_fails_and_saves_the_result(home, world, tmp_path):
    lib, w, did, ate = world
    _, rec = answered(lib, did, "xc-1400")
    ok = run(home, "check-transcript", str(transcript(tmp_path, rec.record_id)), "--record", rec.record_id)
    assert ok.exit_code == 0, ok.output
    assert f"{rec.record_id}: OK" in ok.output
    saved = json.loads((stage(home) / acc.CROSSCHECK_FILE).read_text(encoding="utf-8"))
    assert saved[rec.record_id]["ok"] is True
    bad = run(home, "check-transcript", str(transcript(tmp_path, rec.record_id, drop_show=True)),
              "--record", rec.record_id)
    assert bad.exit_code == 1 and "extra in the log: paper_show" in bad.output
    saved = json.loads((stage(home) / acc.CROSSCHECK_FILE).read_text(encoding="utf-8"))
    assert saved[rec.record_id]["ok"] is False
    worse = run(home, "check-transcript", str(transcript(tmp_path, rec.record_id, extra=True)),
                "--record", rec.record_id, "--no-save", "--json")
    assert worse.exit_code == 1 and json.loads(worse.output)[rec.record_id]["missing"]
    assert json.loads((stage(home) / acc.CROSSCHECK_FILE).read_text(encoding="utf-8"))[rec.record_id]["ok"] is False


def test_check_transcript_without_a_record_uses_the_time_window(home, world, tmp_path):
    lib, w, did, ate = world
    a = Agent(lib, name="win-1400")
    a.search("alpha query")
    a.paper_show(did)
    at = [datetime.fromisoformat(e["at"]) for e in acc.load_events(lib.conn)]
    wr = Writer()
    wr.call("search", {"query": "alpha query", "session": "win-1400"}, {"hits": []}, t=0)
    wr.call("paper_show", {"paper_id": did.paper_id, "session": "win-1400"}, {"ok": 1}, t=1)
    shift = at[0] - T0                                           # restamp the transcript onto the events' clock
    lines = []
    for row in wr.lines:
        if "timestamp" in row:
            row = dict(row, timestamp=(datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) + shift).isoformat())
        lines.append(json.dumps(row))
    path = tmp_path / "w.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = run(home, "check-transcript", str(path))
    assert res.exit_code == 0, res.output
    assert "window: OK: transcript 2 calls, log 2 events" in res.output


def test_check_transcript_refuses_a_missing_file_and_an_unknown_record(home, world, tmp_path):
    assert run(home, "check-transcript", str(tmp_path / "none.jsonl")).exit_code == 3
    lib, w, did, ate = world
    _, rec = answered(lib, did, "xc-1400")
    res = run(home, "check-transcript", str(transcript(tmp_path, rec.record_id)), "--record", "tr_nope")
    assert res.exit_code == 3 and "tr_nope" in res.output
